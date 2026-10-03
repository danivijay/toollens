"""LLM-backed synthetic task generator.

A strong model sees the *whole* tool list at once and is asked for realistic user requests per tool,
including near-miss requests aimed at each tool's look-alike siblings (found by the static linter)
and requests that no tool serves. Output is validated, de-duplicated and written as JSONL so the
dataset can be reviewed, edited and versioned before any eval runs.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from toollens.lint import similar_pairs
from toollens.models import ServerSnapshot, Task, TaskKind

GENERATOR_SYSTEM = """You write evaluation data for AI agents that use tools.
You will be shown every tool an MCP server exposes. Write requests that real users would type:
varied wording, casual or terse, sometimes with typos-free shorthand, never quoting tool names or
copying phrases from tool descriptions. Each request must have exactly one correct tool (or none).
Respond with a single JSON object and nothing else."""

GENERATOR_USER = """TOOLS (JSON):
{tools}

LOOK-ALIKE PAIRS (tools a model may confuse):
{pairs}

Write tasks for the tool `{tool}`:
- {direct} "direct" tasks that clearly need `{tool}`.
- {near_miss} "near_miss" tasks that need `{tool}` but are worded so a look-alike sibling is tempting.
  Set "distractor" to that sibling's name.
For every task, give "expected_args": only the arguments the request states explicitly, using the
tool's parameter names and valid values.

Return: {{"tasks": [{{"prompt": str, "kind": "direct"|"near_miss", "distractor": str|null,
"expected_args": object}}]}}"""

NO_TOOL_USER = """TOOLS (JSON):
{tools}

Write {n} requests a user of this assistant might plausibly make that NONE of these tools can serve
(adjacent to the server's domain, so they are tempting, but genuinely out of scope).
Return: {{"tasks": [{{"prompt": str}}]}}"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model did not return a JSON object")
    return json.loads(text[start : end + 1])


class ChatClient:
    """Minimal JSON-in, JSON-out chat client for Anthropic or any OpenAI-compatible endpoint."""

    def __init__(
        self,
        spec: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300.0,
    ) -> None:
        from toollens.adapters.llm import API_KEY_ENV, LOCAL_PRESETS

        provider, _, model = spec.partition(":")
        if not model:
            raise ValueError(f"generator model {spec!r} must look like provider:model")
        self.provider, self.model = provider, model
        key = api_key or os.environ.get(API_KEY_ENV.get(provider, ""), "") or None
        if provider == "anthropic":
            url = base_url or os.environ.get("TOOLLENS_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            headers = {"anthropic-version": "2023-06-01"}
            if key:
                headers["x-api-key"] = key
        else:
            url = base_url or LOCAL_PRESETS.get(provider)
            if not url:
                raise ValueError(f"unknown provider {provider!r} (or pass --base-url)")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.AsyncClient(
            base_url=url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    async def complete_json(self, system: str, user: str, temperature: float = 0.8) -> dict[str, Any]:
        if self.provider == "anthropic":
            resp = await self._http.post(
                "/v1/messages",
                json={
                    "model": self.model,
                    "max_tokens": 4096,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            text = "".join(
                b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text"
            )
        else:
            resp = await self._http.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
        return _extract_json(text)

    async def aclose(self) -> None:
        await self._http.aclose()


def _tools_json(snapshot: ServerSnapshot) -> str:
    return json.dumps(
        [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in snapshot.tools
        ],
        indent=1,
        ensure_ascii=False,
    )


def _dedupe_key(prompt: str) -> str:
    return re.sub(r"\W+", " ", prompt.lower()).strip()


async def generate_llm_tasks(
    snapshot: ServerSnapshot,
    client: ChatClient,
    *,
    direct: int = 6,
    near_miss: int = 4,
    no_tool: int = 8,
    progress=None,
) -> list[Task]:
    tools_json = _tools_json(snapshot)
    risky = [p for p in similar_pairs(snapshot.tools) if p.risk >= 0.2][:15]
    pairs_text = "\n".join(f"- {p.a} <-> {p.b}" for p in risky) or "- (none detected)"

    tasks: list[Task] = []
    seen: set[str] = set()

    def add(task: Task) -> None:
        key = _dedupe_key(task.prompt)
        if key and key not in seen:
            seen.add(key)
            tasks.append(task)

    for idx, tool in enumerate(snapshot.tools, start=1):
        data = await client.complete_json(
            GENERATOR_SYSTEM,
            GENERATOR_USER.format(
                tools=tools_json, pairs=pairs_text, tool=tool.name, direct=direct, near_miss=near_miss
            ),
        )
        n = 0
        for row in data.get("tasks", []):
            if not isinstance(row, dict):
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            if re.search(rf"\b{re.escape(tool.name)}\b", prompt):
                continue  # label leakage: the prompt names the answer
            kind = TaskKind.NEAR_MISS if row.get("kind") == "near_miss" else TaskKind.DIRECT
            distractor = row.get("distractor")
            if distractor and (distractor == tool.name or not snapshot.tool(distractor)):
                distractor = None
            if kind == TaskKind.NEAR_MISS and not distractor:
                kind = TaskKind.DIRECT  # a near miss without a valid sibling is just a direct task
            args = row.get("expected_args") or {}
            if not isinstance(args, dict):
                args = {}
            args = {k: v for k, v in args.items() if k in tool.properties}
            n += 1
            add(
                Task(
                    id=f"{tool.name}-{n:02d}",
                    prompt=prompt,
                    expected_tool=tool.name,
                    expected_args=args,
                    kind=kind,
                    distractor=distractor if kind == TaskKind.NEAR_MISS else None,
                    notes=f"generated by {client.provider}:{client.model}",
                )
            )
        if progress:
            progress(tool.name, idx, len(snapshot.tools))

    if no_tool:
        data = await client.complete_json(GENERATOR_SYSTEM, NO_TOOL_USER.format(tools=tools_json, n=no_tool))
        rows = [r for r in data.get("tasks", []) if isinstance(r, dict)]
        for i, row in enumerate(rows[:no_tool], start=1):
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                add(
                    Task(
                        id=f"none-{i:02d}",
                        prompt=prompt,
                        expected_tool=None,
                        kind=TaskKind.NO_TOOL,
                        notes=f"generated by {client.provider}:{client.model}",
                    )
                )
    return tasks
