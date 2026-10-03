"""Adapters for real language models.

Two wire formats cover almost everything:

* **OpenAI-compatible** ``/v1/chat/completions`` with ``tools``: Ollama, LM Studio, ``mlx_lm.server``,
  vLLM, llama.cpp server, OpenAI, OpenRouter, Groq and friends.
* **Anthropic** ``/v1/messages`` with ``tools``.

Both are plain ``httpx`` calls (no vendor SDKs) so they are easy to mock in tests.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from toollens.adapters.base import SYSTEM_PROMPT, ModelAdapter
from toollens.models import ServerSnapshot, Task, ToolSpec

LOCAL_PRESETS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "mlx": "http://localhost:8080/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm": "http://localhost:8000/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _schema(tool: ToolSpec) -> dict[str, Any]:
    schema = dict(tool.input_schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _parse_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments are not a JSON object")
        return parsed
    return {}


class OpenAICompatAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        label: str | None = None,
        temperature: float | None = 0.0,
        timeout: float = 120.0,
        concurrency: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.name = label or model
        self.temperature = temperature
        self.concurrency = concurrency
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def tools_payload(self, snapshot: ServerSnapshot) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": _schema(t)},
            }
            for t in snapshot.tools
        ]

    async def _choose(self, snapshot: ServerSnapshot, task: Task) -> tuple[str | None, dict[str, Any], Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ],
            "tools": self.tools_payload(snapshot),
            "tool_choice": "auto",
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        resp = await self._http.post("/chat/completions", json=body)
        if resp.status_code == 400 and "temperature" in body and "temperature" in resp.text.lower():
            # Reasoning models (o-series, gpt-5 class) reject a custom temperature: drop it for this run.
            self.temperature = None
            body.pop("temperature")
            resp = await self._http.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            return None, {}, {"content": message.get("content")}
        fn = calls[0].get("function") or {}
        return fn.get("name"), _parse_args(fn.get("arguments")), {"tool_calls": calls}

    async def aclose(self) -> None:
        await self._http.aclose()


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        label: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        concurrency: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.name = label or f"anthropic:{model}"
        self.temperature = temperature
        self.concurrency = concurrency
        headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def tools_payload(self, snapshot: ServerSnapshot) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": _schema(t)} for t in snapshot.tools
        ]

    async def _choose(self, snapshot: ServerSnapshot, task: Task) -> tuple[str | None, dict[str, Any], Any]:
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": self.temperature,
            "system": SYSTEM_PROMPT,
            "tools": self.tools_payload(snapshot),
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": task.prompt}],
        }
        resp = await self._http.post("/v1/messages", json=body)
        resp.raise_for_status()
        data = resp.json()
        for block in data.get("content") or []:
            if block.get("type") == "tool_use":
                return block.get("name"), _parse_args(block.get("input")), {"tool_use": block}
        text = " ".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
        return None, {}, {"content": text}

    async def aclose(self) -> None:
        await self._http.aclose()


def make_adapter(
    spec: str, *, base_url: str | None = None, api_key: str | None = None, concurrency: int | None = None
):
    """Build an adapter from a CLI model spec.

    ``bm25`` · ``ollama:qwen3:4b`` · ``mlx:mlx-community/Qwen3-4B-4bit`` · ``lmstudio:<model>`` ·
    ``openai:gpt-4.1-mini`` · ``openrouter:<model>`` · ``anthropic:claude-sonnet-4-5`` ·
    ``openai-compat:<model>`` (requires ``--base-url``).
    """
    from toollens.adapters.bm25 import BM25Adapter

    if spec.startswith("replay:"):
        from toollens.adapters.replay import ReplayAdapter

        return ReplayAdapter(spec.split(":", 1)[1])

    if spec == "bm25" or spec.startswith("bm25:"):
        _, _, threshold = spec.partition(":")
        return BM25Adapter(min_score=float(threshold), label=spec) if threshold else BM25Adapter()

    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"model spec {spec!r} must look like provider:model (or 'bm25')")

    key = api_key or (os.environ.get(API_KEY_ENV[provider]) if provider in API_KEY_ENV else None)

    if provider == "anthropic":
        if not key:
            raise ValueError("anthropic models need ANTHROPIC_API_KEY (or --api-key)")
        return AnthropicAdapter(
            model,
            api_key=key,
            base_url=base_url or os.environ.get("TOOLLENS_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            label=spec,
            concurrency=concurrency or 4,
        )

    if provider == "openai-compat":
        if not base_url:
            raise ValueError("openai-compat:<model> needs --base-url")
        url = base_url
    elif provider in LOCAL_PRESETS:
        url = base_url or LOCAL_PRESETS[provider]
    else:
        raise ValueError(
            f"unknown provider {provider!r}; use one of: bm25, anthropic, openai-compat, "
            + ", ".join(sorted(LOCAL_PRESETS))
        )
    if provider in ("openai", "openrouter") and not key:
        raise ValueError(f"{provider} models need {API_KEY_ENV[provider]} (or --api-key)")
    remote = provider in ("openai", "openrouter")
    return OpenAICompatAdapter(
        model, url, api_key=key, label=spec, concurrency=concurrency or (4 if remote else 1)
    )
