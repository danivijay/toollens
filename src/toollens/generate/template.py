"""Offline, deterministic task generator built from tool schemas alone.

It needs no model, so it works in CI and on a plane, but its tasks echo the tools' own words.
Treat it as a smoke test (are tools reachable at all, do arguments validate?) and use the LLM
generator or a hand-written set to measure real confusion.
"""

from __future__ import annotations

import random
from typing import Any

from toollens.models import ServerSnapshot, Task, TaskKind, ToolSpec
from toollens.text import humanize

NO_TOOL_PROMPTS = [
    "What's the capital of Australia?",
    "Write me a two-line poem about autumn.",
    "Explain the difference between TCP and UDP in one paragraph.",
    "How many minutes are there in a week?",
    "Translate 'good morning' into Spanish.",
    "Tell me a fun fact about octopuses.",
    "What is 17% of 240?",
    "Summarise the plot of Romeo and Juliet in one sentence.",
    "Suggest a name for a golden retriever puppy.",
    "Which is heavier, a kilogram of feathers or a kilogram of steel?",
    "Give me three tips for a better night's sleep.",
    "What does the acronym HTTP stand for?",
]

TEMPLATES = [
    "Please {action}{args}.",
    "Can you {action}{args}?",
    "I need you to {action}{args}.",
    "Go ahead and {action}{args}.",
    "Quick one: {action}{args}.",
]

SAMPLE_STRINGS = {
    "email": "priya@example.com",
    "name": "Priya Raman",
    "title": "Quarterly planning",
    "query": "quarterly planning",
    "text": "quarterly planning",
    "id": "abc123",
    "path": "notes/todo.md",
    "url": "https://example.com",
    "date": "2026-10-14",
    "time": "15:00",
    "city": "Bengaluru",
    "currency": "INR",
    "symbol": "INFY",
    "language": "ta",
}


def sample_value(pname: str, schema: dict[str, Any], rng: random.Random) -> Any:
    if "enum" in schema and schema["enum"]:
        return rng.choice(schema["enum"])
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "string")
    if typ == "integer":
        lo = int(schema.get("minimum", 1))
        hi = int(schema.get("maximum", max(lo, 1) + 29))
        return rng.randint(lo, hi)
    if typ == "number":
        return round(rng.uniform(float(schema.get("minimum", 1)), float(schema.get("maximum", 100))), 2)
    if typ == "boolean":
        return True
    if typ == "array":
        item = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
        return [sample_value(pname, item, rng)]
    if typ == "object":
        return {}
    fmt = schema.get("format", "")
    if fmt == "date":
        return "2026-10-14"
    if fmt == "date-time":
        return "2026-10-14T15:00:00+05:30"
    if fmt == "email":
        return SAMPLE_STRINGS["email"]
    lowered = pname.lower()
    for key, value in SAMPLE_STRINGS.items():
        if key in lowered:
            return value
    return f"sample {humanize(pname)}"


def _args_phrase(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts = [
        f"{humanize(k)} {v if not isinstance(v, list) else ', '.join(map(str, v))}" for k, v in args.items()
    ]
    return " with " + " and ".join(parts)


def tasks_for_tool(tool: ToolSpec, per_tool: int, rng: random.Random) -> list[Task]:
    action = humanize(tool.name)
    out: list[Task] = []
    for i in range(per_tool):
        args = {
            p: sample_value(p, s if isinstance(s, dict) else {}, rng)
            for p, s in tool.properties.items()
            if p in tool.required or rng.random() < 0.3
        }
        template = TEMPLATES[i % len(TEMPLATES)]
        out.append(
            Task(
                id=f"{tool.name}-t{i + 1:02d}",
                prompt=template.format(action=action, args=_args_phrase(args)),
                expected_tool=tool.name,
                expected_args={k: v for k, v in args.items() if k in tool.required},
                kind=TaskKind.DIRECT,
                notes="template",
            )
        )
    return out


def generate_template_tasks(
    snapshot: ServerSnapshot, per_tool: int = 5, no_tool: int = 5, seed: int = 7
) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    for tool in snapshot.tools:
        tasks.extend(tasks_for_tool(tool, per_tool, rng))
    for i, prompt in enumerate(rng.sample(NO_TOOL_PROMPTS, k=min(no_tool, len(NO_TOOL_PROMPTS)))):
        tasks.append(
            Task(
                id=f"none-t{i + 1:02d}",
                prompt=prompt,
                expected_tool=None,
                kind=TaskKind.NO_TOOL,
                notes="template",
            )
        )
    return tasks
