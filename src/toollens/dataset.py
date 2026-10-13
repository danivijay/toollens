"""Quality checks for a task set. Synthetic data is only useful if it is trustworthy."""

from __future__ import annotations

import re
from collections import Counter

from toollens.models import ServerSnapshot, Task, TaskKind


def check_tasks(snapshot: ServerSnapshot, tasks: list[Task], min_per_tool: int = 1) -> list[str]:
    """Return human-readable problems. An empty list means the set is usable."""
    problems: list[str] = []
    ids = Counter(t.id for t in tasks)
    problems += [f"duplicate task id {i!r}" for i, n in ids.items() if n > 1]

    prompts = Counter(re.sub(r"\W+", " ", t.prompt.lower()).strip() for t in tasks)
    problems += [f"duplicate prompt {p!r}" for p, n in prompts.items() if n > 1]

    per_tool = Counter(t.expected_tool for t in tasks if t.expected_tool)
    for tool in snapshot.tools:
        if per_tool.get(tool.name, 0) < min_per_tool:
            problems.append(f"tool {tool.name!r} has {per_tool.get(tool.name, 0)} tasks (< {min_per_tool})")

    for t in tasks:
        if not t.prompt.strip():
            problems.append(f"{t.id}: empty prompt")
        if t.kind == TaskKind.NO_TOOL and t.expected_tool is not None:
            problems.append(f"{t.id}: no_tool task expects tool {t.expected_tool!r}")
        if t.kind != TaskKind.NO_TOOL and t.expected_tool is None:
            problems.append(f"{t.id}: {t.kind.value} task has no expected tool")
        if t.expected_tool is None:
            if t.expected_args:
                problems.append(f"{t.id}: no_tool task has expected_args")
            continue
        spec = snapshot.tool(t.expected_tool)
        if spec is None:
            problems.append(f"{t.id}: expects unknown tool {t.expected_tool!r}")
            continue
        unknown = [k for k in t.expected_args if k not in spec.properties]
        if unknown:
            problems.append(f"{t.id}: expected_args not in {spec.name}'s schema: {unknown}")
        if t.kind == TaskKind.NEAR_MISS:
            if not t.distractor:
                problems.append(f"{t.id}: near_miss task has no distractor")
            elif t.distractor == t.expected_tool:
                problems.append(f"{t.id}: distractor equals the expected tool")
            elif snapshot.tool(t.distractor) is None:
                problems.append(f"{t.id}: distractor {t.distractor!r} is not a tool on this server")
        # Leakage: a prompt that names the expected tool verbatim measures string matching, not understanding.
        # Case doesn't make it any less of a giveaway ("Use GET_WEATHER" still names the tool).
        if re.search(rf"\b{re.escape(spec.name)}\b", t.prompt, re.IGNORECASE):
            problems.append(f"{t.id}: prompt names its own expected tool {spec.name!r} (label leakage)")
    return problems
