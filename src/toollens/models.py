"""Core data models shared by every ToolLens stage.

The pipeline is deliberately file-based: each stage reads and writes plain JSON/JSONL
so datasets can be versioned in git, hand-edited, and diffed between runs.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

NO_TOOL = "∅ no tool"
"""Label used in matrices for 'the model (correctly or not) called no tool'."""

ERROR = "⚠ error"
"""Label used in matrices when the model call failed or returned an unknown tool."""


class ToolSpec(BaseModel):
    """One tool as advertised by an MCP server's tools/list."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @property
    def properties(self) -> dict[str, Any]:
        return self.input_schema.get("properties", {}) or {}

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []) or [])


class ServerSnapshot(BaseModel):
    """A frozen copy of a server's tool surface, so evals can run without the server."""

    server: str
    server_name: str | None = None
    protocol_version: str | None = None
    tools: list[ToolSpec]

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ServerSnapshot:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class TaskKind(str, Enum):
    DIRECT = "direct"
    """A plain request that one tool clearly serves."""

    NEAR_MISS = "near_miss"
    """A request worded so that a sibling tool looks tempting, but the expected tool is right."""

    NO_TOOL = "no_tool"
    """A request no tool on the server serves. The right behaviour is to call nothing."""


class Task(BaseModel):
    """A single labelled test case."""

    id: str
    prompt: str
    expected_tool: str | None = None
    expected_args: dict[str, Any] = Field(default_factory=dict)
    kind: TaskKind = TaskKind.DIRECT
    distractor: str | None = None
    """For near-miss tasks: the sibling tool the task is designed to lure the model towards."""
    notes: str | None = None


class Prediction(BaseModel):
    """What a model did for one task."""

    task_id: str
    tool: str | None
    arguments: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    raw: Any = None


class TaskResult(BaseModel):
    """A prediction joined with its task and scored."""

    task: Task
    prediction: Prediction
    tool_correct: bool
    args_valid: bool | None = None
    """Do the arguments validate against the tool's input schema? None when not applicable."""
    args_match: bool | None = None
    """Do the arguments contain the task's expected_args? None when not applicable."""

    @property
    def predicted_label(self) -> str:
        if self.prediction.error:
            return ERROR
        return self.prediction.tool or NO_TOOL

    @property
    def expected_label(self) -> str:
        return self.task.expected_tool or NO_TOOL


def read_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            tasks.append(Task.model_validate_json(line))
        except Exception as exc:  # pragma: no cover - message path
            raise ValueError(f"{path}:{lineno}: invalid task: {exc}") from exc
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate task ids: {sorted(dupes)}")
    return tasks


def write_tasks(tasks: list[Task], path: str | Path) -> None:
    lines = [json.dumps(t.model_dump(mode="json", exclude_none=True), ensure_ascii=False) for t in tasks]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
