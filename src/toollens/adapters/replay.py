"""Score predictions you already have (e.g. from your own agent's logs) without calling any model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toollens.adapters.base import ModelAdapter
from toollens.models import ServerSnapshot, Task


class ReplayAdapter(ModelAdapter):
    """Reads JSONL lines of ``{"task_id": ..., "tool": ... | null, "arguments": {...}}``."""

    concurrency = 64

    def __init__(self, path: str | Path, label: str | None = None) -> None:
        self.name = label or f"replay:{Path(path).name}"
        self._by_id: dict[str, dict[str, Any]] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self._by_id[row["task_id"]] = row

    async def _choose(self, snapshot: ServerSnapshot, task: Task) -> tuple[str | None, dict[str, Any], Any]:
        row = self._by_id.get(task.id)
        if row is None:
            raise KeyError(f"no recorded prediction for task {task.id!r}")
        return row.get("tool"), row.get("arguments") or {}, {"replayed": True}
