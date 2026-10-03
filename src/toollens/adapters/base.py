"""The interface every model adapter implements."""

from __future__ import annotations

import abc
import time
from typing import Any

from toollens.models import Prediction, ServerSnapshot, Task

SYSTEM_PROMPT = (
    "You are an assistant connected to a set of tools. For the user's request, call the single most "
    "appropriate tool with well-formed arguments. If none of the tools can help with the request, do not "
    "call any tool; reply briefly in text instead. Do not ask clarifying questions; make reasonable "
    "assumptions for missing details."
)


class ModelAdapter(abc.ABC):
    """Decides which tool (if any) a model would call for a task."""

    #: Human-readable identifier used in reports, e.g. ``ollama:qwen3:4b`` or ``bm25``.
    name: str

    #: How many tasks may run concurrently against this adapter.
    concurrency: int = 1

    #: False for routers that only pick a tool (argument checks are then reported as n/a).
    produces_arguments: bool = True

    async def predict(self, snapshot: ServerSnapshot, task: Task) -> Prediction:
        start = time.perf_counter()
        try:
            tool, args, raw = await self._choose(snapshot, task)
            error = None
            if tool is not None and snapshot.tool(tool) is None:
                error = f"model called unknown tool {tool!r}"
        except Exception as exc:  # adapters must never take a whole run down
            tool, args, raw, error = None, {}, None, f"{type(exc).__name__}: {exc}"
        return Prediction(
            task_id=task.id,
            tool=tool,
            arguments=args or {},
            error=error,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            raw=raw,
        )

    @abc.abstractmethod
    async def _choose(self, snapshot: ServerSnapshot, task: Task) -> tuple[str | None, dict[str, Any], Any]:
        """Return (tool name or None, arguments, raw response for debugging)."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
