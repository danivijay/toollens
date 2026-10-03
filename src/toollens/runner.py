"""Run a task set against one or more model adapters and persist everything to a run directory."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable
from pathlib import Path

import anyio
from pydantic import BaseModel

from toollens.adapters.base import ModelAdapter
from toollens.lint import LintReport, lint
from toollens.models import Prediction, ServerSnapshot, Task, TaskResult
from toollens.scoring import Metrics, score, summarize

ProgressFn = Callable[[str, int, int], None]


class ModelRun(BaseModel):
    model: str
    produces_arguments: bool
    results: list[TaskResult]
    metrics: Metrics


class Run(BaseModel):
    run_id: str
    created: str
    snapshot: ServerSnapshot
    tasks_file: str | None
    lint: LintReport
    models: list[ModelRun]

    def model(self, name: str) -> ModelRun:
        for m in self.models:
            if m.model == name:
                return m
        raise KeyError(name)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower() or "run"


async def predict_all(
    adapter: ModelAdapter,
    snapshot: ServerSnapshot,
    tasks: list[Task],
    progress: ProgressFn | None = None,
) -> list[Prediction]:
    results: dict[str, Prediction] = {}
    limiter = anyio.CapacityLimiter(max(1, adapter.concurrency))
    done = 0

    async def one(task: Task) -> None:
        nonlocal done
        async with limiter:
            results[task.id] = await adapter.predict(snapshot, task)
        done += 1
        if progress:
            progress(adapter.name, done, len(tasks))

    async with anyio.create_task_group() as tg:
        for t in tasks:
            tg.start_soon(one, t)
    return [results[t.id] for t in tasks]


async def run_eval(
    snapshot: ServerSnapshot,
    tasks: list[Task],
    adapters: list[ModelAdapter],
    *,
    tasks_file: str | None = None,
    progress: ProgressFn | None = None,
) -> Run:
    unknown = sorted(
        {t.expected_tool for t in tasks if t.expected_tool and not snapshot.tool(t.expected_tool)}
    )
    if unknown:
        raise ValueError(f"tasks expect tools the server does not have: {unknown}")
    if not tasks:
        raise ValueError("the task set is empty")
    names = [a.name for a in adapters]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"each model must appear once; duplicated: {dupes}")
    model_runs: list[ModelRun] = []
    for adapter in adapters:
        try:
            preds = await predict_all(adapter, snapshot, tasks, progress)
        finally:
            await adapter.aclose()
        results = score(snapshot, tasks, preds, produces_arguments=adapter.produces_arguments)
        model_runs.append(
            ModelRun(
                model=adapter.name,
                produces_arguments=adapter.produces_arguments,
                results=results,
                metrics=summarize(adapter.name, snapshot, results),
            )
        )
    now = dt.datetime.now(dt.timezone.utc)
    name = snapshot.server_name or "server"
    return Run(
        run_id=f"{now:%Y%m%d-%H%M%S}-{slug(name)}",
        created=now.isoformat(timespec="seconds"),
        snapshot=snapshot,
        tasks_file=tasks_file,
        lint=lint(snapshot),
        models=model_runs,
    )


def save_run(run: Run, out_dir: str | Path) -> Path:
    path = Path(out_dir) / run.run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(run.model_dump_json(indent=1) + "\n", encoding="utf-8")
    for m in run.models:
        lines = [json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in m.results]
        (path / f"results-{slug(m.model)}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_run(path: str | Path) -> Run:
    p = Path(path)
    if p.is_dir():
        p = p / "run.json"
    return Run.model_validate_json(p.read_text(encoding="utf-8"))
