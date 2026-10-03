import json

import anyio
import pytest
from rich.console import Console

from toollens.adapters import BM25Adapter, ReplayAdapter
from toollens.adapters.base import ModelAdapter
from toollens.models import Task
from toollens.report.html import render_html
from toollens.report.terminal import print_run
from toollens.runner import load_run, run_eval, save_run


class SlowCounter(ModelAdapter):
    """Records peak concurrency so we can check the limiter is honoured."""

    def __init__(self, concurrency):
        self.name = "slow"
        self.concurrency = concurrency
        self.active = self.peak = 0

    async def _choose(self, snapshot, task):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await anyio.sleep(0.01)
        self.active -= 1
        return task.expected_tool, {}, None


def _replay(tmp_path, tasks, override=None):
    override = override or {}
    rows = [
        {"task_id": t.id, "tool": override.get(t.id, t.expected_tool), "arguments": t.expected_args}
        for t in tasks
    ]
    path = tmp_path / "preds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return ReplayAdapter(path, label="replay")


def test_run_eval_multiple_models_and_roundtrip(tmp_path, tiny_snapshot, tiny_tasks):
    perfect = _replay(tmp_path, tiny_tasks)
    run = anyio.run(
        lambda: run_eval(tiny_snapshot, tiny_tasks, [perfect, BM25Adapter()], tasks_file="t.jsonl")
    )
    assert [m.model for m in run.models] == ["replay", "bm25"]
    assert run.model("replay").metrics.accuracy == 1.0
    assert run.model("replay").metrics.args_match_rate == 1.0
    assert run.run_id.endswith("-tiny") and run.lint.tool_tokens

    path = save_run(run, tmp_path / "runs")
    assert (path / "run.json").exists() and (path / "results-replay.jsonl").exists()
    again = load_run(path)
    assert again.model("bm25").metrics == run.model("bm25").metrics
    with pytest.raises(KeyError):
        again.model("nope")


def test_run_eval_rejects_tasks_for_unknown_tools(tiny_snapshot):
    tasks = [Task(id="x", prompt="?", expected_tool="ghost")]
    with pytest.raises(ValueError, match="ghost"):
        anyio.run(lambda: run_eval(tiny_snapshot, tasks, [BM25Adapter()]))


def test_run_eval_rejects_empty_task_sets_and_duplicate_models(tiny_snapshot, tiny_tasks):
    with pytest.raises(ValueError, match="empty"):
        anyio.run(lambda: run_eval(tiny_snapshot, [], [BM25Adapter()]))
    with pytest.raises(ValueError, match="duplicated"):
        anyio.run(lambda: run_eval(tiny_snapshot, tiny_tasks, [BM25Adapter(), BM25Adapter()]))


@pytest.mark.parametrize("concurrency", [1, 3])
def test_concurrency_limit_is_respected(tiny_snapshot, concurrency):
    tasks = [Task(id=f"t{i}", prompt="x", expected_tool="get_weather") for i in range(9)]
    adapter = SlowCounter(concurrency)
    progress = []
    run = anyio.run(
        lambda: run_eval(tiny_snapshot, tasks, [adapter], progress=lambda m, d, n: progress.append(d))
    )
    assert adapter.peak == concurrency
    assert sorted(progress) == list(range(1, 10))
    assert run.models[0].metrics.accuracy == 1.0


def test_reports_render(tmp_path, tiny_snapshot, tiny_tasks):
    lured = _replay(tmp_path, tiny_tasks, override={"f1": "get_weather", "n1": "get_weather"})
    run = anyio.run(lambda: run_eval(tiny_snapshot, tiny_tasks, [lured]))

    page = render_html([run, run])
    assert page.startswith("<!doctype html>") and page.count("<svg") == 2
    assert (
        "get_forecast</code> ⇄ <code>get_weather" in page or "get_weather</code> ⇄ <code>get_forecast" in page
    )
    assert "Will it rain in Kochi" in page  # example prompt surfaced
    assert "prefers-color-scheme:dark" in page and "http" not in page.split("<body>")[0].replace(
        "http-equiv", ""
    )

    console = Console(record=True, width=140)
    print_run(console, [run])
    text = console.export_text()
    assert "Scoreboard" in text and "get_forecast ⇄ get_weather" in text.replace(
        "get_weather ⇄ get_forecast", "get_forecast ⇄ get_weather"
    )
    assert "1 false calls" in text
