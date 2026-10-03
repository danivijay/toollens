"""The bundled synthetic dataset is part of the product: hold it to the same bar as code."""

from collections import Counter

from toollens.dataset import check_tasks
from toollens.models import Task, TaskKind


def test_workspace_dataset_is_clean(snapshot_v1, snapshot_v2, workspace_tasks):
    assert check_tasks(snapshot_v1, workspace_tasks, min_per_tool=8) == []
    assert check_tasks(snapshot_v2, workspace_tasks, min_per_tool=8) == []


def test_workspace_dataset_balance(workspace_tasks):
    kinds = Counter(t.kind for t in workspace_tasks)
    assert kinds[TaskKind.DIRECT] >= 60 and kinds[TaskKind.NEAR_MISS] >= 25 and kinds[TaskKind.NO_TOOL] >= 10
    per_tool = Counter(t.expected_tool for t in workspace_tasks if t.expected_tool)
    assert len(per_tool) == 10 and min(per_tool.values()) == max(per_tool.values())


def test_v1_and_v2_expose_identical_contracts(snapshot_v1, snapshot_v2):
    """The demo only changes descriptions; names and argument names must match exactly."""
    assert snapshot_v1.tool_names == snapshot_v2.tool_names
    for a in snapshot_v1.tools:
        b = snapshot_v2.tool(a.name)
        assert set(a.properties) == set(b.properties)
        assert a.required == b.required


def test_check_tasks_catches_problems(tiny_snapshot):
    bad = [
        Task(id="a", prompt="use get_weather for Pune", expected_tool="get_weather"),
        Task(id="a", prompt="dup id", expected_tool="get_weather"),
        Task(id="b", prompt="USE GET_WEATHER for pune", expected_tool="get_weather"),
        Task(id="c", prompt="x", expected_tool="nope"),
        Task(id="d", prompt="y", expected_tool="get_forecast", expected_args={"zip": 1}),
        Task(id="e", prompt="z", expected_tool="get_forecast", kind=TaskKind.NEAR_MISS),
        Task(id="f", prompt="w", expected_tool="get_weather", kind=TaskKind.NO_TOOL),
        Task(id="g", prompt="v", expected_tool=None, kind=TaskKind.DIRECT),
    ]
    text = "\n".join(check_tasks(tiny_snapshot, bad))
    for fragment in [
        "duplicate task id 'a'",
        "duplicate prompt",
        "label leakage",
        "unknown tool 'nope'",
        "not in get_forecast's schema: ['zip']",
        "no distractor",
        "no_tool task expects",
        "direct task has no expected tool",
        "'convert_currency' has 0 tasks",
    ]:
        assert fragment in text, fragment
