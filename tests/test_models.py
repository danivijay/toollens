import pytest

from toollens.models import ServerSnapshot, Task, TaskKind, read_tasks, write_tasks


def test_tasks_roundtrip(tmp_path, tiny_tasks):
    path = tmp_path / "t.jsonl"
    write_tasks(tiny_tasks, path)
    again = read_tasks(path)
    assert [t.model_dump() for t in again] == [t.model_dump() for t in tiny_tasks]


def test_read_tasks_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('// a comment\n\n{"id": "a", "prompt": "hi", "expected_tool": null, "kind": "no_tool"}\n')
    [task] = read_tasks(path)
    assert task.kind is TaskKind.NO_TOOL


def test_read_tasks_rejects_duplicate_ids(tmp_path):
    line = '{"id": "a", "prompt": "hi", "expected_tool": null}\n'
    path = tmp_path / "t.jsonl"
    path.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate task ids"):
        read_tasks(path)


def test_snapshot_save_load_and_lookup(tmp_path, tiny_snapshot):
    path = tmp_path / "s.json"
    tiny_snapshot.save(path)
    loaded = ServerSnapshot.load(path)
    assert loaded.tool_names == ["get_weather", "get_forecast", "convert_currency"]
    assert loaded.tool("get_forecast").required == ["city"]
    assert loaded.tool("nope") is None


def test_task_defaults():
    t = Task(id="x", prompt="p", expected_tool="a")
    assert t.kind is TaskKind.DIRECT and t.expected_args == {}
