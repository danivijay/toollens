import json

from click.testing import CliRunner

from toollens.cli import main

from .conftest import EXAMPLES

V1 = str(EXAMPLES / "snapshots" / "workspace_v1.json")
V2 = str(EXAMPLES / "snapshots" / "workspace_v2.json")
TASKS = str(EXAMPLES / "datasets" / "workspace.jsonl")


def invoke(*args):
    return CliRunner().invoke(main, list(args), catch_exceptions=False)


def test_version_and_help():
    assert "toollens" in invoke("--version").output
    out = invoke("--help").output
    for cmd in ("inspect", "generate", "validate", "run", "report"):
        assert cmd in out


def test_inspect_text_json_and_strict(tmp_path):
    res = invoke("inspect", "--tools", V1, "--save", str(tmp_path / "s.json"))
    assert res.exit_code == 0 and "find_free_slots" in res.output and (tmp_path / "s.json").exists()
    data = json.loads(invoke("inspect", "--tools", V1, "--json").output)
    assert len(data["snapshot"]["tools"]) == 10 and data["lint"]["findings"]
    assert invoke("inspect", "--tools", V1, "--strict").exit_code == 1  # v1 has warnings
    assert invoke("inspect", "--tools", V2).exit_code == 0  # v2 has no errors
    v2 = json.loads(invoke("inspect", "--tools", V2, "--json").output)["lint"]["findings"]
    assert sum(f["severity"] == "warning" for f in v2) <= 2  # two look-alike pairs still lack "not for" hints


def test_inspect_requires_a_source():
    res = CliRunner().invoke(main, ["inspect"])
    assert res.exit_code == 2 and "--server" in res.output


def test_generate_template_then_validate(tmp_path):
    out = tmp_path / "tasks.jsonl"
    res = invoke(
        "generate",
        "--tools",
        V2,
        "--generator",
        "template",
        "--per-tool",
        "3",
        "--no-tool",
        "4",
        "-o",
        str(out),
    )
    assert res.exit_code == 0 and "Wrote 34 tasks" in res.output
    assert invoke("validate", "--tools", V2, "-t", str(out), "--min-per-tool", "3").exit_code == 0
    assert invoke("validate", "--tools", V2, "-t", str(out), "--min-per-tool", "5").exit_code == 1


def test_generate_llm_without_key_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLLENS_ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    res = CliRunner().invoke(main, ["generate", "--tools", V2, "-o", str(tmp_path / "x.jsonl")])
    assert res.exit_code == 2 and "Generation failed" in res.output


def test_run_report_compare_and_fail_under(tmp_path):
    runs = tmp_path / "runs"
    res = invoke("run", "--tools", V1, "-t", TASKS, "-o", str(runs), "--html", str(tmp_path / "v1.html"))
    assert res.exit_code == 0 and "Scoreboard" in res.output and (tmp_path / "v1.html").exists()
    res = invoke("run", "--tools", V2, "-t", TASKS, "-o", str(runs), "--no-matrix", "--fail-under", "0.99")
    assert res.exit_code == 1 and "below --fail-under" in res.output
    v1_dir, v2_dir = sorted(p for p in runs.iterdir())
    if "v2" in v1_dir.name:
        v1_dir, v2_dir = v2_dir, v1_dir
    res = invoke("report", str(v1_dir), str(v2_dir), "--no-matrix", "--html", str(tmp_path / "cmp.html"))
    assert res.exit_code == 0 and "workspace-v1 → workspace-v2" in res.output and "pts" in res.output


def test_run_rejects_bad_model_spec(tmp_path):
    res = CliRunner().invoke(main, ["run", "--tools", V1, "-t", TASKS, "-m", "nonsense", "-o", str(tmp_path)])
    assert res.exit_code == 2 and "provider:model" in res.output


def test_shared_credentials_are_refused_across_providers(tmp_path):
    res = CliRunner().invoke(
        main,
        ["run", "--tools", V1, "-t", TASKS, "-m", "anthropic:a", "-m", "openrouter:b", "--api-key", "sk-x"],
    )
    assert res.exit_code == 2 and "every provider" in res.output


def test_bad_inputs_exit_2_without_tracebacks(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    for args in (
        ["run", "--tools", V1, "-t", str(bad), "-o", str(tmp_path)],
        ["validate", "--tools", V1, "-t", str(bad)],
        ["run", "--tools", V1, "-t", str(empty), "-o", str(tmp_path)],
        ["report", str(tmp_path)],
    ):
        res = CliRunner().invoke(main, args)
        assert res.exit_code == 2, args
        assert "Traceback" not in res.output


def test_server_without_tools_warns(tmp_path):
    snap = tmp_path / "empty.json"
    snap.write_text('{"server": "x", "tools": []}')
    res = CliRunner().invoke(main, ["inspect", "--tools", str(snap)])
    assert res.exit_code == 0 and "no tools" in res.output
