"""End-to-end tests against the real demo servers, over stdio and Streamable HTTP."""

import socket
import subprocess
import time

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from toollens.adapters import BM25Adapter
from toollens.discovery import discover, parse_server_spec
from toollens.runner import run_eval

from .conftest import EXAMPLES, PYTHON

pytestmark = pytest.mark.integration

V1 = f"{PYTHON} {EXAMPLES / 'servers' / 'workspace_v1.py'}"
V2 = f"{PYTHON} {EXAMPLES / 'servers' / 'workspace_v2.py'}"


def test_parse_server_spec():
    assert parse_server_spec("https://example.com/mcp") == "https://example.com/mcp"
    params = parse_server_spec("npx -y '@modelcontextprotocol/server-everything'")
    assert params.command == "npx" and params.args == ["-y", "@modelcontextprotocol/server-everything"]
    with pytest.raises(ValueError):
        parse_server_spec("   ")


@pytest.mark.parametrize("spec, name", [(V1, "workspace-v1"), (V2, "workspace-v2")])
def test_discover_over_stdio_matches_committed_snapshot(spec, name, snapshot_v1, snapshot_v2):
    snap = anyio.run(discover, spec)
    committed = snapshot_v1 if name.endswith("v1") else snapshot_v2
    assert snap.server_name == name and snap.protocol_version
    assert [t.model_dump() for t in snap.tools] == [t.model_dump() for t in committed.tools], (
        "examples/snapshots is stale: re-run `toollens inspect --save`"
    )


def test_demo_tools_really_work():
    async def go():
        params = StdioServerParameters(command=PYTHON, args=[str(EXAMPLES / "servers" / "workspace_v2.py")])
        async with Client(params) as client:
            res = await client.call_tool("find_contact", {"name": "lena"})
            assert "lena@acme.dev" in str(res.content)
            res = await client.call_tool(
                "find_free_slots", {"attendees": ["priya@acme.dev"], "date": "2026-10-14"}
            )
            assert "10:00" not in str(res.content) and "09:00" in str(res.content)

    anyio.run(go)


def test_better_descriptions_measurably_reduce_confusion(workspace_tasks):
    """The headline claim of the demo, kept honest as a regression test."""

    async def go():
        out = {}
        for label, spec in (("v1", V1), ("v2", V2)):
            snap = await discover(spec)
            run = await run_eval(snap, workspace_tasks, [BM25Adapter()])
            out[label] = run
        return out

    runs = anyio.run(go)
    v1, v2 = (runs[k].models[0].metrics for k in ("v1", "v2"))
    assert v2.accuracy >= v1.accuracy + 0.15
    assert v2.accuracy_by_kind["direct"] > v1.accuracy_by_kind["direct"]
    assert runs["v2"].lint.count("warning") < runs["v1"].lint.count("warning")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_discover_over_streamable_http(snapshot_v2):
    port = _free_port()
    proc = subprocess.Popen(
        [PYTHON, str(EXAMPLES / "servers" / "workspace_v2.py"), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        snap = anyio.run(discover, f"http://127.0.0.1:{port}/mcp")
        assert snap.tool_names == snapshot_v2.tool_names
    finally:
        proc.terminate()
        proc.wait(timeout=10)
