"""Connect to an MCP server and capture its tool surface."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from toollens.models import ServerSnapshot, ToolSpec


def parse_server_spec(spec: str) -> str | StdioServerParameters:
    """Turn a user-supplied server string into something `mcp.Client` can connect to.

    * ``http(s)://...``          -> Streamable HTTP URL
    * anything else              -> a command line, launched over stdio
    """
    spec = spec.strip()
    if spec.startswith(("http://", "https://")):
        return spec
    argv = shlex.split(spec)
    if not argv:
        raise ValueError("empty server command")
    return StdioServerParameters(command=argv[0], args=argv[1:], env=dict(os.environ))


async def discover(spec: str, timeout: float = 30.0) -> ServerSnapshot:
    """Connect, page through tools/list, and return a snapshot."""
    target = parse_server_spec(spec)
    async with Client(target, read_timeout_seconds=timeout) as client:
        tools: list[ToolSpec] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            for t in page.tools:
                tools.append(
                    ToolSpec(
                        name=t.name,
                        description=(t.description or "").strip(),
                        input_schema=dict(t.input_schema or {"type": "object", "properties": {}}),
                    )
                )
            cursor = page.next_cursor
            if not cursor:
                break
        info = client.server_info
        return ServerSnapshot(
            server=spec,
            server_name=info.name if info else None,
            protocol_version=client.protocol_version,
            tools=tools,
        )


async def load_or_discover(server: str | None, tools_file: str | Path | None) -> ServerSnapshot:
    if tools_file:
        return ServerSnapshot.load(tools_file)
    if not server:
        raise ValueError("pass --server (a command or URL) or --tools (a saved snapshot)")
    return await discover(server)
