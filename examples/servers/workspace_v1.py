"""Demo MCP server, v1: ten calendar tools with the terse descriptions most servers ship with.

Run it:   python examples/servers/workspace_v1.py         (stdio)
Evaluate: toollens run --server "python examples/servers/workspace_v1.py" \
              --tasks examples/datasets/workspace.jsonl --model bm25
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import workspace_backend as b  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

mcp = MCPServer("workspace-v1")


@mcp.tool(description="Get events.")
def list_events(start_date: str, end_date: str) -> list[dict]:
    return b.list_events(start_date, end_date)


@mcp.tool(description="Search events.")
def search_events(query: str, limit: int = 10) -> list[dict]:
    return b.search_events(query, limit)


@mcp.tool(description="Get an event.")
def get_event(event_id: str) -> dict:
    return b.get_event(event_id)


@mcp.tool(description="Create a new event on the calendar.")
def create_event(
    title: str, start: str, duration_minutes: int = 30, attendees: list[str] | None = None
) -> dict:
    return b.create_event(title, start, duration_minutes, attendees)


@mcp.tool(description="Update an event.")
def update_event(
    event_id: str, title: str | None = None, start: str | None = None, duration_minutes: int | None = None
) -> dict:
    return b.update_event(event_id, title, start, duration_minutes)


@mcp.tool(description="Set event status.")
def cancel_event(event_id: str, notify_attendees: bool = True) -> dict:
    return b.cancel_event(event_id, notify_attendees)


@mcp.tool(description="Check calendar availability for events.")
def find_free_slots(attendees: list[str], date: str, duration_minutes: int = 30) -> list[str]:
    return b.find_free_slots(attendees, date, duration_minutes)


@mcp.tool(description="Add people to an event.")
def invite_attendees(event_id: str, emails: list[str]) -> dict:
    return b.invite_attendees(event_id, emails)


@mcp.tool(description="Search people.")
def find_contact(name: str) -> list[dict]:
    return b.find_contact(name)


@mcp.tool(description="Book for an event.")
def book_room(event_id: str, capacity: int) -> dict:
    return b.book_room(event_id, capacity)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--http":  # python workspace_vN.py --http 8765
        mcp.run("streamable-http", port=int(sys.argv[2]))
    else:
        mcp.run()
