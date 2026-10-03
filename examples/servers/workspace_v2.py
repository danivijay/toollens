"""Demo MCP server, v2: the same ten tools and schemas as v1, with descriptions rewritten after a ToolLens run.

Nothing about behaviour, names or arguments changed. Each description now says what the tool does,
when to use it, and which sibling to use instead, and every parameter is documented.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.dirname(__file__))

import workspace_backend as b  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from pydantic import Field  # noqa: E402

mcp = MCPServer("workspace-v2")

EventId = Annotated[str, Field(description="Event id such as 'evt_1040'.")]
IsoDate = Annotated[str, Field(description="Calendar date, YYYY-MM-DD.")]


@mcp.tool(
    description=(
        "List every calendar event between two dates, e.g. 'what's on my calendar this week', 'my schedule "
        "tomorrow', 'meetings on Friday'. Use when the user gives a day or date range, not keywords. "
        "Not for keyword lookup (use search_events) or open time (use find_free_slots)."
    )
)
def list_events(start_date: IsoDate, end_date: IsoDate) -> list[dict]:
    return b.list_events(start_date, end_date)


@mcp.tool(
    description=(
        "Find events by keyword in the title or topic, e.g. 'when is the billing design review', 'find the "
        "offsite meeting'. Use when the user names a topic but not a date. Not for listing a day's "
        "schedule (use list_events) or details of a known event id (use get_event)."
    )
)
def search_events(
    query: Annotated[str, Field(description="Words that appear in the event title, e.g. 'design review'.")],
    limit: Annotated[int, Field(description="Maximum events to return.")] = 10,
) -> list[dict]:
    return b.search_events(query, limit)


@mcp.tool(
    description=(
        "Fetch full details (time, attendees, room, status) of ONE event when its id like evt_1040 is known. "
        "Not for finding events by keyword (use search_events) or by date (use list_events)."
    )
)
def get_event(event_id: EventId) -> dict:
    return b.get_event(event_id)


@mcp.tool(
    description=(
        "Schedule a brand-new meeting at a chosen time, optionally inviting people. "
        "Not for moving or renaming an existing event (use update_event)."
    )
)
def create_event(
    title: Annotated[str, Field(description="Meeting title.")],
    start: Annotated[str, Field(description="Start time, ISO 8601 local, e.g. 2026-10-14T15:00.")],
    duration_minutes: Annotated[int, Field(description="Length in minutes.")] = 30,
    attendees: Annotated[list[str] | None, Field(description="Attendee email addresses.")] = None,
) -> dict:
    return b.create_event(title, start, duration_minutes, attendees)


@mcp.tool(
    description=(
        "Change an existing event: reschedule or move it to a new time, rename it, or make it longer or "
        "shorter. Not for cancelling (use cancel_event) or adding people (use invite_attendees)."
    )
)
def update_event(
    event_id: EventId,
    title: Annotated[str | None, Field(description="New title.")] = None,
    start: Annotated[str | None, Field(description="New start time, ISO 8601 local.")] = None,
    duration_minutes: Annotated[int | None, Field(description="New length in minutes.")] = None,
) -> dict:
    return b.update_event(event_id, title, start, duration_minutes)


@mcp.tool(
    description=(
        "Cancel, call off or delete a meeting so it no longer happens; attendees are told by default. "
        "Not for moving it to another time (use update_event)."
    )
)
def cancel_event(
    event_id: EventId,
    notify_attendees: Annotated[bool, Field(description="Email attendees about the cancellation.")] = True,
) -> dict:
    return b.cancel_event(event_id, notify_attendees)


@mcp.tool(
    description=(
        "Find open time when everyone given is free on a date: 'when are Priya and I both free', 'find a "
        "30-minute gap with Lena on Thursday'. Use before create_event. Not for listing booked events "
        "(use list_events)."
    )
)
def find_free_slots(
    attendees: Annotated[list[str], Field(description="Email addresses of everyone who must be free.")],
    date: IsoDate,
    duration_minutes: Annotated[int, Field(description="Length of the free gap needed, in minutes.")] = 30,
) -> list[str]:
    return b.find_free_slots(attendees, date, duration_minutes)


@mcp.tool(
    description=(
        "Add guests to a meeting that already exists and send them the invite. "
        "Not for scheduling a new meeting (use create_event)."
    )
)
def invite_attendees(
    event_id: EventId,
    emails: Annotated[list[str], Field(description="Email addresses to add.")],
) -> dict:
    return b.invite_attendees(event_id, emails)


@mcp.tool(
    description=(
        "Look up a colleague in the company directory by name to get their email address. "
        "Not for searching meetings (use search_events)."
    )
)
def find_contact(
    name: Annotated[str, Field(description="Full or partial person name, e.g. 'Lena'.")],
) -> list[dict]:
    return b.find_contact(name)


@mcp.tool(
    description=(
        "Reserve a physical meeting room (conference room) for an existing event, sized to headcount."
    )
)
def book_room(
    event_id: EventId,
    capacity: Annotated[int, Field(description="Number of people the room must seat.")],
) -> dict:
    return b.book_room(event_id, capacity)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--http":  # python workspace_vN.py --http 8765
        mcp.run("streamable-http", port=int(sys.argv[2]))
    else:
        mcp.run()
