"""In-memory calendar/contacts backend shared by the two demo servers.

Both demo servers expose the *same* ten tools with the *same* behaviour and argument names.
Only the tool descriptions differ, which isolates the one variable ToolLens is measuring.
"""

from __future__ import annotations

import datetime as dt
import itertools
from typing import Any

CONTACTS = {
    "priya raman": "priya@acme.dev",
    "arjun mehta": "arjun@acme.dev",
    "lena fischer": "lena@acme.dev",
    "tomás alvarez": "tomas@acme.dev",
    "mei chen": "mei@acme.dev",
}

ROOMS = [("Banyan", 4), ("Neem", 8), ("Peepal", 16)]

_ids = itertools.count(1043)
EVENTS: dict[str, dict[str, Any]] = {
    "evt_1040": {
        "id": "evt_1040",
        "title": "Quarterly planning",
        "start": "2026-10-14T10:00",
        "duration_minutes": 60,
        "attendees": ["priya@acme.dev", "arjun@acme.dev"],
        "status": "confirmed",
        "room": None,
    },
    "evt_1041": {
        "id": "evt_1041",
        "title": "Design review: billing page",
        "start": "2026-10-15T15:00",
        "duration_minutes": 45,
        "attendees": ["lena@acme.dev"],
        "status": "confirmed",
        "room": None,
    },
    "evt_1042": {
        "id": "evt_1042",
        "title": "1:1 Mei",
        "start": "2026-10-16T09:30",
        "duration_minutes": 30,
        "attendees": ["mei@acme.dev"],
        "status": "confirmed",
        "room": None,
    },
}


def _parse(start: str) -> dt.datetime:
    return dt.datetime.fromisoformat(start)


def list_events(start_date: str, end_date: str) -> list[dict[str, Any]]:
    lo, hi = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    return [
        e for e in EVENTS.values() if lo <= _parse(e["start"]).date() <= hi and e["status"] != "cancelled"
    ]


def search_events(query: str, limit: int = 10) -> list[dict[str, Any]]:
    q = query.lower()
    return [e for e in EVENTS.values() if q in e["title"].lower()][:limit]


def get_event(event_id: str) -> dict[str, Any]:
    if event_id not in EVENTS:
        raise ValueError(f"no event {event_id}")
    return EVENTS[event_id]


def create_event(
    title: str, start: str, duration_minutes: int = 30, attendees: list[str] | None = None
) -> dict:
    eid = f"evt_{next(_ids)}"
    EVENTS[eid] = {
        "id": eid,
        "title": title,
        "start": start,
        "duration_minutes": duration_minutes,
        "attendees": attendees or [],
        "status": "confirmed",
        "room": None,
    }
    return EVENTS[eid]


def update_event(
    event_id: str, title: str | None = None, start: str | None = None, duration_minutes: int | None = None
) -> dict[str, Any]:
    event = get_event(event_id)
    for key, value in {"title": title, "start": start, "duration_minutes": duration_minutes}.items():
        if value is not None:
            event[key] = value
    return event


def cancel_event(event_id: str, notify_attendees: bool = True) -> dict[str, Any]:
    event = get_event(event_id)
    event["status"] = "cancelled"
    return {"cancelled": event_id, "notified": event["attendees"] if notify_attendees else []}


def find_free_slots(attendees: list[str], date: str, duration_minutes: int = 30) -> list[str]:
    day = dt.date.fromisoformat(date)
    busy = [
        (_parse(e["start"]), _parse(e["start"]) + dt.timedelta(minutes=e["duration_minutes"]))
        for e in EVENTS.values()
        if _parse(e["start"]).date() == day
        and set(e["attendees"]) & set(attendees)
        and e["status"] != "cancelled"
    ]
    slots, t = [], dt.datetime.combine(day, dt.time(9))
    end = dt.datetime.combine(day, dt.time(18))
    step = dt.timedelta(minutes=30)
    need = dt.timedelta(minutes=duration_minutes)
    while t + need <= end:
        if all(not (t < b_end and t + need > b_start) for b_start, b_end in busy):
            slots.append(t.isoformat(timespec="minutes"))
        t += step
    return slots


def invite_attendees(event_id: str, emails: list[str]) -> dict[str, Any]:
    event = get_event(event_id)
    event["attendees"] = sorted(set(event["attendees"]) | set(emails))
    return event


def find_contact(name: str) -> list[dict[str, str]]:
    n = name.lower()
    return [{"name": k.title(), "email": v} for k, v in CONTACTS.items() if n in k]


def book_room(event_id: str, capacity: int) -> dict[str, Any]:
    event = get_event(event_id)
    room = next((r for r, cap in ROOMS if cap >= capacity), None)
    if room is None:
        raise ValueError(f"no room fits {capacity} people")
    event["room"] = room
    return {"event_id": event_id, "room": room}
