from __future__ import annotations

import sys
from pathlib import Path

import pytest

from toollens.models import ServerSnapshot, Task, TaskKind, ToolSpec, read_tasks

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="session")
def snapshot_v1() -> ServerSnapshot:
    return ServerSnapshot.load(EXAMPLES / "snapshots" / "workspace_v1.json")


@pytest.fixture(scope="session")
def snapshot_v2() -> ServerSnapshot:
    return ServerSnapshot.load(EXAMPLES / "snapshots" / "workspace_v2.json")


@pytest.fixture(scope="session")
def workspace_tasks() -> list[Task]:
    return read_tasks(EXAMPLES / "datasets" / "workspace.jsonl")


@pytest.fixture()
def tiny_snapshot() -> ServerSnapshot:
    """Three tools, two of which look alike on purpose."""
    return ServerSnapshot(
        server="memory://tiny",
        server_name="tiny",
        tools=[
            ToolSpec(
                name="get_weather",
                description="Current weather conditions for a city: temperature, rain, wind.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            ),
            ToolSpec(
                name="get_forecast",
                description="Weather forecast for a city for the coming days.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "days": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Days ahead"},
                    },
                    "required": ["city"],
                },
            ),
            ToolSpec(
                name="convert_currency",
                description="Convert an amount of money from one currency to another at today's rate.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string", "enum": ["INR", "USD", "EUR"]},
                        "to_currency": {"type": "string", "enum": ["INR", "USD", "EUR"]},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            ),
        ],
    )


@pytest.fixture()
def tiny_tasks() -> list[Task]:
    return [
        Task(
            id="w1",
            prompt="Is it raining in Chennai right now?",
            expected_tool="get_weather",
            expected_args={"city": "Chennai"},
        ),
        Task(
            id="w2",
            prompt="How hot is it in Pune?",
            expected_tool="get_weather",
            expected_args={"city": "Pune"},
        ),
        Task(
            id="f1",
            prompt="Will it rain in Kochi over the next 3 days?",
            expected_tool="get_forecast",
            expected_args={"city": "Kochi", "days": 3},
            kind=TaskKind.NEAR_MISS,
            distractor="get_weather",
        ),
        Task(
            id="c1",
            prompt="How much is 100 USD in rupees?",
            expected_tool="convert_currency",
            expected_args={"amount": 100, "from_currency": "USD", "to_currency": "INR"},
        ),
        Task(id="n1", prompt="Write a haiku about the monsoon.", expected_tool=None, kind=TaskKind.NO_TOOL),
    ]


PYTHON = sys.executable
