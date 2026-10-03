import json

import anyio
import httpx
import jsonschema

from toollens.dataset import check_tasks
from toollens.generate import ChatClient, generate_llm_tasks, generate_template_tasks
from toollens.generate.llm import _extract_json
from toollens.models import TaskKind


def test_template_generator_is_deterministic_and_valid(snapshot_v2):
    a = generate_template_tasks(snapshot_v2, per_tool=4, no_tool=3, seed=1)
    b = generate_template_tasks(snapshot_v2, per_tool=4, no_tool=3, seed=1)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]
    assert len(a) == 10 * 4 + 3
    assert sum(t.kind is TaskKind.NO_TOOL for t in a) == 3
    assert check_tasks(snapshot_v2, a, min_per_tool=4) == []
    for t in a:
        if t.expected_tool:
            # every generated required argument validates against the real schema
            schema = snapshot_v2.tool(t.expected_tool).input_schema
            jsonschema.validate(t.expected_args, schema)


def test_template_generator_respects_enums_and_ranges(tiny_snapshot):
    tasks = generate_template_tasks(tiny_snapshot, per_tool=10, no_tool=0)
    conv = [t for t in tasks if t.expected_tool == "convert_currency"]
    assert all(t.expected_args["from_currency"] in {"INR", "USD", "EUR"} for t in conv)


def test_extract_json_handles_fences_and_chatter():
    assert _extract_json('Sure!\n```json\n{"tasks": []}\n```') == {"tasks": []}
    assert _extract_json('noise {"a": {"b": 1}} trailing') == {"a": {"b": 1}}


def _fake_llm(tiny_snapshot):
    """A mock OpenAI-compatible endpoint that returns canned tasks, some deliberately broken."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][1]["content"]
        assert body["response_format"] == {"type": "json_object"}
        if "NONE of these tools" in user:
            payload = {"tasks": [{"prompt": "Book a cab to the airport"}, 42, {"prompt": "Tell me a joke"}]}
        else:
            tool = user.split("Write tasks for the tool `", 1)[1].split("`", 1)[0]
            words = tool.replace("_", " ")
            assert "LOOK-ALIKE PAIRS" in user
            payload = {
                "tasks": [
                    {
                        "prompt": f"plain request: {words} #1",
                        "kind": "direct",
                        "expected_args": {"city": "Pune", "made_up_param": 1},
                    },
                    {
                        "prompt": f"Plain request:  {words} #1!",
                        "kind": "direct",
                    },  # duplicate after normalising
                    {"prompt": f"tricky request: {words}", "kind": "near_miss", "distractor": "get_weather"},
                    {"prompt": f"bad distractor: {words}", "kind": "near_miss", "distractor": "not_a_tool"},
                    {"prompt": f"please call {tool} now", "kind": "direct"},  # leaks the label
                    {"prompt": "   ", "kind": "direct"},  # empty
                    "a bare string instead of an object",
                ]
            }
        content = "```json\n" + json.dumps(payload) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


def test_llm_generator_validates_and_dedupes(tiny_snapshot):
    client = ChatClient("ollama:qwen3:8b", transport=_fake_llm(tiny_snapshot))
    progress = []

    async def go():
        try:
            return await generate_llm_tasks(
                tiny_snapshot,
                client,
                direct=2,
                near_miss=2,
                no_tool=2,
                progress=lambda tool, i, n: progress.append((tool, i, n)),
            )
        finally:
            await client.aclose()

    tasks = anyio.run(go)
    assert progress[-1] == ("convert_currency", 3, 3)
    per_tool = [t for t in tasks if t.expected_tool == "get_forecast"]
    assert len(per_tool) == 3  # duplicate, leaking and empty prompts dropped
    first = per_tool[0]
    assert first.expected_args == {"city": "Pune"}  # unknown parameter stripped
    assert per_tool[1].kind is TaskKind.NEAR_MISS and per_tool[1].distractor == "get_weather"
    assert per_tool[2].kind is TaskKind.DIRECT and per_tool[2].distractor is None  # invented sibling removed
    self_lure = next(t for t in tasks if t.expected_tool == "get_weather" and t.prompt.startswith("tricky"))
    assert self_lure.kind is TaskKind.DIRECT  # a tool cannot be its own distractor
    assert [t.prompt for t in tasks if t.kind is TaskKind.NO_TOOL] == [
        "Book a cab to the airport",
        "Tell me a joke",
    ]
    assert len({t.id for t in tasks}) == len(tasks)
    assert check_tasks(tiny_snapshot, tasks) == []


def test_chat_client_anthropic_wire_format():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": '{"tasks": [{"prompt": "x"}]}'}]}
        )

    client = ChatClient("anthropic:claude-x", api_key="k", transport=httpx.MockTransport(handler))
    out = anyio.run(client.complete_json, "sys", "user")
    assert out == {"tasks": [{"prompt": "x"}]}
    assert seen["url"].endswith("/v1/messages") and seen["body"]["system"] == "sys"
