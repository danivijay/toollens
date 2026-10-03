import json

import anyio
import httpx
import pytest

from toollens.adapters import SYSTEM_PROMPT, AnthropicAdapter, OpenAICompatAdapter, make_adapter
from toollens.adapters.bm25 import BM25Adapter
from toollens.adapters.replay import ReplayAdapter
from toollens.models import Task

TASK = Task(id="t1", prompt="Is it raining in Chennai?", expected_tool="get_weather")


def recording_transport(responder):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request, body))
        return responder(request, body)

    return httpx.MockTransport(handler), seen


def openai_reply(tool=None, args=None, content=None, raw_args=None):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": tool,
                    "arguments": raw_args if raw_args is not None else json.dumps(args or {}),
                },
            }
        ]
    return {"choices": [{"message": message}]}


def run(adapter, snapshot, task=TASK):
    return anyio.run(adapter.predict, snapshot, task)


def test_openai_compat_request_shape_and_tool_call(tiny_snapshot):
    transport, seen = recording_transport(
        lambda r, b: httpx.Response(200, json=openai_reply("get_weather", {"city": "Chennai"}))
    )
    adapter = OpenAICompatAdapter("qwen3:4b", "http://localhost:11434/v1", api_key="k", transport=transport)
    pred = run(adapter, tiny_snapshot)
    assert pred.tool == "get_weather" and pred.arguments == {"city": "Chennai"} and pred.error is None

    request, body = seen[0]
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    assert body["model"] == "qwen3:4b" and body["temperature"] == 0.0 and body["tool_choice"] == "auto"
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert body["messages"][1]["content"] == TASK.prompt
    fns = [t["function"] for t in body["tools"]]
    assert [f["name"] for f in fns] == tiny_snapshot.tool_names
    assert fns[0]["parameters"]["required"] == ["city"]


def test_openai_compat_text_reply_means_no_tool(tiny_snapshot):
    transport, _ = recording_transport(
        lambda r, b: httpx.Response(200, json=openai_reply(content="It's sunny."))
    )
    pred = run(OpenAICompatAdapter("m", "http://x/v1", transport=transport), tiny_snapshot)
    assert pred.tool is None and pred.error is None and pred.raw == {"content": "It's sunny."}


@pytest.mark.parametrize(
    "response, fragment",
    [
        (httpx.Response(500, text="boom"), "HTTPStatusError"),
        (httpx.Response(200, json=openai_reply("get_weather", raw_args="{not json")), "JSONDecodeError"),
        (httpx.Response(200, json=openai_reply("get_weather", raw_args="[1, 2]")), "not a JSON object"),
        (httpx.Response(200, json=openai_reply("launch_rocket", {})), "unknown tool"),
    ],
)
def test_openai_compat_failures_become_errors_not_crashes(tiny_snapshot, response, fragment):
    transport, _ = recording_transport(lambda r, b: response)
    pred = run(OpenAICompatAdapter("m", "http://x/v1", transport=transport), tiny_snapshot)
    assert pred.error and fragment in pred.error


def test_anthropic_request_shape_and_tool_use(tiny_snapshot):
    reply = {
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "tu1", "name": "get_forecast", "input": {"city": "Kochi"}},
        ]
    }
    transport, seen = recording_transport(lambda r, b: httpx.Response(200, json=reply))
    adapter = AnthropicAdapter("claude-x", api_key="sk", transport=transport)
    pred = run(adapter, tiny_snapshot)
    assert pred.tool == "get_forecast" and pred.arguments == {"city": "Kochi"}

    request, body = seen[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk" and request.headers["anthropic-version"]
    assert body["system"] == SYSTEM_PROMPT and body["tool_choice"] == {"type": "auto"}
    assert body["tools"][2]["name"] == "convert_currency"
    assert body["tools"][2]["input_schema"]["required"] == ["amount", "from_currency", "to_currency"]


def test_anthropic_text_only_means_no_tool(tiny_snapshot):
    transport, _ = recording_transport(
        lambda r, b: httpx.Response(200, json={"content": [{"type": "text", "text": "Sorry"}]})
    )
    pred = run(AnthropicAdapter("m", api_key="k", transport=transport), tiny_snapshot)
    assert pred.tool is None and pred.raw == {"content": "Sorry"}


def test_make_adapter_specs(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_adapter("bm25"), BM25Adapter)
    assert make_adapter("bm25:2.5").min_score == 2.5
    assert make_adapter("bm25:2.5").name == "bm25:2.5" and make_adapter("bm25").name == "bm25"

    a = make_adapter("ollama:qwen3:4b")
    assert (
        a.model == "qwen3:4b"
        and str(a._http.base_url).rstrip("/") == "http://localhost:11434/v1"
        and a.concurrency == 1
    )
    assert (
        str(make_adapter("mlx:mlx-community/Qwen3-4B-4bit")._http.base_url).rstrip("/")
        == "http://localhost:8080/v1"
    )
    assert (
        str(make_adapter("lmstudio:m", base_url="http://box:9/v1")._http.base_url).rstrip("/")
        == "http://box:9/v1"
    )
    assert make_adapter("openai-compat:m", base_url="http://h/v1").name == "openai-compat:m"

    preds = tmp_path / "p.jsonl"
    preds.write_text('{"task_id": "t1", "tool": "get_weather"}\n')
    assert isinstance(make_adapter(f"replay:{preds}"), ReplayAdapter)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_adapter("openai:gpt-x")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_adapter("anthropic:claude-x")
    with pytest.raises(ValueError, match="--base-url"):
        make_adapter("openai-compat:m")
    with pytest.raises(ValueError, match="unknown provider"):
        make_adapter("bogus:m")
    with pytest.raises(ValueError, match="provider:model"):
        make_adapter("qwen3")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    remote = make_adapter("openai:gpt-x")
    assert remote.concurrency == 4 and remote._http.headers["authorization"] == "Bearer sk-test"


def test_replay_adapter(tmp_path, tiny_snapshot):
    path = tmp_path / "p.jsonl"
    path.write_text('{"task_id": "t1", "tool": "get_weather", "arguments": {"city": "Chennai"}}\n')
    adapter = ReplayAdapter(path)
    assert run(adapter, tiny_snapshot).arguments == {"city": "Chennai"}
    missing = run(adapter, tiny_snapshot, Task(id="zzz", prompt="?", expected_tool=None))
    assert missing.error and "no recorded prediction" in missing.error


def test_openai_compat_drops_temperature_for_reasoning_models(tiny_snapshot):
    def responder(request, body):
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature'"}})
        return httpx.Response(200, json=openai_reply("get_weather", {"city": "Chennai"}))

    transport, seen = recording_transport(responder)
    adapter = OpenAICompatAdapter("o4-mini", "http://x/v1", transport=transport)
    assert run(adapter, tiny_snapshot).tool == "get_weather"
    assert run(adapter, tiny_snapshot).tool == "get_weather"
    assert len(seen) == 3  # one rejected request, then temperature stays off for the rest of the run
