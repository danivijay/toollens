import anyio

from toollens.adapters.bm25 import BM25Adapter, BM25Index
from toollens.models import Task


def predict(adapter, snapshot, prompt):
    return anyio.run(adapter.predict, snapshot, Task(id="t", prompt=prompt, expected_tool=None))


def test_index_prefers_rare_matching_terms():
    idx = BM25Index({"a": ["weather", "city"], "b": ["currency", "city"]})
    s = idx.scores(["currency"])
    assert s["b"] > 0 and s["a"] == 0


def test_routes_obvious_requests(tiny_snapshot):
    bm25 = BM25Adapter()
    assert predict(bm25, tiny_snapshot, "convert 50 euro currency to INR").tool == "convert_currency"
    assert predict(bm25, tiny_snapshot, "what's the forecast for Delhi").tool == "get_forecast"


def test_abstains_on_unrelated_requests(tiny_snapshot):
    p = predict(BM25Adapter(), tiny_snapshot, "write a limerick about cats")
    assert p.tool is None and p.error is None


def test_threshold_controls_abstention(tiny_snapshot):
    prompt = "forecast"
    assert predict(BM25Adapter(min_score=0.0), tiny_snapshot, prompt).tool == "get_forecast"
    assert predict(BM25Adapter(min_score=100.0), tiny_snapshot, prompt).tool is None


def test_is_deterministic_and_never_produces_arguments(tiny_snapshot):
    bm25 = BM25Adapter()
    a = predict(bm25, tiny_snapshot, "rain in Kochi tomorrow")
    b = predict(bm25, tiny_snapshot, "rain in Kochi tomorrow")
    assert a.tool == b.tool and a.arguments == {} and not bm25.produces_arguments


def test_index_cache_is_keyed_by_content(tiny_snapshot):
    bm25 = BM25Adapter()
    assert predict(bm25, tiny_snapshot, "convert currency").tool == "convert_currency"
    renamed = tiny_snapshot.model_copy(deep=True)
    renamed.tools[2].name = "fx_convert"
    assert predict(bm25, renamed, "convert currency").tool == "fx_convert"
