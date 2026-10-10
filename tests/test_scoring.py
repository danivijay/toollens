import pytest

from toollens.models import ERROR, NO_TOOL, Prediction
from toollens.scoring import args_contain, score, summarize


@pytest.mark.parametrize(
    "expected, actual, ok",
    [
        ({}, {"a": 1}, True),
        ({"city": "Pune"}, {"city": "pune"}, True),
        ({"city": "Pune"}, {"city": "  PUNE "}, True),
        ({"query": "budget"}, {"query": "Q3 budget review"}, True),
        ({"days": 3}, {"days": 3.0}, True),
        ({"days": 3}, {"days": 4}, False),
        ({"city": "Pune"}, {}, False),
        ({"city": "*"}, {"city": "anything"}, True),
        ({"emails": ["b@x", "a@x"]}, {"emails": ["A@x", "b@x"]}, True),
        ({"emails": ["a@x"]}, {"emails": ["a@x", "b@x"]}, True),  # extras allowed
        ({"emails": ["a@x", "c@x"]}, {"emails": ["a@x", "b@x"]}, False),
        ({"emails": ["a@x"]}, {"emails": "a@x"}, False),
        ({"event_id": "evt_1"}, {"event_id": "evt_1041"}, False),  # identifiers match exactly
        ({"email": "mei@acme.dev"}, {"email": "amei@acme.dev"}, False),
        ({"date": "2026-10-1"}, {"date": "2026-10-14"}, False),
        ({"query": ""}, {"query": "anything"}, False),
        ({"filter": {"status": "open"}}, {"filter": {"status": "Open", "x": 1}}, True),
        ({"notify": False}, {"notify": False}, True),
        ({"notify": False}, {"notify": True}, False),
        ({"notify": True}, {"notify": 1}, False),  # booleans never match numbers
        ({"limit": 1}, {"limit": True}, False),
        ({"limit": 0}, {"limit": False}, False),
        ({"notify": "true"}, {"notify": True}, False),
    ],
)
def test_args_contain(expected, actual, ok):
    assert args_contain(expected, actual) is ok


def preds(**by_id):
    out = []
    for task_id, value in by_id.items():
        if isinstance(value, Prediction):
            out.append(value)
        else:
            tool, args = value if isinstance(value, tuple) else (value, {})
            out.append(Prediction(task_id=task_id, tool=tool, arguments=args))
    return out


def test_score_and_summarize(tiny_snapshot, tiny_tasks):
    predictions = preds(
        w1=("get_weather", {"city": "Chennai"}),  # right tool, right args
        w2=("get_forecast", {"city": "Pune"}),  # confused with sibling
        f1=("get_weather", {"city": "Kochi"}),  # near miss: lured
        c1=("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "INR"}),  # bad type
        n1=(None, {}),  # correct abstention
    )
    results = score(tiny_snapshot, tiny_tasks, predictions)
    by_id = {r.task.id: r for r in results}
    assert by_id["w1"].tool_correct and by_id["w1"].args_valid and by_id["w1"].args_match
    assert not by_id["w2"].tool_correct and by_id["w2"].args_valid is None
    assert by_id["c1"].tool_correct and by_id["c1"].args_valid is False
    assert by_id["n1"].tool_correct and by_id["n1"].args_valid is None

    m = summarize("test", tiny_snapshot, results)
    assert m.n == 5 and m.accuracy == pytest.approx(3 / 5)
    assert m.accuracy_by_kind == {"direct": pytest.approx(2 / 3), "near_miss": 0.0, "no_tool": 1.0}
    assert m.abstention_recall == 1.0 and m.abstention_precision == 1.0
    assert m.args_valid_rate == pytest.approx(1 / 2)
    assert m.confusion["get_weather"]["get_forecast"] == 1
    assert m.confusion["get_forecast"]["get_weather"] == 1
    assert m.confusion[NO_TOOL][NO_TOOL] == 1
    [pair] = m.confused_pairs
    assert {pair.a, pair.b} == {"get_weather", "get_forecast"} and pair.total == 2

    stats = {s.tool: s for s in m.per_tool}
    assert stats["get_weather"].support == 2 and stats["get_weather"].recall == 0.5
    assert stats["get_weather"].precision == 0.5
    assert stats["convert_currency"].f1 == 1.0


def test_missed_calls_false_calls_and_errors(tiny_snapshot, tiny_tasks):
    predictions = preds(
        w1=(None, {}),
        w2=Prediction(task_id="w2", tool=None, error="TimeoutError: boom"),
        f1=("get_forecast", {"city": "Kochi", "days": 3}),
        c1=("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "INR"}),
        n1=("get_weather", {"city": "?"}),
    )
    m = summarize("t", tiny_snapshot, score(tiny_snapshot, tiny_tasks, predictions))
    assert m.missed_calls == 1 and m.false_calls == 1 and m.errors == 1

    # an invented tool on an out-of-scope task is an error, not also a false call
    predictions[-1] = Prediction(task_id="n1", tool="launch_rocket", error="model called unknown tool")
    m2 = summarize("t", tiny_snapshot, score(tiny_snapshot, tiny_tasks, predictions))
    assert m2.false_calls == 0 and m2.errors == 2
    assert ERROR in m.labels and m.confusion["get_weather"][ERROR] == 1
    assert m.abstention_precision == 0.0  # the only abstention was wrong
    assert m.abstention_recall == 0.0
    assert m.confused_pairs == []  # abstentions and errors are not tool-to-tool confusions


def test_score_requires_a_prediction_per_task(tiny_snapshot, tiny_tasks):
    with pytest.raises(ValueError, match="no prediction"):
        score(tiny_snapshot, tiny_tasks, preds(w1="get_weather"))


def test_routers_get_na_for_arguments(tiny_snapshot, tiny_tasks):
    p = preds(w1="get_weather", w2="get_weather", f1="get_forecast", c1="convert_currency", n1=None)
    results = score(tiny_snapshot, tiny_tasks, p, produces_arguments=False)
    m = summarize("router", tiny_snapshot, results)
    assert m.accuracy == 1.0 and m.args_valid_rate is None and m.args_match_rate is None
