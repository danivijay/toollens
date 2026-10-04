from toollens.lint import Severity, lint, similar_pairs
from toollens.models import ServerSnapshot, ToolSpec


def codes(report):
    return [f.code for f in report.findings]


def test_terse_server_gets_more_warnings_than_documented_one(snapshot_v1, snapshot_v2):
    v1, v2 = lint(snapshot_v1), lint(snapshot_v2)
    assert v1.count(Severity.WARNING) > v2.count(Severity.WARNING) + 10
    # v1 descriptions are all short and no parameter is documented
    assert codes(v1).count("TL001") == 10
    assert codes(v1).count("TL002") == 10
    assert "TL001" not in codes(v2) and "TL002" not in codes(v2)


def test_richer_descriptions_cost_more_tokens(snapshot_v1, snapshot_v2):
    assert lint(snapshot_v2).total_tokens > lint(snapshot_v1).total_tokens


def test_missing_description_is_an_error():
    snap = ServerSnapshot(server="x", tools=[ToolSpec(name="run")])
    report = lint(snap)
    assert any(f.code == "TL001" and f.severity is Severity.ERROR for f in report.findings)
    assert any(f.code == "TL007" for f in report.findings)  # generic name


def test_duplicate_names_and_undeclared_required_params_are_errors():
    bad = ToolSpec(
        name="a",
        description="A perfectly long description that explains what the tool does.",
        input_schema={"type": "object", "properties": {}, "required": ["ghost"]},
    )
    report = lint(ServerSnapshot(server="x", tools=[bad, bad]))
    assert {"TL000", "TL006"} <= set(codes(report))


def test_look_alikes_are_flagged_and_disambiguation_lowers_risk():
    a = ToolSpec(name="search_orders", description="Search customer orders by keyword and status.")
    b = ToolSpec(name="find_orders", description="Find customer orders by keyword and status.")
    [pair] = similar_pairs([a, b])
    assert pair.risk >= 0.35 and not pair.disambiguated
    risky = pair.risk

    b2 = b.model_copy(update={"description": b.description + " Use search_orders instead for free text."})
    [pair2] = similar_pairs([a, b2])
    assert pair2.disambiguated and pair2.risk < risky

    report = lint(ServerSnapshot(server="x", tools=[a, b]))
    assert any(
        f.code == "TL003" and set(f.tools) == {"search_orders", "find_orders"} for f in report.findings
    )


def test_disambiguation_needs_a_whole_word_reference():
    # 'research' contains 'search' and 'get_event_list' contains 'get_event', but neither names the sibling
    search = ToolSpec(name="search", description="Search the web for pages about a topic and return links.")
    fetch = ToolSpec(
        name="fetch", description="Fetch the pages for research on a topic and return their text."
    )
    get = ToolSpec(name="get_event", description="Get one calendar event and its details by id.")
    get_v2 = ToolSpec(
        name="get_events_v2", description="Get calendar events and details; replaces get_event_list."
    )
    assert not similar_pairs([search, fetch])[0].disambiguated
    assert not similar_pairs([get, get_v2])[0].disambiguated

    # real references still count, by identifier or in plain words, at any position or case
    fetch2 = fetch.model_copy(update={"description": "Fetch one page. Use search first to find URLs."})
    get_v3 = get_v2.model_copy(
        update={"description": "Get events in a range. For a single one, use Get Event."}
    )
    assert similar_pairs([search, fetch2])[0].disambiguated
    assert similar_pairs([get, get_v3])[0].disambiguated


def test_large_definitions_are_noted():
    props = {f"p{i}": {"type": "string", "description": "x" * 60} for i in range(40)}
    big = ToolSpec(
        name="big_tool",
        description="A tool with an enormous schema for testing.",
        input_schema={"type": "object", "properties": props},
    )
    assert "TL005" in codes(lint(ServerSnapshot(server="x", tools=[big])))


def test_findings_are_sorted_by_severity(snapshot_v1):
    order = {"error": 0, "warning": 1, "info": 2}
    sev = [order[f.severity.value] for f in lint(snapshot_v1).findings]
    assert sev == sorted(sev)
