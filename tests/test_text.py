from toollens.text import estimate_tokens, humanize, split_identifier, stem, tokens


def test_split_identifier_handles_snake_camel_and_kebab():
    assert split_identifier("search_events") == ["search", "events"]
    assert split_identifier("searchEvents") == ["search", "events"]
    assert split_identifier("get-HTTPStatus") == ["get", "httpstatus"]
    assert humanize("find_free_slots") == "find free slots"


def test_stem_is_conservative():
    assert stem("meetings") == "meeting"
    assert stem("parties") == "party"
    assert stem("scheduled") == "schedul"
    assert stem("bus") == "bus"  # too short to strip


def test_tokens_drop_stopwords_and_pure_numbers():
    assert tokens("What's on my calendar on 2026-10-14?") == ["calendar"]
    assert "evt" in tokens("cancel evt_1040")
    assert tokens("the a an", keep_stopwords=True) == ["the", "an"]


def test_estimate_tokens_is_positive_and_monotonic():
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 401) > estimate_tokens("x" * 400)
