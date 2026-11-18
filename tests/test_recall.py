"""What the store gives a turn and the period after which a line is swept: the lookup's declaration, its room, its
argument and its result."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pytest
from agents.member import PlaceRef
from agents.tools import ToolEffect
from agents.turn import MAX_EVENT_BYTES

from personas.entries import KeptLine, ViewSeen, Visit, told_bytes, told_entry
from personas.recall import (
    RECALL_DECLARATION,
    SCOPE_TEXT,
    SMALLEST_RESULT_BYTES,
    line_search,
    lookup_result,
    search_text,
    swept_before,
    text_refused_text,
)
from tests.support.history import (
    APP,
    LATER,
    MORNING,
    TABLE,
    a_sentence,
    a_view,
    an_act,
    entry,
    history_holding,
    said,
)

CLIENT = "mozart-client"
LATER_TOLD = "2026-11-10T18:00:00Z"


@pytest.mark.parametrize(
    ("keep_days", "oldest_kept"),
    [
        (30.0, datetime(2026, 10, 11, 18, 0, tzinfo=UTC)),
        (0.5, datetime(2026, 11, 10, 6, 0, tzinfo=UTC)),
        (math.inf, None),
    ],
    ids=["a month", "half a day", "every line"],
)
def test_the_oldest_first_read_that_stays_is_the_keep_period_before_now_and_an_infinite_period_sweeps_nothing(
    keep_days, oldest_kept
):
    assert swept_before(LATER, keep_days) == oldest_kept


def test_a_keep_period_that_reaches_before_the_calendar_sweeps_nothing():
    assert swept_before(LATER, 1e12) is None


def test_the_lookup_is_declared_read_only_under_one_string_parameter_named_text():
    """TURNS-39."""
    assert RECALL_DECLARATION.effect is ToolEffect.READ_ONLY
    assert RECALL_DECLARATION.parameters["properties"].keys() == {"text"}
    assert RECALL_DECLARATION.parameters["properties"]["text"]["type"] == "string"
    assert RECALL_DECLARATION.parameters["required"] == ["text"]


@pytest.mark.parametrize(
    ("first", "its_time"),
    [
        (said(1, "a line"), LATER),
        (a_sentence(), LATER - timedelta(seconds=9)),
        (an_act(), LATER - timedelta(seconds=9)),
        (ViewSeen(APP, TABLE, a_view(1), LATER - timedelta(minutes=1)), LATER - timedelta(minutes=1)),
    ],
    ids=["a line", "a sentence of his", "a call of his", "a view"],
)
def test_a_search_is_for_this_agents_place_and_before_the_first_entry_his_history_still_holds(first, its_time):
    history = history_holding((first, said(2, "newer")))

    search = line_search(CLIENT, APP, TABLE, history, "a few words", 8192)

    assert (search.client_id, search.app, search.place, search.text) == (CLIENT, APP, TABLE, "a few words")
    assert search.first_read_before == its_time


def _smallest_line(seq: int) -> KeptLine:
    return KeptLine(
        Visit("a", PlaceRef(kind="k", id="i"), ""),
        seq,
        LATER,
        "",
        "",
        "",
        own=False,
        spoken=False,
        cut=False,
        act=False,
    )


@pytest.mark.parametrize("result_bytes", [SMALLEST_RESULT_BYTES, 400, 8192, 100_000, MAX_EVENT_BYTES * 3])
def test_a_search_asks_for_one_row_more_than_the_smallest_lines_that_fit_the_result_so_that_no_fitting_line_is_missed(
    result_bytes,
):
    search = line_search(CLIENT, APP, TABLE, history_holding((said(1, "x"),)), "", result_bytes)

    kept = json.loads(lookup_result([_smallest_line(0)] * search.rows, result_bytes))["lines"]

    assert len(kept) == search.rows - 1


def _lines_said(count: int, text: str = "x") -> list[KeptLine]:
    """The lines newest first."""
    return [said(seq, text) for seq in range(count, 0, -1)]


def _told(count: int, text: str = "x") -> list[dict[str, object]]:
    return [
        entry(MORNING.app, MORNING.place.id, text, seq=seq, at=LATER_TOLD, name="Bo") for seq in range(count, 0, -1)
    ]


def test_a_result_holds_its_scope_and_the_lines_in_the_historys_form_newest_first():
    result = json.loads(lookup_result(_lines_said(3), 100_000))

    assert result == {"scope": SCOPE_TEXT, "older_dropped": False, "lines": _told(3)}


def test_a_result_without_a_line_is_the_same_form_with_none():
    assert json.loads(lookup_result([], 100_000)) == {"scope": SCOPE_TEXT, "older_dropped": False, "lines": []}


def test_a_result_holds_the_newest_lines_that_fit_and_says_when_more_matched():
    """TURNS-40, CONFIG-37."""
    whole = len(lookup_result(_lines_said(5), 100_000).encode())

    all_fit = json.loads(lookup_result(_lines_said(5), whole))
    one_byte_short = lookup_result(_lines_said(5), whole - 1)
    kept = json.loads(one_byte_short)

    assert all_fit["lines"] == _told(5)
    assert not all_fit["older_dropped"]
    assert kept["lines"] == _told(5)[: len(kept["lines"])]
    assert 0 < len(kept["lines"]) < 5
    assert kept["older_dropped"]
    assert len(one_byte_short.encode()) <= whole - 1


def test_a_result_within_the_least_bound_holds_no_line_says_more_matched_and_passes_no_bound():
    result = lookup_result(_lines_said(2), SMALLEST_RESULT_BYTES)

    assert json.loads(result) == {"scope": SCOPE_TEXT, "older_dropped": True, "lines": []}
    assert len(result.encode()) <= SMALLEST_RESULT_BYTES


def test_a_result_that_says_more_matched_holds_the_newest_line_when_it_fits_that_result_exactly():
    newest, older = _lines_said(2)
    bound = SMALLEST_RESULT_BYTES + told_bytes(told_entry(newest)) - 1

    result = lookup_result([newest, older], bound)

    assert json.loads(result) == {"scope": SCOPE_TEXT, "older_dropped": True, "lines": _told(2)[:1]}
    assert len(result.encode()) == bound


def test_a_result_never_passes_what_the_wire_carries_in_one_result_whatever_lookup_bytes_allows():
    lines = _lines_said(4, "x" * (MAX_EVENT_BYTES // 3))

    result = lookup_result(lines, MAX_EVENT_BYTES * 10)

    assert len(result.encode()) <= MAX_EVENT_BYTES
    assert json.loads(result)["lines"] == _told(4, "x" * (MAX_EVENT_BYTES // 3))[:2]
    assert json.loads(result)["older_dropped"]


@pytest.mark.parametrize(
    ("arguments", "longest", "text"),
    [
        ({"text": "two words"}, 100, "two words"),
        ({"text": ""}, 100, ""),
        ({"text": "äöü"}, 6, "äöü"),
        ({"text": "two words", "other": 1}, 100, "two words"),
    ],
    ids=["words", "no word", "the longest", "another argument is ignored"],
)
def test_the_search_text_is_the_string_the_model_gave_within_the_longest_a_lookup_accepts(arguments, longest, text):
    assert search_text(arguments, longest) == text


@pytest.mark.parametrize(
    ("arguments", "longest"),
    [
        ({}, 100),
        ({"query": "words"}, 100),
        ({"text": 7}, 100),
        ({"text": None}, 100),
        ({"text": ["words"]}, 100),
        ({"text": "äöü"}, 5),
        ({"text": "\ud800"}, 100),
    ],
    ids=["missing", "misnamed", "a number", "null", "a list", "longer in bytes", "no text in UTF-8"],
)
def test_a_search_text_that_is_missing_no_string_or_longer_than_the_bound_in_bytes_is_refused(arguments, longest):
    assert search_text(arguments, longest) is None


def test_the_refusal_of_a_text_names_the_longest_it_accepts():
    assert "37" in text_refused_text(37)
