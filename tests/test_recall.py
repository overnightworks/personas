"""The day a turn is told beyond its history, and the period after which a line is swept."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from personas.entries import ViewSeen
from personas.recall import EarlierToday, TodayBounds, earlier_today, swept_before, today_since, told_earlier_today
from tests.support.history import (
    APP,
    EMPTY_HISTORY,
    LATER,
    MORNING,
    NOON,
    ROOMY,
    TABLE,
    a_view,
    an_act,
    history_holding,
    kept,
    said,
    told_day_line_bytes,
)


def told(earlier):
    return [[(each.seq, each.text) for each in visit.lines] for visit in earlier.visits]


def test_two_earlier_visits_are_told_in_the_order_they_happened_each_as_its_own_group_in_seq_order():
    lines = [
        kept(NOON, 2, "noon two", hours_before=6),
        kept(MORNING, 1, "morning one", hours_before=9),
        kept(NOON, 1, "noon one", hours_before=6),
        kept(MORNING, 2, "morning two", hours_before=9),
    ]

    earlier = earlier_today(lines, LATER, ROOMY, EMPTY_HISTORY)

    assert told(earlier) == [[(1, "morning one"), (2, "morning two")], [(1, "noon one"), (2, "noon two")]]
    assert [visit.minutes_ago for visit in earlier.visits] == [9 * 60, 6 * 60]
    assert earlier.older_dropped is False


def test_an_act_line_of_an_earlier_visit_is_told_with_the_mark_and_any_other_line_without_it():
    lines = [kept(MORNING, 1, "took a card", hours_before=9, act=True), kept(MORNING, 2, "hello", hours_before=9)]

    [visit] = told_earlier_today(earlier_today(lines, LATER, ROOMY, EMPTY_HISTORY))["visits"]

    assert [each.get("act", False) for each in visit["lines"]] == [True, False]


@pytest.mark.parametrize(
    ("bound_over_the_two_lines", "told_seqs", "older_dropped"),
    [
        pytest.param(1, (1, 2), False, id="the bound lies above the lines"),
        pytest.param(0, (1, 2), False, id="the bound is the lines' size"),
        pytest.param(-1, (2,), True, id="the bound lies one byte below the lines"),
    ],
)
def test_the_day_keeps_the_newest_lines_that_fit_its_bound_counted_in_utf8_bytes_of_their_told_json(
    text_and_its_told_json, bound_over_the_two_lines, told_seqs, older_dropped
):
    text, told_text = text_and_its_told_json
    lines = [kept(MORNING, seq, text, hours_before=1, name="Bo") for seq in (1, 2)]
    bound = TodayBounds(hours=24.0, size_bytes=2 * told_day_line_bytes(told_text, name="Bo") + bound_over_the_two_lines)

    earlier = earlier_today(lines, LATER, bound, EMPTY_HISTORY)

    assert told(earlier) == [[(seq, text) for seq in told_seqs]]
    assert earlier.older_dropped is older_dropped


def test_an_earlier_visit_whose_lines_all_lie_past_the_size_bound_leaves_the_day_whole():
    lines = [
        kept(MORNING, 1, "oldest", hours_before=9, name="Bo"),
        kept(NOON, 1, "middle", hours_before=6, name="Bo"),
        kept(NOON, 2, "newest", hours_before=6, name="Bo"),
    ]
    two_lines = TodayBounds(hours=24.0, size_bytes=2 * told_day_line_bytes("middle", name="Bo"))

    earlier = earlier_today(lines, LATER, two_lines, EMPTY_HISTORY)

    assert told(earlier) == [[(1, "middle"), (2, "newest")]]
    assert [visit.minutes_ago for visit in earlier.visits] == [6 * 60]
    assert earlier.older_dropped is True


def test_no_line_of_the_day_is_told_and_the_day_says_so_when_not_even_the_newest_fits_its_bound():
    lines = [kept(NOON, 1, "middle", hours_before=6, name="Bo")]
    too_small = TodayBounds(hours=24.0, size_bytes=told_day_line_bytes("middle", name="Bo") - 1)

    earlier = earlier_today(lines, LATER, too_small, EMPTY_HISTORY)

    assert earlier == EarlierToday((), older_dropped=True)


def test_the_day_reaches_back_the_configured_hours_and_an_unbounded_day_reaches_back_to_any_line():
    assert today_since(LATER, TodayBounds(hours=24.0, size_bytes=1)) == LATER - timedelta(hours=24)
    assert today_since(LATER, TodayBounds(hours=math.inf, size_bytes=1)) <= datetime(1, 1, 2, tzinfo=UTC)


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


def test_the_day_is_the_same_whether_or_not_the_history_holds_acts_and_holds_no_act():
    kept_lines = [kept(MORNING, 1, "held", hours_before=5), kept(NOON, 1, "not held", hours_before=4)]

    without = earlier_today(kept_lines, LATER, ROOMY, history_holding((said(1, "held"),)))
    with_acts = earlier_today(kept_lines, LATER, ROOMY, history_holding((said(1, "held"), an_act())))

    assert with_acts == without


def test_the_day_leaves_out_the_lines_the_history_holds_by_visit_and_seq_and_those_first_read_after_the_turn():
    kept_lines = [
        kept(MORNING, 1, "held", hours_before=5),
        kept(MORNING, 2, "not held", hours_before=5),
        kept(NOON, 1, "same seq of another visit", hours_before=4),
        kept(NOON, 2, "read as the turn was admitted", hours_before=0),
        kept(NOON, 3, "read after the turn was admitted", hours_before=-1),
    ]

    earlier = earlier_today(kept_lines, LATER, ROOMY, history_holding((said(1, "held"),)))

    assert told(earlier) == [
        [(2, "not held")],
        [(1, "same seq of another visit"), (2, "read as the turn was admitted")],
    ]


def test_the_day_is_the_same_whether_or_not_the_history_holds_views_and_holds_no_view():
    kept_lines = [kept(MORNING, 1, "held", hours_before=5), kept(NOON, 1, "not held", hours_before=4)]
    views = (ViewSeen(APP, TABLE, a_view(1), LATER),)

    without = earlier_today(kept_lines, LATER, ROOMY, history_holding((said(1, "held"),)))
    with_views = earlier_today(kept_lines, LATER, ROOMY, history_holding((said(1, "held"), *views)))

    assert with_views == without
    assert told(with_views) == [[(1, "not held")]]


def test_what_is_left_out_of_the_day_counts_nothing_toward_its_size_bound():
    two_lines = TodayBounds(hours=24.0, size_bytes=2 * told_day_line_bytes("middle", name="Bo"))
    kept_lines = [
        kept(MORNING, 1, "held" * 20, hours_before=5, name="Bo"),
        kept(MORNING, 2, "held" * 20, hours_before=5, name="Bo"),
        kept(NOON, 1, "middle", hours_before=4, name="Bo"),
        kept(NOON, 2, "second", hours_before=4, name="Bo"),
    ]

    earlier = earlier_today(kept_lines, LATER, two_lines, history_holding((said(1, "held"), said(2, "held"))))

    assert told(earlier) == [[(1, "middle"), (2, "second")]]
    assert earlier.older_dropped is False
