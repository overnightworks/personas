"""Which place has a turn due, when the next turn starts, and what a pod that takes him over keeps of his turns."""

from __future__ import annotations

import secrets
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from agents.member import PlaceRef

from personas.decision import AppPlace, Joined, Restarted
from personas.due import (
    Claim,
    Due,
    Turns,
    TurnState,
    WaitReason,
    claimed,
    cut_by_stop,
    cut_for_spoken,
    due_again,
    due_for_hold,
    kept_turns,
    left_standing,
    pause_deadline,
    pause_left,
    paused_for,
    released,
    resumed_turns,
    waited,
    with_due,
    without_due,
    yields_to_spoken,
)
from personas.entries import History, Lines, Reading, Waiting
from tests.support.history import APP, ONCE_DROPPED, TABLE, said
from tests.support.standings import active, paused

CALL_AT_A = AppPlace("https://a.test/mcp", PlaceRef(kind="call", id="room-1"))
CALL_AT_B = AppPlace("https://b.test/mcp", PlaceRef(kind="call", id="room-2"))
TABLE_AT_A = AppPlace("https://a.test/mcp", PlaceRef(kind="table", id="table-1"))
NOW = 1000.0


def _nothing_spoken(_at: AppPlace, _due: Due) -> bool:
    return False


def _spoken_at(*places: AppPlace):
    return lambda at, _due: at in places


def _due_in_order(*due: tuple[AppPlace, Lines | Waiting]) -> Turns:
    turns = Turns()
    for at, occasion in due:
        turns = with_due(turns, at, occasion, NOW)
    return turns


def _occasions(turns: Turns) -> dict[AppPlace, Lines | Waiting]:
    return {at: held.occasion for at, held in turns.due.items()}


def test_lines_due_at_a_place_in_one_generation_join_and_keep_the_place_where_it_was_in_the_order():
    turns = _due_in_order(
        (CALL_AT_A, Lines("g1", frozenset({1}), missed=False)), (CALL_AT_B, Lines("g1", frozenset({4}), missed=False))
    )

    turns = with_due(turns, CALL_AT_A, Lines("g1", frozenset({2, 3}), missed=True), NOW)

    assert _occasions(turns) == {
        CALL_AT_A: Lines("g1", frozenset({1, 2, 3}), missed=True),
        CALL_AT_B: Lines("g1", frozenset({4}), missed=False),
    }
    assert list(turns.due) == [CALL_AT_A, CALL_AT_B]


def test_lines_of_another_generation_and_a_newer_revision_replace_what_was_due_at_the_place_in_its_position():
    turns = _due_in_order(
        (CALL_AT_A, Lines("g1", frozenset({1}), missed=True)), (TABLE_AT_A, Waiting(1)), (CALL_AT_B, Waiting(5))
    )

    turns = with_due(turns, CALL_AT_A, Lines("g2", frozenset({1}), missed=False), NOW)
    turns = with_due(turns, TABLE_AT_A, Waiting(2), NOW)

    assert _occasions(turns) == {
        CALL_AT_A: Lines("g2", frozenset({1}), missed=False),
        TABLE_AT_A: Waiting(2),
        CALL_AT_B: Waiting(5),
    }
    assert list(turns.due) == [CALL_AT_A, TABLE_AT_A, CALL_AT_B]


def test_a_turn_without_an_answer_makes_its_place_due_again_with_what_it_was_for_at_the_back_of_the_order():
    turns = _due_in_order((CALL_AT_B, Waiting(1)))

    for occasion in (Waiting(2), Lines("g1", frozenset({3}), missed=True)):
        again = due_again(turns, CALL_AT_A, occasion, NOW)

        assert _occasions(again) == {CALL_AT_B: Waiting(1), CALL_AT_A: occasion}
        assert list(again.due) == [CALL_AT_B, CALL_AT_A]


def test_a_place_that_became_due_meanwhile_keeps_its_newer_occasion_and_only_lines_of_its_generation_join_the_retry():
    turns = _due_in_order(
        (CALL_AT_A, Lines("g2", frozenset({5}), missed=False)),
        (TABLE_AT_A, Waiting(3)),
        (CALL_AT_B, Lines("g1", frozenset({7}), missed=False)),
    )

    turns = due_again(turns, CALL_AT_A, Lines("g1", frozenset({1}), missed=False), NOW)
    turns = due_again(turns, TABLE_AT_A, Waiting(2), NOW)
    turns = due_again(turns, CALL_AT_B, Lines("g1", frozenset({4}), missed=True), NOW)

    assert _occasions(turns) == {
        CALL_AT_A: Lines("g2", frozenset({5}), missed=False),
        TABLE_AT_A: Waiting(3),
        CALL_AT_B: Lines("g1", frozenset({4, 7}), missed=True),
    }
    assert list(turns.due) == [CALL_AT_A, TABLE_AT_A, CALL_AT_B]


def test_lines_of_one_generation_join_in_the_trace_of_the_newer_lines_whichever_was_due_first():
    older, newer = "trace-of-the-older-lines", "trace-of-the-newer-lines"
    turns = _due_in_order((CALL_AT_A, Lines("g1", frozenset({1, 2}), missed=False, traceparent=older)))

    joined = with_due(turns, CALL_AT_A, Lines("g1", frozenset({3}), missed=False, traceparent=newer), NOW)
    retried = due_again(
        _due_in_order((CALL_AT_A, Lines("g1", frozenset({3}), missed=False, traceparent=newer))),
        CALL_AT_A,
        Lines("g1", frozenset({1, 2}), missed=False, traceparent=older),
        NOW,
    )

    for after in (joined, retried):
        assert _occasions(after) == {CALL_AT_A: Lines("g1", frozenset({1, 2, 3}), missed=False, traceparent=newer)}


def _since(turns: Turns, at: AppPlace) -> float:
    return turns.due[at].since


def test_a_join_and_a_replacement_keep_the_earliest_moment_the_place_became_due():
    """LOG-27."""
    turns = with_due(Turns(), CALL_AT_A, Lines("g1", frozenset({1}), missed=False), NOW)
    turns = with_due(turns, TABLE_AT_A, Waiting(1), NOW + 1)

    joined = with_due(turns, CALL_AT_A, Lines("g1", frozenset({2}), missed=False), NOW + 5)
    replaced = with_due(turns, CALL_AT_A, Lines("g2", frozenset({1}), missed=False), NOW + 5)
    revised = with_due(turns, TABLE_AT_A, Waiting(2), NOW + 5)

    assert (_since(joined, CALL_AT_A), _since(replaced, CALL_AT_A), _since(revised, TABLE_AT_A)) == (NOW, NOW, NOW + 1)


def test_a_turn_due_again_waits_from_the_moment_given_and_a_place_due_meanwhile_keeps_its_own_moment():
    turns = with_due(Turns(), CALL_AT_B, Waiting(1), NOW)
    turns = with_due(turns, CALL_AT_A, Lines("g1", frozenset({2}), missed=False), NOW + 2)

    again = due_again(turns, TABLE_AT_A, Waiting(1), NOW + 30)
    joined = due_again(again, CALL_AT_A, Lines("g1", frozenset({1}), missed=False), NOW + 30)
    newer = due_again(again, CALL_AT_B, Waiting(0), NOW + 30)

    assert (_since(again, TABLE_AT_A), _since(joined, CALL_AT_A), _since(newer, CALL_AT_B)) == (NOW + 30, NOW + 2, NOW)


def test_a_hold_makes_its_place_due_for_no_line_unless_the_place_is_due_already_and_the_turn_it_has_is_told_the_hold():
    lines = Lines("g1", frozenset({4}), missed=False)
    turns = with_due(Turns(), CALL_AT_B, Waiting(1), NOW)

    held = due_for_hold(turns, CALL_AT_A, "g1", NOW + 1)
    already = due_for_hold(held, CALL_AT_B, "g1", NOW + 2)
    joined = with_due(held, CALL_AT_A, lines, NOW + 3)

    assert _occasions(held) == {CALL_AT_B: Waiting(1), CALL_AT_A: Lines("g1", frozenset(), missed=False)}
    assert already is held
    assert _occasions(joined)[CALL_AT_A] == Lines("g1", frozenset({4}), missed=False)
    assert list(joined.due) == [CALL_AT_B, CALL_AT_A]


def test_a_turn_that_the_hold_alone_made_due_and_that_ended_without_an_answer_is_not_due_again():
    turns = with_due(Turns(), CALL_AT_B, Waiting(1), NOW)
    hold_only = Lines("g1", frozenset(), missed=False)

    assert due_again(turns, CALL_AT_A, hold_only, NOW + 5) is turns


def test_the_claimed_turn_says_since_when_it_was_due():
    turns = with_due(Turns(), CALL_AT_A, Waiting(1), NOW - 7)

    _, _, due = claimed(turns, CALL_AT_A.app, NOW, _nothing_spoken) or pytest.fail("the place is due")

    assert due.since == NOW - 7


def test_the_turn_goes_to_the_place_that_waited_longest_and_only_in_its_app():
    turns = _due_in_order((CALL_AT_B, Waiting(1)), (CALL_AT_A, Waiting(2)))

    assert claimed(turns, CALL_AT_A.app, NOW, _nothing_spoken) is None
    after, at, due = claimed(turns, CALL_AT_B.app, NOW, _nothing_spoken) or pytest.fail(
        "the oldest place is at its app"
    )
    assert (after.running, at, due.occasion) == (CALL_AT_B, CALL_AT_B, Waiting(1))
    assert _occasions(after) == {CALL_AT_A: Waiting(2)}


def test_no_turn_starts_while_one_runs_or_when_nothing_is_due():
    assert (
        claimed(
            replace(_due_in_order((CALL_AT_A, Waiting(1))), running=TABLE_AT_A), CALL_AT_A.app, NOW, _nothing_spoken
        )
        is None
    )
    assert claimed(Turns(), CALL_AT_A.app, NOW, _nothing_spoken) is None


@pytest.mark.parametrize(
    ("turns", "reason"),
    [
        (_due_in_order((CALL_AT_A, Waiting(1))), WaitReason.NONE),
        (_due_in_order((CALL_AT_B, Waiting(1)), (CALL_AT_A, Waiting(2))), WaitReason.OLDER_DUE),
        (replace(_due_in_order((CALL_AT_A, Waiting(1))), running=TABLE_AT_A), WaitReason.TURN_HERE),
        (replace(_due_in_order((CALL_AT_A, Waiting(1))), running=CALL_AT_B), WaitReason.TURN_ELSEWHERE),
        (paused_for(_due_in_order((CALL_AT_A, Waiting(1))), NOW, 4), WaitReason.PAUSED),
    ],
    ids=["nothing", "older-place", "turn-here", "turn-elsewhere", "pause"],
)
def test_a_due_place_names_what_stands_directly_in_front_of_it(turns, reason):
    assert waited(turns, NOW, _nothing_spoken).due[CALL_AT_A].waited_on is reason


def test_a_due_place_keeps_the_first_reason_through_a_join_and_names_none_before_its_wait_began():
    lines = Lines("g1", frozenset({1}), missed=False)
    held = waited(replace(_due_in_order((CALL_AT_A, lines)), running=TABLE_AT_A), NOW, _nothing_spoken)
    joined = with_due(released(held), CALL_AT_A, Lines("g1", frozenset({2}), missed=False), NOW + 1)
    after_the_pause = paused_for(with_due(Turns(), CALL_AT_A, Waiting(1), NOW + 4), NOW, 4)

    assert waited(joined, NOW + 1, _nothing_spoken).due[CALL_AT_A].waited_on is WaitReason.TURN_HERE
    assert waited(after_the_pause, NOW + 1, _nothing_spoken).due[CALL_AT_A].waited_on is WaitReason.NONE


def test_a_pause_holds_the_turn_of_the_oldest_place_until_its_end_and_a_release_leaves_it():
    turns = paused_for(_due_in_order((CALL_AT_A, Waiting(1))), NOW, 4)

    assert claimed(turns, CALL_AT_A.app, NOW + 3, _nothing_spoken) is None
    assert pause_left(turns, CALL_AT_A.app, NOW + 3, _nothing_spoken) == 1
    assert pause_left(released(turns), CALL_AT_A.app, NOW + 3, _nothing_spoken) == 1
    assert claimed(turns, CALL_AT_A.app, NOW + 4, _nothing_spoken) is not None
    assert pause_left(turns, CALL_AT_A.app, NOW + 4, _nothing_spoken) == 0


def test_a_pause_is_left_only_to_the_app_of_the_oldest_place_and_only_while_a_turn_waits_for_it():
    waiting = paused_for(_due_in_order((CALL_AT_B, Waiting(1))), NOW, 4)

    assert pause_left(waiting, CALL_AT_A.app, NOW, _nothing_spoken) == 0
    assert pause_left(replace(waiting, running=TABLE_AT_A), CALL_AT_B.app, NOW, _nothing_spoken) == 0
    assert pause_left(paused_for(Turns(), NOW, 4), CALL_AT_B.app, NOW, _nothing_spoken) == 0


def test_a_released_turn_leaves_what_is_due_and_lets_the_next_one_start():
    running = replace(_due_in_order((CALL_AT_A, Waiting(1))), running=TABLE_AT_A)

    after = released(running)

    assert after.running is None
    assert after.due == running.due
    assert claimed(after, CALL_AT_A.app, NOW, _nothing_spoken) is not None


@pytest.mark.parametrize(
    ("meanwhile", "occasion"),
    [
        (Lines("g1", frozenset({3}), missed=False), Lines("g1", frozenset({1, 2, 3}), missed=False)),
        (Lines("g2", frozenset({1}), missed=False), Lines("g2", frozenset({1}), missed=False)),
    ],
    ids=["same-generation", "new-generation"],
)
def test_a_turn_his_stop_cut_before_it_spoke_is_due_first_joined_with_what_made_its_place_due_meanwhile(
    meanwhile, occasion
):
    claim = Claim(CALL_AT_A, Due(Lines("g1", frozenset({1, 2}), missed=False), NOW - 7), mark=0, aloud=False)
    turns = replace(_due_in_order((CALL_AT_B, Waiting(1)), (CALL_AT_A, meanwhile)), running=TABLE_AT_A)

    after = cut_by_stop(turns, claim, History((), left_out=0, heard=0))

    assert list(after.due) == [CALL_AT_A, CALL_AT_B]
    assert (after.due[CALL_AT_A].occasion, after.due[CALL_AT_A].since) == (occasion, NOW - 7)


def test_a_turn_his_stop_cut_after_the_bound_dropped_entries_since_its_claim_counts_as_spoken_and_is_not_due():
    claim = Claim(CALL_AT_A, Due(Waiting(1), NOW), mark=0, aloud=False)

    after = cut_by_stop(Turns(running=CALL_AT_A), claim, ONCE_DROPPED)

    assert after.due == {}


def test_the_places_that_leave_are_no_longer_due_and_the_rest_keep_their_order():
    turns = _due_in_order((CALL_AT_A, Waiting(1)), (CALL_AT_B, Waiting(1)), (TABLE_AT_A, Waiting(1)))

    after = without_due(turns, lambda at: at.app == CALL_AT_A.app)

    assert list(after.due) == [CALL_AT_B]


def test_a_due_place_of_an_app_leaves_with_its_call_or_table_and_stays_while_it_is_active():
    standings = (active("room-1"), active("table-1", "table", serves_view=True), paused("room-9"))
    app = CALL_AT_A.app
    due = [CALL_AT_A, TABLE_AT_A, AppPlace(app, PlaceRef(kind="call", id="room-9")), CALL_AT_B]

    leaving = left_standing(app, standings)

    assert [leaving(at) for at in due] == [False, False, True, False]


def test_the_places_and_lines_a_turn_is_due_for_show_no_name_no_text_and_no_state_in_their_repr():
    marker = secrets.token_hex(8)
    taken = said(1, marker, name=marker)
    at = AppPlace(taken.visit.app, taken.visit.place)
    lines = Lines(taken.visit.generation, frozenset({taken.seq}), missed=False)
    turns = with_due(with_due(Turns(), at, lines, NOW), AppPlace(APP, TABLE), Waiting(revision=1), NOW)

    assert marker not in repr(at)
    assert marker not in repr(lines)
    assert marker not in repr(turns)


def test_turns_kept_and_resumed_keep_the_due_order_and_the_checkpoints_but_no_running_turn_and_only_the_pause_left():
    first, second, said_before = CALL_AT_A, CALL_AT_B, TABLE_AT_A
    lines = Lines("visit", frozenset({4}), missed=False)
    state = TurnState(
        Turns(first, {second: Due(lines, 1.0), first: Due(Waiting(3), 2.0)}, not_before=50.0),
        {first: Joined(), second: Reading("visit", 4), said_before: Restarted()},
        unanswered=3,
    )
    wall = datetime(2026, 11, 17, tzinfo=UTC)

    kept = kept_turns(state, pause_deadline(50.0, 20.0, wall))
    resumed = resumed_turns(kept, 100.0, wall + timedelta(seconds=10))

    assert resumed == TurnState(
        Turns(None, {second: Due(lines, 100.0), first: Due(Waiting(3), 100.0)}, not_before=120.0),
        {first: Joined(), second: Reading("visit", 4)},
        unanswered=3,
    )
    assert list(resumed.turns.due) == [second, first]
    assert pause_deadline(10.0, 20.0, wall) is None


def _claim_in_order(turns: Turns, spoken, now: float = NOW) -> list[AppPlace]:
    """The places in the order they are claimed one after the other, each turn run to its end."""
    order = []
    while turns.due:
        app = next(at.app for at in turns.due if claimed(turns, at.app, now, spoken))
        turns, at, _ = claimed(turns, app, now, spoken) or pytest.fail("a place is claimed")
        order.append(at)
        turns = released(turns)
    return order


def test_a_spoken_place_is_claimed_before_the_written_places_and_each_kind_in_the_order_it_became_due():
    turns = _due_in_order(
        (TABLE_AT_A, Waiting(1)),
        (CALL_AT_B, Lines("g1", frozenset({1}), missed=False)),
        (CALL_AT_A, Lines("g1", frozenset({2}), missed=False)),
    )

    assert list(turns.due) == [TABLE_AT_A, CALL_AT_B, CALL_AT_A]
    first = claimed(turns, CALL_AT_B.app, NOW, _spoken_at(CALL_AT_B, CALL_AT_A))
    assert first is not None
    assert first[1] == CALL_AT_B
    assert claimed(turns, TABLE_AT_A.app, NOW, _spoken_at(CALL_AT_B, CALL_AT_A)) is None


def test_a_written_place_passed_over_for_a_spoken_one_is_claimed_before_the_next_spoken_place():
    spoken = _spoken_at(CALL_AT_A, CALL_AT_B)
    turns = _due_in_order(
        (TABLE_AT_A, Waiting(1)),
        (CALL_AT_A, Lines("g1", frozenset({1}), missed=False)),
        (CALL_AT_B, Lines("g1", frozenset({2}), missed=False)),
    )

    assert _claim_in_order(turns, spoken) == [CALL_AT_A, TABLE_AT_A, CALL_AT_B]


def test_a_written_place_that_became_due_after_the_spoken_place_is_not_passed_over():
    spoken = _spoken_at(CALL_AT_A, CALL_AT_B)
    turns = _due_in_order(
        (CALL_AT_A, Lines("g1", frozenset({1}), missed=False)),
        (TABLE_AT_A, Waiting(1)),
        (CALL_AT_B, Lines("g1", frozenset({2}), missed=False)),
    )

    after, _, _ = claimed(turns, CALL_AT_A.app, NOW, spoken) or pytest.fail("the spoken place is first")

    assert after.gave_way == frozenset()
    assert _claim_in_order(turns, spoken) == [CALL_AT_A, CALL_AT_B, TABLE_AT_A]


def test_a_written_place_whose_turn_ran_to_its_end_gives_way_again_to_the_next_spoken_place():
    spoken = _spoken_at(CALL_AT_A)
    turns = _due_in_order((TABLE_AT_A, Waiting(1)), (CALL_AT_A, Lines("g1", frozenset({1}), missed=False)))
    turns, _, _ = claimed(turns, CALL_AT_A.app, NOW, spoken) or pytest.fail("the spoken place is first")
    turns, table, _ = claimed(released(turns), TABLE_AT_A.app, NOW, spoken) or pytest.fail("the table is next")
    assert turns.gave_way == {TABLE_AT_A}

    after = released(turns)

    assert after.gave_way == frozenset()
    assert table == TABLE_AT_A


def _running_the_table(spoken, *, due=()):
    turns = _due_in_order((TABLE_AT_A, Waiting(1)))
    turns, at, held = claimed(turns, TABLE_AT_A.app, NOW, spoken) or pytest.fail("the table is claimed")
    for place, occasion in due:
        turns = with_due(turns, place, occasion, NOW + 1)
    return turns, Claim(at, held, mark=0, aloud=False)


@pytest.mark.parametrize(
    ("spoken_due", "expected"),
    [
        ((CALL_AT_A, Lines("g1", frozenset({1}), missed=False)), True),
        ((CALL_AT_B, Lines("g1", frozenset({1}), missed=False)), True),
    ],
    ids=["at its app", "at another app"],
)
def test_a_running_written_turn_yields_to_a_spoken_place_that_became_due(spoken_due, expected):
    spoken = _spoken_at(CALL_AT_A, CALL_AT_B)
    turns, claim = _running_the_table(spoken, due=[spoken_due])

    assert yields_to_spoken(turns, claim, spoken) is expected


def test_a_running_written_turn_does_not_yield_when_only_written_places_are_due_or_none_is():
    turns, claim = _running_the_table(_nothing_spoken, due=[(CALL_AT_A, Lines("g1", frozenset({1}), missed=False))])

    assert not yields_to_spoken(turns, claim, _nothing_spoken)
    assert not yields_to_spoken(replace(turns, due={}), claim, _spoken_at(CALL_AT_A))


def test_a_spoken_running_turn_never_yields():
    spoken = _spoken_at(CALL_AT_A, CALL_AT_B)
    turns = _due_in_order((CALL_AT_A, Lines("g1", frozenset({1}), missed=False)))
    turns, at, held = claimed(turns, CALL_AT_A.app, NOW, spoken) or pytest.fail("claimed")
    turns = with_due(turns, CALL_AT_B, Lines("g1", frozenset({2}), missed=False), NOW)

    assert not yields_to_spoken(turns, Claim(at, held, mark=0, aloud=True), spoken)


@pytest.mark.parametrize("gave_way", ["passed over", "cut"])
def test_a_written_turn_that_gave_way_once_is_not_cut_again(gave_way):
    spoken = _spoken_at(CALL_AT_A)
    turns, claim = _running_the_table(spoken, due=[(CALL_AT_A, Lines("g1", frozenset({1}), missed=False))])
    turns = replace(turns, **{"gave_way": {TABLE_AT_A}} if gave_way == "passed over" else {"cut": {TABLE_AT_A}})

    assert not yields_to_spoken(turns, claim, spoken)


def test_a_turn_cut_for_a_spoken_place_is_due_again_first_with_what_arrived_joined_and_goes_after_the_spoken_place():
    spoken = _spoken_at(CALL_AT_A)
    meanwhile = Lines("g1", frozenset({3}), missed=False)
    claim = Claim(TABLE_AT_A, Due(Waiting(2), NOW - 7), mark=0, aloud=False)
    turns = replace(_due_in_order((CALL_AT_A, meanwhile)), running=TABLE_AT_A)

    after = cut_for_spoken(turns, claim)

    assert after.running is None
    assert list(after.due) == [TABLE_AT_A, CALL_AT_A]
    assert (after.due[TABLE_AT_A].occasion, after.due[TABLE_AT_A].since) == (Waiting(2), NOW - 7)
    assert _claim_in_order(after, spoken) == [CALL_AT_A, TABLE_AT_A]


def test_a_place_cut_for_lines_joins_the_lines_that_arrived_meanwhile():
    claim = Claim(CALL_AT_B, Due(Lines("g1", frozenset({1, 2}), missed=False), NOW - 7), mark=0, aloud=False)
    turns = replace(
        _due_in_order((CALL_AT_A, Waiting(1)), (CALL_AT_B, Lines("g1", frozenset({3}), missed=False))),
        running=CALL_AT_B,
    )

    after = cut_for_spoken(turns, claim)

    assert after.due[CALL_AT_B].occasion == Lines("g1", frozenset({1, 2, 3}), missed=False)


def test_a_place_made_due_by_a_hold_alone_is_not_due_again_after_its_turn_was_cut():
    claim = Claim(CALL_AT_B, Due(Lines("g1", frozenset(), missed=False), NOW), mark=0, aloud=False)

    after = cut_for_spoken(Turns(running=CALL_AT_B), claim)

    assert after.due == {}


def test_a_place_that_was_cut_and_whose_spoken_place_is_gone_runs_without_being_cut_again():
    spoken = _spoken_at(CALL_AT_A)
    claim = Claim(TABLE_AT_A, Due(Waiting(2), NOW), mark=0, aloud=False)
    turns = cut_for_spoken(replace(_due_in_order(), running=TABLE_AT_A), claim)
    turns = without_due(
        with_due(turns, CALL_AT_A, Lines("g1", frozenset({1}), missed=False), NOW), lambda at: at == CALL_AT_A
    )
    turns, at, held = claimed(turns, TABLE_AT_A.app, NOW, spoken) or pytest.fail("the table is claimed")
    turns = with_due(turns, CALL_AT_A, Lines("g1", frozenset({2}), missed=False), NOW)

    assert not yields_to_spoken(turns, Claim(at, held, mark=0, aloud=False), spoken)
