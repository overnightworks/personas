"""One agent's history across his apps: what it holds after the lines he took in, within its bound."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime

from agents.member import PlaceRef, PlaceView

from personas.decision import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    AppPlace,
    Lines,
    Occasion,
    TurnTold,
    Waiting,
    turn_messages,
)
from personas.entries import KeptLine, ViewSeen, Visit, entry_size
from personas.mind import Mind

READ_AT = datetime(2026, 11, 11, 20, 0, tzinfo=UTC)
CALL = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-7"), "generation-1")
OTHER_CALL = Visit("https://other.test/mcp", PlaceRef(kind="call", id="room-8"), "generation-1")


NOW = 1000.0
BUDGET_PAUSE_CAP_SECONDS = 300.0
A_TURN = TurnTold(READ_AT, CALL.app, CALL.place, Waiting(1), aloud=False, through=0)


def said(visit: Visit, seq: int, text: str) -> KeptLine:
    return KeptLine(visit, seq, READ_AT, "Ada", "message", text, own=False, spoken=False, cut=False, act=False)


def test_a_mind_starts_with_an_empty_history_that_dropped_nothing():
    history = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS).history

    assert history.entries == ()
    assert history.older_dropped is False


def test_the_lines_of_every_visit_join_his_one_history_in_the_order_taken():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)

    mind.take((said(CALL, 1, "first"), said(CALL, 2, "second")))
    mind.take((said(OTHER_CALL, 1, "third"),))

    assert [(line.visit.app, line.text) for line in mind.history.entries] == [
        (CALL.app, "first"),
        (CALL.app, "second"),
        (OTHER_CALL.app, "third"),
    ]


def test_a_history_read_earlier_does_not_change_when_more_lines_are_taken():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.take((said(CALL, 1, "first"),))
    before = mind.history

    mind.take((said(CALL, 2, "second"),))

    assert [line.text for line in before.entries] == ["first"]


def test_a_take_sets_the_event_read_before_it_and_the_next_take_sets_a_new_one():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    first = mind.next_take()

    mind.take((said(CALL, 1, "first"),))
    second = mind.next_take()

    assert first.is_set() is True
    assert second.is_set() is False
    assert second is not first


def test_taking_nothing_leaves_the_event_read_before_unset():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    waiting = mind.next_take()

    mind.take(())

    assert waiting.is_set() is False
    assert mind.next_take() is waiting


def test_the_views_he_read_join_his_lines_in_the_order_read_and_a_revision_he_holds_adds_nothing():
    table = PlaceRef(kind="table", id="table-1")
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)

    mind.take(
        (said(CALL, 1, "first"), ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=False, state={}), READ_AT))
    )
    mind.take((ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=False, state={}), READ_AT),))
    mind.take(())

    assert [type(each) for each in mind.history.entries] == [KeptLine, ViewSeen]


TABLE_AT_A = AppPlace(CALL.app, PlaceRef(kind="table", id="table-1"))
CALL_AT_B = AppPlace(OTHER_CALL.app, OTHER_CALL.place)


def _claimed(mind: Mind, app: str, now: float) -> tuple[AppPlace, Occasion] | None:
    claim = mind.claim(app, now)
    return None if claim is None else (claim[0], claim[1].occasion)


def test_one_turn_runs_at_a_time_and_the_next_goes_to_the_place_that_waited_longest_once_it_ended():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.due(CALL_AT_B, Waiting(1), NOW)
    mind.due(TABLE_AT_A, Waiting(2), NOW)

    assert mind.claim(TABLE_AT_A.app, NOW) is None
    assert _claimed(mind, CALL_AT_B.app, NOW) == (CALL_AT_B, Waiting(1))
    assert mind.claim(TABLE_AT_A.app, NOW) is None

    mind.release()

    assert _claimed(mind, TABLE_AT_A.app, NOW) == (TABLE_AT_A, Waiting(2))


def test_a_turn_without_an_answer_is_due_again_once_the_running_turn_ends_and_a_newer_occasion_stays():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.due(TABLE_AT_A, Waiting(1), NOW)
    assert _claimed(mind, TABLE_AT_A.app, NOW) == (TABLE_AT_A, Waiting(1))

    mind.due(TABLE_AT_A, Waiting(2), NOW)
    mind.due_again(TABLE_AT_A, Waiting(1), NOW)
    mind.due_again(CALL_AT_B, Waiting(1), NOW)
    assert mind.claim(TABLE_AT_A.app, NOW) is None
    mind.release()

    assert _claimed(mind, TABLE_AT_A.app, NOW) == (TABLE_AT_A, Waiting(2))
    mind.release()
    assert _claimed(mind, CALL_AT_B.app, NOW) == (CALL_AT_B, Waiting(1))


def test_the_wait_after_a_turn_without_an_answer_holds_every_claim_until_it_is_over_whatever_ends_the_turn():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.due(TABLE_AT_A, Waiting(1), NOW)
    mind.due(CALL_AT_B, Waiting(1), NOW)
    assert _claimed(mind, TABLE_AT_A.app, NOW) == (TABLE_AT_A, Waiting(1))

    wait = mind.unanswered(budget_spent=False, now=NOW)
    mind.release()
    before_the_end = NOW + wait / 2

    assert mind.claim(CALL_AT_B.app, before_the_end) is None
    assert mind.pause_left(CALL_AT_B.app, before_the_end) == wait / 2
    assert _claimed(mind, CALL_AT_B.app, NOW + wait) == (CALL_AT_B, Waiting(1))


def test_the_end_of_a_wait_wakes_every_loop_of_his():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    events = [mind.wake_event(), mind.wake_event()]

    mind.wake()

    assert [event.is_set() for event in events] == [True, True]


def test_the_wait_after_turns_without_an_answer_doubles_per_turn_in_a_row_and_starts_over_after_an_answer():
    mind = Mind(1_000, 1_000, BACKOFF_CAP_SECONDS * 4)

    waits = [mind.unanswered(budget_spent=False, now=NOW) for _ in range(2)]
    mind.answered(A_TURN)
    waits.append(mind.unanswered(budget_spent=True, now=NOW))

    assert waits == [BACKOFF_BASE_SECONDS, 2 * BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS]


def test_the_wait_after_a_spent_budget_reaches_its_cap_and_after_any_other_stop_the_backoff_cap():
    cap = BACKOFF_CAP_SECONDS * 4
    mind = Mind(1_000, 1_000, budget_pause_cap_seconds=cap)

    other_stops = [mind.unanswered(budget_spent=False, now=NOW) for _ in range(10)]
    mind.answered(A_TURN)
    budget_stops = [mind.unanswered(budget_spent=True, now=NOW) for _ in range(10)]

    assert (max(other_stops), max(budget_stops)) == (BACKOFF_CAP_SECONDS, cap)


def test_the_end_of_a_turn_wakes_every_loop_of_the_agent():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    events = [mind.wake_event(), mind.wake_event()]

    mind.release()

    assert [each.is_set() for each in events] == [True, True]


def test_a_place_dropped_while_no_turn_runs_wakes_every_loop_and_one_dropped_during_a_turn_wakes_none():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    events = [mind.wake_event(), mind.wake_event()]
    mind.due(CALL_AT_B, Lines("generation-1", frozenset({1}), missed=False), NOW)
    mind.due(TABLE_AT_A, Waiting(1), NOW)
    assert mind.claim(CALL_AT_B.app, NOW) is not None

    mind.undue(lambda at: at == TABLE_AT_A)
    assert [each.is_set() for each in events] == [False, False]

    mind.release()
    for each in events:
        each.clear()
    mind.undue(lambda at: at == CALL_AT_B)
    assert [each.is_set() for each in events] == [False, False]

    mind.due(TABLE_AT_A, Waiting(2), NOW)
    mind.undue(lambda at: at == TABLE_AT_A)
    assert [each.is_set() for each in events] == [True, True]
    assert mind.claim(TABLE_AT_A.app, NOW) is None


def test_a_mind_holding_lines_views_and_due_places_shows_no_name_no_text_and_no_state_in_its_repr():
    marker = secrets.token_hex(8)
    table = PlaceRef(kind="table", id="table-1")
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.take(
        (
            KeptLine(CALL, 1, READ_AT, marker, "message", marker, own=False, spoken=False, cut=False, act=False),
            ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=True, state={"hand": marker}), READ_AT),
        )
    )
    mind.due(CALL_AT_B, Lines(OTHER_CALL.generation, frozenset({1}), missed=False), NOW)

    assert marker not in repr(mind)


def test_he_holds_the_apps_where_he_has_an_active_place_other_than_the_one_asking_until_it_is_recorded_gone():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)

    mind.stand_at(CALL.app, holds=True)
    mind.stand_at(OTHER_CALL.app, holds=True)
    mind.stand_at(OTHER_CALL.app, holds=True)

    assert mind.held_apps(CALL.app) == {OTHER_CALL.app}
    assert mind.held_apps("https://third.test/mcp") == {CALL.app, OTHER_CALL.app}

    mind.stand_at(OTHER_CALL.app, holds=False)
    mind.stand_at("https://third.test/mcp", holds=False)

    assert mind.held_apps(CALL.app) == frozenset()


LINE_BYTES = entry_size(said(CALL, 1, "x"))


def a_lines_turn(mind: Mind, *seqs: int) -> TurnTold:
    return TurnTold(
        READ_AT,
        CALL.app,
        CALL.place,
        Lines(CALL.generation, frozenset(seqs), missed=False),
        aloud=False,
        through=mind.history.taken,
    )


def answer_turn_for(mind: Mind, first_seq: int) -> list[str]:
    """Takes one line, has the turn for it answered and returns the messages that turn was sent."""
    mind.take((said(CALL, first_seq, "x"),))
    turn = a_lines_turn(mind, first_seq)
    sent = turn_messages("Mozart", mind.present, turn)
    mind.answered(turn)
    return list(sent)


def seqs_told(message: str) -> list[int]:
    return [each["seq"] for each in json.loads(message)["entries"]]


def test_passing_the_history_bound_changes_the_earlier_messages_once_and_later_turns_append_again():
    mind = Mind(3 * LINE_BYTES, 2 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    for seq in (1, 2, 3):
        sent = answer_turn_for(mind, seq)
    third_as_sent = sent[-1]

    stepped = answer_turn_for(mind, 4)
    appended = answer_turn_for(mind, 5)

    assert [json.loads(stepped[0])["older_dropped"], stepped[1]] == [True, third_as_sent]
    assert [seqs_told(each) for each in stepped[1:]] == [[3], [4]]
    assert appended[:-1] == stepped
    assert seqs_told(appended[-1]) == [5]


def test_a_turn_in_flight_whose_message_left_in_a_step_is_not_told_and_every_entry_is_told_once():
    mind = Mind(6 * LINE_BYTES, 3 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    mind.take((said(CALL, 1, "x"), said(CALL, 2, "x")))
    in_flight = a_lines_turn(mind, 2)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(3, 9)))
    mind.answered(in_flight)
    mind.take((said(CALL, 9, "x"),))

    sent = turn_messages("Mozart", mind.present, a_lines_turn(mind, 9))

    told = [seq for each in sent[1:] for seq in seqs_told(each)]
    assert told == [entry.seq for entry in mind.history.entries]


def test_a_turn_whose_message_a_step_cut_names_only_the_lines_it_still_tells():
    mind = Mind(6 * LINE_BYTES, 4 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(1, 7)))
    in_flight = a_lines_turn(mind, *range(1, 7))
    mind.take(tuple(said(CALL, seq, "x") for seq in range(7, 10)))
    mind.answered(in_flight)

    sent = turn_messages("Mozart", mind.present, a_lines_turn(mind, 9))

    message = sent[1]
    assert (json.loads(message)["woke"]["seqs"], seqs_told(message)) == ([6], [6])


def test_a_newest_message_past_the_kept_bytes_loses_its_oldest_entries_and_keeps_its_newest():
    mind = Mind(4 * LINE_BYTES, 2 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    answer_turn_for(mind, 1)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(2, 7)))

    sent = turn_messages("Mozart", mind.present, a_lines_turn(mind, 6))

    assert [seqs_told(each) for each in sent[1:]] == [[5, 6]]
    assert json.loads(sent[0])["older_dropped"] is True


def test_the_newest_entry_stays_when_it_alone_is_past_the_kept_bytes():
    mind = Mind(4 * LINE_BYTES, 1, BUDGET_PAUSE_CAP_SECONDS)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(1, 6)))

    sent = turn_messages("Mozart", mind.present, a_lines_turn(mind, 5))

    assert [seqs_told(each) for each in sent[1:]] == [[5]]
