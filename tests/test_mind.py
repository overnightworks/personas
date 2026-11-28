"""One agent's history across his apps: what it holds after the lines he took in, within its bound."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from functools import partial

import pytest
from agents.member import PlaceRef, PlaceView

from personas.decision import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    TURN_CANCELLED_CODE,
    AppPlace,
    Joined,
    Paused,
    Restarted,
)
from personas.entries import (
    ActDone,
    Entry,
    KeptLine,
    Lines,
    Occasion,
    Reading,
    SaidAloud,
    SkillRead,
    ToolsLoaded,
    TurnTold,
    Unanswered,
    ViewSeen,
    Visit,
    Waiting,
    entry_size,
)
from personas.mind import Mind
from personas.present import round_call_id, turn_messages
from personas.speech import Answered, HoldKey
from tests.support.history import message_texts

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


def test_the_views_he_read_join_his_lines_in_the_order_read_and_a_revision_he_holds_adds_nothing():
    table = PlaceRef(kind="table", id="table-1")
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)

    mind.take(
        (said(CALL, 1, "first"), ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=False, state={}), READ_AT))
    )
    mind.take((ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=False, state={}), READ_AT),))
    mind.take(())

    assert [type(each) for each in mind.history.entries] == [KeptLine, ViewSeen]


@pytest.mark.parametrize(
    ("checkpoint", "generation"),
    [
        (Reading("generation-3", 5), "generation-3"),
        (Joined(), None),
        (Restarted(), None),
        (Paused(), None),
        (None, None),
    ],
    ids=["a read window", "a joined place", "a restarted place", "a paused place", "no checkpoint"],
)
def test_the_visit_of_an_entry_is_the_generation_of_the_places_last_read_and_none_without_one(checkpoint, generation):
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    at = AppPlace(CALL.app, CALL.place)
    mind.move_checkpoint(at, checkpoint)

    assert mind.generation_at(at) == generation


TABLE_AT_A = AppPlace(CALL.app, PlaceRef(kind="table", id="table-1"))
CALL_AT_B = AppPlace(OTHER_CALL.app, OTHER_CALL.place)


def _claimed(mind: Mind, app: str, now: float) -> tuple[AppPlace, Occasion] | None:
    claim = mind.claim(app, now)
    return None if claim is None else (claim[0], claim[1].occasion)


def test_one_turn_runs_at_a_time_and_the_next_goes_to_the_place_that_waited_longest_once_it_ended():
    """TURNS-6."""
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.due(CALL_AT_B, Waiting(1), NOW)
    mind.due(TABLE_AT_A, Waiting(2), NOW)

    assert mind.claim(TABLE_AT_A.app, NOW) is None
    assert _claimed(mind, CALL_AT_B.app, NOW) == (CALL_AT_B, Waiting(1))
    assert mind.claim(TABLE_AT_A.app, NOW) is None

    mind.release()

    assert _claimed(mind, TABLE_AT_A.app, NOW) == (TABLE_AT_A, Waiting(2))


CALL_AT_A = AppPlace(CALL.app, CALL.place)
SAID = Lines("generation-1", frozenset({1}), missed=False)


def _mind_with_a_voice_at_a() -> Mind:
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.voice_at(CALL.app, voice=True)
    return mind


def test_a_written_turn_is_to_be_cut_once_a_spoken_place_is_due_and_not_before():
    mind = _mind_with_a_voice_at_a()
    mind.due(TABLE_AT_A, Waiting(1), NOW)
    assert _claimed(mind, TABLE_AT_A.app, NOW) == (TABLE_AT_A, Waiting(1))
    mind.due(CALL_AT_B, Waiting(1), NOW)
    assert not mind.cut_wanted().is_set()

    mind.due(CALL_AT_A, SAID, NOW)

    assert mind.cut_wanted().is_set()


def test_no_cut_is_wanted_where_no_voice_is_or_while_a_spoken_turn_runs():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.due(TABLE_AT_A, Waiting(1), NOW)
    mind.claim(TABLE_AT_A.app, NOW)
    mind.due(CALL_AT_A, SAID, NOW)
    assert not mind.cut_wanted().is_set()
    spoken = _mind_with_a_voice_at_a()
    spoken.due(CALL_AT_A, SAID, NOW)
    spoken.claim(CALL.app, NOW)

    spoken.due(TABLE_AT_A, Waiting(1), NOW)
    spoken.due(CALL_AT_A, SAID, NOW)

    assert not spoken.cut_wanted().is_set()


def test_a_cut_written_turn_is_due_again_behind_the_spoken_place_and_is_not_cut_for_the_next_one():
    mind = _mind_with_a_voice_at_a()
    mind.due(TABLE_AT_A, Waiting(1), NOW)
    mind.claim(TABLE_AT_A.app, NOW)
    mind.due(CALL_AT_A, SAID, NOW)

    mind.cut_for_spoken()
    mind.release()

    assert _claimed(mind, CALL.app, NOW) == (CALL_AT_A, SAID)
    mind.release()
    assert _claimed(mind, CALL.app, NOW) == (TABLE_AT_A, Waiting(1))
    mind.due(CALL_AT_A, SAID, NOW)
    assert not mind.cut_wanted().is_set()


CUT_IN_FLIGHT = Unanswered(TURN_CANCELLED_CODE)
CALLED_AGAIN = (CALL_AT_B, Waiting(1))
TABLE_WAITING = (TABLE_AT_A, Waiting(2))


@pytest.mark.parametrize(
    ("stopping", "taken", "due_after"),
    [
        (True, (), [CALLED_AGAIN, TABLE_WAITING]),
        (True, (said(CALL, 1, "a line of someone else's"),), [CALLED_AGAIN, TABLE_WAITING]),
        (True, (ActDone(CALL.app, "say", {}, CUT_IN_FLIGHT, READ_AT),), [TABLE_WAITING]),
        (True, (SaidAloud(CALL.app, CALL.place, "Hello.", CUT_IN_FLIGHT, READ_AT),), [TABLE_WAITING]),
        (False, (), [TABLE_WAITING]),
    ],
    ids=["stop-before-it-spoke", "stop-after-others-spoke", "stop-after-a-call", "stop-after-a-sentence", "no-stop"],
)
def test_a_turn_his_stop_cuts_before_it_spoke_leaves_its_place_due_first_and_any_other_cut_turn_leaves_none(
    stopping: bool, taken: tuple[Entry, ...], due_after: list[tuple[AppPlace, Occasion]]
):
    """TURNS-38."""
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    mind.due(CALL_AT_B, Waiting(1), NOW)
    mind.due(TABLE_AT_A, Waiting(2), NOW)
    assert _claimed(mind, CALL_AT_B.app, NOW) == (CALL_AT_B, Waiting(1))
    mind.take(taken)

    if stopping:
        mind.stopping()
    mind.cut()
    mind.release()

    assert [(at, due.occasion) for at, due in mind.turn_state().turns.due.items()] == due_after


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
    """TURNS-37, TURNS-38."""
    mind = Mind(1_000, 1_000, BACKOFF_CAP_SECONDS * 4)

    waits = [mind.unanswered(budget_spent=False, now=NOW) for _ in range(2)]
    mind.answered(A_TURN)
    waits.append(mind.unanswered(budget_spent=True, now=NOW))

    assert waits == [BACKOFF_BASE_SECONDS, 2 * BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS]


def test_the_wait_after_a_spent_budget_reaches_its_cap_and_after_any_other_stop_the_backoff_cap():
    """TURNS-37."""
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
    sent = message_texts(turn_messages("Mozart", mind.present, turn, {}))
    mind.answered(turn)
    return list(sent)


def seqs_told(message: str) -> list[int]:
    return [each["seq"] for each in json.loads(message)["entries"]]


def test_passing_the_history_bound_changes_the_earlier_messages_once_and_later_turns_append_again():
    """TURNS-16."""
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
    """TURNS-10."""
    mind = Mind(6 * LINE_BYTES, 3 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    mind.take((said(CALL, 1, "x"), said(CALL, 2, "x")))
    in_flight = a_lines_turn(mind, 2)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(3, 9)))
    mind.answered(in_flight)
    mind.take((said(CALL, 9, "x"),))

    sent = message_texts(turn_messages("Mozart", mind.present, a_lines_turn(mind, 9), {}))

    told = [seq for each in sent[1:] for seq in seqs_told(each)]
    assert told == [entry.seq for entry in mind.history.entries]


def test_a_turn_whose_message_a_step_cut_names_only_the_lines_it_still_tells():
    mind = Mind(6 * LINE_BYTES, 4 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(1, 7)))
    in_flight = a_lines_turn(mind, *range(1, 7))
    mind.take(tuple(said(CALL, seq, "x") for seq in range(7, 10)))
    mind.answered(in_flight)

    sent = message_texts(turn_messages("Mozart", mind.present, a_lines_turn(mind, 9), {}))

    message = sent[1]
    assert (json.loads(message)["woke"]["seqs"], seqs_told(message)) == ([6], [6])


def test_a_newest_message_past_the_kept_bytes_loses_its_oldest_entries_and_keeps_its_newest():
    """TURNS-16."""
    mind = Mind(4 * LINE_BYTES, 2 * LINE_BYTES, BUDGET_PAUSE_CAP_SECONDS)
    answer_turn_for(mind, 1)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(2, 7)))

    sent = message_texts(turn_messages("Mozart", mind.present, a_lines_turn(mind, 6), {}))

    assert [seqs_told(each) for each in sent[1:]] == [[5, 6]]
    assert json.loads(sent[0])["older_dropped"] is True


def test_the_newest_entry_stays_when_it_alone_is_past_the_kept_bytes():
    """TURNS-16."""
    mind = Mind(4 * LINE_BYTES, 1, BUDGET_PAUSE_CAP_SECONDS)
    mind.take(tuple(said(CALL, seq, "x") for seq in range(1, 6)))

    sent = message_texts(turn_messages("Mozart", mind.present, a_lines_turn(mind, 5), {}))

    assert [seqs_told(each) for each in sent[1:]] == [[5]]


def an_own_spoken_line(visit: Visit, seq: int) -> KeptLine:
    return KeptLine(visit, seq, READ_AT, "Mozart", "message", "x", own=True, spoken=True, cut=False, act=False)


def playing_at(mind: Mind, at: AppPlace) -> bool:
    return mind.speech_saying(at) is not None


def one_sentence_sent(mind: Mind, at: AppPlace) -> None:
    """The sender at the place sent a sentence, which stays in flight."""
    mind.sentences_released(at, ("One.",), 100.0, 30.0, replacing=False)
    sentence, _ = mind.sentence_to_send(at, 0.0)
    assert sentence is not None


def one_sentence_playing(mind: Mind, at: AppPlace) -> None:
    one_sentence_sent(mind, at)
    mind.speech_answered(at, Answered.SPOKEN)


def test_his_own_spoken_line_at_a_place_ends_the_sentence_that_plays_there_and_at_no_other_place():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    here, there = AppPlace(CALL.app, CALL.place), AppPlace(OTHER_CALL.app, OTHER_CALL.place)
    for at in (here, there):
        one_sentence_playing(mind, at)

    mind.take((an_own_spoken_line(CALL, 1),))

    assert (playing_at(mind, here), mind.speech_saying(there)) == (False, "One.")


def test_his_own_spoken_line_ends_the_sentence_that_plays_when_the_history_bound_drops_it_at_once():
    mind = Mind(1, 1, BUDGET_PAUSE_CAP_SECONDS)
    at = AppPlace(CALL.app, CALL.place)
    one_sentence_playing(mind, at)

    mind.take((an_own_spoken_line(CALL, 1),))

    assert not playing_at(mind, at)


def test_the_end_of_the_sender_keeps_the_sentence_that_plays_until_his_own_line_enters():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    at = AppPlace(CALL.app, CALL.place)
    one_sentence_playing(mind, at)

    mind.speech_ended(at)

    assert mind.speech_saying(at) == "One."
    mind.take((an_own_spoken_line(CALL, 1),))
    assert not playing_at(mind, at)


@pytest.mark.parametrize(
    "line",
    [
        KeptLine(CALL, 1, READ_AT, "Ada", "message", "x", own=False, spoken=True, cut=False, act=False),
        KeptLine(CALL, 1, READ_AT, "Mozart", "message", "x", own=True, spoken=False, cut=False, act=False),
        KeptLine(OTHER_CALL, 1, READ_AT, "Mozart", "message", "x", own=True, spoken=True, cut=False, act=False),
    ],
    ids=["another's line", "his written line", "his spoken line at another place"],
)
def test_what_enters_that_is_not_his_own_spoken_line_at_the_place_leaves_the_sentence_playing(line):
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    at = AppPlace(CALL.app, CALL.place)
    one_sentence_playing(mind, at)

    mind.take((line,))

    assert mind.speech_saying(at) == "One."


def test_a_sentence_answered_as_spoken_after_his_own_line_for_it_entered_does_not_play():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    at = AppPlace(CALL.app, CALL.place)
    one_sentence_sent(mind, at)

    mind.take((an_own_spoken_line(CALL, 1),))
    mind.speech_answered(at, Answered.SPOKEN)

    assert not playing_at(mind, at)


def test_a_change_of_the_speech_record_wakes_the_sender_of_its_place_only_and_not_the_minds_own_change():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    here, there = AppPlace(CALL.app, CALL.place), AppPlace(OTHER_CALL.app, OTHER_CALL.place)
    waiting_here, waiting_there = mind.speech_wake(here), mind.speech_wake(there)
    mind.changed().clear()

    mind.sentences_released(here, ("One.",), 100.0, 30.0, replacing=False)

    assert (waiting_here.is_set(), waiting_there.is_set(), mind.changed().is_set()) == (True, False, False)


def test_a_place_dropped_from_the_due_places_keeps_its_decided_hold():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    here = AppPlace(CALL.app, CALL.place)
    mind.hold_decided(here, HoldKey(CALL.generation, "hold-1"))
    mind.due(here, Waiting(1), NOW)

    mind.undue(lambda at: at == here)

    assert mind.hold_decided_key(here) == HoldKey(CALL.generation, "hold-1")


def test_a_place_that_leaves_the_standing_forgets_its_decided_hold_and_keeps_the_request_in_flight():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    here, there = AppPlace(CALL.app, CALL.place), AppPlace(OTHER_CALL.app, OTHER_CALL.place)
    for at in (here, there):
        one_sentence_sent(mind, at)
        mind.hold_decided(at, HoldKey(CALL.generation, "hold-1"))

    mind.left(lambda at: at == here)

    assert (mind.hold_decided_key(here), mind.hold_decided_key(there)) == (None, HoldKey(CALL.generation, "hold-1"))
    assert mind.speech_answered(here, Answered.SPOKEN).sentence.text == "One."


def test_a_place_becomes_due_once_however_often_it_is_made_due_and_again_after_its_turn_claimed_it():
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    lines = Lines("generation-1", frozenset({1}), missed=False)

    first = mind.due(CALL_AT_B, lines, NOW)
    joined = mind.due(CALL_AT_B, Lines("generation-1", frozenset({2}), missed=False), NOW)
    held = mind.due_for_hold(CALL_AT_B, "generation-1", NOW)
    assert (first, joined, held) == (True, False, False)

    assert _claimed(mind, CALL_AT_B.app, NOW) is not None
    mind.release()
    assert mind.due_for_hold(CALL_AT_B, "generation-1", NOW) is True


def test_an_active_place_of_any_kind_in_another_app_counts_as_held_elsewhere_and_a_parked_app_s_does_not():
    """TURNS-1"""
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)
    table = PlaceRef(kind="table", id="table-1")
    mind.hold_places(OTHER_CALL.app, frozenset({table}))

    assert mind.holds_active_place_elsewhere(CALL.app)
    assert not mind.holds_active_place_elsewhere(OTHER_CALL.app)

    mind.hold_places(OTHER_CALL.app, frozenset())

    assert not mind.holds_active_place_elsewhere(CALL.app)


def test_a_place_claimed_in_another_app_counts_as_held_until_released_or_read_again():
    """TURNS-1"""
    mind = Mind(1_000, 1_000, BUDGET_PAUSE_CAP_SECONDS)

    mind.claim_places(OTHER_CALL.app, frozenset({OTHER_CALL.place}))
    assert mind.holds_active_place_elsewhere(CALL.app)

    mind.release_places(OTHER_CALL.app, frozenset({OTHER_CALL.place}))
    assert not mind.holds_active_place_elsewhere(CALL.app)

    mind.claim_places(OTHER_CALL.app, frozenset({OTHER_CALL.place}))
    mind.hold_places(OTHER_CALL.app, frozenset())
    assert not mind.holds_active_place_elsewhere(CALL.app)


@pytest.mark.parametrize(
    ("appending", "entry_type"),
    [
        (
            lambda mind: partial(
                mind.skill_read, CALL.app, "greeting", "skill://greeting/SKILL.md", "Greet people by name.", READ_AT
            ),
            SkillRead,
        ),
        (lambda mind: partial(mind.tools_loaded, CALL.app, "greeting", ("say", "play"), READ_AT), ToolsLoaded),
    ],
    ids=["a skill read", "a tools load"],
)
def test_a_round_after_a_cut_gets_a_call_id_no_kept_entry_holds(appending, entry_type):
    mind = Mind(600, 500, BUDGET_PAUSE_CAP_SECONDS)
    append = appending(mind)
    mind.take((said(CALL, 1, "a" * 150),))
    append()
    mind.take((said(CALL, 2, "b" * 150),))
    append()
    mind.take((said(CALL, 3, "c" * 150),))
    kept_before = len(mind.history.entries)
    append()

    ids = [entry.call_id for entry in mind.history.entries if isinstance(entry, entry_type)]

    assert mind.history.left_out > 0
    assert len(ids) == 2
    assert len(set(ids)) == len(ids)
    assert ids[-1] != round_call_id(kept_before)
