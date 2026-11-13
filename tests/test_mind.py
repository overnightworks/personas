"""One agent's history across his apps: what it holds after the lines he took in, within its bound."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from agents.member import PlaceRef, PlaceView

from personas.decision import AppPlace, Lines, Waiting
from personas.entries import KeptLine, ViewSeen, Visit
from personas.mind import Mind

READ_AT = datetime(2026, 11, 11, 20, 0, tzinfo=UTC)
CALL = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-7"), "generation-1")
OTHER_CALL = Visit("https://other.test/mcp", PlaceRef(kind="call", id="room-8"), "generation-1")


def said(visit: Visit, seq: int, text: str) -> KeptLine:
    return KeptLine(visit, seq, READ_AT, "Ada", "message", text, own=False, spoken=False, cut=False, act=False)


def test_a_mind_starts_with_an_empty_history_that_dropped_nothing():
    history = Mind(size_bytes=1_000).history

    assert history.entries == ()
    assert history.older_dropped is False


def test_the_lines_of_every_visit_join_his_one_history_in_the_order_taken():
    mind = Mind(size_bytes=1_000)

    mind.take((said(CALL, 1, "first"), said(CALL, 2, "second")))
    mind.take((said(OTHER_CALL, 1, "third"),))

    assert [(line.visit.app, line.text) for line in mind.history.entries] == [
        (CALL.app, "first"),
        (CALL.app, "second"),
        (OTHER_CALL.app, "third"),
    ]


def test_a_history_read_earlier_does_not_change_when_more_lines_are_taken():
    mind = Mind(size_bytes=1_000)
    mind.take((said(CALL, 1, "first"),))
    before = mind.history

    mind.take((said(CALL, 2, "second"),))

    assert [line.text for line in before.entries] == ["first"]


def test_the_views_he_read_join_his_lines_in_the_order_read_and_a_revision_he_holds_adds_nothing():
    table = PlaceRef(kind="table", id="table-1")
    mind = Mind(size_bytes=1_000)

    mind.take(
        (said(CALL, 1, "first"), ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=False, state={}), READ_AT))
    )
    mind.take((ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=False, state={}), READ_AT),))
    mind.take(())

    assert [type(each) for each in mind.history.entries] == [KeptLine, ViewSeen]


TABLE_AT_A = AppPlace(CALL.app, PlaceRef(kind="table", id="table-1"))
CALL_AT_B = AppPlace(OTHER_CALL.app, OTHER_CALL.place)


def test_one_turn_runs_at_a_time_and_the_next_goes_to_the_place_that_waited_longest_once_it_ended():
    mind = Mind(size_bytes=1_000)
    mind.due(CALL_AT_B, Waiting(1))
    mind.due(TABLE_AT_A, Waiting(2))

    assert mind.claim(TABLE_AT_A.app) is None
    assert mind.claim(CALL_AT_B.app) == (CALL_AT_B, Waiting(1))
    assert mind.claim(TABLE_AT_A.app) is None

    mind.release()

    assert mind.claim(TABLE_AT_A.app) == (TABLE_AT_A, Waiting(2))


def test_the_end_of_a_turn_wakes_every_loop_of_the_agent():
    mind = Mind(size_bytes=1_000)
    events = [mind.wake_event(), mind.wake_event()]

    mind.release()

    assert [each.is_set() for each in events] == [True, True]


def test_a_place_dropped_while_no_turn_runs_wakes_every_loop_and_one_dropped_during_a_turn_wakes_none():
    mind = Mind(size_bytes=1_000)
    events = [mind.wake_event(), mind.wake_event()]
    mind.due(CALL_AT_B, Lines("generation-1", frozenset({1}), missed=False))
    mind.due(TABLE_AT_A, Waiting(1))
    assert mind.claim(CALL_AT_B.app) is not None

    mind.undue(lambda at: at == TABLE_AT_A)
    assert [each.is_set() for each in events] == [False, False]

    mind.release()
    for each in events:
        each.clear()
    mind.undue(lambda at: at == CALL_AT_B)
    assert [each.is_set() for each in events] == [False, False]

    mind.due(TABLE_AT_A, Waiting(2))
    mind.undue(lambda at: at == TABLE_AT_A)
    assert [each.is_set() for each in events] == [True, True]
    assert mind.claim(TABLE_AT_A.app) is None


def test_a_mind_holding_lines_views_and_due_places_shows_no_name_no_text_and_no_state_in_its_repr():
    marker = secrets.token_hex(8)
    table = PlaceRef(kind="table", id="table-1")
    mind = Mind(size_bytes=1_000)
    mind.take(
        (
            KeptLine(CALL, 1, READ_AT, marker, "message", marker, own=False, spoken=False, cut=False, act=False),
            ViewSeen(CALL.app, table, PlaceView(revision=1, awaits_you=True, state={"hand": marker}), READ_AT),
        )
    )
    mind.due(CALL_AT_B, Lines(OTHER_CALL.generation, frozenset({1}), missed=False))

    assert marker not in repr(mind)


def test_he_holds_the_apps_where_he_has_an_active_place_other_than_the_one_asking_until_it_is_recorded_gone():
    mind = Mind(size_bytes=1_000)

    mind.stand_at(CALL.app, holds=True)
    mind.stand_at(OTHER_CALL.app, holds=True)
    mind.stand_at(OTHER_CALL.app, holds=True)

    assert mind.held_apps(CALL.app) == {OTHER_CALL.app}
    assert mind.held_apps("https://third.test/mcp") == {CALL.app, OTHER_CALL.app}

    mind.stand_at(OTHER_CALL.app, holds=False)
    mind.stand_at("https://third.test/mcp", holds=False)

    assert mind.held_apps(CALL.app) == frozenset()
