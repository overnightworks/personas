"""One agent's history across his apps: what it holds after the lines he took in, within its bound."""

from __future__ import annotations

from datetime import UTC, datetime

from agents.member import PlaceRef, PlaceView

from personas.decision import KeptLine, ViewSeen, Visit
from personas.mind import Mind

READ_AT = datetime(2026, 11, 11, 20, 0, tzinfo=UTC)
CALL = Visit("https://app.test/mcp", PlaceRef(kind="call", id="room-7"), "generation-1")
OTHER_CALL = Visit("https://other.test/mcp", PlaceRef(kind="call", id="room-8"), "generation-1")


def said(visit: Visit, seq: int, text: str) -> KeptLine:
    return KeptLine(visit, seq, READ_AT, "Ada", "message", text, own=False)


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
