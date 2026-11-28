"""The memory's search against a real Postgres: what of the record it finds by their words, in which room, and what it
leaves to the lines table."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceRef, PlaceView, Spoken

from personas.entries import (
    NO_TURNS_KEPT,
    ActDone,
    Answered,
    HistoryDelta,
    KeptLine,
    Room,
    SaidAloud,
    ToldEntry,
    ViewSeen,
    Visit,
)
from personas.memory import PostgresHistoryStore, PostgresLineStore
from tests.support.call import PLACE_ID
from tests.support.fakes import WALL_START
from tests.support.memory_search import (
    AGENT,
    APP,
    SECOND_AGENT,
    a_read_in,
    a_search,
    a_visit,
    texts_found,
)
from tests.support.standings import line

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from personas.config import DatabaseTarget

pytestmark = [pytest.mark.anyio, pytest.mark.postgres, pytest.mark.integration]


@pytest.fixture
async def store(database: DatabaseTarget) -> AsyncIterator[PostgresLineStore]:
    kept = PostgresLineStore(database)
    yield kept
    await kept.close()


@pytest.fixture
async def histories(database: DatabaseTarget) -> AsyncIterator[PostgresHistoryStore]:
    kept = PostgresHistoryStore(database)
    yield kept
    await kept.close()


CALL_ROOM = PlaceRef(kind="call", id=PLACE_ID)
VISIT = "visit-1"


def hours_in(hours: float) -> datetime:
    return WALL_START + timedelta(hours=hours)


def a_view_of(state: dict[str, object], *, hours: float = 1, generation: str | None = VISIT) -> ViewSeen:
    return ViewSeen(APP, CALL_ROOM, PlaceView(revision=2, awaits_you=True, state=state), hours_in(hours), generation)


def a_call_of(
    tool: str, arguments: dict[str, object], result: str, *, hours: float = 1, room: Room | None = None
) -> ActDone:
    made_in = Room(APP, CALL_ROOM, VISIT) if room is None else room
    return ActDone(APP, tool, arguments, Answered(result, is_error=False), hours_in(hours), made_in)


def a_sentence_of(text: str, *, hours: float = 1, generation: str | None = VISIT) -> SaidAloud:
    return SaidAloud(APP, CALL_ROOM, text, Spoken(spoken=True), hours_in(hours), generation)


def a_line_of(seq: int, text: str, *, hours: float = 1, generation: str = VISIT) -> KeptLine:
    return KeptLine(
        Visit(APP, CALL_ROOM, generation),
        seq,
        hours_in(hours),
        "Ada",
        "message",
        text,
        own=False,
        spoken=False,
        cut=False,
        act=False,
    )


async def recorded(
    histories: PostgresHistoryStore, *entries: ToldEntry, client_id: str = AGENT, first: int = 0
) -> None:
    """The record takes these entries at the ordinals from ``first``, as his history took them."""
    generation = await histories.take_over(client_id)
    delta = HistoryDelta(first, tuple(entries), 0, (), 0, kept_first=first, kept=tuple(entries))
    await histories.write(client_id, generation, delta, NO_TURNS_KEPT)


A_CALL = a_call_of("auction", {"amount": "seventeen"}, "accepted")


@pytest.mark.parametrize(
    ("words", "found"),
    [
        pytest.param("zebra", a_view_of({"hand": "zebra", "bid": 12}), id="a view revision by a value of its state"),
        pytest.param("12", a_view_of({"hand": "zebra", "bid": 12}), id="a view revision by a number of its state"),
        pytest.param("auction", A_CALL, id="a call by its tool"),
        pytest.param(
            "seventeen", a_call_of("auction", {"amount": "seventeen"}, "accepted"), id="a call by an argument"
        ),
        pytest.param("accepted", a_call_of("auction", {"amount": "seventeen"}, "accepted"), id="a call by its result"),
        pytest.param("Morgen", a_sentence_of("Guten Morgen"), id="a sentence"),
    ],
)
async def test_an_entry_that_left_his_history_is_found_by_a_word_of_its_own_kind_in_the_turns_room(
    store, histories, words, found
):
    """TURNS-39, TURNS-40."""
    await recorded(
        histories,
        a_view_of({"hand": "zebra", "bid": 12}),
        A_CALL,
        a_sentence_of("Guten Morgen"),
        a_line_of(1, "something else"),
    )

    assert await store.search(a_search(words)) == (found,)


@pytest.mark.parametrize(
    "name",
    [
        *("hand", "bid", "amount", "arguments", "outcome", "tool", "text", "is_error", "view", "act", "speech"),
        *("line", "message", "Ada", "app", "test", "mcp", "call", "room", "visit", "state", "revision", "kind"),
    ],
)
async def test_a_key_a_kind_an_app_or_a_place_name_matches_nothing(store, histories, name):
    """TURNS-40."""
    await recorded(
        histories,
        a_view_of({"hand": "zebra", "bid": 12}),
        A_CALL,
        a_sentence_of("Guten Morgen"),
        a_line_of(1, "something else"),
    )

    assert await store.search(a_search(name)) == ()


async def test_every_word_must_match_in_one_entry_and_no_word_gives_every_entry_of_every_kind_newest_first(
    store, histories
):
    """TURNS-40."""
    view = a_view_of({"hand": "red fox"}, hours=1)
    call = a_call_of("auction", {"amount": "red"}, "fox sold", hours=2)
    sentence = a_sentence_of("a red dog", hours=3)
    await recorded(histories, view, call, sentence)

    assert await store.search(a_search("red fox")) == (call, view)
    assert await store.search(a_search("")) == (sentence, call, view)
    assert await store.search(a_search("red dog fox")) == ()


async def test_a_search_gives_only_the_entries_below_the_ordinal_his_history_begins_at_whatever_their_time(
    store, histories
):
    """TURNS-40."""
    first, second, third = a_sentence_of("one", hours=5), a_sentence_of("two", hours=4), a_sentence_of("three", hours=1)
    await recorded(histories, first, second, third)

    below_two = await store.search(a_search(first_ordinal=2, first_read_before=hours_in(0)))
    newest_one = await store.search(a_search(rows=1))

    assert below_two == (first, second)
    assert newest_one == (first,)


async def test_a_search_gives_no_entry_of_another_agent_app_place_or_kind_of_place(store, histories):
    """TURNS-39."""
    elsewhere = [
        replace(a_sentence_of("Said"), app="https://other.test/mcp"),
        replace(a_sentence_of("Said"), place=PlaceRef(kind="call", id="room-9")),
        replace(a_sentence_of("Said"), place=PlaceRef(kind="table", id=PLACE_ID)),
    ]
    await recorded(histories, a_sentence_of("Said here"), *elsewhere)
    await recorded(histories, a_sentence_of("Said to another agent"), client_id=SECOND_AGENT)

    assert await store.search(a_search("Said")) == (a_sentence_of("Said here"),)


async def test_a_call_without_a_room_is_given_to_no_search_and_a_view_and_a_sentence_without_a_visit_are_found(
    store, histories
):
    """TURNS-39."""
    from_before_the_room = ActDone(APP, "auction", {}, Answered("accepted", is_error=False), hours_in(1), None)
    unvisited_view = a_view_of({"hand": "zebra"}, generation=None)
    unvisited_sentence = a_sentence_of("Guten Morgen", generation=None)
    await recorded(histories, from_before_the_room, unvisited_view, unvisited_sentence)

    assert await store.search(a_search("")) == (unvisited_sentence, unvisited_view)
    assert await store.search(a_search("auction")) == ()


async def test_a_line_both_tables_hold_is_given_once_from_the_record_matched_by_its_visit_and_seq_not_by_time(
    store, histories
):
    """TURNS-40."""
    in_both = a_line_of(1, "In both tables", hours=5)
    await store.keep(a_read_in(a_visit(VISIT), line(1, "In both tables"), line(2, "Only a line"), hours_after_start=1))
    await store.keep(a_read_in(a_visit("another"), line(1, "In both tables"), hours_after_start=2))
    await recorded(histories, in_both)

    found = await store.search(a_search("both"))

    assert found == (in_both, replace(in_both, visit=a_visit("another"), first_read_at=hours_in(2)))
    assert texts_found(await store.search(a_search("Only"))) == ["Only a line"]


async def test_a_line_only_the_lines_table_holds_is_found_only_before_the_time_his_history_begins(store, histories):
    """TURNS-40."""
    await store.keep(a_read_in(a_visit(VISIT), line(1, "Before"), hours_after_start=1))
    await store.keep(a_read_in(a_visit(VISIT), line(2, "After"), hours_after_start=3))
    await recorded(histories, a_sentence_of("Recorded", hours=4))

    found = await store.search(a_search(first_read_before=hours_in(2)))

    assert [getattr(each, "text", None) for each in found] == ["Recorded", "Before"]
