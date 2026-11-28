"""The history's store against a real Postgres: an agent's entries and answered turns in order, fenced by the
generation of the pod that owns the agent."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from agents.member import VoiceState

from personas.entries import (
    NO_TURNS_KEPT,
    HistoryDelta,
    KeptCheckpoint,
    KeptDue,
    KeptHistory,
    KeptTurns,
    Lines,
    Reading,
    TurnTold,
    ViewSeen,
)
from personas.failures import StaleGenerationError, StoreDefect
from personas.memory import PostgresHistoryStore, PostgresMemory
from tests.support.history import APP, LATER, MORNING, TABLE, a_sentence, a_view, an_act, said
from tests.support.postgres import run_sql, uncommitted_behind_a_lock, until_a_statement_waits

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from personas.config import DatabaseTarget

pytestmark = [pytest.mark.anyio, pytest.mark.postgres, pytest.mark.integration]

AGENT = "agent-mozart"
OTHER_AGENT = "agent-ada"
LIMIT_SECONDS = 10


@pytest.fixture
async def histories(database: DatabaseTarget) -> AsyncIterator[PostgresHistoryStore]:
    store = PostgresHistoryStore(database)
    yield store
    await store.close()


def a_turn(through: int, minutes: int = 0) -> TurnTold:
    occasion = Lines(MORNING.generation, frozenset({through}), missed=False)
    return TurnTold(LATER + timedelta(minutes=minutes), APP, MORNING.place, occasion, aloud=True, through=through)


def appended(first: int, *entries, told=(), first_told: int = 0, left_out: int = 0) -> HistoryDelta:
    return HistoryDelta(first, tuple(entries), first_told, tuple(told), left_out)


async def test_a_new_owner_gets_the_next_generation_and_reading_it_gives_the_same(histories):
    assert await histories.generation(AGENT) is None

    first = await histories.take_over(AGENT)
    second = await histories.take_over(AGENT)

    assert second > first
    assert await histories.generation(AGENT) == second
    assert await histories.generation(OTHER_AGENT) is None


async def test_the_history_loads_every_appended_entry_and_answered_turn_in_order_and_no_other_agents(histories):
    view = ViewSeen(APP, TABLE, a_view(3), LATER)
    mine = await histories.take_over(AGENT)
    others = await histories.take_over(OTHER_AGENT)

    await histories.write(AGENT, mine, appended(0, said(1, "first"), an_act(), told=[a_turn(2)]), NO_TURNS_KEPT)
    await histories.write(OTHER_AGENT, others, appended(0, said(1, "not his")), NO_TURNS_KEPT)
    await histories.write(
        AGENT, mine, appended(2, view, a_sentence(), told=[a_turn(4, minutes=1)], first_told=1), NO_TURNS_KEPT
    )

    assert await histories.load(AGENT) == KeptHistory(
        0, (said(1, "first"), an_act(), view, a_sentence()), 0, (a_turn(2), a_turn(4, minutes=1))
    )


async def test_answered_turns_load_in_the_order_they_were_appended_whatever_their_wall_time(histories):
    mine = await histories.take_over(AGENT)
    at_the_same_time = [a_turn(1), a_turn(2)]
    earlier = a_turn(2, minutes=-1)
    await histories.write(AGENT, mine, appended(0, said(1, "a"), said(2, "b"), told=at_the_same_time), NO_TURNS_KEPT)

    await histories.write(AGENT, mine, appended(2, told=[earlier], first_told=2), NO_TURNS_KEPT)

    assert (await histories.load(AGENT)).told == (*at_the_same_time, earlier)


async def test_an_agent_with_no_history_loads_an_empty_one(histories):
    assert await histories.load(AGENT) == KeptHistory(0, (), 0, ())


async def test_entries_that_left_the_history_and_the_turns_that_told_only_them_are_gone_after_the_write(histories):
    mine = await histories.take_over(AGENT)
    await histories.write(
        AGENT, mine, appended(0, said(1, "a"), said(2, "b"), told=[a_turn(1), a_turn(2, 1)]), NO_TURNS_KEPT
    )

    await histories.write(
        AGENT, mine, appended(2, said(3, "c"), told=[a_turn(3, 2)], first_told=2, left_out=1), NO_TURNS_KEPT
    )

    assert await histories.load(AGENT) == KeptHistory(1, (said(2, "b"), said(3, "c")), 1, (a_turn(2, 1), a_turn(3, 2)))


async def test_a_load_that_meets_a_write_in_flight_gives_the_history_before_or_after_it_never_a_mix(
    histories, database
):
    mine = await histories.take_over(AGENT)
    await histories.write(
        AGENT, mine, appended(0, said(1, "a"), said(2, "b"), told=[a_turn(1), a_turn(2, 1)]), NO_TURNS_KEPT
    )
    before = await histories.load(AGENT)
    first_left = ("DELETE FROM history_entries WHERE ordinal < 1", "DELETE FROM answered_turns WHERE through <= 1")

    with uncommitted_behind_a_lock(database, "answered_turns", first_left) as commit:
        loading = asyncio.create_task(histories.load(AGENT))
        async with asyncio.timeout(LIMIT_SECONDS):
            await until_a_statement_waits(database)
        commit()
        loaded = await loading

    assert loaded in (before, await histories.load(AGENT))


async def test_a_delta_written_again_after_an_unknown_outcome_changes_nothing(histories):
    mine = await histories.take_over(AGENT)
    delta = appended(0, said(1, "once"), told=[a_turn(1)])
    await histories.write(AGENT, mine, delta, NO_TURNS_KEPT)

    await histories.write(AGENT, mine, delta, NO_TURNS_KEPT)

    assert await histories.load(AGENT) == KeptHistory(0, (said(1, "once"),), 0, (a_turn(1),))


@pytest.mark.parametrize("taken_over", [True, False], ids=["taken over since", "never owned"])
async def test_a_write_with_a_generation_that_is_not_current_is_refused_and_changes_nothing(histories, taken_over):
    stale = 0
    if taken_over:
        stale = await histories.take_over(AGENT)
        current = await histories.take_over(AGENT)
        await histories.write(AGENT, current, appended(0, said(1, "kept"), told=[a_turn(1)]), NO_TURNS_KEPT)
    before = await histories.load(AGENT)
    delta = appended(1, said(2, "stale"), told=[a_turn(2, 1)], first_told=1, left_out=1)

    with pytest.raises(StaleGenerationError):
        await histories.write(AGENT, stale, delta, KeptTurns(unanswered=1))

    assert await histories.load(AGENT) == before


@pytest.mark.parametrize("seqs", [frozenset({3}), frozenset[int]()], ids=["with a line", "a hold with no line"])
async def test_the_turns_written_last_load_with_the_history_whether_a_delta_came_with_them_or_not(histories, seqs):
    mine = await histories.take_over(AGENT)
    due = KeptDue(APP, TABLE, Lines("first-generation", seqs, missed=bool(seqs), traceparent=None))
    reading = Reading("first-generation", 3, VoiceState.UNAVAILABLE)
    await histories.write(AGENT, mine, appended(0, said(1, "kept")), KeptTurns((KeptCheckpoint(APP, TABLE, None),)))
    paused = KeptTurns((KeptCheckpoint(APP, TABLE, reading),), (due,), 2, LATER)

    await histories.write(AGENT, mine, None, paused)

    assert await histories.load(AGENT) == KeptHistory(0, (said(1, "kept"),), 0, (), paused)


def some_turns() -> KeptTurns:
    reading = Reading("first-generation", 3, VoiceState.UNAVAILABLE)
    due = KeptDue(APP, TABLE, Lines("first-generation", frozenset({4}), missed=False, traceparent=None))
    return KeptTurns((KeptCheckpoint(APP, TABLE, reading),), (due,), 1, LATER)


async def nothing_between(database: DatabaseTarget) -> None:
    pass


async def erased(database: DatabaseTarget) -> None:
    memory = PostgresMemory(database)
    try:
        await memory.erase_agent(AGENT)
    finally:
        await memory.close()


async def taken_over_by_a_release_that_keeps_no_turns(database: DatabaseTarget) -> None:
    run_sql(database, "UPDATE agent_owner SET generation = generation + 1")


@pytest.mark.parametrize(
    ("between", "resumed"),
    [
        (nothing_between, some_turns()),
        (erased, some_turns()),
        (taken_over_by_a_release_that_keeps_no_turns, NO_TURNS_KEPT),
    ],
    ids=["nothing", "an erase", "a release that keeps no turns"],
)
async def test_a_takeover_resumes_the_turns_unless_a_release_that_keeps_none_served_him_since(
    histories, database, between, resumed
):
    mine = await histories.take_over(AGENT)
    await histories.write(AGENT, mine, None, some_turns())
    await between(database)

    await histories.take_over(AGENT)

    assert (await histories.load(AGENT)).turns == resumed


@pytest.mark.parametrize(
    "stored",
    ['{"entry": "dream", "text": "held"}', "null", '["held"]', '"held"'],
    ids=["unknown kind", "null", "array", "string"],
)
async def test_a_stored_entry_that_is_no_entry_loads_as_a_defect_that_holds_none_of_it(histories, database, stored):
    mine = await histories.take_over(AGENT)
    await histories.write(AGENT, mine, appended(0, said(1, "kept")), NO_TURNS_KEPT)
    run_sql(database, f"UPDATE history_entries SET entry = '{stored}'")  # noqa: S608 - a test-chosen value

    with pytest.raises(StoreDefect) as defect:
        await histories.load(AGENT)

    assert "held" not in str(defect.value)
    assert defect.value.__cause__ is None
