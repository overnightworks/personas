"""The history's store against a real Postgres: an agent's entries and answered turns in order, fenced by the
generation of the pod that owns the agent."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from personas.entries import HistoryDelta, KeptHistory, Lines, TurnTold, ViewSeen
from personas.failures import StaleGenerationError, StoreDefect
from personas.memory import PostgresHistoryStore
from tests.support.history import APP, LATER, MORNING, TABLE, a_sentence, a_view, an_act, said
from tests.support.postgres import run_sql

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from personas.config import DatabaseTarget

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

AGENT = "agent-mozart"
OTHER_AGENT = "agent-ada"


@pytest.fixture
async def histories(database: DatabaseTarget) -> AsyncIterator[PostgresHistoryStore]:
    store = PostgresHistoryStore(database)
    yield store
    await store.close()


def a_turn(through: int, minutes: int = 0) -> TurnTold:
    occasion = Lines(MORNING.generation, frozenset({through}), missed=False)
    return TurnTold(LATER + timedelta(minutes=minutes), APP, MORNING.place, occasion, aloud=True, through=through)


def appended(first: int, *entries, told=(), left_out: int = 0) -> HistoryDelta:
    return HistoryDelta(first, tuple(entries), tuple(told), left_out)


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

    await histories.append(AGENT, mine, appended(0, said(1, "first"), an_act(), told=[a_turn(2)]))
    await histories.append(OTHER_AGENT, others, appended(0, said(1, "not his")))
    await histories.append(AGENT, mine, appended(2, view, a_sentence(), told=[a_turn(4, minutes=1)]))

    assert await histories.load(AGENT) == KeptHistory(
        0, (said(1, "first"), an_act(), view, a_sentence()), (a_turn(2), a_turn(4, minutes=1))
    )


async def test_an_agent_with_no_history_loads_an_empty_one(histories):
    assert await histories.load(AGENT) == KeptHistory(0, (), ())


async def test_entries_that_left_the_history_and_the_turns_that_told_only_them_are_gone_after_the_write(histories):
    mine = await histories.take_over(AGENT)
    await histories.append(AGENT, mine, appended(0, said(1, "a"), said(2, "b"), told=[a_turn(1), a_turn(2, 1)]))

    await histories.append(AGENT, mine, appended(2, said(3, "c"), told=[a_turn(3, 2)], left_out=1))

    assert await histories.load(AGENT) == KeptHistory(1, (said(2, "b"), said(3, "c")), (a_turn(2, 1), a_turn(3, 2)))


async def test_a_delta_written_again_after_an_unknown_outcome_changes_nothing(histories):
    mine = await histories.take_over(AGENT)
    delta = appended(0, said(1, "once"), told=[a_turn(1)])
    await histories.append(AGENT, mine, delta)

    await histories.append(AGENT, mine, delta)

    assert await histories.load(AGENT) == KeptHistory(0, (said(1, "once"),), (a_turn(1),))


@pytest.mark.parametrize("taken_over", [True, False], ids=["taken over since", "never owned"])
async def test_a_write_with_a_generation_that_is_not_current_is_refused_and_changes_nothing(histories, taken_over):
    stale = 0
    if taken_over:
        stale = await histories.take_over(AGENT)
        current = await histories.take_over(AGENT)
        await histories.append(AGENT, current, appended(0, said(1, "kept"), told=[a_turn(1)]))
    before = await histories.load(AGENT)

    with pytest.raises(StaleGenerationError):
        await histories.append(AGENT, stale, appended(1, said(2, "stale"), told=[a_turn(2, 1)], left_out=1))

    assert await histories.load(AGENT) == before


@pytest.mark.parametrize(
    "stored",
    ['{"entry": "dream", "text": "held"}', "null", '["held"]', '"held"'],
    ids=["unknown kind", "null", "array", "string"],
)
async def test_a_stored_entry_that_is_no_entry_loads_as_a_defect_that_holds_none_of_it(histories, database, stored):
    mine = await histories.take_over(AGENT)
    await histories.append(AGENT, mine, appended(0, said(1, "kept")))
    run_sql(database, f"UPDATE history_entries SET entry = '{stored}'")  # noqa: S608 - a test-chosen value

    with pytest.raises(StoreDefect) as defect:
        await histories.load(AGENT)

    assert "held" not in str(defect.value)
    assert defect.value.__cause__ is None
