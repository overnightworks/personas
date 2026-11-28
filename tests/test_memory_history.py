"""The history's store against a real Postgres: an agent's entries and answered turns in order, fenced by the
generation of the pod that owns the agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceRef, VoiceState

from personas import memory
from personas.entries import (
    NO_TURNS_KEPT,
    HistoryDelta,
    KeptCheckpoint,
    KeptDue,
    KeptHistory,
    KeptTurns,
    Lines,
    Reading,
    Room,
    TurnTold,
    ViewSeen,
    stored_entry,
)
from personas.failures import STORE_DEFECT_CODE, OrdinalReusedError, StaleGenerationError, StoreDefect
from personas.memory import PostgresHistoryStore, PostgresMemory
from tests.support.history import (
    APP,
    LATER,
    MORNING,
    TABLE,
    a_sentence,
    a_skill_read,
    a_view,
    an_act,
    said,
)
from tests.support.postgres import kept_record, run_sql, uncommitted_behind_a_lock, until_a_statement_waits

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


def recorded(first: int, *entries, left_out: int = 0) -> HistoryDelta:
    """A delta that appends the entries to his history, from the ordinal ``first``, and hands them to the record."""
    return HistoryDelta(first, tuple(entries), 0, (), left_out, kept_first=first, kept=tuple(entries))


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


ROW_COLUMNS = "ordinal, kind, app, place_kind, place_id, generation, own, speaker"


async def test_every_told_entry_is_kept_with_its_ordinal_its_kind_the_room_it_was_taken_in_and_who_said_or_did_it(
    histories, database
):
    mine = await histories.take_over(AGENT)
    own_line = replace(said(1, "mine"), own=True)
    view = ViewSeen(APP, TABLE, a_view(2), LATER, generation="morning")
    call = an_act(app="https://target.test/mcp", room=Room(APP, TABLE, "evening"))
    entries = (own_line, said(2, "yours", name="Bo"), view, call, a_sentence(generation="morning"), a_skill_read())

    await histories.write(AGENT, mine, recorded(5, *entries), NO_TURNS_KEPT)

    assert kept_record(database, ROW_COLUMNS) == [
        (5, "line", APP, "call", "room-7", "morning", True, None),
        (6, "line", APP, "call", "room-7", "morning", False, "Bo"),
        (7, "view", APP, "table", "table-1", "morning", False, None),
        (8, "act", APP, "table", "table-1", "evening", True, None),
        (9, "speech", APP, "call", "room-7", "morning", True, None),
    ]


async def test_a_call_a_release_before_the_room_took_is_kept_under_its_target_app_without_place_or_visit(
    histories, database
):
    mine = await histories.take_over(AGENT)

    await histories.write(AGENT, mine, recorded(0, an_act(app="https://target.test/mcp")), NO_TURNS_KEPT)

    assert kept_record(database, ROW_COLUMNS) == [(0, "act", "https://target.test/mcp", None, None, None, True, None)]


async def test_the_record_holds_the_entry_as_json_with_a_call_s_arguments_and_result(histories, database):
    mine = await histories.take_over(AGENT)
    call = an_act(arguments={"card": "jack"}, room=Room(APP, TABLE, "morning"))

    await histories.write(AGENT, mine, recorded(0, call), NO_TURNS_KEPT)

    ((stored,),) = kept_record(database, "entry")
    assert stored == json.loads(json.dumps(stored_entry(call)))
    assert stored["arguments"] == {"card": "jack"}
    assert stored["outcome"]["text"] == "posted"


async def test_what_the_cut_drops_from_his_history_stays_in_the_record_and_so_does_what_entered_and_left_between_writes(
    histories, database
):
    mine = await histories.take_over(AGENT)
    view = ViewSeen(APP, TABLE, a_view(3), LATER)
    await histories.write(AGENT, mine, recorded(0, said(1, "a"), view), NO_TURNS_KEPT)

    await histories.write(AGENT, mine, recorded(2, said(2, "b"), said(3, "c"), left_out=3), NO_TURNS_KEPT)

    assert (await histories.load(AGENT)).first == 3
    assert [(ordinal, kind) for ordinal, kind, *_ in kept_record(database, ROW_COLUMNS)] == [
        (0, "line"),
        (1, "view"),
        (2, "line"),
        (3, "line"),
    ]


async def test_a_delta_that_hands_over_entries_the_history_has_no_more_keeps_them_and_changes_nothing_else(
    histories, database
):
    mine = await histories.take_over(AGENT)
    delta = HistoryDelta(5, (), 0, (), 5, kept_first=2, kept=(said(1, "a"), said(2, "b")))

    await histories.write(AGENT, mine, delta, NO_TURNS_KEPT)

    assert [ordinal for ordinal, *_ in kept_record(database, "ordinal")] == [2, 3]
    assert await histories.load(AGENT) == KeptHistory(0, (), 0, ())


async def test_a_delta_that_hands_over_nothing_to_keep_leaves_the_record_as_it_is(histories, database):
    mine = await histories.take_over(AGENT)

    await histories.write(AGENT, mine, appended(0, said(1, "a")), NO_TURNS_KEPT)

    assert kept_record(database, "ordinal") == []


async def test_writing_an_entry_again_at_its_ordinal_changes_nothing_and_the_held_row_keeps_its_first_form(
    histories, database
):
    mine = await histories.take_over(AGENT)
    first = said(1, "as first written")
    await histories.write(AGENT, mine, recorded(0, first), NO_TURNS_KEPT)
    again = replace(first, text="as a later release writes it", name="Ada")

    await histories.write(AGENT, mine, recorded(0, again), NO_TURNS_KEPT)
    await histories.write(AGENT, mine, recorded(0, first), NO_TURNS_KEPT)

    ((stored, speaker),) = kept_record(database, "entry, speaker")
    assert (stored["text"], speaker) == ("as first written", "Bo")


@pytest.mark.parametrize(
    "reused",
    [
        replace(said(1, "later"), first_read_at=LATER + timedelta(minutes=1)),
        ViewSeen(APP, TABLE, a_view(1), LATER),
    ],
    ids=["another time", "another kind"],
)
async def test_an_ordinal_reused_with_another_time_or_kind_is_a_store_defect_that_takes_nothing_of_the_write(
    histories, database, reused
):
    mine = await histories.take_over(AGENT)
    await histories.write(AGENT, mine, recorded(0, said(1, "held")), NO_TURNS_KEPT)
    delta = HistoryDelta(1, (said(2, "new"),), 0, (), 0, kept_first=0, kept=(reused, said(2, "new")))

    with pytest.raises(OrdinalReusedError) as defect:
        await histories.write(AGENT, mine, delta, KeptTurns(unanswered=3))

    assert str(defect.value) == STORE_DEFECT_CODE
    assert [ordinal for ordinal, *_ in kept_record(database, "ordinal")] == [0]
    assert await histories.load(AGENT) == KeptHistory(0, (said(1, "held"),), 0, ())


async def test_text_no_text_column_takes_is_kept_with_a_replacement_so_the_write_goes_through(histories, database):
    mine = await histories.take_over(AGENT)
    line = replace(
        said(1, "a\x00b \ud800"), name="Ada\x00", visit=replace(MORNING, place=PlaceRef(kind="c", id="r\x00"))
    )

    await histories.write(AGENT, mine, recorded(0, line), NO_TURNS_KEPT)

    ((stored, speaker, place_id),) = kept_record(database, "entry, speaker, place_id")
    assert (stored["text"], speaker, place_id) == ("a\ufffdb \ufffd", "Ada\ufffd", "r\ufffd")


async def test_a_delta_with_more_entries_than_one_statement_takes_is_kept_whole(histories, database):
    count = memory.RECORD_ROWS_PER_STATEMENT + 1
    mine = await histories.take_over(AGENT)

    await histories.write(AGENT, mine, recorded(0, *(said(seq, "a") for seq in range(count))), NO_TURNS_KEPT)

    assert [ordinal for ordinal, *_ in kept_record(database, "ordinal")] == list(range(count))


async def test_a_write_the_generation_refuses_keeps_nothing(histories, database):
    stale = await histories.take_over(AGENT)
    await histories.take_over(AGENT)

    with pytest.raises(StaleGenerationError):
        await histories.write(AGENT, stale, recorded(0, said(1, "stale")), NO_TURNS_KEPT)

    assert kept_record(database, "ordinal") == []


async def test_entries_an_older_release_wrote_are_kept_when_a_load_hands_them_over_and_its_cut_leaves_the_record_alone(
    histories, database
):
    mine = await histories.take_over(AGENT)
    for ordinal, text in enumerate(("old", "older")):
        stored = json.dumps(stored_entry(said(ordinal, text)))
        run_sql(
            database,
            f"INSERT INTO history_entries VALUES ('{AGENT}', {ordinal}, now(), '{stored}')",  # noqa: S608 - test-chosen values
        )
    loaded = await histories.load(AGENT)

    await histories.write(
        AGENT, mine, HistoryDelta(2, (), 0, (), 0, kept_first=loaded.first, kept=loaded.entries), NO_TURNS_KEPT
    )
    run_sql(database, "DELETE FROM history_entries WHERE ordinal < 1")

    assert [ordinal for ordinal, *_ in kept_record(database, "ordinal")] == [0, 1]


async def test_kept_records_count_each_agents_entries_by_kind_with_their_first_and_last_time(histories):
    mine, others = await histories.take_over(AGENT), await histories.take_over(OTHER_AGENT)
    await histories.write(AGENT, mine, recorded(0, said(1, "a"), an_act(), said(2, "b")), NO_TURNS_KEPT)
    await histories.write(OTHER_AGENT, others, recorded(0, a_sentence()), NO_TURNS_KEPT)

    kept = await histories.kept_records(AGENT)

    assert [(each.client_id, each.kind, each.entries) for each in kept] == [(AGENT, "act", 1), (AGENT, "line", 2)]
    assert (kept[0].first_at, kept[0].last_at) == (an_act().at, an_act().at)
    assert (kept[1].first_at, kept[1].last_at) == (LATER, LATER)
    assert [each.client_id for each in await histories.kept_records(None)] == [OTHER_AGENT, AGENT, AGENT]
