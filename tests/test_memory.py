"""The memory against a real Postgres: which lines are kept, what a failed write is, and what leaks nowhere."""

from __future__ import annotations

import asyncio
import math
import secrets
import time
import traceback
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceEntry, PlaceRef, PlaceWindow
from sqlalchemy.exc import IntegrityError

from personas.decision import TodayBounds, Visit, WindowRead, today_since
from personas.failures import STORE_TIMEOUT_CODE, STORE_UNREACHABLE_CODE, StoreDefect, StoreUnavailable
from personas.memory import PostgresLineStore, SchemaNotAtHeadError
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import WALL_START, line, window_of
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import LIMIT_SECONDS, active, formatted, invited, paused
from tests.support.postgres import Forwarder, kept_lines, lines_table_locked, run_sql, waiting_statements

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from personas.config import MemoryTarget

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

APP = "https://app.test/mcp"
AGENT = "agent-mozart"
SECOND_AGENT = "agent-ada"
SECOND_SUBJECT = "ada-subject"
BOUND_SECONDS = 0.5
MARGIN_SECONDS = 1.0
CLOSED_PORT = 1
DARK_WRITE_SECONDS = 1.0
RESET_AFTER_CANCEL_SECONDS = 0.5
BEYOND_64_BITS = 2**63
FRACTIONAL_SEQ_ROW = (
    "INSERT INTO lines VALUES ('agent', 'app', 'call', 'room', 'visit', 1.5, now(), 'Ada', 'message', 'text',"
    " false, false)"
)
REFUSE_EVERY_NEW_ROW = "ALTER TABLE lines ADD CONSTRAINT refuses_every_row CHECK (false) NOT VALID"
"""Its error names the failing row, so only the adapter's own wording keeps the row out of a traceback."""


class ObservedStore(PostgresLineStore):
    """The real store, which tells how many writes have finished, whatever way they ended."""

    def __init__(self, target: MemoryTarget) -> None:
        super().__init__(target)
        self.finished = 0
        self._finishing = asyncio.Event()

    async def keep(self, read: WindowRead) -> None:
        try:
            await super().keep(read)
        finally:
            self.finished += 1
            self._finishing.set()

    async def until_finished(self, count: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while self.finished < count:
                self._finishing.clear()
                await self._finishing.wait()


@pytest.fixture
async def store(database: MemoryTarget) -> AsyncIterator[ObservedStore]:
    kept = ObservedStore(database)
    yield kept
    await kept.close()


def a_read(*lines: PlaceEntry) -> WindowRead:
    return WindowRead(AGENT, APP, PlaceRef(kind="call", id=PLACE_ID), window_of(*lines), WALL_START)


def a_visit(generation: str) -> Visit:
    return Visit(APP, PlaceRef(kind="call", id=PLACE_ID), generation)


def a_read_in(visit: Visit, *lines: PlaceEntry, client_id: str = AGENT, hours_after_start: float = 0.0) -> WindowRead:
    read_at = WALL_START + timedelta(hours=hours_after_start)
    return WindowRead(client_id, visit.app, visit.place, window_of(*lines, generation=visit.generation), read_at)


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_a_window_read_again_and_after_a_restart_keeps_each_line_once_with_the_time_of_its_first_read(
    runner, store, database
):
    runner.lines = store
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "First line")
    await store.until_finished(1)
    first_read_at = runner.clock.wall()
    runner.clock.time += 3600
    say_in_the_call(runner, "Second line")
    await store.until_finished(2)
    second_read_at = runner.clock.wall()
    runner.task.cancel()
    await ended_by(runner.task)
    runner.clock.time += 3600
    runner.start()
    await store.until_finished(3)

    assert kept_lines(database, "seq, text, first_read_at") == [
        (1, "First line", first_read_at),
        (2, "Second line", second_read_at),
    ]


async def test_his_own_line_a_spoken_line_and_one_said_during_a_pause_are_kept_as_the_window_gave_them(
    runner, store, database
):
    runner.lines = store
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "Said aloud", spoken=True)
    await store.until_finished(1)
    runner.app.speak(PLACE_ID, SUBJECT, "Mozart", "My own words")
    await store.until_finished(2)
    pause_seen = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.app.until_read_held()
    say_in_the_call(runner, "During the pause")
    runner.app.stand(SUBJECT, active(PLACE_ID))
    pause_seen.set()
    await store.until_finished(3)

    assert kept_lines(database, "seq, speaker, text, own, spoken") == [
        (1, "Ada", "Said aloud", False, True),
        (2, "Mozart", "My own words", True, False),
        (3, "Ada", "During the pause", False, False),
    ]


async def test_two_agents_in_one_place_keep_each_their_own_rows_under_their_own_client_id(runner, store, database):
    runner.lines = store
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID, 2)
    say_in_the_call(runner, "Hello both")
    await store.until_finished(2)

    assert kept_lines(database, "client_id, app, seq, text") == [
        (SECOND_AGENT, runner.app_url, 1, "Hello both"),
        (runner.agent.client_id, runner.app_url, 1, "Hello both"),
    ]


async def test_a_line_with_u0000_is_kept_with_the_character_replaced_and_the_others_are_kept(store, database):
    entries = (
        PlaceEntry(seq=1, kind="mess\x00age", name="Ad\x00a", text="before\x00after", own=False),
        line(2, "plain"),
    )
    window = PlaceWindow(generation="visit\x00", starts_at=1, entries=entries)
    place = PlaceRef(kind="ca\x00ll", id="room\x00")

    await store.keep(WindowRead(AGENT, APP, place, window, WALL_START))

    assert kept_lines(database, "place_kind, place_id, generation, seq, kind, speaker, text") == [
        ("ca�ll", "room�", "visit�", 1, "mess�age", "Ad�a", "before�after"),
        ("ca�ll", "room�", "visit�", 2, "message", "Ada", "plain"),
    ]


async def test_a_write_waiting_on_a_lock_ends_within_its_bound_releases_its_statement_and_the_next_write_succeeds(
    database,
):
    store = PostgresLineStore(replace(database, write_seconds=BOUND_SECONDS))
    await store.keep(a_read(line(1, "Before the lock")))
    try:
        with lines_table_locked(database):
            started = time.monotonic()
            during_the_lock = a_read(line(2, "During the lock"))
            async with asyncio.timeout(LIMIT_SECONDS):
                with pytest.raises(StoreUnavailable) as cut:
                    await store.keep(during_the_lock)
            elapsed = time.monotonic() - started
            still_waiting = waiting_statements(database)

        await store.keep(a_read(line(3, "After the lock")))
    finally:
        await store.close()

    assert cut.value.code == STORE_TIMEOUT_CODE
    assert still_waiting == 0
    assert BOUND_SECONDS <= elapsed < BOUND_SECONDS + MARGIN_SECONDS
    assert kept_lines(database, "seq") == [(1,), (3,)]


@pytest.mark.parametrize("reset_after_seconds", [None, RESET_AFTER_CANCEL_SECONDS], ids=["dark", "then_reset"])
async def test_a_write_cancelled_while_its_connection_answers_nothing_ends_as_the_cancellation(
    database, reset_after_seconds
):
    forwarder = Forwarder(replace(database, write_seconds=DARK_WRITE_SECONDS))
    async with forwarder.serving() as through_the_forwarder:
        store = PostgresLineStore(through_the_forwarder)
        try:
            await store.keep(a_read(line(1, "Before the dark")))
            forwarder.go_dark()
            write = asyncio.create_task(store.keep(a_read(line(2, "Into the dark"))))
            async with asyncio.timeout(LIMIT_SECONDS):
                await forwarder.dropped.wait()
            write.cancel()
            if reset_after_seconds is not None:
                asyncio.get_running_loop().call_later(reset_after_seconds, forwarder.reset_dark)
            ended = await ended_by(write)
            await store.keep(a_read(line(3, "After the dark")))
        finally:
            await store.close()

    assert isinstance(ended, asyncio.CancelledError)
    assert kept_lines(database, "seq") == [(1,), (3,)]


async def test_a_line_whose_seq_is_beyond_64_bits_is_kept_with_the_other_lines_of_its_window(store, database):
    await store.keep(a_read(line(1, "Ordinary"), line(BEYOND_64_BITS, "Far along")))

    assert kept_lines(database, "seq, text") == [(1, "Ordinary"), (BEYOND_64_BITS, "Far along")]


def test_a_fraction_for_seq_is_refused_by_the_database(database):
    with pytest.raises(IntegrityError):
        run_sql(database, FRACTIONAL_SEQ_ROW)


async def test_a_store_without_a_write_bound_keeps_a_window(database):
    store = PostgresLineStore(replace(database, write_seconds=math.inf))

    await store.keep(a_read(line(1, "Unbounded")))
    await store.close()

    assert kept_lines(database, "seq") == [(1,)]


async def test_a_store_that_cannot_be_reached_is_unavailable_by_its_code(database):
    store = PostgresLineStore(replace(database, port=CLOSED_PORT))

    nowhere_to_go = a_read(line(1, "Nowhere to go"))

    with pytest.raises(StoreUnavailable) as unreachable:
        await store.keep(nowhere_to_go)
    await store.close()

    assert unreachable.value.code == STORE_UNREACHABLE_CODE


async def test_a_schema_behind_its_head_ends_the_write_naming_the_field_and_no_secret(unmigrated_database):
    store = PostgresLineStore(unmigrated_database)

    no_table_yet = a_read(line(1, "No table yet"))

    with pytest.raises(SchemaNotAtHeadError) as behind:
        await store.keep(no_table_yet)
    await store.close()

    assert "memory" in str(behind.value)
    assert unmigrated_database.password.reveal() not in str(behind.value)


async def test_a_database_error_that_is_no_outage_raises_a_code_that_holds_no_line_name_or_password(store, database):
    await store.keep(a_read(line(1, "Kept before")))
    run_sql(database, REFUSE_EVERY_NEW_ROW)
    secret_line = "text-" + secrets.token_hex(8)
    secret_speaker = "speaker-" + secrets.token_hex(8)

    secret_read = a_read(PlaceEntry(seq=2, kind="message", name=secret_speaker, text=secret_line, own=False))

    with pytest.raises(StoreDefect) as defect:
        await store.keep(secret_read)

    shown = "".join(traceback.format_exception(defect.value))
    for held in (secret_line, secret_speaker, database.password.reveal()):
        assert held not in shown
        assert held not in repr(defect.value)


async def test_a_runner_whose_store_cannot_be_reached_starts_and_reports_the_failure_once(runner, database):
    runner.lines = ObservedStore(replace(database, port=CLOSED_PORT))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "First line")
    await runner.lines.until_finished(1)
    say_in_the_call(runner, "Second line")
    await runner.lines.until_finished(2)

    assert runner.reports.events() == [("started", None), ("store_failed", STORE_UNREACHABLE_CODE)]
    await runner.lines.close()


async def test_a_runner_with_a_schema_behind_its_head_ends_loud_with_nothing_but_the_field_in_its_error(
    runner, unmigrated_database
):
    store = ObservedStore(unmigrated_database)
    runner.lines = store
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    ended = await ended_by(runner.task)

    assert "memory" in "".join(traceback.format_exception(ended))
    assert unmigrated_database.password.reveal() not in "".join(traceback.format_exception(ended))
    await store.close()


async def test_a_runner_met_by_a_database_error_that_is_no_outage_ends_with_a_code_and_logs_no_line(
    runner, database, caplog
):
    store = ObservedStore(database)
    runner.lines = store
    await store.keep(a_read(line(1, "Kept before")))
    run_sql(database, REFUSE_EVERY_NEW_ROW)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "A line to be lost")

    ended = await ended_by(runner.task)

    shown = "".join(traceback.format_exception(ended)) + formatted(caplog.get_records("call"))
    assert "A line to be lost" not in shown
    assert database.password.reveal() not in shown
    await store.close()


async def test_back_four_hours_after_a_visit_his_first_turn_is_told_its_lines_in_order_and_not_its_own_window_again(
    runner, store, database
):
    welcome_back = f"{ADDRESSED} Back again?"

    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, welcome_back)

    runner.remember_in(store, database)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Yes."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await store.until_finished(2)
    runner.app.stand(SUBJECT)
    runner.clock.time += 4 * 3600
    runner.on_report = addressed_once_joined
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.app.said_count(2)

    first_visit, back_again = runner.gateway.openings()
    assert first_visit.room["earlier_today"] == {"older_dropped": False, "visits": []}
    assert back_again.room["earlier_today"] == {
        "older_dropped": False,
        "visits": [
            {
                "minutes_ago": 240,
                "lines": [
                    {"name": "Ada", "kind": "message", "text": ADDRESSED, "own": False},
                    {"name": "Mozart", "kind": "message", "text": ANSWER, "own": True},
                ],
            }
        ],
    }
    assert [each["text"] for each in back_again.room["lines"]] == [welcome_back]


async def test_lines_first_read_longer_ago_than_the_day_reaches_are_not_read(store):
    now = WALL_START + timedelta(hours=30)
    await store.keep(a_read_in(a_visit("long-ago"), line(1, "Too long ago"), hours_after_start=5.9))
    await store.keep(a_read_in(a_visit("today"), line(1, "Within the day"), hours_after_start=6.1))

    since = today_since(now, TodayBounds(hours=24.0, size_bytes=1024))
    kept = await store.kept_since(AGENT, since, a_visit("now"))

    assert [each.text for each in kept] == ["Within the day"]


async def test_two_agents_with_lines_in_one_place_each_read_only_their_own_and_never_the_visit_they_are_in(store):
    morning, now = a_visit("morning"), a_visit("now")
    await store.keep(a_read_in(morning, line(1, "Mozart heard this")))
    await store.keep(a_read_in(now, line(1, "Mozart hears this now")))
    await store.keep(a_read_in(morning, line(1, "Ada heard this"), client_id=SECOND_AGENT))

    mozart = await store.kept_since(AGENT, WALL_START, now)
    ada = await store.kept_since(SECOND_AGENT, WALL_START, now)

    assert [(each.visit, each.seq, each.text) for each in mozart] == [(morning, 1, "Mozart heard this")]
    assert [(each.visit, each.seq, each.text) for each in ada] == [(morning, 1, "Ada heard this")]
    assert type(mozart[0].seq) is int


async def test_a_read_slower_than_its_bound_is_cut_as_a_timeout_and_the_turn_answers_without_the_day(runner, database):
    bounded = replace(database, write_seconds=BOUND_SECONDS)
    store = ObservedStore(bounded)
    runner.remember_in(store, bounded)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    try:
        with lines_table_locked(database):
            say_in_the_call(runner, ADDRESSED)
            await runner.app.said_count(1)
    finally:
        await store.close()

    assert "earlier_today" not in runner.gateway.openings()[0].room
    assert runner.reports.events().count(("store_failed", STORE_TIMEOUT_CODE)) == 1
