"""The sweep through the real runner against a real Postgres, its wait on the fake clock."""

from __future__ import annotations

import asyncio
import secrets
import traceback
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceRef

from personas.config import MemoryTarget
from personas.entries import WindowRead
from personas.failures import STORE_TIMEOUT_CODE, STORE_UNREACHABLE_CODE, StoreDefect, StoreUnavailable, leaves_of
from personas.memory import PostgresLineStore
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import SweepClock, line, window_of
from tests.support.gateway import Gateway, says
from tests.support.hopin import LIMIT_SECONDS, formatted, lines_of
from tests.support.postgres import kept_lines, lines_table_locked, run_sql

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from personas.config import DatabaseTarget


pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

KEEP_DAYS = 30.0
SWEEP_MINUTES = 7.0
SWEEP_SECONDS = 420.0
BOUND_SECONDS = 0.5
OLD_DAYS = 40
YOUNG_DAYS = 1
LATER_DAYS = 31
RENAME_FIRST_READ_AT = "ALTER TABLE lines RENAME COLUMN first_read_at TO renamed"


class UnreachableUntilMended(PostgresLineStore):
    """The real store, whose sweeps meet an outage until it is mended."""

    def __init__(self, target: DatabaseTarget) -> None:
        super().__init__(target)
        self.mended = False

    async def erase_before(self, first_read_before) -> int:
        if not self.mended:
            raise StoreUnavailable(STORE_UNREACHABLE_CODE)
        return await super().erase_before(first_read_before)


def read_of(text: str, *, days_ago: float, now, client_id: str = "agent-mozart", speaker: str = "Ada") -> WindowRead:
    """One line of its own visit, first read this many days before ``now``."""
    window = window_of(line(1, text, name=speaker), generation=f"visit-{secrets.token_hex(4)}")
    return WindowRead(
        client_id, "https://app.test/mcp", PlaceRef(kind="call", id="earlier"), window, now - timedelta(days=days_ago)
    )


def sweeping(runner, store: PostgresLineStore, **memory) -> SweepClock:
    """The runner remembers in ``store`` and sweeps every ``SWEEP_MINUTES`` on a clock that waits for its gate."""
    runner.clock = SweepClock(SWEEP_SECONDS)
    runner.remember_in(store, MemoryTarget(lines_keep_days=KEEP_DAYS, sweep_minutes=SWEEP_MINUTES, **memory))
    return runner.clock


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


@pytest.fixture
async def store(database: DatabaseTarget) -> AsyncIterator[PostgresLineStore]:
    kept = PostgresLineStore(database)
    yield kept
    await kept.close()


async def test_the_first_sweep_removes_the_line_first_read_longer_ago_than_the_keep_period_and_keeps_the_younger(
    runner, store, database
):
    now = runner.clock.wall()
    await store.keep(read_of("Older", days_ago=OLD_DAYS, now=now))
    await store.keep(read_of("Younger", days_ago=YOUNG_DAYS, now=now))
    clock = sweeping(runner, store)
    runner.start()
    await clock.until_waiting(1)

    assert kept_lines(database, "text") == [("Younger",)]
    assert runner.reports.lines_removed() == [1]


async def test_a_line_that_ages_past_the_period_while_the_runner_runs_is_removed_at_the_next_sweep_and_not_before(
    runner, store, database
):
    await store.keep(read_of("Ages", days_ago=0, now=runner.clock.wall()))
    clock = sweeping(runner, store)
    runner.start()
    await clock.until_waiting(1)
    clock.time += LATER_DAYS * 24 * 3600
    before_the_next_sweep = kept_lines(database, "text")
    clock.open_gate()
    await runner.reports.until(Event.SWEPT)

    assert before_the_next_sweep == [("Ages",)]
    assert kept_lines(database, "text") == []


async def test_a_store_that_cannot_be_reached_at_a_sweep_is_reported_once_costs_no_answer_and_is_swept_when_back(
    runner, database
):
    store = UnreachableUntilMended(database)
    await store.keep(read_of("Older", days_ago=OLD_DAYS, now=runner.clock.wall()))
    clock = sweeping(runner, store)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    await clock.until_waiting(1)
    clock.open_gate()
    await clock.until_waiting(2)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    store.mended = True
    clock.open_gate()
    await runner.reports.until(Event.SWEPT)

    assert runner.app.said == [(runner.app_principal.sub, PLACE_ID, ANSWER)]
    assert runner.reports.of_the_sweep() == [
        (Event.STORE_FAILED, STORE_UNREACHABLE_CODE),
        (Event.STORE_RESUMED, None),
        (Event.SWEPT, None),
    ]
    assert "Older" not in [row[0] for row in kept_lines(database, "text")]
    await store.close()


async def test_a_database_error_at_a_sweep_that_is_no_outage_ends_the_runner_with_a_code_and_no_secret(
    runner, store, database
):
    sweeping(runner, store)
    run_sql(database, RENAME_FIRST_READ_AT)
    runner.start()

    ended = await ended_by(runner.task)

    assert [type(leaf) for leaf in leaves_of(ended)] == [StoreDefect]
    assert database.password.reveal() not in "".join(traceback.format_exception(ended))


async def test_a_sweep_slower_than_the_write_bound_is_cut_as_a_timeout_logged_and_every_line_stays(runner, database):
    bounded = replace(database, write_seconds=BOUND_SECONDS)
    store = PostgresLineStore(bounded)
    await store.keep(read_of("Older", days_ago=OLD_DAYS, now=runner.clock.wall()))
    clock = sweeping(runner, store)
    try:
        with lines_table_locked(database):
            runner.start()
            await clock.until_waiting(1)
    finally:
        await store.close()

    assert runner.reports.of_the_sweep() == [(Event.STORE_FAILED, STORE_TIMEOUT_CODE)]
    assert kept_lines(database, "text") == [("Older",)]


async def test_no_line_text_speaker_or_password_reaches_a_log_line_of_a_sweep(runner, store, database, caplog):
    text, speaker = f"text-{secrets.token_hex(8)}", f"speaker-{secrets.token_hex(8)}"
    await store.keep(read_of(text, days_ago=OLD_DAYS, now=runner.clock.wall(), speaker=speaker))
    clock = sweeping(runner, store)
    runner.start()
    await clock.until_waiting(1)

    logged = formatted(caplog.get_records("call"))
    assert lines_of(caplog.get_records("call"), "lines.swept")
    for held in (text, speaker, database.password.reveal()):
        assert held not in logged
