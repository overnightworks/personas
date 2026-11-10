"""The sweep task: when it sends a statement, how it ends and what it leaves running."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace

import pytest

from personas.sweep import sweep_lines
from tests.support.call import restarted_in_the_call
from tests.support.fakes import FakeClock, Reports, SweepClock, memory_target
from tests.support.hopin import LIMIT_SECONDS

pytestmark = pytest.mark.anyio

SWEEP_MINUTES = 7.0
SWEEP_SECONDS = 420.0


class ScriptedSweeps:
    """Counts the statements it is sent; it can stay inside the first one."""

    def __init__(self, *, stays_inside: bool = False) -> None:
        self.statements = 0
        self._stays_inside = stays_inside
        self._inside = asyncio.Event()

    async def erase_before(self, first_read_before) -> int:
        self.statements += 1
        self._inside.set()
        if self._stays_inside:
            await asyncio.Event().wait()
        return 0

    async def until_inside(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self._inside.wait()


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


@pytest.mark.parametrize(
    ("keep_days", "sweep_minutes", "statements"),
    [(math.inf, SWEEP_MINUTES, 0), (30.0, math.inf, 1)],
    ids=["every line kept: none sent", "no next sweep: only the first"],
)
async def test_a_task_with_nothing_to_remove_or_no_next_sweep_ends_after_the_statements_it_sent(
    keep_days, sweep_minutes, statements
):
    store = ScriptedSweeps()
    memory = replace(memory_target(), lines_keep_days=keep_days, sweep_minutes=sweep_minutes)

    async with asyncio.timeout(LIMIT_SECONDS):
        await sweep_lines(store, memory, FakeClock(), Reports())

    assert store.statements == statements


async def test_a_runner_without_a_memory_table_sweeps_nothing(runner):
    store = ScriptedSweeps()
    runner.lines = store
    await restarted_in_the_call(runner)
    await runner.app.until_standing_read(1)

    assert store.statements == 0


@pytest.mark.parametrize("stays_inside", [False, True], ids=["in its wait", "in a sweep"])
async def test_a_runner_stopped_in_a_sweep_or_in_its_wait_ends_cancelled_with_no_defect(runner, stays_inside):
    store = ScriptedSweeps(stays_inside=stays_inside)
    runner.clock = SweepClock(SWEEP_SECONDS)
    runner.remember_in(store, replace(memory_target(), sweep_minutes=SWEEP_MINUTES))
    runner.start()
    await (store.until_inside() if stays_inside else runner.clock.until_waiting(1))

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
