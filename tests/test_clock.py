"""The real clock: a sleep moves the time on."""

from __future__ import annotations

import pytest

from personas.clock import SystemClock

pytestmark = pytest.mark.anyio


async def test_a_sleep_moves_the_time_on():
    clock = SystemClock()
    slept = 0.01
    before = clock.now()

    await clock.sleep(slept)

    assert clock.now() - before > slept / 2
