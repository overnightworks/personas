"""The real clock: a sleep moves the time on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from personas.clock import SystemClock

pytestmark = pytest.mark.anyio


async def test_a_sleep_moves_the_time_on():
    clock = SystemClock()
    slept = 0.01
    before = clock.now()

    await clock.sleep(slept)

    assert clock.now() - before > slept / 2


def test_the_wall_reading_is_the_calendar_time_now_in_utc():
    reading = SystemClock().wall()

    assert reading.utcoffset() == timedelta(0)
    assert abs(datetime.now(UTC) - reading) < timedelta(minutes=1)
