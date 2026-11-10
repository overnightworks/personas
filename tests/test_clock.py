"""The real clock: time never runs backwards and a sleep returns."""

from __future__ import annotations

import pytest

from personas.clock import SystemClock

pytestmark = pytest.mark.anyio


def test_the_time_never_runs_backwards():
    clock = SystemClock()

    first = clock.now()

    assert clock.now() >= first


async def test_a_sleep_of_zero_seconds_returns():
    await SystemClock().sleep(0)
