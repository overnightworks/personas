"""The connections a store holds against the database: how many at most, and how long a dark one keeps a write."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceRef

from personas.entries import WindowRead
from personas.failures import STORE_TIMEOUT_CODE, StoreUnavailable
from personas.memory import PostgresLineStore
from tests.support.call import PLACE_ID
from tests.support.fakes import WALL_START
from tests.support.hopin import LIMIT_SECONDS
from tests.support.postgres import Forwarder, lines_table_locked, waiting_statements
from tests.support.standings import line, window_of

if TYPE_CHECKING:
    from personas.config import DatabaseTarget

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

AGENT = "agent-mozart"
APP = "https://app.test/mcp"
POOL_WITH_OVERFLOW = 15
WRITES_BEYOND_THE_POOL = 5
HOLD_SECONDS = 1.0
DARK_WRITE_SECONDS = 1.0
CANCEL_WAIT_AT_LEAST_SECONDS = 4.0
"""The driver gives a connection up after waiting about five seconds for a cancelled statement's answer."""
CANCEL_WAIT_AT_MOST_SECONDS = 11.0
"""About ten seconds, the most the reference gives the cancel wait, with a second's margin."""
DARK_LIMIT_SECONDS = 30.0


def a_read(seq: int) -> WindowRead:
    return WindowRead(AGENT, APP, PlaceRef(kind="call", id=PLACE_ID), window_of(line(seq, f"Line {seq}")), WALL_START)


async def until_waiting(target: DatabaseTarget, count: int) -> None:
    async with asyncio.timeout(LIMIT_SECONDS):
        while await asyncio.to_thread(waiting_statements, target) < count:
            continue


async def test_a_store_holds_at_most_a_pool_of_five_with_an_overflow_of_ten_connections(database):
    """CONFIG-38."""
    store = PostgresLineStore(replace(database, write_seconds=math.inf))
    await store.keep(a_read(1))
    writes: set[asyncio.Task[None]] = set()
    try:
        with lines_table_locked(database):
            writes = {
                asyncio.create_task(store.keep(a_read(seq)))
                for seq in range(2, 2 + POOL_WITH_OVERFLOW + WRITES_BEYOND_THE_POOL)
            }
            await until_waiting(database, POOL_WITH_OVERFLOW)
            _, pending = await asyncio.wait(writes, timeout=HOLD_SECONDS)
            held = await asyncio.to_thread(waiting_statements, database)
        async with asyncio.timeout(LIMIT_SECONDS):
            await asyncio.gather(*writes)
    finally:
        for write in writes:
            write.cancel()
        await store.close()

    assert held == POOL_WITH_OVERFLOW
    assert pending == writes


async def test_a_write_whose_connection_goes_dark_is_given_up_only_after_the_drivers_cancel_wait(database):
    """CONFIG-25."""
    forwarder = Forwarder(replace(database, write_seconds=DARK_WRITE_SECONDS))
    async with forwarder.serving() as through_the_forwarder:
        store = PostgresLineStore(through_the_forwarder)
        try:
            await store.keep(a_read(1))
            in_the_dark = a_read(2)
            forwarder.go_dark()
            started = time.monotonic()
            async with asyncio.timeout(DARK_LIMIT_SECONDS):
                with pytest.raises(StoreUnavailable) as given_up:
                    await store.keep(in_the_dark)
            elapsed = time.monotonic() - started
        finally:
            await store.close()

    assert given_up.value.code == STORE_TIMEOUT_CODE
    assert (
        DARK_WRITE_SECONDS + CANCEL_WAIT_AT_LEAST_SECONDS <= elapsed < DARK_WRITE_SECONDS + CANCEL_WAIT_AT_MOST_SECONDS
    )
