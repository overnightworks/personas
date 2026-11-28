"""The pod's lock connection against a real Postgres: what it holds, what a second pod gets, and how it fails."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from sqlalchemy import text

from personas.config import OwnershipTimings
from personas.locks import PostgresAgentLocks
from personas.memory import MIGRATION_LOCK_KEY
from personas.ownership import LockConnectionLost
from tests.support.hopin import LIMIT_SECONDS
from tests.support.postgres import Forwarder, engine_of, lock_holders, terminate_backend

pytestmark = [pytest.mark.anyio, pytest.mark.postgres, pytest.mark.integration]

AGENT = "agent-mozart"
OTHER_AGENT = "agent-ada"
CLOSED_PORT = 1
CLOSE_PATIENCE_SECONDS = 0.2
TIMINGS = OwnershipTimings(heartbeat_timeout_seconds=1.0)


@pytest.fixture
async def pod(database):
    locks = PostgresAgentLocks(database, TIMINGS)
    yield locks
    await locks.close()


@pytest.fixture
async def other_pod(database):
    locks = PostgresAgentLocks(database, TIMINGS)
    yield locks
    await locks.close()


async def test_a_pod_takes_the_lock_of_each_agent_and_a_second_pod_takes_none_of_them(pod, other_pod):
    assert await pod.hold([AGENT, OTHER_AGENT]) == {AGENT, OTHER_AGENT}

    assert await other_pod.hold([AGENT, OTHER_AGENT]) == frozenset()


async def test_a_second_pod_takes_the_agents_the_first_does_not_contend_for(pod, other_pod):
    await pod.hold([AGENT])

    assert await other_pod.hold([AGENT, OTHER_AGENT]) == {OTHER_AGENT}


async def test_holding_again_confirms_the_locks_without_taking_each_a_second_time(database, pod):
    await pod.hold([AGENT, OTHER_AGENT])

    assert await pod.hold([AGENT, OTHER_AGENT]) == {AGENT, OTHER_AGENT}
    assert len(lock_holders(database)) == 2


async def test_a_pod_that_closes_its_connection_frees_its_agents_for_the_second_pod(pod, other_pod):
    await pod.hold([AGENT])
    await pod.close()

    assert await other_pod.hold([AGENT]) == {AGENT}


async def test_the_lock_of_an_agent_is_apart_from_the_migration_lock(database, pod):
    engine = engine_of(database.dbname)
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK_KEY})

        assert await pod.hold([AGENT]) == {AGENT}
    engine.dispose()


async def test_a_session_ended_by_the_server_is_a_lost_connection_and_the_next_round_opens_a_new_one(database, pod):
    await pod.hold([AGENT])
    (holder,) = set(lock_holders(database))
    terminate_backend(database, holder)

    with pytest.raises(LockConnectionLost):
        await pod.hold([AGENT])

    assert await pod.hold([AGENT]) == {AGENT}


async def test_a_pod_whose_session_was_ended_leaves_its_agents_to_the_second_pod(database, pod, other_pod):
    await pod.hold([AGENT])
    (holder,) = set(lock_holders(database))
    terminate_backend(database, holder)

    assert await other_pod.hold([AGENT]) == {AGENT}


async def test_a_round_on_a_connection_that_goes_dark_is_cut_at_the_heartbeat_timeout(database):
    """CONFIG-27."""
    forwarder = Forwarder(database)
    async with forwarder.serving() as through:
        locks = PostgresAgentLocks(through, TIMINGS)
        try:
            await locks.hold([AGENT])
            forwarder.go_dark()

            started = time.monotonic()
            async with asyncio.timeout(LIMIT_SECONDS):
                with pytest.raises(LockConnectionLost):
                    await locks.hold([AGENT])

            assert time.monotonic() - started < TIMINGS.heartbeat_timeout_seconds + TIMINGS.cancel_margin_seconds
        finally:
            await locks.close()


async def test_a_round_cut_at_the_heartbeat_timeout_leaves_the_pod_no_second_connection_while_it_unwinds(database):
    forwarder = Forwarder(database)
    async with forwarder.serving() as through:
        locks = PostgresAgentLocks(through, TIMINGS)
        try:
            await locks.hold([AGENT])
            forwarder.go_dark()
            with pytest.raises(LockConnectionLost):
                await locks.hold([AGENT])

            with pytest.raises(LockConnectionLost):
                await locks.hold([AGENT])
        finally:
            forwarder.reset_dark()
            await locks.close()


async def test_closing_a_pod_whose_round_was_cancelled_waits_until_the_session_the_round_used_has_ended(database):
    forwarder = Forwarder(database)
    async with forwarder.serving() as through:
        locks = PostgresAgentLocks(through, TIMINGS)
        await locks.hold([AGENT])
        forwarder.go_dark()
        round_ = asyncio.create_task(locks.hold([AGENT]))
        await forwarder.dropped.wait()
        round_.cancel()
        with pytest.raises(asyncio.CancelledError):
            await round_

        closing = asyncio.create_task(locks.close())
        closed_early, _ = await asyncio.wait({closing}, timeout=CLOSE_PATIENCE_SECONDS)
        forwarder.reset_dark()
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing

        assert not closed_early


async def test_closing_a_pod_whose_caller_was_cancelled_after_its_round_succeeded_frees_its_agents(pod, other_pod):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    caller = asyncio.create_task(pod.hold([AGENT]))

    def cancel_the_caller_when_the_round_completes(loop, coroutine, **keywords):
        round_ = asyncio.Task(coroutine, loop=loop, **keywords)
        round_.add_done_callback(lambda _: caller.cancel())
        loop.set_task_factory(previous_factory)
        return round_

    loop.set_task_factory(cancel_the_caller_when_the_round_completes)
    try:
        with pytest.raises(asyncio.CancelledError):
            await caller
    finally:
        loop.set_task_factory(previous_factory)

    await pod.close()

    assert await other_pod.hold([AGENT]) == {AGENT}


async def test_a_server_that_cannot_be_reached_is_a_lost_connection(database):
    locks = PostgresAgentLocks(replace(database, host="127.0.0.1", port=CLOSED_PORT), TIMINGS)

    with pytest.raises(LockConnectionLost):
        await locks.hold([AGENT])
