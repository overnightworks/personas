"""The whole runner, the memory: a window read is handed to the store, and the store's failure costs no answer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from personas.failures import STORE_TIMEOUT_CODE, STORE_UNREACHABLE_CODE, StoreUnavailable
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import LIMIT_SECONDS

if TYPE_CHECKING:
    from personas.decision import WindowRead

pytestmark = pytest.mark.anyio

OUTAGE_EVENTS = {"backoff", "suspended"}


class ScriptedStore:
    """Keeps what it is handed, or fails with the failure it is told to, or waits for a release first."""

    def __init__(self) -> None:
        self.kept: list[WindowRead] = []
        self.failure: StoreUnavailable | None = None
        self.waiting = asyncio.Event()
        self.release: asyncio.Event | None = None
        self.attempts = 0
        self._attempted = asyncio.Event()

    async def until_waiting(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.waiting.wait()

    async def until_attempts(self, count: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while self.attempts < count:
                self._attempted.clear()
                await self._attempted.wait()

    async def keep(self, read: WindowRead) -> None:
        self.attempts += 1
        self._attempted.set()
        if self.release is not None:
            self.waiting.set()
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        self.kept.append(read)

    def seqs_kept(self) -> list[int]:
        return [line.seq for read in self.kept for line in read.window.entries]


async def test_a_store_that_is_down_is_reported_once_and_costs_no_answer_and_the_lines_are_kept_when_it_is_back(runner):
    store = ScriptedStore()
    store.failure = StoreUnavailable(STORE_UNREACHABLE_CODE)
    runner.lines = store
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)
    await store.until_attempts(2)
    store.failure = None
    say_in_the_call(runner, "A line once it is back")
    await runner.reports.until("store_resumed")

    assert runner.app.said == [(runner.app_principal.sub, PLACE_ID, ANSWER)]
    assert runner.reports.events() == [
        ("started", None),
        ("store_failed", STORE_UNREACHABLE_CODE),
        ("store_resumed", None),
    ]
    assert store.seqs_kept()[-3:] == [1, 2, 3]


async def test_a_write_that_ends_in_a_timeout_is_reported_by_its_code_and_does_not_delay_the_turn_it_followed(runner):
    store = ScriptedStore()
    store.release = asyncio.Event()
    store.failure = StoreUnavailable(STORE_TIMEOUT_CODE)
    runner.lines = store
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await store.until_waiting()
    await runner.app.said_count(1)
    store.release.set()
    await runner.reports.until("store_failed")

    assert ("store_failed", STORE_TIMEOUT_CODE) in runner.reports.events()
    assert not OUTAGE_EVENTS & {event for event, _ in runner.reports.events()}
