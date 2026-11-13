"""The whole runner, the memory: a window read is handed to the store, and the store's failure costs no answer."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from personas.failures import STORE_TIMEOUT_CODE, STORE_UNREACHABLE_CODE, StoreUnavailable
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import memory_target
from tests.support.gateway import Gateway, says
from tests.support.history import TABLE, a_state, at_table
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import LIMIT_SECONDS

if TYPE_CHECKING:
    from datetime import datetime

    from personas.entries import KeptLine, WindowRead

pytestmark = pytest.mark.anyio

OUTAGE_EVENTS = {Event.BACKOFF, Event.SUSPENDED}


class ScriptedStore:
    """Keeps what it is handed, or fails with the failure it is told to, or waits for a release first."""

    def __init__(self) -> None:
        self.kept: list[WindowRead] = []
        self.failure: StoreUnavailable | None = None
        self.read_failure: StoreUnavailable | None = None
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

    async def kept_since(self, client_id: str, since: datetime) -> tuple[KeptLine, ...]:
        if self.read_failure is not None:
            raise self.read_failure
        return ()

    async def erase_before(self, first_read_before: datetime) -> int:
        raise AssertionError("a test about the loop swept")

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
    await runner.reports.until(Event.STORE_RESUMED)

    assert runner.app.said == [(runner.app_principal.sub, PLACE_ID, ANSWER)]
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.STORE_FAILED, STORE_UNREACHABLE_CODE),
        (Event.STORE_RESUMED, None),
    ]
    assert store.seqs_kept()[-3:] == [1, 2, 3]


async def test_the_window_of_a_view_place_is_handed_to_the_store_under_its_kind_and_id_with_the_act_marks(runner):
    store = ScriptedStore()
    runner.lines = store
    runner.app.stand(SUBJECT, at_table())
    runner.app.speak(TABLE.id, "visitor", "Ada", "took a card", act=True)
    runner.app.show_view(TABLE.id, 1, a_state(), awaits_you=False)
    runner.start()
    await store.until_attempts(1)

    (read, *_) = store.kept
    assert read.place == TABLE
    assert [(each.text, each.act) for each in read.window.entries] == [("took a card", True)]


async def test_a_write_that_ends_in_a_timeout_is_reported_by_its_code_and_the_turn_it_held_up_follows(runner):
    store = ScriptedStore()
    store.release = asyncio.Event()
    store.failure = StoreUnavailable(STORE_TIMEOUT_CODE)
    runner.lines = store
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await store.until_waiting()
    assert runner.gateway.requests == []
    store.release.set()
    await runner.app.said_count(1)
    await runner.reports.until(Event.STORE_FAILED)

    assert (Event.STORE_FAILED, STORE_TIMEOUT_CODE) in runner.reports.reported()
    assert not OUTAGE_EVENTS & {event for event, _ in runner.reports.reported()}


async def test_a_store_that_cannot_be_read_when_a_turn_starts_costs_no_answer_and_is_reported_once(runner):
    store = ScriptedStore()
    store.read_failure = StoreUnavailable(STORE_UNREACHABLE_CODE)
    runner.remember_in(store, replace(memory_target(), lines_keep_days=math.inf))
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)
    await runner.reports.until(Event.STORE_RESUMED)

    assert runner.app.said == [(runner.app_principal.sub, PLACE_ID, ANSWER)]
    assert "earlier_today" not in runner.gateway.openings()[0].room
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.STORE_FAILED, STORE_UNREACHABLE_CODE),
        (Event.STORE_RESUMED, None),
    ]
