"""The whole runner, the spoken turn against the real store: what it keeps and what it is told of the day."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion

from personas.failures import STORE_TIMEOUT_CODE
from personas.memory import PostgresLineStore
from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import LIMIT_SECONDS, active
from tests.support.postgres import kept_lines, lines_table_locked

if TYPE_CHECKING:
    from personas.config import MemoryTarget
    from personas.entries import WindowRead

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

ANSWER = "Here I am."
BOUND_SECONDS = 0.5


class WatchedStore(PostgresLineStore):
    """The real store, which tells which sequence numbers it has kept."""

    def __init__(self, target: MemoryTarget) -> None:
        super().__init__(target)
        self._kept: set[int] = set()
        self._kept_more = asyncio.Event()

    async def keep(self, read: WindowRead) -> None:
        await super().keep(read)
        self._kept.update(entry.seq for entry in read.window.entries)
        self._kept_more.set()

    async def until_kept(self, seq: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while seq not in self._kept:
                self._kept_more.clear()
                await self._kept_more.wait()


async def test_a_line_heard_in_a_voice_place_is_kept_once_with_the_sentence_he_spoke_in_answer(runner, database):
    store = WatchedStore(database)
    runner.remember_in(store, database)
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    say_in_the_call(runner, "Nothing for him in this one.")
    await store.until_kept(3)
    await store.close()

    assert kept_lines(database, "seq, text, own, spoken") == [
        (1, ADDRESSED, False, True),
        (2, ANSWER, True, True),
        (3, "Nothing for him in this one.", False, False),
    ]


async def test_back_after_a_restart_his_first_turn_at_the_voice_place_is_told_the_lines_of_the_day_in_order(
    runner, database
):
    welcome_back = f"{ADDRESSED} Back again?"
    store = WatchedStore(database)
    runner.remember_in(store, database)
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER), completion("Welcome."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=True, cut=True)
    await runner.app.spoken_count(1)
    await store.until_kept(2)
    runner.task.cancel()
    await asyncio.gather(runner.task, return_exceptions=True)
    reads_before = runner.app.window_reads(PLACE_ID)
    runner.start()
    await runner.app.until_window_read(PLACE_ID, reads_before + 1)
    say_in_the_call(runner, welcome_back, spoken=True)
    await runner.app.spoken_count(2)
    await store.close()

    first, second = runner.gateway.turns()
    assert first.room["earlier_today"] == {"older_dropped": False, "visits": []}
    (visit,) = second.room["earlier_today"]["visits"]
    assert [(each["text"], each["spoken"]) for each in visit["lines"]] == [(ADDRESSED, True), (ANSWER, True)]
    assert [each.get("cut", False) for each in visit["lines"]] == [True, False]
    assert [each["text"] for each in second.room["history"]["entries"] if each["new"]] == [welcome_back]


async def test_a_store_that_cannot_be_read_lets_the_turn_speak_without_the_day_and_is_logged_once(runner, database):
    bounded = replace(database, write_seconds=BOUND_SECONDS)
    store = WatchedStore(bounded)
    runner.remember_in(store, bounded)
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    try:
        with lines_table_locked(database):
            runner.start()
            await runner.app.until_window_read(PLACE_ID)
            say_in_the_call(runner, ADDRESSED, spoken=True)
            await runner.app.spoken_count(1)
    finally:
        await store.close()

    assert runner.app.spoken == [(SUBJECT, PLACE_ID, ANSWER)]
    assert "earlier_today" not in runner.gateway.turns()[0].room
    assert runner.reports.reported().count((Event.STORE_FAILED, STORE_TIMEOUT_CODE)) == 1
