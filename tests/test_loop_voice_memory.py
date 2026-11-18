"""The whole runner, the spoken turn against the real store: what it keeps and the lookup it is handed."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL
from agents.tools import ToolEffect

from personas.failures import STORE_TIMEOUT_CODE
from personas.memory import PostgresLineStore
from personas.recall import RECALL_TOOL
from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import LIMIT_SECONDS, SAY_TOOL
from tests.support.postgres import kept_lines, lines_table_locked
from tests.support.standings import active

if TYPE_CHECKING:
    from personas.config import DatabaseTarget
    from personas.entries import WindowRead

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

ANSWER = "Here I am."
BOUND_SECONDS = 0.5


class WatchedStore(PostgresLineStore):
    """The real store, which tells which sequence numbers it has kept."""

    def __init__(self, target: DatabaseTarget) -> None:
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
    runner.remember_in(store)
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


async def test_a_write_cut_as_a_timeout_lets_the_turn_speak_and_is_logged_once(runner, database):
    bounded = replace(database, write_seconds=BOUND_SECONDS)
    store = WatchedStore(bounded)
    runner.remember_in(store)
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
    assert runner.reports.reported().count((Event.STORE_FAILED, STORE_TIMEOUT_CODE)) == 1


async def test_a_spoken_turn_with_a_memory_is_handed_the_lookup_first_declared_read_only_before_the_apps_tools(
    runner, database
):
    store = WatchedStore(database)
    runner.remember_in(store)
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    await store.close()

    (handed,) = runner.handed
    assert runner.gateway.offered_tools() == [RECALL_TOOL, SAY_TOOL, CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL]
    assert (handed[0].declaration.name, handed[0].declaration.effect) == (RECALL_TOOL, ToolEffect.READ_ONLY)
