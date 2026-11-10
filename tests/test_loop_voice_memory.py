"""The whole runner, the voice loop against the real store: what it keeps and what it is told of the day."""

from __future__ import annotations

import asyncio
import json
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
from tests.support.hopin import LIMIT_SECONDS, active, invited
from tests.support.postgres import kept_lines, lines_table_locked

if TYPE_CHECKING:
    from personas.config import MemoryTarget
    from personas.decision import WindowRead

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


def context_of(gateway: Gateway, request_number: int) -> dict[str, object]:
    """The part of a decision's system text the runner adds after the library's: the last block."""
    return json.loads(gateway.system_of(request_number).rsplit("\n\n", 1)[-1])


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


async def test_back_four_hours_after_a_visit_his_first_decision_is_told_its_lines_in_order_and_not_its_own_window_again(
    runner, database
):
    welcome_back = f"{ADDRESSED} Back again?"

    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, welcome_back, spoken=True)

    store = WatchedStore(database)
    runner.remember_in(store, database)
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    await store.until_kept(2)
    runner.app.stand(SUBJECT)
    runner.clock.time += 4 * 3600
    runner.on_report = addressed_once_joined
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.app.spoken_count(2)
    await store.close()

    assert context_of(runner.gateway, 1) == {"earlier_today": {"older_dropped": False, "visits": []}}
    assert context_of(runner.gateway, 2) == {
        "earlier_today": {
            "older_dropped": False,
            "visits": [
                {
                    "minutes_ago": 240,
                    "lines": [
                        {"name": "Ada", "kind": "message", "text": ADDRESSED, "own": False},
                        {"name": "Mozart", "kind": "message", "text": ANSWER, "own": True},
                    ],
                }
            ],
        }
    }
    assert welcome_back in runner.gateway.told_in(2)
    assert ADDRESSED not in runner.gateway.told_in(2).replace(welcome_back, "")


async def test_a_store_that_cannot_be_read_when_the_loop_starts_lets_it_answer_without_the_day_and_is_logged_once(
    runner, database
):
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
            await runner.reports.until("store_failed")
            say_in_the_call(runner, ADDRESSED, spoken=True)
            await runner.app.spoken_count(1)
    finally:
        await store.close()

    assert runner.app.spoken == [(SUBJECT, PLACE_ID, ANSWER)]
    assert "earlier_today" not in runner.gateway.system_of(1)
    assert runner.reports.events().count(("store_failed", STORE_TIMEOUT_CODE)) == 1
