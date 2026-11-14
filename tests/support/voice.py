"""What the tests of a spoken turn share: a call of a speaking app, and a clock whose waits a test can hold."""

from __future__ import annotations

import asyncio

from tests.support.call import PLACE_ID, restarted_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.gateway import Gateway
from tests.support.history import at_table
from tests.support.hopin import AGENT_SUBJECT, LIMIT_SECONDS, active

TWO_SENTENCES = "First sentence. Second sentence."


class HeldSleeps:
    """The clock's sleeps wait until released; a test sees when each began and whether one was cancelled."""

    def __init__(self, runner) -> None:
        self.began: list[float] = []
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self._began = asyncio.Event()
        runner.clock.sleep = self._sleep
        runner.sleeps = self

    async def _sleep(self, seconds: float) -> None:
        self.began.append(seconds)
        self._began.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def until_began(self, count: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while len(self.began) < count:
                self._began.clear()
                await self._began.wait()

    def let_through(self) -> None:
        """The sleeps that wait now return; the ones that begin later are held again."""
        self.release.set()
        self.release.clear()

    async def until_cancelled(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.cancelled.wait()


def gated(runner) -> HeldSleeps:
    """The runner's held sleeps, made once."""
    return runner.sleeps or HeldSleeps(runner)


def holding_sentence_ends(runner) -> HeldSleeps:
    """The app's voiced sentences end only when the test calls ``runner.app.end_sentence()``, and the bound of the
    wait for that end, which is the turn's deadline, passes only when the test releases the returned sleeps."""
    runner.app.sentence_ends_held = True
    return gated(runner)


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def a_voice_call(runner, *replies, max_duration_seconds=PERSONA_MAX_DURATION_SECONDS):
    """The runner restarted in a call of a speaking app, for an agent whose entry has a voice table."""
    runner.give_voice(max_duration_seconds)
    runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)


async def in_a_voice_call_and_at_a_table(runner, *replies):
    """The runner started for an agent whose entry has a voice table, standing in a call and at a table of the same
    app, and has read the call's window once."""
    runner.give_voice()
    runner.gateway = Gateway(*replies)
    runner.app.stand(AGENT_SUBJECT, active(PLACE_ID), at_table())
    runner.start()
    await runner.app.until_window_read(PLACE_ID)


def sentences_spoken(runner) -> list[str]:
    return [text for _, _, text in runner.app.spoken]
