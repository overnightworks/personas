"""What the tests of a spoken turn share: a call of a speaking app, and a clock whose waits a test can hold."""

from __future__ import annotations

import asyncio

from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import SpeakRefusal, SpeakRefused
from agents.tools import ToolOutcome

from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.gateway import Gateway, say
from tests.support.history import at_table, settled
from tests.support.hopin import CALL, LIMIT_SECONDS
from tests.support.standings import AGENT_SUBJECT, active
from tests.support.wire import hangs_once_entered

TWO_SENTENCES = "First sentence. Second sentence."
HOLD_ID = "hold-1"
SAID = ("Ada", "wait a moment")
FIRST, SECOND = "First sentence.", "Second sentence."


class HeldSleeps:
    """The clock's sleeps wait until released and then take their time off the clock; a test sees when each began and
    whether one was cancelled."""

    def __init__(self, runner) -> None:
        self._clock = runner.clock
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
        self._clock.time += seconds

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


async def a_turn_whose_speak_is_in_flight(runner, said: asyncio.Event | None = None):
    """The runner restarted in a voice call; the turn's first sentence reached the app's ``speak``, which never
    answers."""
    entered = asyncio.Event()
    runner.app.script(CALL, hangs_once_entered(entered))
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    if said is not None:
        said.set()
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()


async def in_a_voice_call_and_at_a_table(runner, *replies):
    """The runner started for an agent whose entry has a voice table, standing in a call and at a table of the same
    app, and has read the call's window once."""
    runner.give_voice()
    runner.gateway = Gateway(*replies)
    runner.app.stand(AGENT_SUBJECT, active(PLACE_ID), at_table())
    runner.start()
    await runner.app.until_window_read(PLACE_ID)


async def calling_while_held(runner, *after_the_hold, hold_round: bool = False, words: bool = True):
    """A spoken turn whose first round writes a sentence and calls ``say`` while it plays, whose second round, held back
    until a hold without words stands at the app, calls ``say`` again; people then talk over him. With ``hold_round``
    the round that decides the hold waits for the returned event; with no ``words`` nobody talks over him."""
    holding_sentence_ends(runner)
    first, second = tool_round(say("before", PLACE_ID), content=FIRST), tool_round(say("waits", PLACE_ID))
    await a_voice_call(runner, first, second, *after_the_hold)
    second_round = runner.gateway.hold_request(2)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.said_count(1)
    await runner.app.spoken_count(1)
    runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", ""))
    await settled(runner.app)
    deciding_round = runner.gateway.hold_request(3) if hold_round else None
    second_round.set()
    await runner.gateway.until_answered(2)
    if words:
        runner.app.talk_over(PLACE_ID, SAID)
    return deciding_round


def deciding(tool: str, hold_id: str = HOLD_ID):
    return tool_round(tool_call(f"{tool}-{hold_id}", tool, {"id": hold_id}))


async def held_in_the_first_sentence(runner, *replies, said=(SAID,), hold_id: str = HOLD_ID):
    """A spoken turn whose first sentence plays and whose second waits for it, and the app then holds him."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, *replies)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold(PLACE_ID, hold_id, *said)


def sentences_spoken(runner) -> list[str]:
    return [text for _, _, text in runner.app.spoken]


THREE_SENTENCES = "First sentence. Second sentence. Third sentence."


async def decided(runner, count: int = 1) -> list[str | None]:
    await runner.reports.until(Event.HOLD_DECIDED, count=count)
    return runner.reports.codes_of(Event.HOLD_DECIDED)


def refusing(refusal: SpeakRefusal) -> ToolOutcome:
    return ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True)
