"""What the tests of a spoken turn share: a call of a speaking app, and a clock whose waits a test can hold."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, SpeakRefusal, SpeakRefused
from agents.tools import ToolOutcome

from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.gateway import Gateway, say
from tests.support.history import at_table
from tests.support.hopin import CALL, LIMIT_SECONDS
from tests.support.standings import AGENT_SUBJECT, active
from tests.support.wire import hangs_once_entered

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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


async def calling_while_held(
    runner, *after_the_hold, hold_round: bool = False, words: bool = True, wordless: bool = False
):
    """A spoken turn whose first round writes a sentence and calls ``say`` while it plays, whose second round, held back
    until a hold without words stands at the app, calls ``say`` again; people then talk over him. With ``hold_round``
    the round that decides the hold waits for the returned event; with no ``words`` nobody talks over him; a
    ``wordless`` hold is one the app ends by itself."""
    holding_sentence_ends(runner)
    first, second = tool_round(say("before", PLACE_ID), content=FIRST), tool_round(say("waits", PLACE_ID))
    await a_voice_call(runner, first, second, *after_the_hold)
    second_round = runner.gateway.hold_request(2)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.said_count(1)
    await runner.app.spoken_count(1)
    if wordless:
        runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    else:
        runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", ""))
    await settled(runner.app)
    deciding_round = runner.gateway.hold_request(3) if hold_round else None
    second_round.set()
    await runner.gateway.until_answered(2)
    if words:
        runner.app.talk_over(PLACE_ID, SAID)
    return deciding_round


def continuing(arguments: dict[str, object], hold_id: str = HOLD_ID):
    return tool_round(tool_call(f"continue-{hold_id}", CONTINUE_SPEAKING_TOOL, {"id": hold_id, **arguments}))


def continuing_hold(hold_id: str):
    return continuing({}, hold_id)


def deciding(tool: str, hold_id: str = HOLD_ID, *, time: int = 1):
    """A reply that decides the hold with ``tool``; the call of a decision made again is told apart by its ``time``."""
    return tool_round(tool_call(f"{tool}-{hold_id}-{time}", tool, {"id": hold_id}))


async def held_in_the_first_sentence(runner, *replies, said=(SAID,), hold_id: str = HOLD_ID):
    """A spoken turn whose first sentence plays and whose second waits for it, and the app then holds him."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, *replies)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold(PLACE_ID, hold_id, *said)


async def play_out(runner, count: int) -> None:
    """Each of the first ``count`` sentences that plays ends."""
    for spoken in range(1, count + 1):
        await runner.app.spoken_count(spoken)
        if runner.app.playing:
            runner.app.end_sentence()


def sentences_spoken(runner) -> list[str]:
    return [text for _, _, text in runner.app.spoken]


THIRD = "Third sentence."
THREE_SENTENCES = "First sentence. Second sentence. Third sentence."
SECOND_HOLD = "hold-2"
SECOND_LINE = "Yes, one moment."
SECOND_ANSWER = "I hear you."
SECOND_HOLD_DECISIONS = pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (deciding(CONTINUE_SPEAKING_TOOL, SECOND_HOLD), "continued"),
        (deciding(STOP_SPEAKING_TOOL, SECOND_HOLD), "stopped"),
        (completion("[silent]"), "undecided"),
        (continuing({"line": SECOND_LINE}, SECOND_HOLD), "short_line"),
        (completion(SECOND_ANSWER), "answered"),
    ],
    ids=["go on", "stop", "no decision", "short line", "answer instead"],
)


def heard_after_the_second_hold(outcome: str, spoken: list[str], rest: list[str]) -> list[str]:
    """What the app has played once the second hold is decided: what had played, and then what the decision lets
    follow it, the rest of the turn unless he stops or answers instead."""
    follows = {
        "continued": rest,
        "short_line": [SECOND_LINE, *rest],
        "answered": [SECOND_ANSWER],
        "stopped": [],
        "undecided": [],
    }
    return [*spoken, *follows[outcome]]


async def decided(runner, count: int = 1) -> list[str | None]:
    await runner.reports.until(Event.HOLD_DECIDED, count=count)
    return runner.reports.codes_of(Event.HOLD_DECIDED)


def refusing(refusal: SpeakRefusal) -> ToolOutcome:
    return ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True)


REFUSALS = pytest.mark.parametrize(
    ("refusal", "outcome", "stops"),
    [
        pytest.param(SpeakRefusal.TOO_LONG, "undecided", [(STOP_SPEAKING_TOOL, HOLD_ID)], id="refused"),
    ],
)


async def refused_ends_the_hold_as_the_app_says(
    runner,
    arrange: Callable[[], Awaitable[None]],
    refusal: SpeakRefusal,
    outcome: str,
    stops: list[tuple[str, str]],
    unsent: str,
    *,
    taken_before: int = 2,
    cut: str | None = SECOND,
) -> None:
    """The app refuses the speak after ``taken_before`` taken ones: the hold ends as its refusal says, the ``unsent``
    sentence is never sent and the sentence his stop ``cut`` is sent once, never put back. ``arrange`` sets up the held
    turn."""
    runner.app.speak_outcomes.extend([*[None] * taken_before, refusing(refusal)])
    await arrange()

    assert await decided(runner) == [outcome]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == stops
    assert runner.reports.codes_of(Event.SPEAK_REFUSED) == [refusal.value]
    sent = [text for text, _ in runner.app.speaks_naming_holds]
    assert unsent not in sent
    if cut is not None:
        assert sent.count(cut) == 1
