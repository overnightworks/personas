"""The voice loop's ports: the lines it hears, the sentences it speaks and the log."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agents.member import SpeakRefusal
from agents.voice import DecisionFailed, VoiceRefused

from personas.decision import spoken_wait_seconds
from personas.reporting import Event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents.member import PlaceEntry, PlaceRef
    from agents.voice import VoiceEvent

    from personas.app_session import AppSession
    from personas.clock import Clock

type Report = Callable[[Event, str | None], None]


class Hearing:
    """The lines a voice loop hears: the runner puts in what it reads, and the loop takes them in order, for good."""

    def __init__(self) -> None:
        self._lines: asyncio.Queue[PlaceEntry] = asyncio.Queue()

    def put(self, line: PlaceEntry) -> None:
        self._lines.put_nowait(line)

    def __aiter__(self) -> Hearing:
        return self

    async def __anext__(self) -> PlaceEntry:
        return await self._lines.get()


@dataclass(frozen=True)
class ActiveVoice:
    """A place's running voice loop: its task, the window generation it hears and the lines it is yet to take."""

    task: asyncio.Task[None]
    generation: str
    hearing: Hearing


@dataclass(frozen=True)
class SpeakingVoice:
    """The app's ``speak`` as the loop's voice: a sentence is rendered and queued by the app when it is played.

    The session is the loop's own, and its token is renewed before each call.
    """

    session: AppSession
    place: PlaceRef
    clock: Clock
    report: Report
    renew_token: Callable[[], Awaitable[None]]

    # The library's port is an ``async def`` that pyright holds this to, and there is nothing to await: the app renders
    # the sentence when it is played.
    async def render(self, sentence: str) -> str:  # NOSONAR(S7503)
        return sentence

    async def play(self, audio: str) -> None:
        """Speaks the sentence and waits for as long as the app says it plays, to a bound; a refusal ends the answer."""
        await self.renew_token()
        outcome = await self.session.speak(self.place, audio)
        if isinstance(outcome, SpeakRefusal):
            self.report(Event.SPEAK_REFUSED, outcome.value)
            raise VoiceRefused
        await self.clock.sleep(spoken_wait_seconds(outcome.duration_ms))


def reporting_decisions(report: Report) -> Callable[[VoiceEvent], None]:
    """A failed decision is logged by its code; nothing the loop hears or says is logged."""

    def on_event(event: VoiceEvent) -> None:
        if not isinstance(event, DecisionFailed):
            return
        report(Event.TURN_ENDED, event.code.value)

    return on_event
