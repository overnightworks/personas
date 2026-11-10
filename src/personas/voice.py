"""The voice loop's ports: the lines it hears, the sentences it speaks, the backend that carries the cap and the log."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.member import SpeakRefusal
from agents.turn import RejectionCode
from agents.voice import DecisionFailed, VoiceRefused

from personas.decision import CAPPED_DETAIL, spoken_wait_seconds
from personas.reporting import Event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from agents.agent_file import TurnRequest
    from agents.backend import AgentBackend, TurnEvent, TurnPerson
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

    async def render(self, sentence: str) -> str:
        return sentence

    async def play(self, audio: str) -> None:
        """Speaks the sentence and waits for as long as the app says it plays, to a bound; a refusal ends the answer."""
        await self.renew_token()
        outcome = await self.session.speak(self.place, audio)
        if isinstance(outcome, SpeakRefusal):
            self.report(Event.SPEAK_REFUSED, outcome.value)
            raise VoiceRefused
        await self.clock.sleep(spoken_wait_seconds(outcome.duration_ms))


@dataclass(frozen=True)
class CappedBackend:
    """A backend whose every run first reserves a turn against the agent's cap.

    A run the cap holds back ends refused with the cap's label, so that the loop's lines stay undecided.
    """

    backend: AgentBackend
    reserve: Callable[[], bool]

    async def run(self, request: TurnRequest, person: TurnPerson) -> AsyncGenerator[TurnEvent]:
        if not self.reserve():
            raise TurnStopped(TurnStopKind.REFUSED, RejectionCode.BUSY, detail=CAPPED_DETAIL)
        async with aclosing(self.backend.run(request, person)) as events:
            async for event in events:
                yield event


def reporting_decisions(report: Report) -> Callable[[VoiceEvent], None]:
    """A failed decision is logged by its code, or as capped; nothing the loop hears or says is logged."""

    def on_event(event: VoiceEvent) -> None:
        if not isinstance(event, DecisionFailed):
            return
        if event.detail == CAPPED_DETAIL:
            report(Event.CAPPED, None)
        else:
            report(Event.TURN_ENDED, event.code.value)

    return on_event
