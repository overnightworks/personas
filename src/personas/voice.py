"""What a spoken turn does with its text: the sentences it releases, sent in order to the app's ``speak``, one ahead of
the one playing."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.member import SpeakRefusal, Spoken
from agents.turn import FailureCode
from agents.voice import StayedSilent
from agents.voice.reply import ReplyReader

from personas.decision import AppPlace, speech_outcome
from personas.entries import SaidAloud
from personas.failures import task_is_being_cancelled
from personas.reporting import Event
from personas.speech import may_send_next
from personas.task_record import RelayFailed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from agents.backend import TurnEvent
    from agents.member import PlaceRef, SpeakAnswer

    from personas.app_session import AppSession
    from personas.clock import Clock
    from personas.mind import Mind
    from personas.reporting import TurnTally
    from personas.turn_bound import TurnBound


class Speaking:
    """One spoken turn's speech: the sentences its reply releases, sent to ``speak`` in order, the next one as soon as
    the one before it is answered as spoken, so that the app renders it while that one plays.

    The app answers a sentence when its turn to play comes, so at most one request is open. Each sentence answered as
    spoken owes one line of his own, whole or cut, in the mind's history, and a sentence goes out only while at most the
    one playing is still owed. The turn hands every event to ``hear`` and runs ``play`` beside its events. A tool call
    waits until every owed line has entered, also once a refusal ended the speech; the speech itself ends at the
    answer of its last sentence, and the line of that sentence enters after the turn, which owes it in the mind's
    record. A request is sent and a line awaited only before the turn's bound: a request or line that has not come by
    then ends the turn as ``limit_exceeded``. Each sentence sent enters the agent's history as a ``SaidAloud`` however
    its request ends. A refusal, a stop included, ends the speech and is reported; the turn goes on without it. The
    turn's tally counts each sentence sent and each silent round. His own stop (``cut``) is no refusal and is not
    reported.

    A turn told a hold hands ``released``: no sentence is sent until it is set, which a decision does, and his own stop
    or a failed turn does too, so that the sentences waiting end unsent.
    """

    def __init__(
        self,
        session: AppSession,
        app: str,
        place: PlaceRef,
        clock: Clock,
        bound: TurnBound,
        renew_token: Callable[[], Awaitable[None]],
        report: Callable[[Event, str | None], None],
        mind: Mind,
        tally: TurnTally,
        released: asyncio.Event | None = None,
    ) -> None:
        self._session = session
        self._app = app
        self._place = place
        self._clock = clock
        self._bound = bound
        self._renew_token = renew_token
        self._report = report
        self._mind = mind
        self._tally = tally
        self._released = released
        self._at = AppPlace(app, place)
        self._reader = ReplyReader()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._silenced = False
        self.was_cut = False
        self.unsaid: deque[str] = deque()
        """The sentences heard that the app has not answered as spoken, the one in flight first."""
        self._idle = asyncio.Event()
        self._idle.set()

    def hear(self, event: TurnEvent) -> None:
        """Queues the sentences the event completed, and counts a round that opened with the silent marker."""
        reading = self._reader.read(event)
        if isinstance(reading.opening, StayedSilent):
            self._tally.silent_count += 1
        for sentence in reading.sentences:
            self.unsaid.append(sentence)
            self._queue.put_nowait(sentence)

    def cut(self) -> tuple[str, ...]:
        """His own stop: no sentence is sent from now on and the speech ends once the one in flight is answered; the
        sentences not yet said are returned."""
        unsaid = tuple(self.unsaid)
        self.was_cut = True
        self._silence()
        self.unsaid.clear()
        self._queue.put_nowait(None)
        return unsaid

    def seconds_left(self) -> float:
        return self._bound.seconds_left()

    def end(self) -> None:
        """No more events come: ``play`` ends once the queued sentences are spoken."""
        self._queue.put_nowait(None)

    async def settle(self) -> None:
        """The turn's events failed: no sentence is sent from now on, and one in flight is answered and entered."""
        self._silence()
        await self._idle.wait()

    def _silence(self) -> None:
        self._silenced = True
        if self._released is not None:
            self._released.set()

    async def spoken(self) -> None:
        """Returns once every sentence heard so far is answered and every line owed for them has entered; a turn past
        its bound by then ends."""
        await self._queue.join()
        await self._bound.until_lines(lambda owed: owed == 0, self._at)
        self._bound.stop_if_past()

    async def play(self) -> None:
        """Speaks the queued sentences in order until the queue ends, which is at the answer of the last one; the line
        of that sentence is not awaited."""
        while (sentence := await self._queue.get()) is not None:
            try:
                if not self._silenced:
                    await self._speak(sentence)
            finally:
                self._queue.task_done()

    async def _until_released(self) -> None:
        """Returns once the hold the turn was told is decided or the speech is silenced; the turn's deadline ends the
        turn first when it passes before."""
        if self._released is None or self._released.is_set():
            return
        waiting = asyncio.create_task(self._released.wait())
        try:
            await asyncio.wait({waiting, self._bound.deadline()}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()
        if not self._released.is_set():
            self._bound.deadline().result()
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)

    async def _speak(self, sentence: str) -> None:
        self._bound.stop_if_past()
        await self._until_released()
        await self._bound.until_lines(may_send_next, self._at)
        try:
            outcome = await self._entered_as_said(sentence)
        except TurnStopped:
            raise
        except Exception as error:
            raise RelayFailed(error) from None
        if outcome is None:
            if not self._silenced:
                self._bound.stop_if_past()
            return
        if isinstance(outcome, SpeakRefusal):
            if not (self.was_cut and outcome is SpeakRefusal.STOPPED):
                self._report(Event.SPEAK_REFUSED, outcome.value)
            self._silenced = True
        else:
            self._said()

    def _said(self) -> None:
        """The app took the sentence, voiced or written: it is no longer unsaid."""
        if not self.was_cut:
            self.unsaid.popleft()

    async def _answered_before_the_deadline(
        self, request: Coroutine[None, None, SpeakAnswer | SpeakRefusal]
    ) -> SpeakAnswer | SpeakRefusal:
        """The answer of the request, which the app holds back while the sentence before plays; a request not answered
        by the turn's deadline is given up, the turn ending ``limit_exceeded``."""
        sending = asyncio.create_task(request)
        try:
            while True:
                taken = self._mind.next_take()
                if sending.done():
                    return sending.result()
                if self._bound.deadline().done():
                    raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
                await self._bound.first_of(taken, sending, self._bound.deadline())
        finally:
            if not sending.done():
                sending.cancel()
                await asyncio.gather(sending, return_exceptions=True)

    async def _entered_as_said(self, sentence: str) -> SpeakAnswer | SpeakRefusal | None:
        """Sends the sentence and enters it into the history however its request ends; ``None`` when the token's
        renewal ran into the deadline or a silence, so that nothing was sent and nothing is entered.

        Whether the turn is being cancelled is asked once the request and every scope inside it have exited.
        """
        ended: SpeakAnswer | SpeakRefusal | BaseException | None = None
        sent = False
        self._idle.clear()
        try:
            await self._renew_token()
            if self._silenced or self._bound.past():
                return None
            self._tally.sentence_sent()
            self._mind.speech_sent(self._at, self._bound.ends_at)
            sent = True
            answer = await self._answered_before_the_deadline(self._session.speak(self._place, sentence))
            ended = answer
            return answer
        except Exception as error:
            ended = error
            raise
        except asyncio.CancelledError as cancelled:
            ended = cancelled
            raise
        finally:
            if sent:
                self._mind.speech_answered(self._at, sentence, spoken=isinstance(ended, Spoken))
            self._idle.set()
            outcome = None if ended is None else speech_outcome(ended, cancelling=task_is_being_cancelled())
            if outcome is not None:
                self._mind.take((SaidAloud(self._app, self._place, sentence, outcome, self._clock.wall()),))
