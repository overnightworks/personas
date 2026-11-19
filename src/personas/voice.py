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

from personas.decision import AppPlace, may_send_next, past_deadline, remaining_seconds, speech_outcome
from personas.entries import SaidAloud
from personas.failures import task_is_being_cancelled
from personas.reporting import Event
from personas.task_record import RelayFailed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from agents.backend import TurnEvent
    from agents.member import PlaceRef, SpeakAnswer

    from personas.app_session import AppSession
    from personas.clock import Clock
    from personas.mind import Mind
    from personas.reporting import TurnTally


class Speaking:
    """One spoken turn's speech: the sentences its reply releases, sent to ``speak`` in order, the next one as soon as
    the one before it is answered as spoken, so that the app renders it while that one plays.

    The app answers a sentence when its turn to play comes, so at most one request is open. Each sentence answered as
    spoken owes one line of his own, whole or cut, in the mind's history, and a sentence goes out only while at most the
    one playing is still owed. The turn hands every event to ``hear`` and runs ``play`` beside its events. A tool call
    and the end of the speech wait until every owed line has entered, also once a refusal ended the speech. A request is
    sent and its line awaited only before the turn's deadline: a request or line that has not come by then ends the
    turn as ``limit_exceeded``. Each sentence sent enters the agent's history as a ``SaidAloud`` however its request
    ends. A refusal, a stop included, ends the speech and is reported; the turn goes on without it. The turn's tally
    counts each sentence sent and each silent round. His own stop (``cut``) is no refusal and is not reported.

    A turn told a hold hands ``released``: no sentence is sent until it is set, which a decision does, and his own stop
    or a failed turn does too, so that the sentences waiting end unsent.
    """

    def __init__(
        self,
        session: AppSession,
        app: str,
        place: PlaceRef,
        clock: Clock,
        max_duration_seconds: float,
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
        self._started = clock.now()
        self._max_duration_seconds = max_duration_seconds
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
        self._deadline: asyncio.Task[None] | None = None

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
        return remaining_seconds(self._clock.now(), self._started, self._max_duration_seconds)

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
        its deadline by then ends."""
        await self._queue.join()
        await self._until(lambda: self._owed == 0)
        self._stop_past_deadline()

    async def play(self) -> None:
        """Speaks the queued sentences in order until the queue ends and every owed line has entered; a turn past its
        deadline by then ends."""
        spoke = False
        try:
            while (sentence := await self._queue.get()) is not None:
                try:
                    if not self._silenced:
                        spoke = True
                        await self._speak(sentence)
                finally:
                    self._queue.task_done()
            await self._until(lambda: self._owed == 0)
            if spoke and not self._silenced:
                self._stop_past_deadline()
        finally:
            if self._deadline is not None:
                self._deadline.cancel()

    @property
    def _owed(self) -> int:
        """The own spoken lines that sentences answered as spoken have not yet brought into the history."""
        return self._mind.speech_owed(self._at)

    async def _until_released(self) -> None:
        """Returns once the hold the turn was told is decided or the speech is silenced; the turn's deadline ends the
        turn first when it passes before."""
        if self._released is None or self._released.is_set():
            return
        waiting = asyncio.create_task(self._released.wait())
        try:
            await asyncio.wait({waiting, self._deadline_bound()}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()
        if not self._released.is_set():
            self._deadline_bound().result()
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)

    async def _speak(self, sentence: str) -> None:
        self._stop_past_deadline()
        await self._until_released()
        if self._silenced:
            return
        await self._until(lambda: may_send_next(self._owed))
        try:
            outcome = await self._entered_as_said(sentence)
        except TurnStopped:
            raise
        except Exception as error:
            raise RelayFailed(error) from None
        if outcome is None:
            if not self._silenced:
                self._stop_past_deadline()
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

    async def _until(self, reached: Callable[[], bool]) -> None:
        """Returns once ``reached`` holds, judged at every take of the mind; the turn's deadline, the one bound of all
        the waits of the speech, ends the turn first when it passes before, a tie going to it."""
        while True:
            taken = self._mind.next_take()
            if self._deadline is not None and self._deadline.done():
                self._deadline.result()
                raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
            if reached():
                return
            self._stop_past_deadline()
            await self._first_of(taken, self._deadline_bound())

    @staticmethod
    async def _first_of(taken: asyncio.Event, *ends: asyncio.Task[object]) -> None:
        """Returns at the mind's next take or once one of ``ends`` is done, whichever comes first."""
        woken = asyncio.create_task(taken.wait())
        try:
            await asyncio.wait({*ends, woken}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            woken.cancel()

    def _deadline_bound(self) -> asyncio.Task[None]:
        """The task that ends when the turn's deadline passes, made once for the whole speech."""
        if self._deadline is None:
            remaining = remaining_seconds(self._clock.now(), self._started, self._max_duration_seconds)
            self._deadline = asyncio.create_task(self._clock.sleep(remaining))
        return self._deadline

    def _past_deadline(self) -> bool:
        return past_deadline(self._clock.now(), self._started, self._max_duration_seconds)

    def _stop_past_deadline(self) -> None:
        if self._past_deadline():
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)

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
                if self._deadline_bound().done():
                    raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
                await self._first_of(taken, sending, self._deadline_bound())
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
            if self._silenced or self._past_deadline():
                return None
            self._tally.sentence_sent()
            self._mind.speech_sent(self._at)
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
