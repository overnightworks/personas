"""What a spoken turn does with its text: the sentences it releases, sent one at a time to the app's ``speak``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.member import SpeakRefusal, Spoken
from agents.turn import FailureCode
from agents.voice.reply import ReplyReader

from personas.decision import AppPlace, past_deadline, remaining_seconds, sentence_ended, speech_outcome
from personas.entries import SaidAloud
from personas.failures import task_is_being_cancelled
from personas.reporting import Event
from personas.task_record import RelayFailed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents.backend import TurnEvent
    from agents.member import PlaceRef, SpeakAnswer

    from personas.app_session import AppSession
    from personas.clock import Clock
    from personas.mind import Mind


class Speaking:
    """One spoken turn's speech: the sentences its reply releases, spoken in order, each after the one before has been
    answered and his own spoken line for it has entered the mind's history.

    The turn hands every event to ``hear`` and runs ``play`` beside its events. A sentence is sent only before the
    turn's deadline, and the wait for its line ends at that deadline: a line that has not entered by then ends the
    turn as ``limit_exceeded``. Each sentence sent enters the agent's history as a ``SaidAloud`` however its request
    ends. A refusal ends the speech and is reported; the turn goes on without it.
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
        self._at = AppPlace(app, place)
        self._sent_at = 0
        self._reader = ReplyReader()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._silenced = False
        self._idle = asyncio.Event()
        self._idle.set()

    def hear(self, event: TurnEvent) -> None:
        """Queues the sentences the event completed."""
        for sentence in self._reader.read(event).sentences:
            self._queue.put_nowait(sentence)

    def end(self) -> None:
        """No more events come: ``play`` ends once the queued sentences are spoken."""
        self._queue.put_nowait(None)

    async def settle(self) -> None:
        """The turn's events failed: no sentence is sent from now on, and one in flight is answered and entered."""
        self._silenced = True
        await self._idle.wait()

    async def spoken(self) -> None:
        """Returns once every sentence heard so far is answered and his own line for it has entered; a turn past its
        deadline by then ends."""
        await self._queue.join()
        self._stop_past_deadline()

    async def play(self) -> None:
        """Speaks the queued sentences in order until the queue ends; a turn past its deadline by the last one's end
        ends."""
        spoke = False
        while (sentence := await self._queue.get()) is not None:
            try:
                if not self._silenced:
                    spoke = True
                    await self._speak(sentence)
            finally:
                self._queue.task_done()
        if spoke and not self._silenced:
            self._stop_past_deadline()

    async def _speak(self, sentence: str) -> None:
        self._stop_past_deadline()
        try:
            outcome = await self._entered_as_said(sentence)
        except Exception as error:
            raise RelayFailed(error) from None
        if outcome is None:
            if not self._silenced:
                self._stop_past_deadline()
            return
        if isinstance(outcome, SpeakRefusal):
            self._report(Event.SPEAK_REFUSED, outcome.value)
            self._silenced = True
        elif isinstance(outcome, Spoken) and not self._silenced:
            await self._until_ended(self._sent_at)

    async def _until_ended(self, mark: int) -> None:
        """Returns once his own spoken line for the sentence sent at ``mark`` has entered; the turn's deadline, the one
        bound of all the waits for that sentence, ends the turn first when it passes before, a tie going to it."""
        remaining = remaining_seconds(self._clock.now(), self._started, self._max_duration_seconds)
        if remaining == 0:
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
        bound = asyncio.create_task(self._clock.sleep(remaining))
        try:
            while True:
                if bound.done():
                    bound.result()
                    raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
                taken = self._mind.next_take()
                if sentence_ended(self._mind.history, mark, self._at):
                    return
                woken = asyncio.create_task(taken.wait())
                try:
                    await asyncio.wait({bound, woken}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    woken.cancel()
        finally:
            bound.cancel()

    def _past_deadline(self) -> bool:
        return past_deadline(self._clock.now(), self._started, self._max_duration_seconds)

    def _stop_past_deadline(self) -> None:
        if self._past_deadline():
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)

    async def _entered_as_said(self, sentence: str) -> SpeakAnswer | SpeakRefusal | None:
        """Sends the sentence and enters it into the history however its request ends; ``None`` when the token's
        renewal ran into the deadline or a silence, so that nothing was sent and nothing is entered.

        Whether the turn is being cancelled is asked once the request and every scope inside it have exited.
        """
        ended: SpeakAnswer | SpeakRefusal | BaseException | None = None
        self._idle.clear()
        try:
            await self._renew_token()
            if self._silenced or self._past_deadline():
                return None
            self._sent_at = self._mind.history.taken
            answer = await self._session.speak(self._place, sentence)
            ended = answer
            return answer
        except Exception as error:
            ended = error
            raise
        except asyncio.CancelledError as cancelled:
            ended = cancelled
            raise
        finally:
            self._idle.set()
            outcome = None if ended is None else speech_outcome(ended, cancelling=task_is_being_cancelled())
            if outcome is not None:
                self._mind.take((SaidAloud(self._app, self._place, sentence, outcome, self._clock.wall()),))
