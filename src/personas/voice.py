"""What a spoken turn does with its text and who sends it: the turn releases its sentences into the mind's record, and
the sender of the place sends them to the app's ``speak``, one request open at a time."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.member import SpeakRefusal
from agents.turn import FailureCode
from agents.voice import StayedSilent
from agents.voice.reply import ReplyReader

from personas.decision import AppPlace, speech_outcome
from personas.entries import SaidAloud
from personas.failures import task_is_being_cancelled
from personas.reporting import Event
from personas.speech import Answered, DropReason, answered_as
from personas.task_record import TaskRecord, ending_loud_when_cancelled, ending_with_recorded_defect

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from agents.backend import TurnEvent
    from agents.member import PlaceRef, SpeakAnswer

    from personas.app_session import AppSession
    from personas.clock import Clock
    from personas.mind import Mind
    from personas.reporting import TurnTally
    from personas.speech import Sentence
    from personas.turn_bound import TurnBound

type DropReporter = Callable[[DropReason, int], None]


class Speaking:
    """One spoken turn's speech: the sentences its reply releases, put into the mind's record at the place with the
    turn's bound, and a round that opened with the silent marker counted. The turn's tally counts each sentence
    released. The first sentence a turn told no hold releases replaces the unsent rest of the turns before it, which
    is logged as dropped. His own stop (``cut``) ends the releasing; the sentences the stop drops are the record's.
    Sending them is the place's ``Sender``'s, also after the turn.
    """

    def __init__(
        self,
        app: str,
        place: PlaceRef,
        bound: TurnBound,
        mind: Mind,
        tally: TurnTally,
        report_dropped: DropReporter,
        *,
        replacing: bool,
    ) -> None:
        self._at = AppPlace(app, place)
        self._bound = bound
        self._mind = mind
        self._tally = tally
        self._report_dropped = report_dropped
        self._replacing = replacing
        self._reader = ReplyReader()
        self.was_cut = False

    def hear(self, event: TurnEvent) -> None:
        """Releases the sentences the event completed, and counts a round that opened with the silent marker."""
        reading = self._reader.read(event)
        if isinstance(reading.opening, StayedSilent):
            self._tally.silent_count += 1
        if self.was_cut or not reading.sentences:
            return
        for _ in reading.sentences:
            self._tally.sentence_released()
        dropped = self._mind.sentences_released(
            self._at, reading.sentences, self._bound.ends_at(), replacing=self._replacing
        )
        self._replacing = False
        self._report_dropped(DropReason.REPLACED, dropped)

    def cut(self) -> None:
        """His own stop: the turn releases no sentence from now on."""
        self.was_cut = True

    def seconds_left(self) -> float:
        return self._bound.seconds_left()


class Sender:
    """The sender of one place's sentences on the loop's session: it takes the next sentence of the mind's record and
    sends it to ``speak``, one request open at a time, which the app answers when its sentence starts to play, so he is
    never more than one sentence ahead. Nothing goes out while a hold stands. A sentence unsent at its bound is dropped
    and logged, and a request not answered by its bound is given up. Each sentence sent enters the agent's history as
    a ``SaidAloud`` however its request ends, unless the app answered it as held, which keeps it first among the
    unsent. Any other refusal drops the unsent sentences and is reported; one that his own stop caused is not. It wakes
    on the place's own event, never on the mind's change.

    A ``speak`` that fails as a call ends the sender with the failure, which ends the loop's session. A defect ends the
    runner loud, a cancel included. At its end the request in flight is gone and the unsent sentences drop.
    """

    def __init__(
        self,
        session: AppSession,
        app: str,
        place: PlaceRef,
        clock: Clock,
        renew_token: Callable[[], Awaitable[None]],
        report: Callable[[Event, str | None], None],
        report_dropped: DropReporter,
        mind: Mind,
    ) -> None:
        self._session = session
        self._app = app
        self._place = place
        self._clock = clock
        self._renew_token = renew_token
        self._report = report
        self._report_dropped = report_dropped
        self._mind = mind
        self._at = AppPlace(app, place)
        self._wake = mind.speech_wake(self._at)
        self._record = TaskRecord()

    async def run(self) -> None:
        """Sends until it is cancelled, wrapped like a turn's task: a defect ends the runner loud whatever way the
        task ends."""
        record = self._record
        await ending_loud_when_cancelled(ending_with_recorded_defect(self._serve(), record), record.loud_end)

    async def _serve(self) -> None:
        try:
            await self._send_forever()
        except Exception as error:
            self._record.note(error)
            raise
        finally:
            self._report_dropped(self._mind.drop_reason_at_end(), self._mind.speech_ended(self._at))

    async def _send_forever(self) -> None:
        given_up = 0
        while True:
            self._wake.clear()
            await self._renew_token()
            sentence, expired = self._mind.sentence_to_send(self._at, self._clock.now())
            self._report_dropped(DropReason.LIMIT_EXCEEDED, expired + given_up)
            given_up = 0
            if sentence is None:
                await self._wake.wait()
                continue
            try:
                await self._sent(sentence)
            except TurnStopped:
                given_up = 1

    async def _answered_before_its_bound(
        self, sentence: Sentence, request: Coroutine[None, None, SpeakAnswer | SpeakRefusal]
    ) -> SpeakAnswer | SpeakRefusal:
        """The answer of the request, which the app holds back while the sentence before plays; a request not answered
        by the sentence's bound is given up."""
        sending = asyncio.create_task(request)
        giving_up = asyncio.create_task(self._clock.sleep(max(0.0, sentence.bound - self._clock.now())))
        try:
            await asyncio.wait({sending, giving_up}, return_when=asyncio.FIRST_COMPLETED)
            if sending.done():
                return sending.result()
            raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
        finally:
            giving_up.cancel()
            if not sending.done():
                sending.cancel()
            # A cancel can reach this wait after the request raised, which the cancel would lose.
            self._note_failure_of(sending)
            await asyncio.gather(sending, giving_up, return_exceptions=True)
            self._note_failure_of(sending)

    def _note_failure_of(self, request: asyncio.Task[object]) -> None:
        if request.done() and not request.cancelled() and isinstance(failure := request.exception(), Exception):
            self._record.note(failure)

    async def _sent(self, sentence: Sentence) -> None:
        """Sends the sentence and enters it into the history however its request ends.

        Whether the sender is being cancelled is asked once the request and every scope inside it have exited.
        """
        ended: SpeakAnswer | SpeakRefusal | BaseException | None = None
        try:
            answer = await self._answered_before_its_bound(sentence, self._session.speak(self._place, sentence.text))
            ended = answer
        except Exception as error:
            ended = error
            raise
        except asyncio.CancelledError as cancelled:
            ended = cancelled
            raise
        finally:
            answered = answered_as(ended)
            request = self._mind.speech_answered(self._at, answered)
            outcome = None if ended is None else speech_outcome(ended, cancelling=task_is_being_cancelled())
            if outcome is not None and answered is not Answered.HELD:
                self._mind.take((SaidAloud(self._app, self._place, sentence.text, outcome, self._clock.wall()),))
        if isinstance(answer, SpeakRefusal) and answered is Answered.REFUSED and not request.cut_by_stop(answer):
            self._report(Event.SPEAK_REFUSED, answer.value)
