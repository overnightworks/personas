"""What a spoken turn does with its text and who sends it: the turn releases its sentences into the mind's record, and
the sender of the place sends them to the app's ``speak``, one request open at a time; and how his text answers a hold
of his instead."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from agents.backend import TurnStopKind, TurnStopped
from agents.member import SpeakRefusal, Spoken, Written
from agents.member_outcomes import CallFailure
from agents.tools import ToolOutcome
from agents.turn import FailureCode
from agents.turn_events import FinalEvent, ToolCallEvent
from agents.voice import StayedSilent
from agents.voice.reply import ReplyReader

from personas.decision import AppPlace, failure_code, speech_outcome
from personas.entries import SaidAloud
from personas.failures import task_is_being_cancelled
from personas.holds import (
    APP_FAILED_TEXT,
    CONTINUED_TEXT,
    DECISIONS_BOUND_TEXT,
    GONE_TEXT,
    LINE_ARGUMENT,
    STYLE_ARGUMENT,
    dropping,
)
from personas.reporting import Event, failure_event
from personas.speech import Answered, DropReason, HoldOutcome, NotDecided, answered_as, bound_reached, hold_ended_as
from personas.task_record import (
    RelayFailed,
    TaskRecord,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine

    from agents.backend import TurnEvent
    from agents.mcp_client import MemberSession
    from agents.member import PlaceRef, SpeakAnswer
    from agents.turn import JsonObject

    from personas.clock import Clock
    from personas.config import AppTarget
    from personas.holds import HoldInbox, TurnHold
    from personas.mind import Mind
    from personas.reporting import LoopReport, TurnTally
    from personas.speech import Held, Sentence
    from personas.turn_bound import TurnBound
    from personas.turn_sessions import TurnSessions

type DropReporter = Callable[[DropReason, int], None]

_LINE_TEXT: Final = "line must be a non-empty string"
_STYLE_TEXT: Final = "style must be a non-empty string"
_STYLE_ALONE_TEXT: Final = "style comes only with a line"
_REFUSED: Final = frozenset({Answered.REFUSED, Answered.STOPPED})


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _line_told(ended: HoldOutcome | NotDecided) -> ToolOutcome:
    """What his short line's call is told once the hold ended as ``ended``, other than as stale."""
    match ended:
        case HoldOutcome.SHORT_LINE:
            return ToolOutcome(CONTINUED_TEXT, is_error=False)
        case NotDecided.GONE:
            return ToolOutcome(GONE_TEXT, is_error=False)
        case _:
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)


@dataclass(frozen=True)
class Voicing:
    """What the spoken turns of one loop share: the sessions they ask other apps in, the app, the report, the clock,
    the mind and the reporter of dropped sentences."""

    sessions: TurnSessions
    app: AppTarget
    report: LoopReport
    clock: Clock
    mind: Mind
    report_dropped: DropReporter

    def speaking(
        self,
        place: PlaceRef,
        record: TaskRecord,
        tally: TurnTally,
        bound: TurnBound,
        inbox: HoldInbox,
        told: TurnHold | None,
        decided: Callable[[Held, HoldOutcome], None],
        ended_undecided: Callable[[Held], Awaitable[HoldOutcome | NotDecided | None]],
        retold: Callable[[TurnHold], Awaitable[ToolOutcome]],
    ) -> tuple[Speaking, Answering]:
        """One spoken turn's speech and its answer instead of a hold of his."""
        speaking = Speaking(
            self.app.mcp_url, place, bound, self.mind, tally, self.report_dropped, told=told, replacing=told is None
        )
        return speaking, Answering(self, place, record, tally, bound, inbox, told, decided, ended_undecided, retold)


class Answering:
    """His answer instead of a hold of his: the text of a reply while the hold he was told is undecided. It drops the
    unsent sentences at the place, logged as replaced, once the app took the first sentence, and cuts the parked run
    then. The first sentence goes to ``speak`` naming the hold and the changes of the talk-over he was told, in a
    session of its own, since the sender's open request answers only once the app took it; the sender sends nothing
    meanwhile. The rest joins the mind's record with the deciding turn's bound once the app took the first, and the
    place's sender sends it. A first sentence the app refused or never answered sends no further one: the refusal or
    the failure is reported, and a hold that no longer stood decides nothing and drops nothing; a hold that stands ends
    undecided, as the answer stale is no decision to tell him of.
    """

    def __init__(
        self,
        loop: Voicing,
        place: PlaceRef,
        record: TaskRecord,
        tally: TurnTally,
        bound: TurnBound,
        inbox: HoldInbox,
        told: TurnHold | None,
        decided: Callable[[Held, HoldOutcome], None],
        ended_undecided: Callable[[Held], Awaitable[HoldOutcome | NotDecided | None]],
        retold: Callable[[TurnHold], Awaitable[ToolOutcome]],
    ) -> None:
        self._told = told
        self._sessions = loop.sessions
        self._app = loop.app
        self._report = loop.report
        self._clock = loop.clock
        self._mind = loop.mind
        self._report_dropped = loop.report_dropped
        self._place = place
        self._at = AppPlace(loop.app.mcp_url, place)
        self._record = record
        self._tally = tally
        self._bound = bound
        self._inbox = inbox
        self._decided = decided
        self._ended_undecided = ended_undecided
        self._retold = retold

    async def hear_round(
        self, events: AsyncIterator[TurnEvent], told: TurnHold, cut_parked_run: Callable[[], Awaitable[object]]
    ) -> bool:
        """Reads the events of the round that decides the hold ``told``: its first text while the hold is undecided
        answers instead, and the parked run is cut with its waiting call unsent once the app took the answer or the
        stop that follows its failure; the rest of that reply follows the answer. Text of a later reply says nothing;
        the round's calls are answered as before. It tells whether it tried to answer, which ended the hold or left it
        as the app says, once."""
        reader = ReplyReader()
        answer: HoldOutcome | NotDecided | None = None
        tried = False
        async for event in events:
            sentences = reader.read(event).sentences
            if sentences and answer is HoldOutcome.ANSWERED:
                self._release(sentences)
            elif sentences and not tried and self._mind.hold_decided_key(self._at) != told.held.key:
                tried = True
                answer = await self.instead(told, sentences)
                if isinstance(answer, HoldOutcome):
                    await cut_parked_run()
            if isinstance(event, ToolCallEvent | FinalEvent):
                answer = None
        return tried

    async def instead_told(self, sentences: tuple[str, ...]) -> None:
        """The sentences answer the hold the turn was told instead; the hold is decided for the turn's calls."""
        assert self._told is not None  # noqa: S101 - only a turn told a hold has text that answers it
        ended = await self.instead(self._told, sentences)
        self._told.outcome = HoldOutcome.UNDECIDED if ended is NotDecided.GONE else ended

    async def instead(self, told: TurnHold, sentences: tuple[str, ...]) -> HoldOutcome | NotDecided:
        """The sentences answer the hold ``told`` instead of what he had not yet said; how the hold ended. The unsent
        sentences drop once the app took the first sentence; one it refused as stale, for the words that came while he
        answered, ends the hold undecided, since no call of his can be told what was said since."""
        self._inbox.answering()
        first, *rest = sentences
        with dropping(self._mind, self._at, self._inbox):
            ended = await self._sent_naming(told, first, HoldOutcome.ANSWERED, rest)
            if ended is NotDecided.STALE:
                ended = await self._ended_undecided(told.held) or HoldOutcome.UNDECIDED
        if ended is NotDecided.GONE:
            self._inbox.answer_refused()
        return ended

    async def line_decided(self, told: TurnHold, arguments: JsonObject) -> ToolOutcome:
        """His ``continue_speaking`` of the hold ``told`` with a ``line`` and maybe a ``style``: the line says at once,
        in that style, what the call tells him, which for a line told too few changes is what was said since; a line or
        style that is no non-empty string, or a style without a line, decides nothing. The hold the turn was told is
        decided for the turn's calls."""
        line, style = arguments.get(LINE_ARGUMENT), arguments.get(STYLE_ARGUMENT)
        if LINE_ARGUMENT in arguments and not _is_text(line):
            return ToolOutcome(_LINE_TEXT, is_error=True)
        if STYLE_ARGUMENT in arguments and not _is_text(style):
            return ToolOutcome(_STYLE_TEXT, is_error=True)
        if not isinstance(line, str):
            return ToolOutcome(_STYLE_ALONE_TEXT, is_error=True)
        if not told.take_decision():
            return ToolOutcome(DECISIONS_BOUND_TEXT, is_error=True)
        ended = await self._line(told, line, style if isinstance(style, str) else None)
        if ended is NotDecided.STALE:
            return await self._retold(told)
        if self._told is told:
            told.outcome = ended
        return _line_told(ended)

    async def _line(self, told: TurnHold, line: str, style: str | None) -> HoldOutcome | NotDecided:
        """The line goes to ``speak`` naming the hold ``told``, the hold's sentence cut and the unsent ones kept for
        after it; a line the app refused or never answered sends no further sentence. The inbox stays open: a hold
        that stands after it is a new hold."""
        self._mind.speech_line_said(self._at, self._bound.ends_at())
        return await self._sent_naming(told, line, HoldOutcome.SHORT_LINE, (), style)

    async def _sent_naming(
        self,
        told: TurnHold,
        first: str,
        ended_as: HoldOutcome,
        rest: list[str] | tuple[str, ...],
        style: str | None = None,
    ) -> HoldOutcome | NotDecided:
        """The sentence goes to ``speak`` naming the hold ``told`` and the changes he was told; once the app took it the
        hold ends as ``ended_as``, what he had not said drops if his answer replaces it, and the rest follows; how the
        hold ended."""
        hold = told.held
        self._tally.sentence_released()
        answer = await self._speak(first, hold.id, style, hold.worded_changes)
        outcome = speech_outcome(answer, cancelling=False)
        assert outcome is not None  # noqa: S101 - a sent sentence always ends with an outcome
        self._mind.take(
            (
                SaidAloud(
                    self._app.mcp_url,
                    self._place,
                    first,
                    outcome,
                    self._clock.wall(),
                    self._mind.generation_at(self._at),
                ),
            )
        )
        if isinstance(answer, Spoken | Written):
            if isinstance(answer, Spoken):
                self._mind.speech_started(self._at, first)
            self._decided(hold, ended_as)
            if ended_as is HoldOutcome.ANSWERED:
                _, dropped = self._mind.speech_stopped(self._at)
                self._report_dropped(DropReason.REPLACED, dropped)
            self._release(rest)
            return ended_as
        if isinstance(answer, CallFailure):
            self._report(failure_event(self._app), failure_code(answer), self._app, answer)
        else:
            self._report(Event.SPEAK_REFUSED, answer.value)
            if isinstance(not_decided := hold_ended_as(ended_as, answer), NotDecided):
                return not_decided
        return await self._ended_undecided(hold) or HoldOutcome.UNDECIDED

    def _release(self, sentences: list[str] | tuple[str, ...]) -> None:
        """More sentences of the reply that answered join the record, to be sent plain with the turn's bound."""
        for _ in sentences:
            self._tally.sentence_released()
        self._mind.sentences_released(self._at, sentences, self._bound.ends_at(), self._bound.allowed, replacing=False)

    async def _speak(
        self, text: str, hold_id: str, style: str | None, told_changes: int | None
    ) -> SpeakAnswer | SpeakRefusal | CallFailure:
        try:
            return await self._sessions.ask_elsewhere(
                self._app,
                lambda session: session.speak(self._place, text, style, interrupted=hold_id, told_changes=told_changes),
                self._record,
            )
        except Exception as error:
            raise RelayFailed(error) from None


class Speaking:
    """One spoken turn's speech: the sentences its reply releases, put into the mind's record at the place with the
    turn's bound, and a round that opened with the silent marker counted. The turn's tally counts each sentence
    released. The first sentence a turn told no hold releases replaces the unsent rest of the turns before it, which
    is logged as dropped; a turn told a hold releases nothing while he has not decided it, its text is his answer. His
    own stop (``cut``) ends the releasing; the sentences the stop drops are the record's. Sending them is the place's
    ``Sender``'s, also after the turn.
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
        told: TurnHold | None,
        replacing: bool,
    ) -> None:
        self._at = AppPlace(app, place)
        self._told = told
        self._bound = bound
        self._mind = mind
        self._tally = tally
        self._report_dropped = report_dropped
        self._replacing = replacing
        self._reader = ReplyReader()
        self.was_cut = False

    def hear(self, event: TurnEvent) -> tuple[str, ...]:
        """Releases the sentences the event completed, and counts a round that opened with the silent marker. The
        sentences that answer instead the hold the turn was told, while he has not decided it, are returned, not
        released; once that answer failed the rest of the reply is dropped."""
        reading = self._reader.read(event)
        if isinstance(reading.opening, StayedSilent):
            self._tally.silent_count += 1
        if self.was_cut or not reading.sentences:
            return ()
        if self._told is not None and self._told.silenced:
            return ()
        if self._told is not None and self._told.pending:
            return reading.sentences
        for _ in reading.sentences:
            self._tally.sentence_released()
        dropped = self._mind.sentences_released(
            self._at, reading.sentences, self._bound.ends_at(), self._bound.allowed, replacing=self._replacing
        )
        self._replacing = False
        self._report_dropped(DropReason.REPLACED, dropped)
        return ()

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
        session: MemberSession,
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
        by the sentence's bound is given up, which a hold going on by itself renews while it waits."""
        sending = asyncio.create_task(request)
        giving_up = asyncio.create_task(self._clock.sleep(max(0.0, sentence.bound - self._clock.now())))
        try:
            while True:
                await asyncio.wait({sending, giving_up}, return_when=asyncio.FIRST_COMPLETED)
                if sending.done():
                    return sending.result()
                bound = self._mind.sentence_bound(self._at, sentence)
                if bound_reached(self._clock.now(), bound):
                    raise TurnStopped(TurnStopKind.FAILED, FailureCode.LIMIT_EXCEEDED)
                giving_up = asyncio.create_task(self._clock.sleep(bound - self._clock.now()))
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
                self._mind.take(
                    (
                        SaidAloud(
                            self._app,
                            self._place,
                            sentence.text,
                            outcome,
                            self._clock.wall(),
                            self._mind.generation_at(self._at),
                        ),
                    )
                )
        if isinstance(answer, SpeakRefusal) and answered in _REFUSED and not request.cut_by_stop(answer):
            self._report(Event.SPEAK_REFUSED, answer.value)
