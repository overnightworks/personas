"""What a spoken turn hears of a hold of his while it speaks: the round, its two tools, the told decision, the end."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.backend import TurnPerson, TurnStopped
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, hold_place
from agents.member_outcomes import CallFailure, Delivery
from agents.tools import HandTool, ToolDeclaration, ToolEffect, ToolOutcome
from agents.turn import Turn

from personas.decision import TURN_ROLE, AppPlace, failure_code, round_seconds
from personas.entries import TurnTold, told_hold, told_stop, told_went_on
from personas.present import turn_messages, user_message
from personas.reporting import Event, failure_event
from personas.speech import (
    DropReason,
    Held,
    HoldKey,
    HoldOutcome,
    hold_ended_as,
    hold_stated,
    hold_tellable,
    untold_words,
)
from personas.task_record import RelayFailed

if TYPE_CHECKING:
    from collections.abc import Generator

    from agents.agent_file import TurnRequest
    from agents.member import PartialSpeech, PlaceRef, PlaceWindow
    from agents.turn import JsonObject, Principal
    from agents.turn_events import FinalEvent, ToolCallEvent

    from personas.admission import Admitted
    from personas.config import AgentCredentials, AppTarget
    from personas.held_tools import HeldTools
    from personas.mind import Mind
    from personas.reporting import LoopReport
    from personas.task_record import TaskRecord
    from personas.turn_bound import TurnBound
    from personas.turn_sessions import TurnSessions
    from personas.turning import TurnSetup
    from personas.voice import Answering, DropReporter, Speaking

DECISION_CALLS: Final = 1
LINE_ARGUMENT: Final = "line"
STYLE_ARGUMENT: Final = "style"
_NO_SPEECH_FOR_LINE_TEXT: Final = "this turn does not speak: continue_speaking without a line"
APP_FAILED_TEXT: Final = "the app did not answer"
CONTINUED_TEXT: Final = '{"continued": true}'
GONE_TEXT: Final = '{"gone": true}'
DECIDING_FIRST_TEXT: Final = "a hold is decided first: call continue_speaking or stop_speaking"
DECIDED_TEXT: Final = "this hold is decided"
_ANSWER_INSTEAD: Final = (
    " Or reply with text instead, which answers the hold: the text is what you say now, and everything you had not "
    "yet said is dropped."
)
_HOLD_ID: Final = {"id": {"type": "string", "description": "The id of the hold you were told."}}
_SHORT_LINE: Final = {
    LINE_ARGUMENT: {
        "type": "string",
        "minLength": 1,
        "description": (
            "One short line to say at once. The sentence you were held in is cut, not said again, and your next "
            "sentences follow the line."
        ),
    },
    STYLE_ARGUMENT: {
        "type": "string",
        "minLength": 1,
        "description": (
            "How the line sounds, in a few words, such as `louder and angrier`. Only with `line`; your next "
            "sentences follow without it."
        ),
    },
}
DECISION_DECLARATIONS: Final = tuple(
    ToolDeclaration(
        name=name,
        description=description + _ANSWER_INSTEAD,
        parameters={
            "type": "object",
            "properties": {**_HOLD_ID, **properties},
            "required": ["id"],
            "additionalProperties": False,
        },
        effect=ToolEffect.SIDE_EFFECTING,
    )
    for name, description, properties in (
        (
            CONTINUE_SPEAKING_TOOL,
            "Lets the sentence you were held in go on to its end, and your next sentences follow. With `line`, that "
            "one short line is said at once instead, the sentence you were held in is cut, and your next sentences "
            "follow the line.",
            _SHORT_LINE,
        ),
        (
            STOP_SPEAKING_TOOL,
            "Drops the rest of the sentence you were held in and everything you had not yet said.",
            {},
        ),
    )
)


class TurnHold:
    """The hold a turn was told at its start and has not decided yet: until he decides, its every other call is
    answered that he decides first, and the sentences wait unsent in the mind's record, where the hold stands."""

    def __init__(self, held: Held) -> None:
        self.held = held
        self.outcome: HoldOutcome | None = None

    @property
    def pending(self) -> bool:
        return self.outcome is None

    @property
    def silenced(self) -> bool:
        """His answer instead of this hold failed: the rest of its reply is dropped."""
        return self.outcome in {HoldOutcome.UNDECIDED, HoldOutcome.GONE}

    def dropped(self) -> bool:
        """The turn's speech was cut before he decided this hold: it counts as undecided, once. Whether it was still
        pending."""
        if not self.pending:
            return False
        self.outcome = HoldOutcome.UNDECIDED
        return True


def carries_line(arguments: JsonObject) -> bool:
    """Whether a ``continue_speaking`` names a short line or a style, which `Answering.line_decided` then judges."""
    return LINE_ARGUMENT in arguments or STYLE_ARGUMENT in arguments


def deciding_first(told: TurnHold | None) -> bool:
    """Whether the turn was told a hold that he has not decided."""
    return told is not None and told.pending


async def close_when_over(inbox: HoldInbox, *speech: asyncio.Task[object]) -> None:
    await asyncio.wait(speech)
    inbox.close()


class HoldInbox:
    """What the window says of him while a spoken turn runs: the hold that stands, the call that waits for it or is on
    its way to its app, and what people say over him, with what a round told him of it."""

    def __init__(self, standing: Held | None = None, *, told: bool = False) -> None:
        self.hold = standing
        self.waiting: ToolCallEvent | None = None
        self.told: tuple[PartialSpeech, ...] | None = standing.speech if told and standing is not None else None
        self._said: tuple[PartialSpeech, ...] = () if standing is None else standing.speech
        self._taken: HoldKey | None = standing.key if told and standing is not None else None
        self._decided: HoldKey | None = None
        self._answering = False
        self._calls_sent = 0
        self._calls_answered = asyncio.Event()
        self._calls_answered.set()
        self._closed = False
        self._changed = asyncio.Event()
        self._freed = asyncio.Event()
        """Beside ``_changed`` because each of the two waits clears the event it waits on, which would swallow the
        other's wake-up."""

    def _wake(self) -> None:
        self._changed.set()
        self._freed.set()

    def hear(self, window: PlaceWindow) -> None:
        stated = hold_stated(window, self.hold)
        self.hold = None if stated is not None and stated.key == self._decided else stated
        self._said = () if window.talked_over is None else window.talked_over.speech
        self._wake()

    def answering(self) -> None:
        """His answer instead of a hold is on its way: no round decides a hold from now on; one that stands is
        left for his next turn."""
        self._answering = True
        self._wake()

    def decided(self, key: HoldKey) -> None:
        """The hold ``key`` was decided: it stands no more, whatever a window read before the decision says."""
        self._decided = key
        if self.hold is not None and self.hold.key == key:
            self.hold = None
        self._wake()

    async def until_none_stands(self, bound: TurnBound) -> None:
        """Returns once no hold stands; the turn's deadline ends the turn first when it passes before."""
        while self.hold is not None and not self._closed:
            self._freed.clear()
            await bound.first_of(self._freed, bound.deadline())
            bound.stop_if_deadline_done()

    @contextmanager
    def sending(self) -> Generator[None]:
        """A call of the turn is on its way to its app, until its answer has entered his history."""
        self._calls_sent += 1
        self._calls_answered.clear()
        try:
            yield
        finally:
            self._calls_sent -= 1
            if not self._calls_sent:
                self._calls_answered.set()

    async def calls_answered(self) -> None:
        """Returns once no call of the turn is on its way to its app."""
        await self._calls_answered.wait()

    def untold(self) -> tuple[PartialSpeech, ...]:
        """What people said over him since a round told him of it, once; nothing before a round has."""
        if self.told is None:
            return ()
        words = untold_words(self._said, self.told)
        if words:
            self.told = self._said
        return words

    def close(self) -> None:
        self._closed = True
        self._wake()

    @property
    def undecided(self) -> Held | None:
        return self.hold if self.hold is not None and self.hold.key != self._taken else None

    async def next_hold(self) -> Held | None:
        while not self._closed and not self._answering:
            hold = self.undecided
            if hold is not None and hold_tellable(hold):
                self._taken = hold.key
                return hold
            self._changed.clear()
            await self._changed.wait()
        return None


type _Hearing = asyncio.Task[FinalEvent | None]


@dataclass
class Voiced:
    """One spoken turn's speech, with what a round at a hold of his starts from."""

    setup: TurnSetup
    admitted: Admitted
    turn: TurnTold
    request: TurnRequest
    held: HeldTools
    speaking: Speaking
    answering: Answering
    record: TaskRecord
    inbox: HoldInbox
    bound: TurnBound


class HoldDeciding:
    """What decides a hold of his for one agent in one app: the round beside the run parked at it, the decision of the
    turn that was told it, and the end of a hold in the app. It holds the spoken turn that listens at a place, which
    hears what a read of the place's window says of him."""

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        sessions: TurnSessions,
        mind: Mind,
        report: LoopReport,
        report_dropped: DropReporter,
    ) -> None:
        self._agent = agent
        self._app = app
        self._sessions = sessions
        self._mind = mind
        self._report = report
        self._report_dropped = report_dropped
        self._listening: tuple[PlaceRef, HoldInbox] | None = None

    def _at(self, place: PlaceRef) -> AppPlace:
        return AppPlace(self._app.mcp_url, place)

    @contextmanager
    def listening(self, place: PlaceRef, inbox: HoldInbox) -> Generator[None]:
        """The spoken turn at ``place`` hears what its window says of him, until the block ends."""
        self._listening = (place, inbox)
        try:
            yield
        finally:
            self._listening = None

    def heard_at(self, place: PlaceRef, window: PlaceWindow) -> bool:
        """Whether a spoken turn listens at ``place``, which then hears the window."""
        if self._listening is None or self._listening[0] != place:
            return False
        self._listening[1].hear(window)
        return True

    def hold_decided(self, place: PlaceRef, held: Held, outcome: HoldOutcome) -> None:
        self._mind.hold_decided(self._at(place), held.key)
        if self._listening is not None and self._listening[0] == place:
            self._listening[1].decided(held.key)
        self._report(Event.HOLD_DECIDED, outcome.value, None, None, hold_id=held.id)

    async def decide_holds(self, voiced: Voiced, hearing: _Hearing) -> None:
        """Starts a round for each hold that may be told, until the speech is over; a hold no round took by then stands
        in the mind's record for the turn that decides it."""
        while (hold := await voiced.inbox.next_hold()) is not None:
            await self._decide_hold(voiced, hearing, hold)

    async def _decide_hold(self, voiced: Voiced, hearing: _Hearing, hold: Held) -> None:
        """One round on a backend of its own beside the run parked at the hold: his history as it stands and the
        telling of the hold, on his persona and tools. Without a decision he stops and says nothing."""
        admitted, speaking, inbox = voiced.admitted, voiced.speaking, voiced.inbox
        inbox.told = hold.speech
        turn = voiced.turn
        current = TurnTold(
            turn.at, turn.app, turn.place, turn.occasion, aloud=turn.aloud, through=self._mind.history.taken
        )
        standing = self.told_standing(
            admitted.place, hold, None if inbox.waiting is None else (inbox.waiting.name, inbox.waiting.input)
        )
        told = [*turn_messages(admitted.facts.name, self._mind.present, current, voiced.held.deferred)]
        if self._mind.went_on_to_tell(self._at(admitted.place)):
            told.append(user_message(told_went_on()))
        told.append(user_message(standing))
        request = voiced.request.model_copy(
            update={
                "turn": Turn(
                    system=voiced.request.turn.system,
                    messages=told,
                ),
                "model": self._agent.hold_model or voiced.request.model,
                "max_duration_seconds": round_seconds(speaking.seconds_left()),
            }
        )
        tools = tuple(
            HandTool(each.declaration, partial(self._answer_round, voiced, hearing, hold, each.declaration.name))
            for each in voiced.held.tools
        )
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        answered_instead = False
        try:
            async with voiced.setup.backend(tools) as backend, aclosing(backend.run(request, person)) as events:
                cut_parked_run = partial(self._cut_speech, voiced, hearing, DropReason.REPLACED)
                answered_instead = await voiced.answering.hear_round(events, hold, cut_parked_run)
        except RelayFailed:
            raise
        except TurnStopped:
            pass
        if not answered_instead and self._mind.hold_decided_key(self._at(admitted.place)) != hold.key:
            await self._cut_speech(voiced, hearing)
            ended = await self.end_hold(admitted.place, voiced.record, hold, HoldOutcome.UNDECIDED, voiced.bound)
            if ended is not None:
                self._report_told_hold_dropped(voiced)

    async def _answer_round(
        self, voiced: Voiced, hearing: _Hearing, hold: Held, name: str, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        """A call of the round: only his two decisions reach the app, for the hold he was told."""
        outcomes = {CONTINUE_SPEAKING_TOOL: HoldOutcome.CONTINUED, STOP_SPEAKING_TOOL: HoldOutcome.STOPPED}
        outcome = outcomes.get(name)
        if outcome is None:
            return ToolOutcome(DECIDING_FIRST_TEXT, is_error=True)
        if self._mind.hold_decided_key(self._at(voiced.admitted.place)) == hold.key:
            return ToolOutcome(DECIDED_TEXT, is_error=True)
        if outcome is HoldOutcome.CONTINUED and carries_line(arguments):
            return await voiced.answering.line_decided(hold, arguments)
        told = CONTINUED_TEXT if outcome is HoldOutcome.CONTINUED else await self._cut_speech(voiced, hearing)
        ended = await self.end_hold(voiced.admitted.place, voiced.record, hold, outcome, voiced.bound)
        if ended is None:
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        if outcome is HoldOutcome.STOPPED:
            self._report_told_hold_dropped(voiced)
        return ToolOutcome(GONE_TEXT if ended is HoldOutcome.GONE else told, is_error=False)

    async def _cut_speech(self, voiced: Voiced, hearing: _Hearing, reason: DropReason = DropReason.STOPPED) -> str:
        """His stop at once: no unsent sentence is sent and the parked reply is dropped with its waiting call, unsent; a
        call already sent runs to its answer, which enters his history, before the reply is dropped. The result tells
        what was left unsaid."""
        waiting = voiced.inbox.waiting
        voiced.speaking.cut()
        told = told_stop(self.stop_speech(voiced.admitted.place, reason), None if waiting is None else waiting.name)
        await voiced.inbox.calls_answered()
        hearing.cancel()
        return told

    def stop_speech(self, place: PlaceRef, reason: DropReason = DropReason.STOPPED) -> tuple[str, ...]:
        """The unsent sentences at the place drop; the sentences not said yet, which a request in flight may still
        answer."""
        unsaid, dropped = self._mind.speech_stopped(self._at(place))
        self._report_dropped(reason, dropped)
        return unsaid

    def _report_told_hold_dropped(self, voiced: Voiced) -> None:
        """A round for a newer hold cut the speech before the turn decided the hold it was told, which the app may have
        ended already: the older id gets its line."""
        told = voiced.held.told
        if told is not None and told.dropped():
            self._report(Event.HOLD_DECIDED, HoldOutcome.UNDECIDED.value, None, None, hold_id=told.held.id)

    async def end_hold(
        self, place: PlaceRef, record: TaskRecord, hold: Held, outcome: HoldOutcome, bound: TurnBound
    ) -> HoldOutcome | None:
        """Sends his decision in a session of its own; how the hold ended, or ``None`` when the app did not answer.

        Before it goes out a stop drops the unsent sentences, so that none follows it, and a continue gives them the
        deciding turn's ``bound``, since a read of the window can end the hold in the record before the app's answer
        arrives.
        """
        if outcome is HoldOutcome.CONTINUED:
            self._mind.speech_continued(self._at(place), bound.ends_at())
        else:
            self.stop_speech(place)
        try:
            answer = await self._sessions.ask_elsewhere(
                self._app,
                lambda session: (
                    session.continue_speaking(place, hold.id)
                    if outcome is HoldOutcome.CONTINUED
                    else session.stop_speaking(place, hold.id)
                ),
                record,
            )
        except Exception as error:
            raise RelayFailed(error) from None
        if isinstance(answer, CallFailure):
            self._report(failure_event(self._app), failure_code(answer), self._app, answer)
            return None
        assert answer is not Delivery.NOT_SERVED  # noqa: S101 - a continue without ``finish`` is always sent
        ended = hold_ended_as(outcome, answer)
        self.hold_decided(place, hold, ended)
        return ended

    def told_standing(self, place: PlaceRef, held: Held, waiting: tuple[str, JsonObject] | None = None) -> str:
        """The telling of the hold: what was said over him, the place the app states for it or, when it states none, his
        playing sentence as unknown, his unsaid sentences and the call of his that waits."""
        at = self._at(place)
        stopped_at = held.place or hold_place(self._mind.speech_saying(at), None)
        return told_hold(held.id, held.speech, stopped_at, self._mind.speech_unsaid(at), waiting)

    async def decide_told(
        self,
        place: PlaceRef,
        record: TaskRecord,
        deciding: TurnHold,
        answering: Answering | None,
        bound: TurnBound,
        name: str,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        """His decision of the hold the turn was told, sent to the app with the standing id; a second one is refused."""
        outcome = HoldOutcome.CONTINUED if name == CONTINUE_SPEAKING_TOOL else HoldOutcome.STOPPED
        if not deciding.pending:
            return ToolOutcome(DECIDED_TEXT, is_error=True)
        if outcome is HoldOutcome.CONTINUED and carries_line(arguments):
            if answering is None:
                return ToolOutcome(_NO_SPEECH_FOR_LINE_TEXT, is_error=True)
            return await answering.line_decided(deciding.held, arguments)
        deciding.outcome = outcome
        unsaid = self._mind.speech_unsaid(self._at(place))
        ended = await self.end_hold(place, record, deciding.held, outcome, bound)
        if ended is None:
            deciding.outcome = None
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        if ended is HoldOutcome.GONE:
            return ToolOutcome(GONE_TEXT, is_error=False)
        told = CONTINUED_TEXT if outcome is HoldOutcome.CONTINUED else told_stop(unsaid, None)
        return ToolOutcome(told, is_error=False)
