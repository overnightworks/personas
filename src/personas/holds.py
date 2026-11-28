"""What a spoken turn hears of a hold of his while it speaks: the round, its two tools, the told decision, the end."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.backend import TurnPerson, TurnStopped
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, hold_place
from agents.member_outcomes import CallFailure, Delivery
from agents.tools import HandTool, ToolDeclaration, ToolEffect, ToolOutcome
from agents.turn import Turn

from personas.decision import TURN_ROLE, AppPlace, call_within_bound, failure_code, round_seconds
from personas.entries import TurnTold, told_hold, told_stop, told_went_on, told_words
from personas.present import turn_messages, user_message
from personas.reporting import Event, failure_event
from personas.speech import (
    DropReason,
    Held,
    HoldKey,
    HoldOutcome,
    NotDecided,
    hold_ended_as,
    hold_stated,
    hold_tellable,
    untold_words,
)
from personas.task_record import RelayFailed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from agents.agent_file import TurnRequest
    from agents.mcp_client import MemberSession
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

DECISION_CALLS: Final = 8
"""The decisions one hold is told, in a round or in a turn: his first and seven more after the app took none, one it
refused as stale counting as any other. Past them the hold ends undecided and stops him, as any hold without a
decision does; the turn's own bound stays the turn's."""
ROUND_CALLS: Final = DECISION_CALLS + 1
"""The calls a round's run may make whatever the turn it began beside is bound to: the decisions and the call past
them, which the hold answers that it takes no more."""
_APART_DECISION_CALLS: Final = 1
DECISIONS_BOUND_TEXT: Final = "this hold takes no more decisions"
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


def decision_calls(deciding: TurnHold | None) -> int:
    """The calls of the decision tools one turn counts: ``DECISION_CALLS`` for a turn told a hold, one for any other,
    whose decision tools are answered apart with an error."""
    return _APART_DECISION_CALLS if deciding is None else DECISION_CALLS


class TurnHold:
    """The hold a turn or a round was told, as it stands in what it was told last: after the app took no decision of
    his for want of the latest words he is told what was said since and decides again. Until he decides the hold a turn
    was told, its every other call is answered that he decides first, and the sentences wait unsent in the mind's
    record, where the hold stands."""

    def __init__(self, held: Held) -> None:
        self.held = held
        self.outcome: HoldOutcome | NotDecided | None = None
        self._decisions = 0

    def take_decision(self) -> bool:
        """Counts a decision of his that reaches the app; ``False`` once the hold took ``DECISION_CALLS`` of them."""
        if not call_within_bound(self._decisions, DECISION_CALLS):
            return False
        self._decisions += 1
        return True

    @property
    def pending(self) -> bool:
        return self.outcome is None

    @property
    def silenced(self) -> bool:
        """His answer instead of this hold failed: the rest of its reply is dropped."""
        return self.outcome is HoldOutcome.UNDECIDED

    def dropped(self) -> bool:
        """The turn's speech was cut before he decided this hold: it counts as undecided, once. Whether it was still
        pending."""
        if not self.pending:
            return False
        self.outcome = HoldOutcome.UNDECIDED
        return True


@contextmanager
def dropping(mind: Mind, at: AppPlace, inbox: HoldInbox | None) -> Generator[None]:
    """His decision to drop what waited is on its way to the app: nothing is sent and no call of his parked run goes out
    meanwhile, and what waited stays unless the app takes the decision and the caller drops it."""
    mind.speech_dropping(at)
    try:
        with nullcontext() if inbox is None else inbox.stopping():
            yield
    finally:
        mind.speech_dropping_ended(at)


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
        self._stopping = False
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

    def answer_refused(self) -> None:
        """The app took no answer of his, since the hold no longer stood: a new hold gets its round."""
        self._answering = False
        self._wake()

    @contextmanager
    def stopping(self) -> Generator[None]:
        """His decision to drop what waited is on its way: the hold stands for the calls of the turn until the app
        answers, so none goes out before the stop is known."""
        self._stopping = True
        try:
            yield
        finally:
            self._stopping = False
            self._wake()

    def decided(self, key: HoldKey) -> None:
        """The hold ``key`` was decided: it stands no more, whatever a window read before the decision says."""
        self._decided = key
        if self.hold is not None and self.hold.key == key:
            self.hold = None
        self._wake()

    async def until_none_stands(self, bound: TurnBound) -> None:
        """Returns once no hold stands and none of his stops is on its way; the turn's deadline ends the turn first when
        it passes before."""
        while (self.hold is not None or self._stopping) and not self._closed:
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

    def _inbox_at(self, place: PlaceRef) -> HoldInbox | None:
        return self._listening[1] if self._listening is not None and self._listening[0] == place else None

    def hold_decided(self, place: PlaceRef, held: Held, outcome: HoldOutcome) -> None:
        self._mind.hold_decided(self._at(place), held.key)
        if (inbox := self._inbox_at(place)) is not None:
            inbox.decided(held.key)
        self._report(Event.HOLD_DECIDED, outcome.value, None, None, hold_id=held.id)

    async def decide_holds(self, voiced: Voiced, hearing: _Hearing) -> None:
        """Starts a round for each hold that may be told, until the speech is over; a hold no round took by then stands
        in the mind's record for the turn that decides it."""
        while (hold := await voiced.inbox.next_hold()) is not None:
            await self._decide_hold(voiced, hearing, hold)

    async def _decide_hold(self, voiced: Voiced, hearing: _Hearing, hold: Held) -> None:
        """One round on a backend of its own beside the run parked at the hold: his history as it stands and the
        telling of the hold, on his persona and tools, bound to its own calls. Without a decision he stops and says
        nothing."""
        admitted, speaking, inbox = voiced.admitted, voiced.speaking, voiced.inbox
        inbox.told = hold.speech
        told = TurnHold(hold)
        turn = voiced.turn
        current = TurnTold(
            turn.at, turn.app, turn.place, turn.occasion, aloud=turn.aloud, through=self._mind.history.taken
        )
        standing = self.told_standing(
            admitted.place, hold, None if inbox.waiting is None else (inbox.waiting.name, inbox.waiting.input)
        )
        messages = [*turn_messages(admitted.facts.name, self._mind.present, current, voiced.held.deferred)]
        if self._mind.went_on_to_tell(self._at(admitted.place)):
            messages.append(user_message(told_went_on()))
        messages.append(user_message(standing))
        request = voiced.request.model_copy(
            update={
                "turn": Turn(
                    system=voiced.request.turn.system,
                    messages=messages,
                ),
                "model": self._agent.hold_model or voiced.request.model,
                "max_duration_seconds": round_seconds(speaking.seconds_left()),
                "max_tool_calls": ROUND_CALLS,
            }
        )
        tools = tuple(
            HandTool(each.declaration, partial(self._answer_round, voiced, hearing, told, each.declaration.name))
            for each in voiced.held.tools
        )
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        answered_instead = False
        try:
            async with voiced.setup.backend(tools) as backend, aclosing(backend.run(request, person)) as events:
                cut_parked_run = partial(self._cut_parked_run, voiced, hearing)
                answered_instead = await voiced.answering.hear_round(events, told, cut_parked_run)
        except RelayFailed:
            raise
        except TurnStopped:
            pass
        if not answered_instead and self._mind.hold_decided_key(self._at(admitted.place)) != hold.key:
            ended = await self.end_hold(admitted.place, voiced.record, told.held, HoldOutcome.UNDECIDED, voiced.bound)
            if not isinstance(ended, NotDecided):
                await self._cut_parked_run(voiced, hearing)
            if isinstance(ended, HoldOutcome):
                self._report_told_hold_dropped(voiced)

    async def _answer_round(
        self, voiced: Voiced, hearing: _Hearing, told: TurnHold, name: str, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        """A call of the round: only his two decisions reach the app, for the hold he was told."""
        outcomes = {CONTINUE_SPEAKING_TOOL: HoldOutcome.CONTINUED, STOP_SPEAKING_TOOL: HoldOutcome.STOPPED}
        outcome = outcomes.get(name)
        if outcome is None:
            return ToolOutcome(DECIDING_FIRST_TEXT, is_error=True)
        at = self._at(voiced.admitted.place)
        if self._mind.hold_decided_key(at) == told.held.key:
            return ToolOutcome(DECIDED_TEXT, is_error=True)
        if outcome is HoldOutcome.CONTINUED and carries_line(arguments):
            return await voiced.answering.line_decided(told, arguments)
        if not told.take_decision():
            return ToolOutcome(DECISIONS_BOUND_TEXT, is_error=True)
        unsaid = self._mind.speech_unsaid(at)
        ended = await self.end_hold(voiced.admitted.place, voiced.record, told.held, outcome, voiced.bound)
        if ended is None:
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        if isinstance(ended, NotDecided):
            return await self._not_decided(voiced.admitted.place, voiced.record, told, ended)
        if outcome is HoldOutcome.CONTINUED:
            return ToolOutcome(CONTINUED_TEXT, is_error=False)
        waiting = await self._cut_parked_run(voiced, hearing)
        self._report_told_hold_dropped(voiced)
        return ToolOutcome(told_stop(unsaid, waiting), is_error=False)

    async def _cut_parked_run(self, voiced: Voiced, hearing: _Hearing) -> str | None:
        """The app took his stop: the turn releases no sentence from now on and the parked reply is dropped with its
        waiting call unsent; a call already sent runs to its answer, which enters his history, before the reply is
        dropped. The tool of the call that was never sent."""
        waiting = voiced.inbox.waiting
        voiced.speaking.cut()
        await voiced.inbox.calls_answered()
        hearing.cancel()
        return None if waiting is None else waiting.name

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

    async def _asked[Answer](
        self, record: TaskRecord, ask: Callable[[MemberSession], Awaitable[Answer]]
    ) -> Answer | CallFailure:
        """The app's answer in a session of its own, or the failure of the call, which is reported."""
        try:
            answer = await self._sessions.ask_elsewhere(self._app, ask, record)
        except Exception as error:
            raise RelayFailed(error) from None
        if isinstance(answer, CallFailure):
            self._report(failure_event(self._app), failure_code(answer), self._app, answer)
        return answer

    async def end_hold(
        self, place: PlaceRef, record: TaskRecord, hold: Held, outcome: HoldOutcome, bound: TurnBound
    ) -> HoldOutcome | NotDecided | None:
        """Sends his decision in a session of its own; how the hold ended, ``NotDecided`` when the app took no decision
        of his, or ``None`` when the app did not answer.

        A continue gives the unsent sentences the deciding turn's ``bound``, since a read of the window can end the hold
        in the record before the app's answer arrives, and names the changes of the talk-over he was told. A stop of
        his names them too; one of the runner's own (``UNDECIDED``) names none, so it acts at once. A stop drops the
        unsent sentences once the app took it; until then the sender sends nothing and what waited stays.
        """
        at = self._at(place)
        stops = outcome is not HoldOutcome.CONTINUED
        told_changes = None if outcome is HoldOutcome.UNDECIDED else hold.worded_changes
        if not stops:
            self._mind.speech_continued(at, bound.ends_at())
        with dropping(self._mind, at, self._inbox_at(place)) if stops else nullcontext():
            answer = await self._asked(
                record,
                lambda session: (
                    session.stop_speaking(place, hold.id, told_changes=told_changes)
                    if stops
                    else session.continue_speaking(place, hold.id, told_changes=told_changes)
                ),
            )
            if isinstance(answer, CallFailure):
                return None
            assert answer is not Delivery.NOT_SERVED  # noqa: S101 - a continue without ``finish`` is always sent
            ended = hold_ended_as(outcome, answer)
            if isinstance(ended, HoldOutcome):
                if stops:
                    self.stop_speech(place)
                self.hold_decided(place, hold, ended)
            return ended

    def told_standing(self, place: PlaceRef, held: Held, waiting: tuple[str, JsonObject] | None = None) -> str:
        """The telling of the hold: what was said over him, the place the app states for it or, when it states none, his
        playing sentence as unknown, his unsaid sentences and the call of his that waits. The words told are noted, so
        that a line of theirs that repeats them says nothing new."""
        at = self._at(place)
        self._mind.talk_told(at, held.speech)
        stopped_at = held.place or hold_place(self._mind.speech_saying(at), None)
        return told_hold(held.id, held.speech, stopped_at, self._mind.speech_unsaid(at), waiting)

    async def retold(self, place: PlaceRef, record: TaskRecord, told: TurnHold) -> ToolOutcome:
        """The app took no decision of his because people said more over him than he was told: the result tells what
        was said since, in the shape of a call held back, and the hold stands told that. He decides again."""
        window = await self._asked(record, lambda session: session.read_place(place))
        if isinstance(window, CallFailure) or window is None:
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        stated = hold_stated(window, told.held)
        if stated is None or stated.key != told.held.key:
            told.outcome = NotDecided.GONE
            return ToolOutcome(GONE_TEXT, is_error=False)
        said = untold_words(stated.speech, told.held.speech)
        told.held = stated
        self._mind.talk_told(self._at(place), stated.speech)
        if (inbox := self._inbox_at(place)) is not None:
            inbox.told = stated.speech
        return ToolOutcome(told_words(said), is_error=False)

    async def _not_decided(self, place: PlaceRef, record: TaskRecord, told: TurnHold, why: NotDecided) -> ToolOutcome:
        """The result of a decision the app took none of: what was said since when it was told too few changes, that the
        hold is gone otherwise, which leaves what waited and the parked run as they were."""
        if why is NotDecided.STALE:
            return await self.retold(place, record, told)
        told.outcome = NotDecided.GONE
        return ToolOutcome(GONE_TEXT, is_error=False)

    async def stop_undecided(self, session: MemberSession, place: PlaceRef, held: Held) -> None:
        """The runner's own stop of a hold whose turn ended without deciding it, in the loop's session: it names no
        changes, so it acts at once, and drops the unsent sentences once the app took it."""
        with dropping(self._mind, self._at(place), None):
            ended = hold_ended_as(HoldOutcome.UNDECIDED, await session.stop_speaking(place, held.id))
            if isinstance(ended, HoldOutcome):
                self.stop_speech(place)
                self.hold_decided(place, held, ended)

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
            return await answering.line_decided(deciding, arguments)
        if not deciding.take_decision():
            return ToolOutcome(DECISIONS_BOUND_TEXT, is_error=True)
        deciding.outcome = outcome
        unsaid = self._mind.speech_unsaid(self._at(place))
        ended = await self.end_hold(place, record, deciding.held, outcome, bound)
        if ended is None:
            deciding.outcome = None
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        if isinstance(ended, NotDecided):
            deciding.outcome = None
            return await self._not_decided(place, record, deciding, ended)
        told = CONTINUED_TEXT if outcome is HoldOutcome.CONTINUED else told_stop(unsaid, None)
        return ToolOutcome(told, is_error=False)
