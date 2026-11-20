"""What one turn of an agent in an app does: the tools held, the sessions its requests run on and how the turn ends."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.agent_file import TurnRequest
from agents.backend import TurnPerson, TurnStopped
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL
from agents.tools import HandTool, ToolDeclaration, ToolOutcome
from agents.turn import RejectionCode, Turn, TurnMessage, TurnRole
from agents.turn_events import FinalEvent, ToolCallEvent
from agents.voice.reply import REPLY_RULES

from personas.decision import (
    TURN_ROLE,
    AppPlace,
    act_outcome,
    failure_code,
    offered_declarations,
    round_seconds,
    turn_call_bound,
)
from personas.entries import (
    ActDone,
    TurnTold,
    told_hold,
    told_stop,
    told_words,
)
from personas.failures import (
    CallFailure,
    leaves_of,
    raise_if_cancelled,
    task_is_being_cancelled,
)
from personas.held_tools import Calls, HeldTools, HoldBack
from personas.holds import (
    APP_FAILED_TEXT,
    CONTINUED_TEXT,
    DECISION_CALLS,
    DECISION_DECLARATIONS,
    GONE_TEXT,
    LINE_ARGUMENT,
    HoldInbox,
    TurnHold,
    close_when_over,
)
from personas.present import held_told, turn_messages
from personas.recall import (
    LOOKUPS_SPENT_TEXT,
    RECALL_DECLARATION,
    STORE_DID_NOT_ANSWER_TEXT,
    line_search,
    lookup_result,
    search_text,
    text_refused_text,
)
from personas.reporting import Event, LoopReport, TurnTally, failure_event, step_timer, stepping
from personas.sizing import Sizing
from personas.speech import DropReason, Held, HoldOutcome, Telling, hold_ended_as
from personas.task_record import (
    RelayFailed,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    failure_held_by,
    failures_leaving_with,
)
from personas.tool_lists import Declared, ToolLists
from personas.turn_bound import TurnBound
from personas.turn_sessions import TurnSessions
from personas.voice import Answering, DropReporter, Speaking, Voicing

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
    from contextlib import AbstractAsyncContextManager

    from agents.agent_file import AgentDefinition
    from agents.backend import AgentBackend, TurnEvent
    from agents.member import PlaceRef, PlaceWindow
    from agents.turn import FailureCode, JsonObject, Principal
    from logline import StepTimer

    from personas.admission import Admitted
    from personas.app_session import AppSession, SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget
    from personas.mind import Mind
    from personas.recall import LookupBounds
    from personas.remembering import Remembering
    from personas.task_record import TaskRecord
    from personas.tokens import TokenSource

# The backend a turn runs on, which it closes when the turn ends.
type BackendFactory = Callable[[Iterable[HandTool]], AbstractAsyncContextManager[AgentBackend]]

_CALL_BOUND_TEXT: Final = "this app takes no more calls in this turn"
_APART_TEXT: Final = (
    "a hold is decided in a round of its own when one starts: nothing is held now, so there is nothing to decide"
)
_DECIDING_FIRST_TEXT: Final = "a hold is decided first: call continue_speaking or stop_speaking"
_DECIDED_TEXT: Final = "this hold is decided"


async def _final_event_of(events: AsyncIterator[TurnEvent], tally: TurnTally) -> FinalEvent | None:
    """Reads the turn's events to their end, counting its tool calls, and returns its final event, if it had one."""
    final = None
    async for event in events:
        if isinstance(event, ToolCallEvent):
            tally.tool_call_count += 1
        elif isinstance(event, FinalEvent):
            final = event
    return final


@dataclass(frozen=True)
class TurnSetup:
    """What an agent's turn in an app is made of: the persona and the backend that runs it."""

    persona: AgentDefinition
    backend: BackendFactory


type _Hearing = asyncio.Task[FinalEvent | None]


@dataclass
class _Voiced:
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


class Turning:
    """Runs the turns of one agent in one app, each over the loop's session, holding the tools of every app in the
    configuration's order: the app's own, those of the apps in ``admitted.apps`` and those of the tools-only apps,
    after the lookup when ``[memory]`` is set.

    It holds no state across turns beyond what it records into the agent's mind: each relayed call that passed its
    app's bound, once its request ended, and each tools-only app the realm refuses the agent, which no turn holds
    until its retry is due. A call is held back, once per turn, when something not his own arrived in
    his history since the turn's admission: it is not sent and the model is told what arrived. The token, its use on
    a session and the two reports are the loop's, handed over as callables and read at each call. A lookup reads the
    memory through the loop's ``Remembering``, takes nothing into the agent's history and is never held back.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        apps: tuple[AppTarget, ...],
        tokens: TokenSource,
        open_session: SessionOpener,
        mind: Mind,
        clock: Clock,
        use_current_token: Callable[[AppSession], Awaitable[None]],
        report: LoopReport,
        report_pause: Callable[[float, str], None],
        report_tools_left_out: Callable[[AppTarget, int], None],
        report_dropped: DropReporter,
        remembering: Remembering,
    ) -> None:
        self._agent = agent
        self._app = app
        self._apps = tuple((each, each.max_tool_calls) for each in apps if each.max_tool_calls is not None)
        self._sessions = TurnSessions(agent, tokens, open_session)
        self._lists = ToolLists(
            agent, app, tuple(each for each, _ in self._apps), self._sessions, mind, clock, use_current_token, report
        )
        self._mind = mind
        self._clock = clock
        self._use_current_token = use_current_token
        self._report = report
        self._report_pause = report_pause
        self._report_tools_left_out = report_tools_left_out
        self._report_dropped = report_dropped
        self._voicing = Voicing(self._sessions, app, report, clock, mind, report_dropped)
        self._left_out_reported: set[str] = set()
        self._remembering = remembering
        self._listening: tuple[PlaceRef, HoldInbox] | None = None
        self._aloud_at: PlaceRef | None = None

    async def take(
        self,
        session: AppSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        tally: TurnTally,
        gate: Callable[[], Awaitable[None]],
    ) -> None:
        """Runs one turn once its ``gate`` is passed; whatever way it ends, a defect a request of it met ends the
        runner loud. The turn lists the tools of its apps beside the gate, since a listing is a read.

        A session whose close cuts a request short can lose the defect from what leaves, so the turn's record
        joins it. A cancelled turn ends as the cancellation unless a defect was met or nobody cancelled it.
        """
        self._aloud_at = admitted.place if admitted.aloud else None
        try:
            await ending_loud_when_cancelled(
                ending_with_recorded_defect(self._judge_turn(session, setup, admitted, record, tally, gate), record),
                record.loud_end,
            )
        finally:
            self._aloud_at = None
            if admitted.aloud and not tally.answered:
                at = self._at(admitted.place)
                self._report_dropped(self._mind.drop_reason_at_end(), self._mind.unsent_dropped(at))

    @asynccontextmanager
    async def _listing_beside(
        self, session: AppSession, admitted: Admitted, record: TaskRecord
    ) -> AsyncGenerator[asyncio.Task[tuple[Declared, ...]]]:
        """Lists the tools of the apps in a task beside the block, which awaits it where it needs what they listed.

        The listing is a task of its own, not in a group with the gate, so a listing that fails never cuts the gate
        and its failure ends the turn after the thinking. A turn that ends before it awaited the listing cancels it,
        and a defect the listing met joins the turn's record.
        """
        listing = asyncio.create_task(self._lists.listed(session, admitted, record))
        try:
            yield listing
        finally:
            listing.cancel()
            await asyncio.wait((listing,))
            if not listing.cancelled() and isinstance(error := listing.exception(), Exception):
                record.note(error)

    def _at(self, place: PlaceRef) -> AppPlace:
        return AppPlace(self._app.mcp_url, place)

    async def _judge_turn(
        self,
        session: AppSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        tally: TurnTally,
        gate: Callable[[], Awaitable[None]],
    ) -> None:
        """Runs one turn over the loop's session.

        A turn the gateway ends short is told to the tally unless a defect was met; a failed relayed call and any other
        error go to the group. Every error of a cancelled turn ends as the cancellation.
        """
        sizing = Sizing()
        try:
            await self._run_turn(session, setup, admitted, record, sizing, tally, gate)
        except Exception as error:
            raise_if_cancelled()
            leaves = list(leaves_of(error))
            relay_stop = next((leaf for leaf in leaves if isinstance(leaf, RelayFailed)), None)
            stopped = relay_stop or next((leaf for leaf in leaves if isinstance(leaf, TurnStopped)), None)
            if stopped is None:
                raise
            # The stop only carries the failure out of the backend; the failure goes to the group, never the stop.
            held = failure_held_by(stopped)
            if held is None and record.defect is None:
                tally.stop_code = stopped.code.value
                self._leave_unanswered(stopped.code, admitted)
            # What left the session beside the stop (a late answer, the failure another stop holds) still goes to the
            # group, to be judged there.
            rest = failures_leaving_with(held, (leaf for leaf in leaves if leaf is not stopped))
            if len(rest) == 1:
                raise rest[0] from None
            if rest:
                raise BaseExceptionGroup("the turn's session failed", rest) from None
        finally:
            tally.sizes = sizing.sizes()

    def _leave_unanswered(self, code: RejectionCode | FailureCode, admitted: Admitted) -> None:
        """A turn that ended without an answer leaves his mind a pause that holds every turn of his, so the lines that
        arrive meanwhile are taken into his history and make one turn after it, with the turn's own place, due again
        from the pause's end. The pause is the mind's, so it outlives this task."""
        now = self._clock.now()
        seconds = self._mind.unanswered(budget_spent=code is RejectionCode.BUDGET_SPENT, now=now)
        self._report_pause(seconds, code.value)
        self._mind.due_again(self._at(admitted.place), admitted.occasion, now + seconds)

    async def _run_turn(
        self,
        session: AppSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        sizing: Sizing,
        tally: TurnTally,
        gate: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._listing_beside(session, admitted, record) as listing:
            await gate()
            with stepping(step_timer("turn_setup.ended", self._agent.client_id, self._app.mcp_url)) as hand_off:
                lists = await listing
                turn = TurnTold(
                    admitted.at,
                    self._app.mcp_url,
                    admitted.place,
                    admitted.occasion,
                    aloud=admitted.aloud,
                    through=admitted.history.taken,
                )
                told = turn_messages(admitted.facts.name, admitted.present, turn)
                if admitted.held is not None:
                    told = (*told, self._told_standing(admitted, admitted.held))
                bound = TurnBound(self._clock, setup.persona.max_duration_seconds)
                try:
                    await self._run_session_body(
                        setup, admitted, turn, told, session, record, sizing, tally, hand_off, bound, lists
                    )
                except Exception as error:
                    record.note(error)
                    raise
                finally:
                    bound.close()

    async def _run_session_body(
        self,
        setup: TurnSetup,
        admitted: Admitted,
        turn: TurnTold,
        told: tuple[str, ...],
        session: AppSession,
        record: TaskRecord,
        sizing: Sizing,
        tally: TurnTally,
        hand_off: StepTimer,
        bound: TurnBound,
        lists: tuple[Declared, ...],
    ) -> None:
        raise_if_cancelled()
        held = self._hold_tools(session, record, admitted, bound, lists)
        system = f"{setup.persona.prompt}\n\n{REPLY_RULES}" if admitted.aloud else setup.persona.prompt
        request = TurnRequest(
            agent=setup.persona.name,
            turn=Turn(system=system, messages=[TurnMessage(role=TurnRole.USER, text=each) for each in told]),
            roles=setup.persona.roles,
            tools=held.declarations,
            model=(self._agent.hold_model if admitted.held else None) or setup.persona.model,
            max_duration_seconds=setup.persona.max_duration_seconds,
            max_tool_calls=held.max_tool_calls,
            one_call_per_reply=admitted.aloud,
        )
        sizing.request_built(held.declarations, told)
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        async with (
            setup.backend(tuple(sizing.measuring(each) for each in held.tools)) as backend,
            aclosing(backend.run(request, person)) as events,
        ):
            hand_off.end()
            cut = False
            if admitted.aloud and held.inbox is not None:
                place = admitted.place
                speaking, answering = self._voicing.speaking(
                    place,
                    record,
                    tally,
                    bound,
                    held.inbox,
                    held.told,
                    partial(self._hold_decided, place),
                    lambda hold: self._end_hold(place, record, hold, HoldOutcome.UNDECIDED, bound),
                )
                if held.told is not None:
                    held.told.say_line = partial(answering.line_decided, held.told.held)
                voiced = _Voiced(setup, admitted, turn, request, held, speaking, answering, record, held.inbox, bound)
                final = await self._speak_along(voiced, events, tally)
                cut = speaking.was_cut
            else:
                final = await _final_event_of(events, tally)
        if final is not None or cut:
            tally.answered = True
            tally.usage = None if final is None else final.usage
            self._mind.answered(turn)

    async def _speak_along(
        self, voiced: _Voiced, events: AsyncIterator[TurnEvent], tally: TurnTally
    ) -> FinalEvent | None:
        """Reads the turn's events while each hold of his is decided beside them; the sentences they release go to the
        mind's record, whose place's sender sends them after the turn too.

        A call is relayed when the events are asked for again, at once unless a hold stands, whatever of his plays. The
        turn ends once its events have. The final event, if the turn had one, is returned; his stop ends the turn
        without one.
        """
        self._listening = (voiced.admitted.place, voiced.inbox)
        try:
            async with asyncio.TaskGroup() as group:
                hearing = group.create_task(self._hear_events(events, voiced, tally))
                group.create_task(self._decide_holds(voiced, hearing))
                group.create_task(close_when_over(voiced.inbox, hearing))
        finally:
            self._listening = None
        return None if hearing.cancelled() else hearing.result()

    async def _hear_events(
        self, events: AsyncIterator[TurnEvent], voiced: _Voiced, tally: TurnTally
    ) -> FinalEvent | None:
        speaking = voiced.speaking
        held = voiced.held
        final = None
        async for event in events:
            if answer := speaking.hear(event):
                await voiced.answering.instead_told(answer)
            if isinstance(event, ToolCallEvent):
                tally.tool_call_count += 1
                voiced.inbox.waiting = event
                if not held.deciding_first:
                    await voiced.inbox.until_none_stands(voiced.bound)
                voiced.inbox.waiting = None
            elif isinstance(event, FinalEvent):
                final = event
        if held.deciding_first:
            speaking.cut()
        return final

    async def hold(self, session: AppSession, place: PlaceRef, window: PlaceWindow) -> bool:
        """What a read of the place's window says of him. The hold it states stands in the mind's record of the place.
        The spoken turn at the place hears it; with none there, a hold that may be told makes the place due, and a hold
        whose turn ended without deciding it is stopped with its id, the loop's session taking the call, and the unsent
        sentences at the place drop. Whether the hold made the place due."""
        at = self._at(place)
        held = self._mind.hold_read(at, window)
        if self._listening is not None and self._listening[0] == place:
            self._listening[1].hear(window)
        elif held is None or self._aloud_at == place:
            return False
        elif held.telling is Telling.UNDECIDED:
            self._stop_speech(place)
            refusal = await session.stop_speaking(place, held.id)
            self._hold_decided(place, held, hold_ended_as(HoldOutcome.UNDECIDED, refusal))
        elif self._mind.hold_makes_due(at):
            return self._mind.due_for_hold(at, window.generation, self._clock.now())
        return False

    def _hold_decided(self, place: PlaceRef, held: Held, outcome: HoldOutcome) -> None:
        self._mind.hold_decided(self._at(place), held.key)
        if self._listening is not None and self._listening[0] == place:
            self._listening[1].decided(held.key)
        self._report(Event.HOLD_DECIDED, outcome.value, None, None, hold_id=held.id)

    async def _decide_holds(self, voiced: _Voiced, hearing: _Hearing) -> None:
        """Starts a round for each hold that may be told, until the speech is over; a hold no round took by then stands
        in the mind's record for the turn that decides it."""
        while (hold := await voiced.inbox.next_hold()) is not None:
            await self._decide_hold(voiced, hearing, hold)

    async def _decide_hold(self, voiced: _Voiced, hearing: _Hearing, hold: Held) -> None:
        """One round on a backend of its own beside the run parked at the hold: his history as it stands and the
        telling of the hold, on his persona and tools. Without a decision he stops and says nothing."""
        admitted, speaking, inbox = voiced.admitted, voiced.speaking, voiced.inbox
        inbox.told = hold.speech
        turn = voiced.turn
        current = TurnTold(
            turn.at, turn.app, turn.place, turn.occasion, aloud=turn.aloud, through=self._mind.history.taken
        )
        told = (
            *turn_messages(admitted.facts.name, self._mind.present, current),
            self._told_standing(
                admitted, hold, None if inbox.waiting is None else (inbox.waiting.name, inbox.waiting.input)
            ),
        )
        request = voiced.request.model_copy(
            update={
                "turn": Turn(
                    system=voiced.request.turn.system,
                    messages=[TurnMessage(role=TurnRole.USER, text=each) for each in told],
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
            ended = await self._end_hold(admitted.place, voiced.record, hold, HoldOutcome.UNDECIDED, voiced.bound)
            if ended is not None:
                self._report_told_hold_dropped(voiced)

    async def _answer_round(
        self, voiced: _Voiced, hearing: _Hearing, hold: Held, name: str, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        """A call of the round: only his two decisions reach the app, for the hold he was told."""
        outcomes = {CONTINUE_SPEAKING_TOOL: HoldOutcome.CONTINUED, STOP_SPEAKING_TOOL: HoldOutcome.STOPPED}
        outcome = outcomes.get(name)
        if outcome is None:
            return ToolOutcome(_DECIDING_FIRST_TEXT, is_error=True)
        if self._mind.hold_decided_key(self._at(voiced.admitted.place)) == hold.key:
            return ToolOutcome(_DECIDED_TEXT, is_error=True)
        if outcome is HoldOutcome.CONTINUED and LINE_ARGUMENT in arguments:
            return await voiced.answering.line_decided(hold, arguments[LINE_ARGUMENT])
        told = CONTINUED_TEXT if outcome is HoldOutcome.CONTINUED else await self._cut_speech(voiced, hearing)
        ended = await self._end_hold(voiced.admitted.place, voiced.record, hold, outcome, voiced.bound)
        if ended is None:
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        if outcome is HoldOutcome.STOPPED:
            self._report_told_hold_dropped(voiced)
        return ToolOutcome(GONE_TEXT if ended is HoldOutcome.GONE else told, is_error=False)

    async def _cut_speech(self, voiced: _Voiced, hearing: _Hearing, reason: DropReason = DropReason.STOPPED) -> str:
        """His stop at once: no unsent sentence is sent and the parked reply is dropped with its waiting call, unsent; a
        call already sent runs to its answer, which enters his history, before the reply is dropped. The result tells
        what was left unsaid."""
        waiting = voiced.inbox.waiting
        voiced.speaking.cut()
        told = told_stop(self._stop_speech(voiced.admitted.place, reason), None if waiting is None else waiting.name)
        await voiced.inbox.calls_answered()
        hearing.cancel()
        return told

    def _stop_speech(self, place: PlaceRef, reason: DropReason = DropReason.STOPPED) -> tuple[str, ...]:
        """The unsent sentences at the place drop; the sentences not said yet, which a request in flight may still
        answer."""
        unsaid, dropped = self._mind.speech_stopped(self._at(place))
        self._report_dropped(reason, dropped)
        return unsaid

    def _report_told_hold_dropped(self, voiced: _Voiced) -> None:
        """A round for a newer hold cut the speech before the turn decided the hold it was told, which the app may have
        ended already: the older id gets its line."""
        told = voiced.held.told
        if told is not None and told.dropped():
            self._report(Event.HOLD_DECIDED, HoldOutcome.UNDECIDED.value, None, None, hold_id=told.held.id)

    async def _end_hold(
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
            self._stop_speech(place)
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
        ended = hold_ended_as(outcome, answer)
        self._hold_decided(place, hold, ended)
        return ended

    def _told_standing(self, admitted: Admitted, held: Held, waiting: tuple[str, JsonObject] | None = None) -> str:
        """The telling of the hold: what was said over him, the sentence he was in, which is unknown when another pod
        sent it, his unsaid sentences and the call of his that waits."""
        at = self._at(admitted.place)
        return told_hold(held.id, held.speech, self._mind.speech_saying(at), self._mind.speech_unsaid(at), waiting)

    async def _decide_told(
        self,
        admitted: Admitted,
        record: TaskRecord,
        deciding: TurnHold,
        bound: TurnBound,
        name: str,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        """His decision of the hold the turn was told, sent to the app with the standing id; a second one is refused."""
        outcome = HoldOutcome.CONTINUED if name == CONTINUE_SPEAKING_TOOL else HoldOutcome.STOPPED
        if not deciding.pending:
            return ToolOutcome(_DECIDED_TEXT, is_error=True)
        if outcome is HoldOutcome.CONTINUED and LINE_ARGUMENT in arguments:
            return await deciding.line_said(arguments[LINE_ARGUMENT])
        deciding.outcome = outcome
        unsaid = self._mind.speech_unsaid(self._at(admitted.place))
        ended = await self._end_hold(admitted.place, record, deciding.held, outcome, bound)
        if ended is None:
            deciding.outcome = None
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        if ended is HoldOutcome.GONE:
            return ToolOutcome(GONE_TEXT, is_error=False)
        told = CONTINUED_TEXT if outcome is HoldOutcome.CONTINUED else told_stop(unsaid, None)
        return ToolOutcome(told, is_error=False)

    def _hold_tools(
        self, session: AppSession, record: TaskRecord, admitted: Admitted, bound: TurnBound, lists: tuple[Declared, ...]
    ) -> HeldTools:
        """The lookup when the configuration has a memory, then the tools of the place's app, of each app named in the
        admission and of each tools-only app the realm grants and that answers, in the configuration's order, each app
        counting its own calls.

        Of each app every tool it lists now is held but the runner's reserved names and those an earlier app holds,
        none when it lists none. Another app whose listing fails, and the place's own app when it does not answer within
        its call time, is left out and reported, and the turn holds the rest.
        """
        deciding = None if admitted.held is None else TurnHold(admitted.held)
        inbox = self._inbox_of(admitted) if admitted.aloud else None
        hold = HoldBack(admitted.history, inbox, deciding)
        tools: list[HandTool] = []
        bounds: list[int] = []
        taken = frozenset[str]()
        lookup = self._remembering.lookup
        if lookup is not None:
            tools.append(HandTool(RECALL_DECLARATION, partial(self._look, lookup, Calls(lookup.calls), admitted, hold)))
            bounds.append(lookup.calls)
        for (app, max_calls), declared in zip(self._apps, lists, strict=True):
            calls = Calls(max_calls)
            if declared is None:
                continue
            offered, left_out = offered_declarations(declared, taken)
            self._tools_left_out(app, left_out)
            taken |= {each.name for each in offered}
            if offered:
                tools.extend(self._hand_tools(app, session, offered, calls, hold, record))
                bounds.append(calls.bound)
        if deciding is not None:
            tools.extend(
                HandTool(each, partial(self._decide_told, admitted, record, deciding, bound, each.name))
                for each in DECISION_DECLARATIONS
            )
            bounds.append(DECISION_CALLS)
        elif inbox is not None:
            tools.extend(HandTool(each, self._decided_apart) for each in DECISION_DECLARATIONS)
            bounds.append(DECISION_CALLS)
        return HeldTools(tuple(tools), turn_call_bound(bounds), inbox, deciding)

    def _inbox_of(self, admitted: Admitted) -> HoldInbox:
        """What the spoken turn hears of a hold: the one that stands in the record, told at the turn's start or not."""
        if admitted.held is not None:
            return HoldInbox(admitted.held, told=True)
        return HoldInbox(self._mind.hold_standing(self._at(admitted.place)))

    def _hand_tools(
        self,
        app: AppTarget,
        session: AppSession,
        offered: tuple[ToolDeclaration, ...],
        calls: Calls,
        hold: HoldBack,
        record: TaskRecord,
    ) -> list[HandTool]:
        if app.mcp_url == self._app.mcp_url:
            return [HandTool(each, partial(self._relay, session, each.name, calls, hold)) for each in offered]
        return [HandTool(each, partial(self._relay_elsewhere, app, each.name, calls, hold, record)) for each in offered]

    def _tools_left_out(self, app: AppTarget, count: int) -> None:
        """Reports an app's names left out for an earlier app's, once until the loop's session ends."""
        if count and app.mcp_url not in self._left_out_reported:
            self._left_out_reported.add(app.mcp_url)
            self._report_tools_left_out(app, count)

    def new_session(self) -> None:
        """The loop's app session starts: a name left out is reported again."""
        self._left_out_reported.clear()

    async def _look(
        self,
        lookup: LookupBounds,
        calls: Calls,
        admitted: Admitted,
        hold: HoldBack,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        """Looks up what the agent's own place held before its history: nothing enters the history, and a lookup is
        never held back by what arrived, only by a hold he has not decided."""
        if hold.deciding_first:
            return ToolOutcome(_DECIDING_FIRST_TEXT, is_error=True)
        if not calls.take():
            return ToolOutcome(LOOKUPS_SPENT_TEXT, is_error=True)
        text = search_text(arguments, lookup.result_bytes)
        if text is None:
            return ToolOutcome(text_refused_text(lookup.result_bytes), is_error=True)
        search = line_search(
            self._agent.client_id, self._app.mcp_url, admitted.place, admitted.history, text, lookup.result_bytes
        )
        try:
            found = await self._remembering.search(search)
        except Exception as error:
            raise RelayFailed(error) from None
        if found is None:
            return ToolOutcome(STORE_DID_NOT_ANSWER_TEXT, is_error=True)
        return ToolOutcome(lookup_result(found, lookup.result_bytes), is_error=False)

    async def _decided_apart(  # NOSONAR(S7503) HandToolHandler is async
        self, _principal: Principal, _arguments: JsonObject
    ) -> ToolOutcome:
        return ToolOutcome(_APART_TEXT, is_error=True)

    def _held(self, hold: HoldBack) -> ToolOutcome | None:
        """The result of a call that is not sent, or ``None`` when the call goes on."""
        if hold.deciding_first:
            return ToolOutcome(_DECIDING_FIRST_TEXT, is_error=True)
        arrived = hold.holding(self._mind.history)
        if arrived is not None:
            return ToolOutcome(held_told(arrived), is_error=False)
        if words := hold.said_over_him():
            return ToolOutcome(told_words(words), is_error=False)
        return None

    async def _relay(
        self, session: AppSession, name: str, calls: Calls, hold: HoldBack, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        if (held := self._held(hold)) is not None:
            return held
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        try:
            with hold.sending():
                return await self._recorded_as_act(self._app, name, arguments, self._invoke(session, name, arguments))
        except Exception as error:
            raise RelayFailed(error) from None

    async def _invoke(self, session: AppSession, name: str, arguments: JsonObject) -> ToolOutcome:
        await self._use_current_token(session)
        return await session.invoke(name, arguments)

    async def _relay_elsewhere(
        self,
        app: AppTarget,
        name: str,
        calls: Calls,
        hold: HoldBack,
        record: TaskRecord,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        if (held := self._held(hold)) is not None:
            return held
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        try:
            with hold.sending():
                answer = await self._recorded_as_act(
                    app,
                    name,
                    arguments,
                    self._sessions.ask_elsewhere(app, lambda session: session.invoke(name, arguments), record),
                )
        except Exception as error:
            raise RelayFailed(error) from None
        if isinstance(answer, CallFailure):
            calls.close()
            self._report(failure_event(app), failure_code(answer), app, answer)
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        return answer

    async def _recorded_as_act[Answer: ToolOutcome | CallFailure](
        self, app: AppTarget, name: str, arguments: JsonObject, request: Awaitable[Answer]
    ) -> Answer:
        """Awaits the request of a relayed call and records it as an act of the agent however it ends.

        Whether the turn is being cancelled is asked once the request and every scope inside it have exited.
        """
        ended: ToolOutcome | BaseException | None = None
        try:
            answer = await request
            ended = answer
            return answer
        except Exception as error:
            ended = error
            raise
        except asyncio.CancelledError as cancelled:
            ended = cancelled
            raise
        finally:
            outcome = None if ended is None else act_outcome(ended, cancelling=task_is_being_cancelled())
            if outcome is not None:
                self._mind.take((ActDone(app.mcp_url, name, arguments, outcome, self._clock.wall()),))
