"""What one turn of an agent in an app does: the session, the tools held and how the turn ends."""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack, aclosing
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.agent_file import TurnRequest
from agents.backend import TurnPerson, TurnStopped
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL
from agents.tools import HandTool, ToolDeclaration, ToolEffect, ToolOutcome
from agents.turn import RejectionCode, Turn, TurnMessage, TurnRole
from agents.turn_events import FinalEvent, ToolCallEvent
from agents.voice.reply import REPLY_RULES

from personas.decision import (
    TURN_ROLE,
    AppPlace,
    Held,
    HoldOutcome,
    TurnTold,
    act_outcome,
    arrived_since,
    call_within_bound,
    decisive_failure,
    failure_code,
    held_told,
    hold_stated,
    hold_tellable,
    is_not_granted,
    offered_declarations,
    round_seconds,
    turn_call_bound,
    turn_messages,
)
from personas.entries import ActDone, InputSizes, message_bytes, picture_bytes, told_bytes, told_hold, told_stop
from personas.failures import CallFailure, Unreachable, leaves_of, raise_if_cancelled, task_is_being_cancelled
from personas.recall import (
    LOOKUPS_SPENT_TEXT,
    RECALL_DECLARATION,
    STORE_DID_NOT_ANSWER_TEXT,
    line_search,
    lookup_result,
    search_text,
    text_refused_text,
)
from personas.reporting import Event, TurnTally, step_timer, stepping
from personas.task_record import (
    RelayFailed,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    failure_held_by,
    failures_leaving_with,
    unreachable_unless_cancelling,
)
from personas.voice import Speaking

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime

    from agents.agent_file import AgentDefinition
    from agents.backend import AgentBackend, TurnEvent
    from agents.member import PlaceRef, PlaceWindow
    from agents.turn import FailureCode, JsonObject, Principal
    from logline import StepTimer

    from personas.app_session import AppSession, SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget
    from personas.decision import Arrived, Occasion, Present, Token
    from personas.entries import History
    from personas.mind import Mind
    from personas.recall import LookupBounds
    from personas.remembering import Remembering
    from personas.secret import Secret
    from personas.task_record import TaskRecord
    from personas.tokens import TokenFacts, TokenSource

# The backend a turn runs on, which it closes when the turn ends.
type BackendFactory = Callable[[Iterable[HandTool]], AbstractAsyncContextManager[AgentBackend]]

_CALL_BOUND_TEXT: Final = "this app takes no more calls in this turn"
_APP_FAILED_TEXT: Final = "the app did not answer"
_MODEL_FACING: Final = {"name", "description", "parameters"}
_RESULT_ROLE: Final = "tool"
_APART_TEXT: Final = (
    "a hold is decided in a round of its own when one starts: nothing is held now, so there is nothing to decide"
)
_DECIDING_FIRST_TEXT: Final = "a hold is decided first: call continue_speaking or stop_speaking"
_DECIDED_TEXT: Final = "this hold is decided"
_CONTINUED_TEXT: Final = '{"continued": true}'
_DECISION_CALLS: Final = 1
_DECISION_DECLARATIONS: Final = tuple(
    ToolDeclaration(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string", "description": "The id of the hold you were told."}},
            "required": ["id"],
            "additionalProperties": False,
        },
        effect=ToolEffect.SIDE_EFFECTING,
    )
    for name, description in (
        (
            CONTINUE_SPEAKING_TOOL,
            "Lets the sentence you were held in go on to its end, and your next sentences follow.",
        ),
        (STOP_SPEAKING_TOOL, "Drops the rest of the sentence you were held in and everything you had not yet said."),
    )
)


def _failure_event(app: AppTarget) -> Event:
    return Event.HELD_APP_FAILED if app.listens else Event.TOOLS_ONLY_APP_FAILED


async def _final_event_of(events: AsyncIterator[TurnEvent], tally: TurnTally) -> FinalEvent | None:
    """Reads the turn's events to their end, counting its tool calls, and returns its final event, if it had one."""
    final = None
    async for event in events:
        if isinstance(event, ToolCallEvent):
            tally.tool_call_count += 1
        elif isinstance(event, FinalEvent):
            final = event
    return final


async def _close_when_over(inbox: _HoldInbox, *speech: asyncio.Task[object]) -> None:
    await asyncio.wait(speech)
    inbox.close()


@dataclass(frozen=True)
class TurnSetup:
    """What an agent's turn in an app is made of: the persona and the backend that runs it."""

    persona: AgentDefinition
    backend: BackendFactory


class _Sizing:
    """What the turn's input weighs: the request's tools and told messages once it is built, and every tool result
    relayed back to the model, held and refused calls included. Only bytes are kept."""

    def __init__(self) -> None:
        self._request: tuple[int, int] | None = None
        self._results = 0
        self._images = 0

    def request_built(self, declarations: Iterable[ToolDeclaration], told: Iterable[str]) -> None:
        tools = sum(told_bytes(each.model_dump(mode="json", include=_MODEL_FACING)) for each in declarations)
        self._request = (tools, sum(message_bytes(TurnRole.USER.value, each) for each in told))

    def measuring(self, tool: HandTool) -> HandTool:
        """The tool with a handler that counts the result it returns."""

        async def handler(principal: Principal, arguments: JsonObject) -> ToolOutcome:
            outcome = await tool.handler(principal, arguments)
            self._results += message_bytes(_RESULT_ROLE, outcome.content)
            self._images += picture_bytes(outcome.image)
            return outcome

        return HandTool(tool.declaration, handler)

    def sizes(self) -> InputSizes | None:
        """The weights, or none while the request is not built."""
        if self._request is None:
            return None
        return InputSizes(self._request[0], self._request[1], self._results, self._images)


@dataclass
class _Calls:
    """How many calls an app took in this turn, against its own bound."""

    bound: int
    made: int = 0

    def take(self) -> bool:
        if not call_within_bound(self.made, self.bound):
            return False
        self.made += 1
        return True

    def close(self) -> None:
        """The app takes no more calls in this turn."""
        self.made = self.bound


class _HoldInbox:
    """What the window says of him while a spoken turn runs: the hold that stands, and the call that waits."""

    def __init__(self) -> None:
        self.hold: Held | None = None
        self.waiting: ToolCallEvent | None = None
        self._taken: str | None = None
        self._closed = False
        self._changed = asyncio.Event()

    def hear(self, window: PlaceWindow) -> None:
        self.hold = hold_stated(window, self.hold)
        self._changed.set()

    def close(self) -> None:
        self._closed = True
        self._changed.set()

    @property
    def undecided(self) -> Held | None:
        return self.hold if self.hold is not None and self.hold.id != self._taken else None

    async def next_hold(self) -> Held | None:
        while not self._closed:
            hold = self.undecided
            if hold is not None and hold_tellable(hold):
                self._taken = hold.id
                return hold
            self._changed.clear()
            await self._changed.wait()
        return None


@dataclass
class _Hold:
    """Whether a call of the turn is held back: once, when an entry not his own entered his history since the turn's
    admission."""

    admitted: History = field(repr=False)
    spent: bool = False

    def holding(self, current: History) -> Arrived | None:
        """What arrived for the first call that finds news; every later call of the turn is let through."""
        if self.spent:
            return None
        arrived = arrived_since(current, self.admitted)
        self.spent = arrived is not None
        return arrived


@dataclass(frozen=True)
class _HeldTools:
    """What a turn holds: the hand tools of every app that answered, and the calls the turn may make in all."""

    tools: tuple[HandTool, ...]
    max_tool_calls: int | None
    inbox: _HoldInbox | None = None

    @property
    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(each.declaration for each in self.tools)


@dataclass
class _Voiced:
    """One spoken turn's speech, with what a round at a hold of his starts from."""

    setup: TurnSetup
    admitted: Admitted
    turn: TurnTold
    request: TurnRequest
    held: _HeldTools
    speaking: Speaking
    record: TaskRecord
    inbox: _HoldInbox
    hearing: asyncio.Task[FinalEvent | None] | None = None


@dataclass(frozen=True)
class Admitted:
    """A turn decided on, with the facts it was decided from: the token's, what woke it, what its agent had taken in,
    the wall time it was admitted at, whether its text is spoken and the other apps whose tools it holds."""

    facts: TokenFacts
    place: PlaceRef
    occasion: Occasion
    present: Present = field(repr=False)
    at: datetime
    aloud: bool
    apps: frozenset[str]

    @property
    def history(self) -> History:
        return self.present.history

    def groups(self) -> Coroutine[None, None, frozenset[str]]:
        """The library's GroupSource wants an awaitable; the groups are already known."""
        return asyncio.sleep(0, self.facts.groups)


class Turning:
    """Runs the turns of one agent in one app, each over a session of its own, holding the tools of every app in the
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
        current_token: Callable[[], Awaitable[Token]],
        use_current_token: Callable[[AppSession], Awaitable[None]],
        report: Callable[[Event, str | None, AppTarget | None, CallFailure | None], None],
        report_pause: Callable[[float, str], None],
        report_tools_left_out: Callable[[AppTarget, int], None],
        remembering: Remembering,
    ) -> None:
        self._agent = agent
        self._app = app
        self._apps = tuple((each, each.max_tool_calls) for each in apps if each.max_tool_calls is not None)
        self._tokens = tokens
        self._open_session = open_session
        self._mind = mind
        self._clock = clock
        self._current_token = current_token
        self._use_current_token = use_current_token
        self._report = report
        self._report_pause = report_pause
        self._report_tools_left_out = report_tools_left_out
        self._left_out_reported: set[str] = set()
        self._remembering = remembering
        self._listening: tuple[PlaceRef, _HoldInbox] | None = None
        self._decided: dict[PlaceRef, str] = {}

    async def take(self, setup: TurnSetup, admitted: Admitted, record: TaskRecord, tally: TurnTally) -> None:
        """Runs one turn; whatever way it ends, a defect a request of it met ends the runner loud.

        A session whose close cuts a request short can lose the defect from what leaves, so the turn's record
        joins it. A cancelled turn ends as the cancellation unless a defect was met or nobody cancelled it.
        """
        await ending_loud_when_cancelled(
            ending_with_recorded_defect(self._judge_turn(setup, admitted, record, tally), record), record.loud_end
        )

    async def _judge_turn(self, setup: TurnSetup, admitted: Admitted, record: TaskRecord, tally: TurnTally) -> None:
        """Runs one turn over its own session.

        A turn the gateway ends short is told to the tally unless a defect was met; a failed relayed call and any other
        error go to the group. Every error of a cancelled turn ends as the cancellation.
        """
        sizing = _Sizing()
        try:
            await self._run_turn(setup, admitted, record, sizing, tally)
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
        from the pause's end on the ``time.perf_counter`` scale of the step timers. The pause is the mind's, so it
        outlives this task."""
        seconds = self._mind.unanswered(budget_spent=code is RejectionCode.BUDGET_SPENT, now=self._clock.now())
        self._report_pause(seconds, code.value)
        self._mind.due_again(
            AppPlace(self._app.mcp_url, admitted.place), admitted.occasion, time.perf_counter() + seconds
        )

    async def _run_turn(
        self, setup: TurnSetup, admitted: Admitted, record: TaskRecord, sizing: _Sizing, tally: TurnTally
    ) -> None:
        agent = self._agent.client_id
        begun = time.perf_counter()
        with stepping(step_timer("turn_setup.ended", agent, self._app.mcp_url, started_at=begun)) as hand_off:
            turn = TurnTold(
                admitted.at,
                self._app.mcp_url,
                admitted.place,
                admitted.occasion,
                aloud=admitted.aloud,
                through=admitted.history.taken,
            )
            told = turn_messages(admitted.facts.name, admitted.present, turn)
            with stepping(step_timer("token_hold.ended", agent, self._app.mcp_url, started_at=begun)):
                token = await self._current_token()
            async with AsyncExitStack() as opened:
                session = await self._opened(opened, self._app, token.value, measured=True)
                try:
                    await self._run_session_body(setup, admitted, turn, told, session, record, sizing, tally, hand_off)
                except Exception as error:
                    record.note(error)
                    raise

    async def _run_session_body(
        self,
        setup: TurnSetup,
        admitted: Admitted,
        turn: TurnTold,
        told: tuple[str, ...],
        session: AppSession,
        record: TaskRecord,
        sizing: _Sizing,
        tally: TurnTally,
        hand_off: StepTimer,
    ) -> None:
        raise_if_cancelled()
        held = await self._hold_tools(session, record, admitted)
        system = f"{setup.persona.prompt}\n\n{REPLY_RULES}" if admitted.aloud else setup.persona.prompt
        request = TurnRequest(
            agent=setup.persona.name,
            turn=Turn(system=system, messages=[TurnMessage(role=TurnRole.USER, text=each) for each in told]),
            roles=setup.persona.roles,
            tools=held.declarations,
            model=setup.persona.model,
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
                speaking = Speaking(
                    session,
                    self._app.mcp_url,
                    admitted.place,
                    self._clock,
                    setup.persona.max_duration_seconds,
                    partial(self._use_current_token, session),
                    lambda event, code: self._report(event, code, None, None),
                    self._mind,
                    tally,
                )
                voiced = _Voiced(setup, admitted, turn, request, held, speaking, record, held.inbox)
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
        """Reads the turn's events while the sentences they release are spoken and each hold of his is decided beside
        them.

        A call is relayed only when the events are asked for again, so it goes out after his own line for every
        sentence before it has entered. When the events end in an error, the sentence in flight is answered and entered
        before it is raised; only a cancel or his own stop cuts the speech. The turn ends once the speech has, and its
        final event, if it had one, is returned; his stop ends the turn without one.
        """
        self._listening = (voiced.admitted.place, voiced.inbox)
        try:
            async with asyncio.TaskGroup() as group:
                playing = group.create_task(voiced.speaking.play())
                hearing = voiced.hearing = group.create_task(self._hear_events(events, voiced, tally))
                group.create_task(self._decide_holds(voiced))
                group.create_task(_close_when_over(voiced.inbox, playing, hearing))
        finally:
            self._listening = None
        return None if hearing.cancelled() else hearing.result()

    async def _hear_events(
        self, events: AsyncIterator[TurnEvent], voiced: _Voiced, tally: TurnTally
    ) -> FinalEvent | None:
        speaking = voiced.speaking
        final = None
        try:
            async for event in events:
                speaking.hear(event)
                if isinstance(event, ToolCallEvent):
                    tally.tool_call_count += 1
                    voiced.inbox.waiting = event
                    await speaking.spoken()
                    voiced.inbox.waiting = None
                elif isinstance(event, FinalEvent):
                    final = event
        except Exception:
            await speaking.settle()
            raise
        speaking.end()
        return final

    async def hold(self, session: AppSession, place: PlaceRef, window: PlaceWindow) -> None:
        """What a read of the place's window says of him: the spoken turn at the place hears it; with none there, a hold
        that stands is stopped at once, the loop's session taking the call."""
        if self._listening is not None and self._listening[0] == place:
            self._listening[1].hear(window)
        elif window.interrupted is not None and self._decided.get(place) != window.interrupted.id:
            await session.stop_speaking(place, window.interrupted.id)
            self._hold_decided(place, window.interrupted.id, HoldOutcome.NO_TURN)

    def _hold_decided(self, place: PlaceRef, hold_id: str, outcome: HoldOutcome) -> None:
        self._decided[place] = hold_id
        self._report(Event.HOLD_DECIDED, outcome.value, None, None)

    async def _decide_holds(self, voiced: _Voiced) -> None:
        """Starts a round for each hold that may be told, until the speech is over; a hold no round took by then is
        stopped."""
        while (hold := await voiced.inbox.next_hold()) is not None:
            await self._decide_hold(voiced, hold)
        if (hold := voiced.inbox.undecided) is not None:
            await self._end_hold(voiced, hold, HoldOutcome.UNDECIDED)

    async def _decide_hold(self, voiced: _Voiced, hold: Held) -> None:
        """One round on a backend of its own beside the run parked at the hold: his history as it stands and the
        telling of the hold, on his persona and tools. Without a decision he stops and says nothing."""
        admitted, speaking, inbox = voiced.admitted, voiced.speaking, voiced.inbox
        told = (
            *turn_messages(
                admitted.facts.name, self._mind.present, replace(voiced.turn, through=self._mind.history.taken)
            ),
            told_hold(
                hold.id,
                hold.speech,
                speaking.saying,
                speaking.unsaid,
                None if inbox.waiting is None else (inbox.waiting.name, inbox.waiting.input),
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
            HandTool(each.declaration, partial(self._answer_round, voiced, hold, each.declaration.name))
            for each in voiced.held.tools
        )
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        try:
            async with voiced.setup.backend(tools) as backend, aclosing(backend.run(request, person)) as events:
                async for _ in events:
                    pass
        except RelayFailed:
            raise
        except TurnStopped:
            pass
        if self._decided.get(admitted.place) != hold.id:
            self._cut_speech(voiced)
            await self._end_hold(voiced, hold, HoldOutcome.UNDECIDED)

    async def _answer_round(
        self, voiced: _Voiced, hold: Held, name: str, _principal: Principal, _arguments: JsonObject
    ) -> ToolOutcome:
        """A call of the round: only his two decisions reach the app, for the hold he was told."""
        outcomes = {CONTINUE_SPEAKING_TOOL: HoldOutcome.CONTINUED, STOP_SPEAKING_TOOL: HoldOutcome.STOPPED}
        outcome = outcomes.get(name)
        if outcome is None:
            return ToolOutcome(_DECIDING_FIRST_TEXT, is_error=True)
        if self._decided.get(voiced.admitted.place) == hold.id:
            return ToolOutcome(_DECIDED_TEXT, is_error=True)
        told = _CONTINUED_TEXT if outcome is HoldOutcome.CONTINUED else self._cut_speech(voiced)
        if not await self._end_hold(voiced, hold, outcome):
            return ToolOutcome(_APP_FAILED_TEXT, is_error=True)
        return ToolOutcome(told, is_error=False)

    @staticmethod
    def _cut_speech(voiced: _Voiced) -> str:
        """His stop at once: no unsent sentence is spoken, the parked reply is dropped with its waiting call; the result
        tells what was left unsaid."""
        waiting = voiced.inbox.waiting
        told = told_stop(voiced.speaking.cut(), None if waiting is None else waiting.name)
        if voiced.hearing is not None:
            voiced.hearing.cancel()
        return told

    async def _end_hold(self, voiced: _Voiced, hold: Held, outcome: HoldOutcome) -> bool:
        """Sends his decision in a session of its own; says if the app answered."""
        place = voiced.admitted.place
        try:
            answer = await self._ask_elsewhere(
                self._app,
                lambda session: (
                    session.continue_speaking(place, hold.id)
                    if outcome is HoldOutcome.CONTINUED
                    else session.stop_speaking(place, hold.id)
                ),
                voiced.record,
            )
        except Exception as error:
            raise RelayFailed(error) from None
        if isinstance(answer, CallFailure):
            self._report(_failure_event(self._app), failure_code(answer), self._app, answer)
            return False
        self._hold_decided(place, hold.id, outcome)
        return True

    async def _hold_tools(self, session: AppSession, record: TaskRecord, admitted: Admitted) -> _HeldTools:
        """The lookup when the configuration has a memory, then the tools of the place's app, of each app named in the
        admission and of each tools-only app the realm grants and that answers, in the configuration's order, each app
        counting its own calls.

        Of each app every tool it lists now is held but the runner's reserved names and those an earlier app holds,
        none when it lists none. Another app whose listing fails is left out and reported once, and the turn holds the
        rest.
        """
        own = await self._list_tools(self._app, session)
        inbox = _HoldInbox() if admitted.aloud else None
        hold = _Hold(admitted.history)
        tools: list[HandTool] = []
        bounds: list[int] = []
        taken = frozenset[str]()
        lookup = self._remembering.lookup
        if lookup is not None:
            tools.append(HandTool(RECALL_DECLARATION, partial(self._look, lookup, _Calls(lookup.calls), admitted)))
            bounds.append(lookup.calls)
        for app, max_calls in self._apps:
            calls = _Calls(max_calls)
            if app.mcp_url == self._app.mcp_url:
                declared = own
            elif (app.listens and app.mcp_url not in admitted.apps) or (
                not app.listens and self._mind.holds_back(app.mcp_url, self._clock.now())
            ):
                continue
            else:
                found = await self._list_elsewhere(app, record)
                if found is None:
                    continue
                declared = found
            offered, left_out = offered_declarations(declared, taken)
            self._tools_left_out(app, left_out)
            taken |= {each.name for each in offered}
            if offered:
                tools.extend(self._hand_tools(app, session, offered, calls, hold, record))
                bounds.append(calls.bound)
        if inbox is not None:
            tools.extend(HandTool(each, self._decided_apart) for each in _DECISION_DECLARATIONS)
            bounds.append(_DECISION_CALLS)
        return _HeldTools(tuple(tools), turn_call_bound(bounds), inbox)

    def _hand_tools(
        self,
        app: AppTarget,
        session: AppSession,
        offered: tuple[ToolDeclaration, ...],
        calls: _Calls,
        hold: _Hold,
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
        self, lookup: LookupBounds, calls: _Calls, admitted: Admitted, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        """Looks up what the agent's own place held before its history: nothing enters the history, and a lookup is
        never held back."""
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

    async def _list_elsewhere(self, app: AppTarget, record: TaskRecord) -> tuple[ToolDeclaration, ...] | None:
        """The tools another app lists, or ``None`` after the report of its failed listing."""
        declared = await self._ask_elsewhere(app, partial(self._list_tools, app), record, opening=True)
        if isinstance(declared, CallFailure):
            if not app.listens and is_not_granted(declared):
                if self._mind.refused(app.mcp_url, self._clock.now()):
                    self._report(Event.SUSPENDED, failure_code(declared), app, declared)
            else:
                self._report(_failure_event(app), failure_code(declared), app, declared)
            return None
        if not app.listens and self._mind.granted(app.mcp_url):
            self._report(Event.RESUMED, None, app, None)
        return declared

    async def _list_tools(self, app: AppTarget, session: AppSession) -> tuple[ToolDeclaration, ...]:
        with stepping(step_timer("tool_list.ended", self._agent.client_id, app.mcp_url), turn_goes_on=app != self._app):
            return await session.declared_tools()

    async def _decided_apart(self, _principal: Principal, _arguments: JsonObject) -> ToolOutcome:
        return ToolOutcome(_APART_TEXT, is_error=True)

    def _held(self, hold: _Hold) -> ToolOutcome | None:
        """The result of a call that is not sent, or ``None`` when the call goes on. It is asked before any await."""
        arrived = hold.holding(self._mind.history)
        if arrived is None:
            return None
        return ToolOutcome(held_told(arrived), is_error=False)

    async def _relay(
        self, session: AppSession, name: str, calls: _Calls, hold: _Hold, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        if (held := self._held(hold)) is not None:
            return held
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        try:
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
        calls: _Calls,
        hold: _Hold,
        record: TaskRecord,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        if (held := self._held(hold)) is not None:
            return held
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        try:
            answer = await self._recorded_as_act(
                app, name, arguments, self._ask_elsewhere(app, lambda session: session.invoke(name, arguments), record)
            )
        except Exception as error:
            raise RelayFailed(error) from None
        if isinstance(answer, CallFailure):
            calls.close()
            self._report(_failure_event(app), failure_code(answer), app, answer)
            return ToolOutcome(_APP_FAILED_TEXT, is_error=True)
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

    async def _ask_elsewhere[Answer](
        self,
        app: AppTarget,
        ask: Callable[[AppSession], Awaitable[Answer]],
        record: TaskRecord,
        *,
        opening: bool = False,
    ) -> Answer | CallFailure:
        """Asks another app in a session of its own, under the token for its own scope; ``opening`` says the turn
        waits for the session's opening, which is then a step of its own.

        The session lives only for this request. A failure of the call, a session that died of its own child and
        an expiry of the app's call time come back as the failure; any other error is raised, and a turn whose
        place left stops. A defect recorded in the turn ends the request, raised from the failure.
        """
        try:
            return await ending_loud_when_cancelled(
                self._within_call_time(app, ask, record, opening=opening), unreachable_unless_cancelling
            )
        except Exception as error:
            raise_if_cancelled()
            failure = decisive_failure(tuple(leaves_of(error)), already_renewed=False)
            if failure is None:
                raise
            defect = record.defect
            if defect is not None:
                raise defect from failure
            return failure

    async def _opened(self, opened: AsyncExitStack, app: AppTarget, token: Secret, *, measured: bool) -> AppSession:
        """The session at ``app``, closed with ``opened``; its opening is a step when ``measured``."""
        if not measured:
            return await opened.enter_async_context(self._open_session(app, token))
        with stepping(
            step_timer("session_open.ended", self._agent.client_id, app.mcp_url), turn_goes_on=app != self._app
        ):
            return await opened.enter_async_context(self._open_session(app, token))

    async def _within_call_time[Answer](
        self, app: AppTarget, ask: Callable[[AppSession], Awaitable[Answer]], record: TaskRecord, *, opening: bool
    ) -> Answer:
        """The token, the session and the request, all under the call time of the app."""
        bound = asyncio.timeout(app.call_seconds)
        try:
            async with bound:
                granted = await self._tokens.fetch(self._agent, app.scope)
                async with AsyncExitStack() as opened:
                    session = await self._opened(opened, app, granted.value, measured=opening)
                    try:
                        return await ask(session)
                    except Exception as error:
                        record.note(error)
                        raise
        except TimeoutError:
            if not bound.expired():
                raise
            raise Unreachable("the app did not answer in time") from None
