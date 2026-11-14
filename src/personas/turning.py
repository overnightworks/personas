"""What one turn of an agent in an app does: the day read, the session, the tools held and how the turn ends."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.agent_file import TurnRequest
from agents.backend import TurnPerson, TurnStopped
from agents.tools import HandTool, ToolOutcome
from agents.turn import RejectionCode, Turn, TurnMessage, TurnRole
from agents.turn_events import FinalEvent, ToolCallEvent

from personas.decision import (
    TURN_ROLE,
    act_outcome,
    arrived_since,
    bound_declarations,
    call_within_bound,
    decisive_failure,
    failure_code,
    held_told,
    turn_call_bound,
    turn_told,
)
from personas.entries import ActDone, InputSizes, message_bytes, picture_bytes, told_bytes
from personas.failures import CallFailure, Unreachable, leaves_of, raise_if_cancelled, task_is_being_cancelled
from personas.reporting import Event
from personas.task_record import (
    RelayFailed,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    failure_held_by,
    unreachable_unless_cancelling,
)
from personas.voice import Speaking

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
    from datetime import datetime

    from agents.agent_file import AgentDefinition
    from agents.backend import AgentBackend, TurnEvent
    from agents.member import PlaceRef
    from agents.tools import ToolDeclaration
    from agents.turn import FailureCode, JsonObject, Principal, TurnUsage

    from personas.app_session import AppSession, SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget, ToolBinding
    from personas.decision import Arrived, Occasion, Token
    from personas.entries import History
    from personas.mind import Mind
    from personas.remembering import Remembering
    from personas.task_record import TaskRecord
    from personas.tokens import TokenFacts, TokenSource

type BackendFactory = Callable[[Iterable[HandTool]], AgentBackend]

_CALL_BOUND_TEXT: Final = "this app takes no more calls in this turn"
_APP_FAILED_TEXT: Final = "the app did not answer"
_MODEL_FACING: Final = {"name", "description", "parameters"}
_RESULT_ROLE: Final = "tool"


def _failure_event(app: AppTarget) -> Event:
    return Event.HELD_APP_FAILED if app.listens else Event.TOOLS_ONLY_APP_FAILED


async def _final_event_of(events: AsyncIterator[TurnEvent]) -> FinalEvent | None:
    """Reads the turn's events to their end and returns its final event, if it had one."""
    final = None
    async for event in events:
        if isinstance(event, FinalEvent):
            final = event
    return final


@dataclass(frozen=True)
class TurnSetup:
    """What an agent's turn in an app is made of: the persona, the tools bound and the backend that runs it."""

    persona: AgentDefinition
    binding: ToolBinding
    backend: BackendFactory


class _Sizing:
    """What the turn's input weighs: the request's tools and told message once it is built, and every tool result
    relayed back to the model, held and refused calls included. Only bytes are kept."""

    def __init__(self) -> None:
        self._request = (0, 0)
        self._results = 0
        self._images = 0

    def request_built(self, declarations: Iterable[ToolDeclaration], told: str) -> None:
        tools = sum(told_bytes(each.model_dump(mode="json", include=_MODEL_FACING)) for each in declarations)
        self._request = (tools, message_bytes(TurnRole.USER.value, told))

    def measuring(self, tool: HandTool) -> HandTool:
        """The tool with a handler that counts the result it returns."""

        async def handler(principal: Principal, arguments: JsonObject) -> ToolOutcome:
            outcome = await tool.handler(principal, arguments)
            self._results += message_bytes(_RESULT_ROLE, outcome.content)
            self._images += picture_bytes(outcome.image)
            return outcome

        return HandTool(tool.declaration, handler)

    def sizes(self) -> InputSizes:
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

    @property
    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(each.declaration for each in self.tools)


@dataclass(frozen=True)
class Admitted:
    """A turn decided on, with the facts it was decided from: the token's, what woke it, what its agent had taken in,
    the wall time it was admitted at, whether its text is spoken and the other apps whose tools it holds."""

    facts: TokenFacts
    place: PlaceRef
    occasion: Occasion
    history: History = field(repr=False)
    at: datetime
    aloud: bool
    apps: frozenset[str]

    def groups(self) -> Coroutine[None, None, frozenset[str]]:
        """The library's GroupSource wants an awaitable; the groups are already known."""
        return asyncio.sleep(0, self.facts.groups)


class Turning:
    """Runs the turns of one agent in one app, each over a session of its own, holding the tools of every app in the
    configuration's order: the app's own, those of the apps in ``admitted.apps`` and those of the tools-only apps.

    It holds no state across turns beyond what it records into the agent's mind: each relayed call that passed its
    app's bound, once its request ended. A call is held back, once per turn, when something not his own arrived in
    his history since the turn's admission: it is not sent and the model is told what arrived. The token, its use on
    a session and the two reports are the loop's, handed over as callables and read at each call.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        apps: tuple[AppTarget, ...],
        tokens: TokenSource,
        open_session: SessionOpener,
        remembering: Remembering,
        mind: Mind,
        clock: Clock,
        current_token: Callable[[], Awaitable[Token]],
        use_current_token: Callable[[AppSession], Awaitable[None]],
        report: Callable[[Event, str | None, AppTarget | None], None],
        report_turn_end: Callable[[str | None, TurnUsage | None, InputSizes | None], None],
        report_pause: Callable[[float], None],
    ) -> None:
        self._agent = agent
        self._app = app
        self._apps = tuple((each, each.binding) for each in apps if each.binding is not None)
        self._tokens = tokens
        self._open_session = open_session
        self._remembering = remembering
        self._mind = mind
        self._clock = clock
        self._current_token = current_token
        self._use_current_token = use_current_token
        self._report = report
        self._report_turn_end = report_turn_end
        self._report_pause = report_pause

    async def take(self, setup: TurnSetup, admitted: Admitted, record: TaskRecord) -> None:
        """Runs one turn; whatever way it ends, a defect a request of it met ends the runner loud.

        A session whose close cuts a request short can lose the defect from what leaves, so the turn's record
        joins it. A cancelled turn ends as the cancellation unless a defect was met or nobody cancelled it.
        """
        await ending_loud_when_cancelled(
            ending_with_recorded_defect(self._judge_turn(setup, admitted, record), record), record.loud_end
        )

    async def _judge_turn(self, setup: TurnSetup, admitted: Admitted, record: TaskRecord) -> None:
        """Runs one turn over its own session.

        A turn the gateway ends short is reported unless a defect was met; a failed relayed call and any other
        error go to the group. Every error of a cancelled turn ends as the cancellation.
        """
        sizing = _Sizing()
        try:
            await self._run_turn(setup, admitted, record, sizing)
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
                self._report_turn_end(stopped.code.value, None, sizing.sizes())
                await self._pause_when_budget_spent(stopped.code)
            # What left the session beside the stop (a late answer) still goes to the group, to be judged there.
            rest: list[BaseException] = [held] if held is not None else []
            rest.extend(leaf for leaf in leaves if leaf is not stopped)
            if len(rest) == 1:
                raise rest[0] from None
            if rest:
                raise BaseExceptionGroup("the turn's session failed", rest) from None

    async def _pause_when_budget_spent(self, code: RejectionCode | FailureCode) -> None:
        """A turn the gateway refused for a spent budget holds his one turn slot for the wait, so the lines that arrive
        meanwhile are taken into his history and make one turn after it; a turn ended otherwise ends the wait."""
        if code is not RejectionCode.BUDGET_SPENT:
            self._mind.budget_not_spent()
            return
        seconds = self._mind.budget_spent()
        self._report_pause(seconds)
        await self._clock.sleep(seconds)

    async def _run_turn(self, setup: TurnSetup, admitted: Admitted, record: TaskRecord, sizing: _Sizing) -> None:
        told = await self._told(admitted)
        token = await self._current_token()
        async with self._open_session(self._app, token.value) as session:
            try:
                await self._run_session_body(setup, admitted, told, session, record, sizing)
            except Exception as error:
                record.note(error)
                raise

    async def _told(self, admitted: Admitted) -> str:
        """The user message of the turn: the day is read at its admission and tells what the history does not hold."""
        earlier = await self._remembering.read_day(admitted.at, admitted.history)
        return turn_told(
            admitted.facts.name,
            self._app.mcp_url,
            admitted.place,
            admitted.occasion,
            admitted.history,
            earlier,
            admitted.at,
            aloud=admitted.aloud,
        )

    async def _run_session_body(
        self,
        setup: TurnSetup,
        admitted: Admitted,
        told: str,
        session: AppSession,
        record: TaskRecord,
        sizing: _Sizing,
    ) -> None:
        raise_if_cancelled()
        held = await self._hold_tools(setup, session, record, admitted)
        request = TurnRequest(
            agent=setup.persona.name,
            turn=Turn(system=setup.persona.prompt, messages=[TurnMessage(role=TurnRole.USER, text=told)]),
            roles=setup.persona.roles,
            tools=held.declarations,
            model=setup.persona.model,
            max_duration_seconds=setup.persona.max_duration_seconds,
            max_tool_calls=held.max_tool_calls,
        )
        sizing.request_built(held.declarations, told)
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        async with aclosing(
            setup.backend(tuple(sizing.measuring(each) for each in held.tools)).run(request, person)
        ) as events:
            if admitted.aloud:
                speaking = Speaking(
                    session,
                    self._app.mcp_url,
                    admitted.place,
                    self._clock,
                    setup.persona.max_duration_seconds,
                    partial(self._use_current_token, session),
                    lambda event, code: self._report(event, code, None),
                    self._mind.take,
                )
                final = await self._speak_along(events, speaking)
            else:
                final = await _final_event_of(events)
        if final is not None:
            self._report_turn_end(None, final.usage, sizing.sizes())
            self._mind.budget_not_spent()

    async def _speak_along(self, events: AsyncIterator[TurnEvent], speaking: Speaking) -> FinalEvent | None:
        """Reads the turn's events while the sentences they release are spoken beside them.

        A call is relayed only when the events are asked for again, so it goes out after every sentence before it was
        spoken. When the events end in an error, the sentence in flight is answered and entered before it is raised;
        only a cancel cuts the speech. The turn ends once the speech has, and its final event, if it had one, is
        returned.
        """
        async with asyncio.TaskGroup() as group:
            group.create_task(speaking.play())
            hearing = group.create_task(self._hear_events(events, speaking))
        return hearing.result()

    async def _hear_events(self, events: AsyncIterator[TurnEvent], speaking: Speaking) -> FinalEvent | None:
        final = None
        try:
            async for event in events:
                speaking.hear(event)
                if isinstance(event, ToolCallEvent):
                    await speaking.spoken()
                elif isinstance(event, FinalEvent):
                    final = event
        except Exception:
            await speaking.settle()
            raise
        speaking.end()
        return final

    async def _hold_tools(
        self, setup: TurnSetup, session: AppSession, record: TaskRecord, admitted: Admitted
    ) -> _HeldTools:
        """The tools of the place's app, of each app named in the admission and of each tools-only app that answers,
        in the configuration's order, each app counting its own calls.

        Of each app only the bound tools it lists now are held, none when it lists none. Another app whose listing
        fails is left out and reported once, and the turn holds the rest.
        """
        own = bound_declarations(await session.declared_tools(), setup.binding.tools)
        hold = _Hold(admitted.history)
        tools: list[HandTool] = []
        bounds: list[int] = []
        for app, binding in self._apps:
            calls = _Calls(binding.max_tool_calls)
            if app.mcp_url == self._app.mcp_url:
                lent = [HandTool(each, partial(self._relay, session, each.name, calls, hold)) for each in own]
            elif not app.listens or app.mcp_url in admitted.apps:
                lent = await self._list_elsewhere(app, binding, calls, hold, record)
            else:
                continue
            if lent:
                tools.extend(lent)
                bounds.append(calls.bound)
        return _HeldTools(tuple(tools), turn_call_bound(bounds))

    async def _list_elsewhere(
        self, app: AppTarget, binding: ToolBinding, calls: _Calls, hold: _Hold, record: TaskRecord
    ) -> list[HandTool] | None:
        """The hand tools of the bound tools another app lists, or ``None`` after the report of its failed listing."""
        declared = await self._ask_elsewhere(app, lambda session: session.declared_tools(), record)
        if isinstance(declared, CallFailure):
            self._report(_failure_event(app), failure_code(declared), app)
            return None
        listed = bound_declarations(declared, binding.tools)
        return [HandTool(each, partial(self._relay_elsewhere, app, each.name, calls, hold, record)) for each in listed]

    def _held(self, hold: _Hold) -> ToolOutcome | None:
        """The result of a call that is not sent, or ``None`` when the call goes on. It is asked before any await."""
        arrived = hold.holding(self._mind.history)
        if arrived is None:
            return None
        return ToolOutcome(held_told(arrived, self._clock.wall()), is_error=False)

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
            self._report(_failure_event(app), failure_code(answer), app)
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
        self, app: AppTarget, ask: Callable[[AppSession], Awaitable[Answer]], record: TaskRecord
    ) -> Answer | CallFailure:
        """Asks another app in a session of its own, under the token for its own scope.

        The session lives only for this request. A failure of the call, a session that died of its own child and
        an expiry of the app's call time come back as the failure; any other error is raised, and a turn whose
        place left stops. A defect recorded in the turn ends the request, raised from the failure.
        """
        try:
            return await ending_loud_when_cancelled(
                self._within_call_time(app, ask, record), unreachable_unless_cancelling
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

    async def _within_call_time[Answer](
        self, app: AppTarget, ask: Callable[[AppSession], Awaitable[Answer]], record: TaskRecord
    ) -> Answer:
        """The token, the session and the request, all under the call time of the app."""
        bound = asyncio.timeout(app.call_seconds)
        try:
            async with bound:
                granted = await self._tokens.fetch(self._agent, app.scope)
                async with self._open_session(app, granted.value) as session:
                    try:
                        return await ask(session)
                    except Exception as error:
                        record.note(error)
                        raise
        except TimeoutError:
            if not bound.expired():
                raise
            raise Unreachable("the app did not answer in time") from None
