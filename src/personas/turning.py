"""What one turn of an agent in an app does: the day read, the session, the tools held and how the turn ends."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.agent_file import TurnRequest
from agents.backend import TurnPerson, TurnStopped
from agents.tools import HandTool, ToolOutcome
from agents.turn import Turn, TurnMessage, TurnRole

from personas.decision import (
    TOOL_NOT_DECLARED_CODE,
    TURN_ROLE,
    Visit,
    bound_declarations,
    call_within_bound,
    decisive_failure,
    failure_code,
    turn_call_bound,
    turn_input,
)
from personas.failures import CallFailure, Unreachable, leaves_of, raise_if_cancelled
from personas.reporting import Event
from personas.task_record import (
    RelayFailed,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    failure_held_by,
    unreachable_unless_cancelling,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterable

    from agents.agent_file import AgentDefinition
    from agents.backend import AgentBackend
    from agents.member import PlaceEntry, PlaceRef, PlaceWindow
    from agents.tools import ToolDeclaration
    from agents.turn import JsonObject, Principal

    from personas.app_session import AppSession, SessionOpener
    from personas.config import AgentCredentials, AppTarget, ToolBinding
    from personas.decision import Token
    from personas.remembering import Remembering
    from personas.task_record import TaskRecord
    from personas.tokens import TokenFacts, TokenSource

type BackendFactory = Callable[[Iterable[HandTool]], AgentBackend]

_CALL_BOUND_TEXT: Final = "this app takes no more calls in this turn"
_APP_FAILED_TEXT: Final = "the app did not answer"


@dataclass(frozen=True)
class TurnSetup:
    """What an agent's turn in an app is made of: the persona, the tools bound and the backend that runs it."""

    persona: AgentDefinition
    binding: ToolBinding
    backend: BackendFactory


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


@dataclass(frozen=True)
class _HeldTools:
    """What a turn holds: the hand tools of every app that answered, and the calls the turn may make in all."""

    tools: tuple[HandTool, ...]
    max_tool_calls: int

    @property
    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(each.declaration for each in self.tools)


@dataclass(frozen=True)
class Admitted:
    """A turn decided on, with the facts it was decided from: the token's and what the place's window gave."""

    facts: TokenFacts
    place: PlaceRef
    window: PlaceWindow
    new: tuple[PlaceEntry, ...]
    missed: bool

    def groups(self) -> Coroutine[None, None, frozenset[str]]:
        """The library's GroupSource wants an awaitable; the groups are already known."""
        return asyncio.sleep(0, self.facts.groups)


class Turning:
    """Runs the turns of one agent in one app, each over a session of its own.

    It holds no state across turns. The token, its use on a session and the two reports are the loop's, handed over
    as callables and read at each call.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        tools_only: tuple[AppTarget, ...],
        tokens: TokenSource,
        open_session: SessionOpener,
        remembering: Remembering,
        current_token: Callable[[], Awaitable[Token]],
        use_current_token: Callable[[AppSession], Awaitable[None]],
        report: Callable[[Event, str | None, AppTarget | None], None],
        report_turn_end: Callable[[str], None],
    ) -> None:
        self._agent = agent
        self._app = app
        self._tools_only = tuple((each, each.binding) for each in tools_only if each.binding is not None)
        self._tokens = tokens
        self._open_session = open_session
        self._remembering = remembering
        self._current_token = current_token
        self._use_current_token = use_current_token
        self._report = report
        self._report_turn_end = report_turn_end

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
        try:
            await self._run_turn(setup, admitted, record)
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
                self._report_turn_end(stopped.code.value)
            # What left the session beside the stop (a late answer) still goes to the group, to be judged there.
            rest: list[BaseException] = [held] if held is not None else []
            rest.extend(leaf for leaf in leaves if leaf is not stopped)
            if len(rest) == 1:
                raise rest[0] from None
            if rest:
                raise BaseExceptionGroup("the turn's session failed", rest) from None

    async def _run_turn(self, setup: TurnSetup, admitted: Admitted, record: TaskRecord) -> None:
        earlier = await self._remembering.read_day(Visit(self._app.mcp_url, admitted.place, admitted.window.generation))
        told = turn_input(
            admitted.place, admitted.facts.name, admitted.window, admitted.new, missed=admitted.missed, earlier=earlier
        )
        token = await self._current_token()
        async with self._open_session(self._app, token.value) as session:
            try:
                await self._run_session_body(setup, admitted, told, session, record)
            except Exception as error:
                record.note(error)
                raise

    async def _run_session_body(
        self, setup: TurnSetup, admitted: Admitted, told: str, session: AppSession, record: TaskRecord
    ) -> None:
        raise_if_cancelled()
        held = await self._hold_tools(setup, session, record)
        if held is None:
            self._report_turn_end(TOOL_NOT_DECLARED_CODE)
            return
        request = TurnRequest(
            agent=setup.persona.name,
            turn=Turn(system=setup.persona.prompt, messages=[TurnMessage(role=TurnRole.USER, text=told)]),
            roles=setup.persona.roles,
            tools=held.declarations,
            model=setup.persona.model,
            max_duration_seconds=setup.persona.max_duration_seconds,
            max_tool_calls=held.max_tool_calls,
        )
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        async with aclosing(setup.backend(held.tools).run(request, person)) as events:
            async for _ in events:
                continue

    async def _hold_tools(self, setup: TurnSetup, session: AppSession, record: TaskRecord) -> _HeldTools | None:
        """The tools of the place's app and of each tools-only app that answers, each app counting its own calls.

        ``None`` when the place's app declares a bound tool no more. A tools-only app that fails or declares one
        no more is left out, and the turn holds the rest.
        """
        own = bound_declarations(await session.declared_tools(), setup.binding.tools)
        if own is None:
            return None
        own_calls = _Calls(setup.binding.max_tool_calls)
        tools = [HandTool(each, partial(self._relay, session, each.name, own_calls)) for each in own]
        bounds = [own_calls.bound]
        failed_apps: set[str] = set()
        for app, binding in self._tools_only:
            declared = await self._ask_tools_only(app, lambda session: session.declared_tools(), record)
            if isinstance(declared, CallFailure):
                self._report_tools_only_failure(app, failed_apps, failure_code(declared))
                continue
            listed = bound_declarations(declared, binding.tools)
            if listed is None:
                self._report_tools_only_failure(app, failed_apps, TOOL_NOT_DECLARED_CODE)
                continue
            calls = _Calls(binding.max_tool_calls)
            tools.extend(
                HandTool(each, partial(self._relay_tools_only, app, each.name, calls, failed_apps, record))
                for each in listed
            )
            bounds.append(calls.bound)
        return _HeldTools(tuple(tools), turn_call_bound(bounds))

    async def _relay(
        self, session: AppSession, name: str, calls: _Calls, _principal: Principal, arguments: JsonObject
    ) -> ToolOutcome:
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        try:
            await self._use_current_token(session)
            return await session.invoke(name, arguments)
        except Exception as error:
            raise RelayFailed(error) from None

    async def _relay_tools_only(
        self,
        app: AppTarget,
        name: str,
        calls: _Calls,
        failed_apps: set[str],
        record: TaskRecord,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        try:
            answer = await self._ask_tools_only(app, lambda session: session.invoke(name, arguments), record)
        except Exception as error:
            raise RelayFailed(error) from None
        if isinstance(answer, CallFailure):
            self._report_tools_only_failure(app, failed_apps, failure_code(answer))
            return ToolOutcome(_APP_FAILED_TEXT, is_error=True)
        return answer

    async def _ask_tools_only[Answer](
        self, app: AppTarget, ask: Callable[[AppSession], Awaitable[Answer]], record: TaskRecord
    ) -> Answer | CallFailure:
        """Asks a tools-only app in a session of its own, under the token for its own scope.

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

    def _report_tools_only_failure(self, app: AppTarget, failed_apps: set[str], code: str) -> None:
        """One report per app and turn."""
        if app.mcp_url not in failed_apps:
            failed_apps.add(app.mcp_url)
            self._report(Event.TOOLS_ONLY_APP_FAILED, code, app)
