"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final, NoReturn

from agent_runtime_client.agents import TurnRequest
from agent_runtime_client.backend import TurnPerson, TurnStopped
from agent_runtime_client.tools import HandTool, ToolOutcome
from agent_runtime_client.turn import AgentRole, Principal, Turn, TurnMessage, TurnRole

from agents.decision import (
    SUSPENDED_RETRY_SECONDS,
    TOOL_NOT_DECLARED_CODE,
    Memory,
    Reaction,
    Seen,
    Suspend,
    Token,
    accepts_for,
    active_call_places,
    addressed,
    after_refusal,
    backoff_delay,
    backoff_reason,
    bound_declarations,
    decisive_failure,
    forget_gone,
    reaction_to,
    renewal_due,
    turn_input,
    unread,
)
from agents.failures import CallFailure, leaves_of
from agents.reporting import Event, Reporter, StateChange
from agents.tokens import TokenFacts, facts_of

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable

    from agent_runtime_client.agents import AgentDefinition
    from agent_runtime_client.backend import AgentBackend
    from agent_runtime_client.member import PlaceRef, Standing
    from agent_runtime_client.turn import JsonObject

    from agents.app_session import AppSession, SessionOpener
    from agents.clock import Clock
    from agents.config import AgentCredentials, AppTarget, Config, ToolBinding
    from agents.readiness import ReadyMarker
    from agents.tokens import TokenSource

type BackendFactory = Callable[[Iterable[HandTool]], AgentBackend]

TURN_ROLE: Final[AgentRole] = "user"
"""The app role a turn acts under: an agent holds no admin right."""
_STREAM_ENDED: Final = object()


@dataclass(frozen=True)
class Boundaries:
    """Everything a loop touches outside itself."""

    tokens: TokenSource
    open_session: SessionOpener
    clock: Clock
    report: Reporter
    ready: ReadyMarker
    backend: BackendFactory | None = None


def _raise_if_cancelled() -> None:
    """Stops a loop whose cancellation an SDK task group swallowed while a session was opening.

    The task still counts the request; without this the loop would open the next session instead of ending.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


@dataclass(frozen=True)
class _TurnSetup:
    """What an agent's turn in an app is made of: the persona, the tools bound and the backend that runs it."""

    persona: AgentDefinition
    binding: ToolBinding
    backend: BackendFactory


@dataclass(frozen=True)
class _Admitted:
    """A turn decided on, with the facts it was decided from."""

    facts: TokenFacts
    input: str


async def _granted_groups(facts: TokenFacts) -> frozenset[str]:
    return facts.groups


class AgentLoop:
    """Keeps one stream of changes open for one agent in one app, accepts what it is invited to and answers.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next
    retry, and resumes once a read answers again. A network failure, a lost stream or a 5xx backs off,
    logged once until a read answers. In an app that binds tools, the lines of the agent's call places are
    read too, and one addressed to him starts a turn; at most one turn runs per loop, over its own session,
    and a failure of the turn's calls goes into the one reaction of the loop.
    """

    def __init__(self, agent: AgentCredentials, app: AppTarget, boundaries: Boundaries) -> None:
        self._agent = agent
        self._app = app
        self._boundaries = boundaries
        self._token: Token | None = None
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None
        self._backing_off = False
        self._seen: dict[PlaceRef, Seen] = {}
        self._turn: asyncio.Task[None] | None = None
        self._turn_ended = asyncio.Event()
        persona, binding, backend = agent.persona, app.binding, boundaries.backend
        self._turn_setup = (
            None if persona is None or binding is None or backend is None else _TurnSetup(persona, binding, backend)
        )

    async def run(self) -> NoReturn:
        self._report(Event.STARTED)
        while True:
            _raise_if_cancelled()
            try:
                await self._serve()
            except Exception as error:
                failure = decisive_failure(tuple(leaves_of(error)), already_renewed=self._renewed_after_401)
                if failure is None:
                    raise
                await self._react(failure)

    async def _serve(self) -> NoReturn:
        token = await self._current_token()
        async with asyncio.TaskGroup() as turns, self._boundaries.open_session(self._app, token.value) as session:
            while True:
                await self._use_current_token(session)
                async with session.changes() as changes, aclosing(self._wakes(changes)) as wakes:
                    await self._reconcile(session, turns)
                    async for _ in wakes:
                        await self._reconcile(session, turns)

    async def _wakes(self, changes: AsyncIterator[object]) -> AsyncGenerator[None]:
        """A wake for each change the app signals and for each end of a turn; it ends when the stream does."""
        change = asyncio.ensure_future(anext(changes, _STREAM_ENDED))
        ended = asyncio.ensure_future(self._turn_ended.wait())
        try:
            while True:
                await asyncio.wait({change, ended}, return_when=asyncio.FIRST_COMPLETED)
                if ended.done():
                    self._turn_ended.clear()
                    ended = asyncio.ensure_future(self._turn_ended.wait())
                    yield
                if change.done():
                    if change.result() is _STREAM_ENDED:
                        return
                    change = asyncio.ensure_future(anext(changes, _STREAM_ENDED))
                    yield
        finally:
            change.cancel()
            ended.cancel()

    async def _reconcile(self, session: AppSession, turns: asyncio.TaskGroup) -> None:
        """Reads the standing as it is now and acts on it; a wake says only that it may have changed."""
        await self._use_current_token(session)
        standing = await session.read_standing()
        self._resume_if_unwell()
        self._boundaries.ready.announce()
        self._memory = forget_gone(self._memory, standing.standings)
        await self._accept_invitations(session, standing.standings)
        await self._answer_addressed_lines(session, turns, standing.standings)
        self._served()

    async def _use_current_token(self, session: AppSession) -> None:
        session.use_token((await self._current_token()).value)

    async def _current_token(self) -> Token:
        if self._token is None or renewal_due(self._token, self._boundaries.clock.now()):
            granted = await self._boundaries.tokens.fetch(self._agent, self._app.scope)
            expires_at = self._boundaries.clock.now() + granted.lifetime_seconds
            self._token = Token(granted.value, expires_at)
        return self._token

    def _served(self) -> None:
        """A read and every accept it led to went through: the failure counters start over."""
        self._failures = 0
        self._renewed_after_401 = False

    def _resume_if_unwell(self) -> None:
        """A read answered: the app is reachable and accepts the token, whatever an accept then does."""
        if self._suspension is not None or self._backing_off:
            self._suspension = None
            self._backing_off = False
            self._report(Event.RESUMED)

    async def _accept_invitations(self, session: AppSession, standings: tuple[Standing, ...]) -> None:
        for accept in accepts_for(standings, self._memory):
            await self._use_current_token(session)
            refusal = await session.accept(accept)
            if refusal is None:
                self._seen.pop(accept.place, None)
                self._report(Event.JOINED)
                continue
            self._memory, is_news = after_refusal(self._memory, accept, refusal)
            if is_news:
                self._report(Event.REFUSED, refusal.value)

    async def _answer_addressed_lines(
        self, session: AppSession, turns: asyncio.TaskGroup, standings: tuple[Standing, ...]
    ) -> None:
        """Reads the active call places' windows; a line addressed to the agent starts the one turn."""
        if self._turn_setup is None or self._turn is not None:
            return
        facts = facts_of((await self._current_token()).value)
        for place in active_call_places(standings):
            await self._use_current_token(session)
            window = await session.read_place(place)
            if window is None:
                continue
            new = unread(window, self._seen.get(place))
            if not new:
                continue
            # The read position moves, the turn is created and its callback added with no await between,
            # so a line starts at most one turn.
            self._seen[place] = Seen(window.generation, new[-1].seq)
            if any(addressed(line, facts.name) for line in new):
                self._start_turn(turns, self._turn_setup, _Admitted(facts, turn_input(place, facts.name, window, new)))
                return

    def _start_turn(self, turns: asyncio.TaskGroup, setup: _TurnSetup, admitted: _Admitted) -> None:
        self._turn = turns.create_task(self._take_turn(setup, admitted))
        self._turn.add_done_callback(self._turn_over)

    def _turn_over(self, _turn: asyncio.Task[None]) -> None:
        self._turn = None
        self._turn_ended.set()

    async def _take_turn(self, setup: _TurnSetup, admitted: _Admitted) -> None:
        """Runs one turn over its own session; a turn that ends short is reported, any other error is the group's."""
        try:
            await self._run_turn(setup, admitted)
        except TurnStopped as stopped:
            _raise_if_cancelled()
            self._report(Event.TURN_ENDED, stopped.code.value)
        except Exception:
            _raise_if_cancelled()
            raise

    async def _run_turn(self, setup: _TurnSetup, admitted: _Admitted) -> None:
        token = await self._current_token()
        async with self._boundaries.open_session(self._app, token.value) as session:
            _raise_if_cancelled()
            declarations = bound_declarations(await session.declared_tools(), setup.binding.tools)
            if declarations is None:
                self._report(Event.TURN_ENDED, TOOL_NOT_DECLARED_CODE)
                return
            tools = tuple(HandTool(each, partial(self._relay, session, each.name)) for each in declarations)
            request = TurnRequest(
                agent=setup.persona.name,
                turn=Turn(system=setup.persona.prompt, messages=[TurnMessage(role=TurnRole.USER, text=admitted.input)]),
                roles=setup.persona.roles,
                tools=declarations,
                model=setup.persona.model,
                max_duration_seconds=setup.persona.max_duration_seconds,
                max_tool_calls=setup.binding.max_tool_calls,
            )
            person = TurnPerson(admitted.facts.principal, TURN_ROLE, partial(_granted_groups, admitted.facts))
            async with aclosing(setup.backend(tools).run(request, person)) as events:
                async for _ in events:
                    pass

    async def _relay(self, session: AppSession, name: str, _principal: Principal, arguments: JsonObject) -> ToolOutcome:
        await self._use_current_token(session)
        return await session.invoke(name, arguments)

    async def _react(self, failure: CallFailure) -> None:
        match reaction_to(failure, already_renewed=self._renewed_after_401):
            case Reaction.RENEW:
                self._renewed_after_401 = True
                self._token = None
            case Suspend(code=code):
                await self._suspend(code)
            case Reaction.BACKOFF:
                await self._back_off(failure)

    async def _suspend(self, code: str) -> None:
        if self._suspension != code:
            self._suspension = code
            self._report(Event.SUSPENDED, code)
        self._backing_off = False
        self._token = None
        await self._boundaries.clock.sleep(SUSPENDED_RETRY_SECONDS)

    async def _back_off(self, failure: CallFailure) -> None:
        if not self._backing_off:
            self._backing_off = True
            self._report(Event.BACKOFF, backoff_reason(failure))
        self._failures += 1
        await self._boundaries.clock.sleep(backoff_delay(self._failures))

    def _report(self, event: Event, code: str | None = None) -> None:
        self._boundaries.report(StateChange(event, self._agent.client_id, self._app.mcp_url, code))


async def run_all(config: Config, boundaries: Boundaries) -> NoReturn:
    """One loop for every agent and every app; a loop never ends on its own."""
    async with asyncio.TaskGroup() as group:
        for agent in config.agents:
            for app in config.apps:
                group.create_task(AgentLoop(agent, app, boundaries).run())
    raise AssertionError("a loop ended")  # pragma: no cover
