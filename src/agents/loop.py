"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Final, NoReturn

from agent_runtime_client.agents import TurnRequest
from agent_runtime_client.backend import TurnPerson, TurnStopKind, TurnStopped
from agent_runtime_client.tools import HandTool, ToolOutcome
from agent_runtime_client.turn import FailureCode, Principal, Turn, TurnMessage, TurnRole

from agents.decision import (
    SUSPENDED_RETRY_SECONDS,
    TOOL_NOT_DECLARED_CODE,
    TURN_CANCELLED_CODE,
    TURN_ROLE,
    Joined,
    Memory,
    Reaction,
    Suspend,
    Token,
    TurnCap,
    accepts_for,
    active_call_places,
    addressed,
    after_refusal,
    backoff_delay,
    backoff_reason,
    bound_declarations,
    decisive_failure,
    forget_gone,
    is_defect,
    perceive,
    reaction_to,
    renewal_due,
    reserve_turn,
    sets_where_reading_starts,
    sync_checkpoints,
    turn_input,
    turn_invalid,
)
from agents.failures import CallFailure, leaves_of
from agents.reporting import Event, Reporter, StateChange
from agents.tokens import TokenFacts, facts_of

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, Iterable

    from agent_runtime_client.agents import AgentDefinition
    from agent_runtime_client.backend import AgentBackend
    from agent_runtime_client.member import PlaceRef, Standing
    from agent_runtime_client.turn import JsonObject

    from agents.app_session import AppSession, SessionOpener
    from agents.clock import Clock
    from agents.config import AgentCredentials, AppTarget, Config, ToolBinding
    from agents.decision import Checkpoint
    from agents.readiness import ReadyMarker
    from agents.tokens import TokenSource

type BackendFactory = Callable[[Iterable[HandTool]], AgentBackend]

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
    """Stops a task whose cancellation was swallowed on the way, as by a fetch or an SDK task group.

    The task still counts the request; without this the loop would go on instead of ending.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


def _defect_of(task: asyncio.Task[object]) -> BaseException | None:
    """The error a finished task raised when a leaf of it is a defect; reading it marks it retrieved."""
    if not task.done() or task.cancelled():
        return None
    error = task.exception()
    if error is None or not any(is_defect(leaf) for leaf in leaves_of(error)):
        return None
    return error


class UnrequestedCancellationError(Exception):
    """A task ended cancelled while the task that should have been cancelled was not."""


def _unrequested_unless_cancelling(asked: asyncio.Task[object]) -> UnrequestedCancellationError | None:
    return None if asked.cancelling() else UnrequestedCancellationError()


async def _ending_loud_when_cancelled[T](work: Awaitable[T], loud_end: Callable[[], Exception | None]) -> T:
    """Runs work; when it is cancelled and ``loud_end`` gives an error, that error ends it, caused by the cancellation.

    Every cancellation is re-raised where it is caught; the loud end is decided in the ``finally``.
    """
    cancelled: asyncio.CancelledError | None = None
    try:
        return await work
    except asyncio.CancelledError as error:
        cancelled = error
        raise
    finally:
        if cancelled is not None and (loud := loud_end()) is not None:
            raise loud from cancelled


class _RelayFailed(TurnStopped):
    """The stop a failed relayed call gives the backend, which ends the turn at once; it holds the failure."""

    def __init__(self, failure: Exception) -> None:
        super().__init__(TurnStopKind.FAILED, FailureCode.PROVIDER_ERROR)
        self.failure = failure


@dataclass(frozen=True)
class _TurnSetup:
    """What an agent's turn in an app is made of: the persona, the tools bound and the backend that runs it."""

    persona: AgentDefinition
    binding: ToolBinding
    backend: BackendFactory
    cap: TurnCap


@dataclass
class TurnCounter:
    """The start times of one agent's recent turns, shared by all its loops and kept for the process."""

    started: tuple[float, ...] = ()


@dataclass(frozen=True)
class ActiveTurn:
    """The running turn: its task and place, the loop's request to end it, its end report and a relayed defect."""

    task: asyncio.Task[None]
    place: PlaceRef
    cancel_requested: bool = False
    end_reported: bool = False
    relay_defect: Exception | None = None


@dataclass(frozen=True)
class _Admitted:
    """A turn decided on, with the facts it was decided from."""

    facts: TokenFacts
    input: str

    def groups(self) -> Coroutine[None, None, frozenset[str]]:
        """The library's GroupSource wants an awaitable; the groups are already known."""
        return asyncio.sleep(0, self.facts.groups)


class AgentLoop:
    """Keeps one stream of changes open for one agent in one app, accepts what it is invited to and answers.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next
    retry, and resumes once a read answers again. A network failure, a lost stream or a 5xx backs off,
    logged once until a read answers. In an app that binds tools, the lines of the agent's call places are
    read too, and one addressed to him starts a turn; at most one turn runs per loop, over its own session,
    and a failure of the turn's calls goes into the one reaction of the loop.
    """

    def __init__(self, agent: AgentCredentials, app: AppTarget, boundaries: Boundaries, counter: TurnCounter) -> None:
        self._agent = agent
        self._counter = counter
        self._app = app
        self._boundaries = boundaries
        self._token: Token | None = None
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None
        self._backing_off = False
        self._checkpoints: dict[PlaceRef, Checkpoint] = {}
        self._turn: ActiveTurn | None = None
        self._turn_ended = asyncio.Event()
        persona, binding, backend, cap = agent.persona, app.binding, boundaries.backend, agent.turn_cap
        self._turn_setup = (
            None
            if persona is None or binding is None or backend is None or cap is None
            else _TurnSetup(persona, binding, backend, cap)
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
        async with (
            asyncio.TaskGroup() as turns,
            self._boundaries.open_session(self._app, token.value) as session,
        ):
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
            # A defect of the change task comes out of aclose, so it ends the runner even when it lost the race to
            # the body's error, which its traceback does not keep.
            if (defect := _defect_of(change)) is not None:
                raise defect

    async def _reconcile(self, session: AppSession, turns: asyncio.TaskGroup) -> None:
        """Reads the standing as it is now and acts on it; a wake says only that it may have changed."""
        standings = await self._take_standing(session)
        if await self._answer_addressed_lines(session, turns, standings):
            # A window the app lost dropped its place's checkpoint: the standing says where the place stands now.
            await self._take_standing(session)
        self._served()

    async def _take_standing(self, session: AppSession) -> tuple[Standing, ...]:
        """Reads the standing, keeps the checkpoints in line with it and accepts what it invites to."""
        await self._use_current_token(session)
        standing = await session.read_standing()
        self._resume_if_unwell()
        self._boundaries.ready.announce()
        self._memory = forget_gone(self._memory, standing.standings)
        self._checkpoints = sync_checkpoints(self._checkpoints, standing.standings)
        self._cancel_turn_if_its_place_left(standing.standings)
        await self._accept_invitations(session, standing.standings)
        return standing.standings

    def _cancel_turn_if_its_place_left(self, standings: tuple[Standing, ...]) -> None:
        """A turn ends with its place; its slot stays taken until its task has ended."""
        turn = self._turn
        if turn is not None and not turn.cancel_requested and turn_invalid(turn.place, standings):
            self._turn = replace(turn, cancel_requested=True)
            turn.task.cancel()

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
                self._checkpoints[accept.place] = Joined()
                self._report(Event.JOINED)
                continue
            self._memory, is_news = after_refusal(self._memory, accept, refusal)
            if is_news:
                self._report(Event.REFUSED, refusal.value)

    async def _answer_addressed_lines(
        self, session: AppSession, turns: asyncio.TaskGroup, standings: tuple[Standing, ...]
    ) -> bool:
        """Reads the active call places' windows; a line addressed to the agent starts the one turn.

        While a turn runs it reads only the windows that set where reading starts, which take no line.
        It says whether a window was lost, which dropped its place's checkpoint.
        """
        if self._turn_setup is None:
            return False
        places = active_call_places(standings)
        if self._turn is not None:
            places = tuple(place for place in places if sets_where_reading_starts(self._checkpoints[place]))
        facts = facts_of((await self._current_token()).value)
        lost = False
        for place in places:
            await self._use_current_token(session)
            window = await session.read_place(place)
            if window is None:
                del self._checkpoints[place]
                lost = True
                continue
            perception = perceive(window, self._checkpoints[place])
            # The checkpoint moves, the turn is reserved and created and its callback added with no await
            # between, so a line starts at most one turn and a capped line is never answered later.
            self._checkpoints[place] = perception.checkpoint
            if not any(addressed(line, facts.name) for line in perception.new):
                continue
            started = reserve_turn(self._counter.started, self._boundaries.clock.now(), self._turn_setup.cap)
            if started is None:
                self._report(Event.CAPPED)
                continue
            self._counter.started = started
            admitted = _Admitted(facts, turn_input(place, facts.name, window, perception.new, missed=perception.missed))
            self._start_turn(turns, self._turn_setup, place, admitted)
            return lost
        return lost

    def _start_turn(self, turns: asyncio.TaskGroup, setup: _TurnSetup, place: PlaceRef, admitted: _Admitted) -> None:
        task = turns.create_task(self._take_turn(setup, admitted))
        self._turn = ActiveTurn(task, place)
        task.add_done_callback(self._turn_over)

    def _turn_over(self, task: asyncio.Task[None]) -> None:
        turn, self._turn = self._turn, None
        if turn is not None and turn.cancel_requested and task.cancelled() and not turn.end_reported:
            self._report(Event.TURN_ENDED, TURN_CANCELLED_CODE)
        self._turn_ended.set()

    def _report_turn_end(self, code: str) -> None:
        """A turn reports its end at most once; the record tells the loop, which then reports no cancel."""
        assert self._turn is not None  # noqa: S101 - only the running turn's task reports its end
        self._turn = replace(self._turn, end_reported=True)
        self._report(Event.TURN_ENDED, code)

    async def _take_turn(self, setup: _TurnSetup, admitted: _Admitted) -> None:
        """Runs one turn; a cancelled one ends the runner loud with a defect it met, or when nobody cancelled it."""
        task = asyncio.current_task()
        assert task is not None  # noqa: S101 - a turn runs inside a task
        await _ending_loud_when_cancelled(self._judge_turn(setup, admitted), partial(self._loud_end_of, task))

    def _loud_end_of(self, task: asyncio.Task[object]) -> Exception | None:
        """What ends the runner when the turn's session has left and the turn is cancelled.

        A defect a relayed call raised comes first: the cancellation can reach the session's teardown after it
        and leave the defect only as the cancellation's context. A turn's own count is above 0 on a cancel by
        the loop, the group or the runner; it is 0 when only anyio's own scope cancellation ended its session.
        """
        assert self._turn is not None  # noqa: S101 - the turn's record lives until its task has ended
        return self._turn.relay_defect or _unrequested_unless_cancelling(task)

    async def _judge_turn(self, setup: _TurnSetup, admitted: _Admitted) -> None:
        """Runs one turn over its own session.

        A turn the gateway ends short is reported; a failed relayed call and any other error go to the group.
        A cancelled turn hands the group only a defect, so a failure of its call and late answers end as the
        cancellation.
        """
        try:
            await self._run_turn(setup, admitted)
        except Exception as error:
            leaves = list(leaves_of(error))
            relay_stop = next((leaf for leaf in leaves if isinstance(leaf, _RelayFailed)), None)
            stopped = relay_stop or next((leaf for leaf in leaves if isinstance(leaf, TurnStopped)), None)
            # The stop only carries the failure out of the backend; the failure goes to the group, never the stop.
            # What left the session beside the stop (a late answer) goes there too, to be judged there.
            rest: list[BaseException] = [relay_stop.failure] if relay_stop is not None else []
            rest.extend(leaf for leaf in leaves if leaf is not stopped)
            if not any(is_defect(leaf) for leaf in rest):
                _raise_if_cancelled()
            if stopped is None:
                raise
            if relay_stop is None:
                self._report_turn_end(stopped.code.value)
            if len(rest) == 1:
                raise rest[0] from None
            if rest:
                raise BaseExceptionGroup("the turn's session failed", rest) from None

    async def _run_turn(self, setup: _TurnSetup, admitted: _Admitted) -> None:
        token = await self._current_token()
        async with self._boundaries.open_session(self._app, token.value) as session:
            _raise_if_cancelled()
            declarations = bound_declarations(await session.declared_tools(), setup.binding.tools)
            if declarations is None:
                self._report_turn_end(TOOL_NOT_DECLARED_CODE)
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
            person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
            async with aclosing(setup.backend(tools).run(request, person)) as events:
                async for _ in events:
                    continue

    async def _relay(self, session: AppSession, name: str, _principal: Principal, arguments: JsonObject) -> ToolOutcome:
        try:
            await self._use_current_token(session)
            return await session.invoke(name, arguments)
        except Exception as error:
            if is_defect(error):
                assert self._turn is not None  # noqa: S101 - only the running turn relays
                self._turn = replace(self._turn, relay_defect=error)
            raise _RelayFailed(error) from None

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


async def _run_loop(loop: AgentLoop, runner: asyncio.Task[object]) -> None:
    """Runs a loop; one that ends cancelled while its runner is not being cancelled ends the runner loud.

    The runner's own count says so: it is above zero on a cancellation of the runner and while its group aborts
    after a loop's loud end. A loop task's count is not read, a cancelled lifecycle session lowers it.
    """
    return await _ending_loud_when_cancelled(loop.run(), partial(_unrequested_unless_cancelling, runner))


async def run_all(config: Config, boundaries: Boundaries) -> NoReturn:
    """One loop for every agent and every app; a loop never ends on its own."""
    runner = asyncio.current_task()
    assert runner is not None  # noqa: S101 - run_all runs inside a task
    async with asyncio.TaskGroup() as group:
        for agent in config.agents:
            counter = TurnCounter()
            for app in config.apps:
                group.create_task(_run_loop(AgentLoop(agent, app, boundaries, counter), runner))
    raise AssertionError("a loop ended")  # pragma: no cover
