"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from agents.agent_file import TurnRequest
from agents.backend import TurnPerson, TurnStopped
from agents.tools import HandTool, ToolOutcome
from agents.turn import Principal, Turn, TurnMessage, TurnRole
from agents.voice.loop import run_voice_loop
from agents.voice.wake import addressed_or_asked

from personas.decision import (
    SUSPENDED_RETRY_SECONDS,
    TOOL_NOT_DECLARED_CODE,
    TURN_CANCELLED_CODE,
    TURN_ROLE,
    Joined,
    Memory,
    Perception,
    Reaction,
    Suspend,
    Token,
    Visit,
    WindowRead,
    accepts_for,
    active_call_places,
    addressed,
    after_refusal,
    backoff_delay,
    bound_declarations,
    call_within_bound,
    decisive_failure,
    failure_code,
    forget_gone,
    is_defect,
    is_voice_app,
    perceive,
    reaction_to,
    renewal_due,
    sets_where_reading_starts,
    sync_checkpoints,
    turn_call_bound,
    turn_input,
    turn_invalid,
    voice_context,
)
from personas.failures import CallFailure, Unreachable, leaves_of, task_is_being_cancelled
from personas.remembering import LineStore, Remembering
from personas.reporting import Event, Reporter, StateChange
from personas.task_record import (
    RelayFailed,
    TaskRecord,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    failure_held_by,
    unreachable_unless_cancelling,
    unrequested_unless_cancelling,
)
from personas.tokens import TokenFacts, facts_of
from personas.voice import ActiveVoice, Hearing, SpeakingVoice, reporting_decisions

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, Iterable

    from agents.agent_file import AgentDefinition
    from agents.backend import AgentBackend
    from agents.member import PlaceEntry, PlaceRef, PlaceWindow, Standing
    from agents.tools import ToolDeclaration
    from agents.turn import JsonObject

    from personas.app_session import AppSession, SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget, Config, ToolBinding
    from personas.decision import Checkpoint, TodayBounds
    from personas.readiness import ReadyMarker
    from personas.tokens import TokenSource

type BackendFactory = Callable[[Iterable[HandTool]], AgentBackend]

_STREAM_ENDED: Final = object()
_CALL_BOUND_TEXT: Final = "this app takes no more calls in this turn"
_APP_FAILED_TEXT: Final = "the app did not answer"


@dataclass(frozen=True)
class Boundaries:
    """Everything a loop touches outside itself."""

    tokens: TokenSource
    open_session: SessionOpener
    clock: Clock
    report: Reporter
    ready: ReadyMarker
    backend: BackendFactory | None = None
    lines: LineStore | None = None


def _raise_if_cancelled() -> None:
    """Stops a task whose cancellation was swallowed on the way, as by a fetch or an SDK task group.

    The task still counts the request; without this the loop would go on instead of ending.
    """
    if task_is_being_cancelled():
        raise asyncio.CancelledError


def _defect_of(task: asyncio.Task[object]) -> BaseException | None:
    """The error a finished task raised when a leaf of it is a defect; reading it marks it retrieved."""
    if not task.done() or task.cancelled():
        return None
    error = task.exception()
    if error is None or not any(is_defect(leaf) for leaf in leaves_of(error)):
        return None
    return error


@dataclass(frozen=True)
class _TurnSetup:
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
class ActiveTurn:
    """The running turn: its task and place, the loop's request to end it and its end report."""

    task: asyncio.Task[None]
    place: PlaceRef
    cancel_requested: bool = False
    end_reported: bool = False


@dataclass(frozen=True)
class _Admitted:
    """A turn decided on, with the facts it was decided from: the token's and what the place's window gave."""

    facts: TokenFacts
    place: PlaceRef
    window: PlaceWindow
    new: tuple[PlaceEntry, ...]
    missed: bool

    def groups(self) -> Coroutine[None, None, frozenset[str]]:
        """The library's GroupSource wants an awaitable; the groups are already known."""
        return asyncio.sleep(0, self.facts.groups)


class AgentLoop:
    """Keeps one stream of changes open for one agent in one app, accepts what it is invited to and answers.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next
    retry, and resumes once a read answers again. A network failure, a lost stream or a 5xx backs off,
    logged once until a read answers. In an app that binds tools, the lines of the agent's call places are
    read too, and one addressed to him starts a turn; at most one turn runs per loop, over its own session,
    and a failure of the turn's calls to this app goes into the one reaction of the loop. The turn also holds the
    bound tools of the tools-only apps: each request to one runs in a session of its own under the token for its
    own scope, and its failure never reaches the reaction. In an app that declares ``speak``, for an agent whose
    persona has a voice section, each active call place instead runs one voice loop, which hears every line the place
    gives and speaks through ``speak``.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        boundaries: Boundaries,
        tools_only: tuple[AppTarget, ...] = (),
        today: TodayBounds | None = None,
    ) -> None:
        self._agent = agent
        self._app = app
        self._tools_only = tuple((each, each.binding) for each in tools_only if each.binding is not None)
        self._boundaries = boundaries
        self._remembering = Remembering(boundaries.lines, today, agent.client_id, boundaries.clock, self._report)
        self._token: Token | None = None
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None
        self._backing_off = False
        self._checkpoints: dict[PlaceRef, Checkpoint] = {}
        self._turn: ActiveTurn | None = None
        self._turn_ended = asyncio.Event()
        self._voices: dict[PlaceRef, ActiveVoice] = {}
        self._voice_app = False
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
        async with (
            asyncio.TaskGroup() as turns,
            self._boundaries.open_session(self._app, token.value) as session,
        ):
            self._voices = {}
            self._voice_app = await self._is_voice_app(session)
            while True:
                await self._use_current_token(session)
                async with session.changes() as changes, aclosing(self._wakes(changes)) as wakes:
                    await self._reconcile(session, turns)
                    async for _ in wakes:
                        await self._reconcile(session, turns)

    async def _is_voice_app(self, session: AppSession) -> bool:
        """Whether this app's places run the voice loop, decided once for the lifecycle session."""
        has_voice_section = self._turn_setup is not None and self._turn_setup.persona.voice_window_bytes is not None
        declared = await session.declared_tools() if has_voice_section else ()
        return is_voice_app(has_voice_section, declared)

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
        """A turn and a voice loop end with their place; a turn's slot stays taken until its task has ended."""
        for place in [each for each in self._voices if turn_invalid(each, standings)]:
            self._voices.pop(place).task.cancel()
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
            read_at = self._boundaries.clock.wall()
            turn_started = self._take_lines(turns, self._turn_setup, place, window, facts)
            await self._remembering.keep(WindowRead(self._agent.client_id, self._app.mcp_url, place, window, read_at))
            if turn_started:
                return lost
        return lost

    def _take_lines(
        self, turns: asyncio.TaskGroup, setup: _TurnSetup, place: PlaceRef, window: PlaceWindow, facts: TokenFacts
    ) -> bool:
        """Moves the place's checkpoint and takes the new lines: a voice loop hears all of them, and else one
        addressed to the agent starts a turn, which this says. No await stands in it."""
        perception = perceive(window, self._checkpoints[place])
        self._checkpoints[place] = perception.checkpoint
        if not self._voice_app:
            return self._start_turn_if_addressed(turns, setup, place, window, facts, perception)
        voice = self._voices.get(place)
        if voice is None or voice.generation != window.generation:
            if voice is not None:
                voice.task.cancel()
            voice = self._start_voice(turns, setup, Visit(self._app.mcp_url, place, window.generation), facts)
        for line in perception.new:
            voice.hearing.put(line)
        return False

    def _start_turn_if_addressed(
        self,
        turns: asyncio.TaskGroup,
        setup: _TurnSetup,
        place: PlaceRef,
        window: PlaceWindow,
        facts: TokenFacts,
        perception: Perception,
    ) -> bool:
        """Starts the turn when a new line addresses the agent. A line starts at most one turn."""
        if not any(addressed(line, facts.name) for line in perception.new):
            return False
        admitted = _Admitted(facts, place, window, perception.new, perception.missed)
        self._start_turn(turns, setup, place, admitted)
        return True

    def _start_turn(self, turns: asyncio.TaskGroup, setup: _TurnSetup, place: PlaceRef, admitted: _Admitted) -> None:
        task = turns.create_task(self._take_turn(setup, admitted, TaskRecord()))
        self._turn = ActiveTurn(task, place)
        task.add_done_callback(self._turn_over)

    def _start_voice(self, turns: asyncio.TaskGroup, setup: _TurnSetup, visit: Visit, facts: TokenFacts) -> ActiveVoice:
        hearing = Hearing()
        task = turns.create_task(self._take_voice(setup, visit, facts, hearing, TaskRecord()))
        self._voices[visit.place] = voice = ActiveVoice(task, visit.generation, hearing)
        return voice

    async def _take_voice(
        self, setup: _TurnSetup, visit: Visit, facts: TokenFacts, hearing: Hearing, record: TaskRecord
    ) -> None:
        """Runs a place's voice loop; whatever way it ends, a defect met in it ends the runner loud.

        The loud end is asked here, in the task's own coroutine, and the record joins a defect the session's close
        can lose from what leaves.
        """
        await ending_loud_when_cancelled(
            ending_with_recorded_defect(self._run_voice(setup, visit, facts, hearing, record), record), record.loud_end
        )

    async def _run_voice(
        self, setup: _TurnSetup, visit: Visit, facts: TokenFacts, hearing: Hearing, record: TaskRecord
    ) -> None:
        """Reads the day, then hears and speaks over a session of its own until it is cancelled.

        Lines heard meanwhile wait in the hearing. A decision's tools are none, so nothing relays a call.
        """
        context = voice_context(await self._remembering.read_day(visit))
        token = await self._current_token()
        async with self._boundaries.open_session(self._app, token.value) as session:
            try:
                voice = SpeakingVoice(
                    session,
                    visit.place,
                    self._boundaries.clock,
                    self._report,
                    partial(self._use_current_token, session),
                )
                await run_voice_loop(
                    setup.persona,
                    setup.backend(()),
                    TurnPerson(facts.principal, TURN_ROLE, self._current_groups),
                    hearing,
                    voice,
                    wakes=addressed_or_asked(frozenset({facts.name})),
                    context=context,
                    report=reporting_decisions(self._report),
                )
            except Exception as error:
                record.note(error)
                raise

    async def _current_groups(self) -> frozenset[str]:
        return facts_of((await self._current_token()).value).groups

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

    async def _take_turn(self, setup: _TurnSetup, admitted: _Admitted, record: TaskRecord) -> None:
        """Runs one turn; whatever way it ends, a defect a request of it met ends the runner loud.

        A session whose close cuts a request short can lose the defect from what leaves, so the turn's record
        joins it. A cancelled turn ends as the cancellation unless a defect was met or nobody cancelled it.
        """
        await ending_loud_when_cancelled(
            ending_with_recorded_defect(self._judge_turn(setup, admitted, record), record), record.loud_end
        )

    async def _judge_turn(self, setup: _TurnSetup, admitted: _Admitted, record: TaskRecord) -> None:
        """Runs one turn over its own session.

        A turn the gateway ends short is reported unless a defect was met; a failed relayed call and any other
        error go to the group. Every error of a cancelled turn ends as the cancellation.
        """
        try:
            await self._run_turn(setup, admitted, record)
        except Exception as error:
            _raise_if_cancelled()
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

    async def _run_turn(self, setup: _TurnSetup, admitted: _Admitted, record: TaskRecord) -> None:
        earlier = await self._remembering.read_day(Visit(self._app.mcp_url, admitted.place, admitted.window.generation))
        told = turn_input(
            admitted.place, admitted.facts.name, admitted.window, admitted.new, missed=admitted.missed, earlier=earlier
        )
        token = await self._current_token()
        async with self._boundaries.open_session(self._app, token.value) as session:
            try:
                await self._run_session_body(setup, admitted, told, session, record)
            except Exception as error:
                record.note(error)
                raise

    async def _run_session_body(
        self, setup: _TurnSetup, admitted: _Admitted, told: str, session: AppSession, record: TaskRecord
    ) -> None:
        _raise_if_cancelled()
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

    async def _hold_tools(self, setup: _TurnSetup, session: AppSession, record: TaskRecord) -> _HeldTools | None:
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
            _raise_if_cancelled()
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
                granted = await self._boundaries.tokens.fetch(self._agent, app.scope)
                async with self._boundaries.open_session(app, granted.value) as session:
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

    async def _react(self, failure: CallFailure) -> None:
        match reaction_to(failure, already_renewed=self._renewed_after_401):
            case Reaction.RENEW:
                self._renewed_after_401 = True
                self._token = None
            case Suspend(code=code):
                await self._suspend(code)
            case remaining:
                # pyright narrows `remaining` to the one reaction left, so a new reaction fails the type check here.
                _: Literal[Reaction.BACKOFF] = remaining
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
            self._report(Event.BACKOFF, failure_code(failure))
        self._failures += 1
        await self._boundaries.clock.sleep(backoff_delay(self._failures))

    def _report(self, event: Event, code: str | None = None, app: AppTarget | None = None) -> None:
        self._boundaries.report(StateChange(event, self._agent.client_id, (app or self._app).mcp_url, code))


async def _run_loop(loop: AgentLoop, runner: asyncio.Task[object]) -> None:
    """Runs a loop; one that ends cancelled while its runner is not being cancelled ends the runner loud.

    The runner's own count says so: it is above zero on a cancellation of the runner and while its group aborts
    after a loop's loud end. A loop task's count is not read, a cancelled lifecycle session lowers it.
    """
    return await ending_loud_when_cancelled(loop.run(), partial(unrequested_unless_cancelling, runner))


async def run_all(config: Config, boundaries: Boundaries) -> NoReturn:
    """One loop for every agent and every app that listens; a loop never ends on its own."""
    runner = asyncio.current_task()
    assert runner is not None  # noqa: S101 - run_all runs inside a task
    async with asyncio.TaskGroup() as group:
        tools_only = tuple(app for app in config.apps if not app.listens)
        today = None if config.memory is None else config.memory.today
        for agent in config.agents:
            for app in config.apps:
                if app.listens:
                    loop = AgentLoop(agent, app, boundaries, tools_only, today)
                    group.create_task(_run_loop(loop, runner))
    raise AssertionError("a loop ended")  # pragma: no cover
