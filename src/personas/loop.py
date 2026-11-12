"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from operator import eq
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from personas.decision import (
    SUSPENDED_RETRY_SECONDS,
    TURN_CANCELLED_CODE,
    AppPlace,
    Joined,
    Lines,
    Memory,
    Reaction,
    Suspend,
    Token,
    ViewSeen,
    Visit,
    Waiting,
    WindowRead,
    accepts_for,
    active_call_places,
    active_view_places,
    after_refusal,
    backoff_delay,
    decisive_failure,
    failure_code,
    forget_gone,
    is_defect,
    is_voice_app,
    left_standing,
    lends_tools,
    perceive,
    reaction_to,
    renewal_due,
    speaks_aloud,
    sync_checkpoints,
    sync_turned_revisions,
    taken_lines,
    turn_invalid,
    view_turn_due,
    wakes_him,
)
from personas.failures import CallFailure, leaves_of, raise_if_cancelled
from personas.mind import Mind
from personas.remembering import LineStore, Remembering
from personas.reporting import Event, Reporter, StateChange
from personas.task_record import (
    TaskRecord,
    ending_loud_when_cancelled,
    unrequested_unless_cancelling,
)
from personas.tokens import facts_of
from personas.turning import Admitted, Turning, TurnSetup

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from datetime import datetime

    from agents.member import PlaceRef, PlaceWindow, Standing

    from personas.app_session import AppSession, SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget, Config
    from personas.decision import Checkpoint, TodayBounds
    from personas.readiness import ReadyMarker
    from personas.tokens import TokenSource
    from personas.turning import BackendFactory

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
    lines: LineStore | None = None


def _defect_of(task: asyncio.Task[object]) -> BaseException | None:
    """The error a finished task raised when a leaf of it is a defect; reading it marks it retrieved."""
    if not task.done() or task.cancelled():
        return None
    error = task.exception()
    if error is None or not any(is_defect(leaf) for leaf in leaves_of(error)):
        return None
    return error


@dataclass(frozen=True)
class ActiveTurn:
    """The running turn: its task and place, the loop's request to end it and its end report."""

    task: asyncio.Task[None]
    place: PlaceRef
    cancel_requested: bool = False
    end_reported: bool = False


class AgentLoop:
    """Keeps one stream of changes open for one agent in one app, accepts what it is invited to and answers.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next
    retry, and resumes once a read answers again. A network failure, a lost stream or a 5xx backs off,
    logged once until a read answers. In an app that binds tools, the lines of the agent's call places are read too,
    while any turn of his runs, and every new line and every view read goes into the agent's history. A line that is
    not his own and a view that waits for him make a turn due in the agent's mind, and after each read the loop starts
    the turn of the place that has waited longest when it is in this app; at most one turn of the agent runs across all
    his loops, over its own session, and a failure of the turn's calls to this app goes into the one reaction of the
    loop. The turn also holds the bound tools of the tools-only apps and of the other apps where the agent holds an
    active place when it is admitted, which the loops of those apps record in his mind: each request to one runs in a
    session of its own under the token for its own scope, and its failure never reaches the reaction. Each call the
    turn relays, to this app or another, enters the agent's history when its request ends. In an app
    that declares ``speak``, for an agent whose persona has a voice section, a turn at a call place is spoken: its text
    is said sentence by sentence through ``speak``.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        boundaries: Boundaries,
        mind: Mind,
        apps: tuple[AppTarget, ...],
        today: TodayBounds | None = None,
    ) -> None:
        self._agent = agent
        self._app = app
        self._boundaries = boundaries
        self._mind = mind
        self._remembering = Remembering(boundaries.lines, today, agent.client_id, self._report)
        self._turning = Turning(
            agent,
            app,
            apps,
            boundaries.tokens,
            boundaries.open_session,
            self._remembering,
            mind,
            boundaries.clock,
            self._current_token,
            self._use_current_token,
            self._report,
            self._report_turn_end,
        )
        self._token: Token | None = None
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None
        self._backing_off = False
        self._checkpoints: dict[PlaceRef, Checkpoint] = {}
        self._turned_revisions: dict[PlaceRef, int] = {}
        self._turn: ActiveTurn | None = None
        self._turn_ended = mind.wake_event()
        self._voice_app = False
        persona, binding, backend = agent.persona, app.binding, boundaries.backend
        self._turn_setup = (
            None if persona is None or binding is None or backend is None else TurnSetup(persona, binding, backend)
        )

    async def run(self) -> NoReturn:
        self._report(Event.STARTED)
        while True:
            raise_if_cancelled()
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
            self._voice_app = await self._is_voice_app(session)
            while True:
                await self._use_current_token(session)
                async with session.changes() as changes, aclosing(self._wakes(changes)) as wakes:
                    await self._reconcile(session, turns)
                    async for _ in wakes:
                        await self._reconcile(session, turns)

    async def _is_voice_app(self, session: AppSession) -> bool:
        """Whether the turns at this app's call places are spoken, decided once for the lifecycle session."""
        has_voice_section = self._turn_setup is not None and self._turn_setup.persona.voice_window_bytes is not None
        declared = await session.declared_tools() if has_voice_section else ()
        return is_voice_app(has_voice_section, declared)

    async def _wakes(self, changes: AsyncIterator[object]) -> AsyncGenerator[None]:
        """A wake for each change the app signals and for each wake of his mind; it ends when the stream does."""
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
        if await self._read_call_places(session, standings):
            # A window the app lost dropped its place's checkpoint: the standing says where the place stands now.
            standings = await self._take_standing(session)
        await self._read_views(session, standings)
        await self._start_due_turn(turns)
        self._served()

    async def _take_standing(self, session: AppSession) -> tuple[Standing, ...]:
        """Reads the standing, keeps the checkpoints in line with it and accepts what it invites to."""
        await self._use_current_token(session)
        standing = await session.read_standing()
        self._resume_if_unwell()
        self._boundaries.ready.announce()
        self._memory = forget_gone(self._memory, standing.standings)
        self._checkpoints = sync_checkpoints(self._checkpoints, standing.standings)
        self._turned_revisions = sync_turned_revisions(self._turned_revisions, standing.standings)
        self._mind.undue(left_standing(self._app.mcp_url, standing.standings))
        self._mind.stand_at(self._app.mcp_url, holds=lends_tools(self._app.binding is not None, standing.standings))
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
        for accept in accepts_for(
            standings, self._memory, takes_turns=self._turn_setup is not None, figure=self._agent.figure
        ):
            await self._use_current_token(session)
            refusal = await session.accept(accept)
            if refusal is None:
                self._checkpoints[accept.place] = Joined()
                self._report(Event.JOINED)
                continue
            self._memory, is_news = after_refusal(self._memory, accept, refusal)
            if is_news:
                self._report(Event.REFUSED, refusal.value)

    async def _read_call_places(self, session: AppSession, standings: tuple[Standing, ...]) -> bool:
        """Reads the active call places' windows, a turn of his running or not, and takes their lines in.

        It says whether a window was lost, which dropped its place's checkpoint.
        """
        if self._turn_setup is None:
            return False
        lost = False
        for place in active_call_places(standings):
            await self._use_current_token(session)
            window = await session.read_place(place)
            if window is None:
                del self._checkpoints[place]
                lost = True
                continue
            read_at = self._boundaries.clock.wall()
            self._take_lines(place, window, read_at)
            await self._remembering.keep(WindowRead(self._agent.client_id, self._app.mcp_url, place, window, read_at))
        return lost

    async def _read_views(self, session: AppSession, standings: tuple[Standing, ...]) -> None:
        """Reads the views of the active view places in the standing's order and takes each into the agent's history.

        A revision that waits for him and has no turn yet makes a turn due at its place; one that does not wait, and a
        view the app gives none of, take the place's due away. A revision gets one turn, whatever that turn did.
        """
        if self._turn_setup is None:
            return
        for place in active_view_places(standings):
            await self._use_current_token(session)
            view = await session.read_view(place)
            at = AppPlace(self._app.mcp_url, place)
            if view is None:
                self._mind.undue(partial(eq, at))
                continue
            self._mind.take((ViewSeen(at.app, place, view, self._boundaries.clock.wall()),))
            if view_turn_due(self._turned_revisions.get(place), view):
                self._mind.due(at, Waiting(view.revision))
            else:
                self._mind.undue(partial(eq, at))

    async def _start_due_turn(self, turns: asyncio.TaskGroup) -> None:
        """Starts the turn of the place that has waited longest when it is in this app and none of his runs. The claim
        and the turn's start share one synchronous step."""
        setup = self._turn_setup
        if setup is None:
            return
        facts = facts_of((await self._current_token()).value)
        claim = self._mind.claim(self._app.mcp_url)
        if claim is None:
            return
        at, occasion = claim
        if isinstance(occasion, Waiting):
            self._turned_revisions[at.place] = occasion.revision
        aloud = speaks_aloud(self._voice_app, occasion)
        apps = self._mind.held_apps(self._app.mcp_url)
        admitted = Admitted(facts, at.place, occasion, self._mind.history, self._boundaries.clock.wall(), aloud, apps)
        self._start_turn(turns, setup, at.place, admitted)

    def _take_lines(self, place: PlaceRef, window: PlaceWindow, read_at: datetime) -> None:
        """Moves the place's checkpoint and takes the new lines into the agent's history; a line that is not his own
        makes a turn due. No await stands in it."""
        perception = perceive(window, self._checkpoints[place])
        self._checkpoints[place] = perception.checkpoint
        self._mind.take(taken_lines(Visit(self._app.mcp_url, place, window.generation), perception.new, read_at))
        if wakes_him(perception.new):
            lines = Lines(window.generation, frozenset(line.seq for line in perception.new), perception.missed)
            self._mind.due(AppPlace(self._app.mcp_url, place), lines)

    def _start_turn(self, turns: asyncio.TaskGroup, setup: TurnSetup, place: PlaceRef, admitted: Admitted) -> None:
        task = turns.create_task(self._turning.take(setup, admitted, TaskRecord()))
        self._turn = ActiveTurn(task, place)
        task.add_done_callback(self._turn_over)

    def _turn_over(self, task: asyncio.Task[None]) -> None:
        turn, self._turn = self._turn, None
        if turn is not None and turn.cancel_requested and task.cancelled() and not turn.end_reported:
            self._report(Event.TURN_ENDED, TURN_CANCELLED_CODE)
        self._mind.release()

    def _report_turn_end(self, code: str) -> None:
        """A turn reports its end at most once; the record tells the loop, which then reports no cancel."""
        assert self._turn is not None  # noqa: S101 - only the running turn's task reports its end
        self._turn = replace(self._turn, end_reported=True)
        self._report(Event.TURN_ENDED, code)

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
        self._park()
        if self._suspension != code:
            self._suspension = code
            self._report(Event.SUSPENDED, code)
        self._backing_off = False
        self._token = None
        await self._boundaries.clock.sleep(SUSPENDED_RETRY_SECONDS)

    async def _back_off(self, failure: CallFailure) -> None:
        self._park()
        if not self._backing_off:
            self._backing_off = True
            self._report(Event.BACKOFF, failure_code(failure))
        self._failures += 1
        await self._boundaries.clock.sleep(backoff_delay(self._failures))

    def _park(self) -> None:
        """A parked app holds up no place of his other apps and lends no tools to their turns: its due places are
        dropped and its place unrecorded; it reads them again after."""
        self._mind.undue(lambda at: at.app == self._app.mcp_url)
        self._mind.stand_at(self._app.mcp_url, holds=False)

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
        today = None if config.memory is None else config.memory.today
        for agent in config.agents:
            mind = Mind(config.history_bytes)
            for app in config.apps:
                if app.listens:
                    loop = AgentLoop(agent, app, boundaries, mind, config.apps, today)
                    group.create_task(_run_loop(loop, runner))
    raise AssertionError("a loop ended")  # pragma: no cover
