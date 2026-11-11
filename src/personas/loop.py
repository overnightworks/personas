"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from agents.backend import TurnPerson
from agents.voice.loop import run_voice_loop
from agents.voice.wake import addressed_or_asked

from personas.decision import (
    SUSPENDED_RETRY_SECONDS,
    TURN_CANCELLED_CODE,
    TURN_ROLE,
    Joined,
    Lines,
    Memory,
    Perception,
    Reaction,
    Suspend,
    Token,
    Visit,
    Waiting,
    WindowRead,
    accepts_for,
    active_call_places,
    active_view_places,
    addressed,
    after_refusal,
    backoff_delay,
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
    sync_turned_revisions,
    taken_lines,
    turn_invalid,
    view_turn_due,
    voice_context,
)
from personas.failures import CallFailure, leaves_of, raise_if_cancelled
from personas.mind import Mind
from personas.remembering import LineStore, Remembering
from personas.reporting import Event, Reporter, StateChange
from personas.task_record import (
    TaskRecord,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    unrequested_unless_cancelling,
)
from personas.tokens import TokenFacts, facts_of
from personas.turning import Admitted, Turning, TurnSetup
from personas.voice import ActiveVoice, Hearing, SpeakingVoice, reporting_decisions

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
    logged once until a read answers. In an app that binds tools, the lines of the agent's call places are
    read too, every new line goes into the agent's history, and one addressed to him starts a turn; at most one turn
    runs per loop, over its own session, and a failure of the turn's calls to this app goes into the one reaction of
    the loop. The turn also holds the bound tools of the tools-only apps: each request to one runs in a session of
    its own under the token for its own scope, and its failure never reaches the reaction. In an app that declares
    ``speak``, for an agent whose persona has a voice section, each active call place instead runs one voice loop,
    which hears every line the place gives and speaks through ``speak``.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        boundaries: Boundaries,
        mind: Mind,
        tools_only: tuple[AppTarget, ...] = (),
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
            tools_only,
            boundaries.tokens,
            boundaries.open_session,
            self._remembering,
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
        self._turn_ended = asyncio.Event()
        self._voices: dict[PlaceRef, ActiveVoice] = {}
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
            standings = await self._take_standing(session)
        await self._answer_waiting_view(session, turns, standings)
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
        for accept in accepts_for(standings, self._memory, takes_turns=self._turn_setup is not None):
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
            turn_started = self._take_lines(turns, self._turn_setup, place, window, facts, read_at)
            await self._remembering.keep(WindowRead(self._agent.client_id, self._app.mcp_url, place, window, read_at))
            if turn_started:
                return lost
        return lost

    async def _answer_waiting_view(
        self, session: AppSession, turns: asyncio.TaskGroup, standings: tuple[Standing, ...]
    ) -> None:
        """Reads the views of the active view places in the standing's order; the first that is due starts the turn.

        It reads none while a turn runs or was just started. A revision starts one turn, whatever that turn did.
        """
        setup = self._turn_setup
        if setup is None or self._turn is not None:
            return
        facts = facts_of((await self._current_token()).value)
        for place in active_view_places(standings):
            await self._use_current_token(session)
            view = await session.read_view(place)
            if view is None or not view_turn_due(self._turned_revisions.get(place), view):
                continue
            self._turned_revisions[place] = view.revision
            admitted = Admitted(facts, place, Waiting(view), self._mind.history, self._boundaries.clock.wall())
            self._start_turn(turns, setup, place, admitted)
            return

    def _take_lines(
        self,
        turns: asyncio.TaskGroup,
        setup: TurnSetup,
        place: PlaceRef,
        window: PlaceWindow,
        facts: TokenFacts,
        read_at: datetime,
    ) -> bool:
        """Moves the place's checkpoint and takes the new lines into the agent's history: a voice loop hears all of
        them, and else one addressed to the agent starts a turn, which this says. No await stands in it."""
        perception = perceive(window, self._checkpoints[place])
        self._checkpoints[place] = perception.checkpoint
        visit = Visit(self._app.mcp_url, place, window.generation)
        self._mind.take(taken_lines(visit, perception.new, read_at))
        if not self._voice_app:
            return self._start_turn_if_addressed(turns, setup, place, window, facts, perception, read_at)
        voice = self._voices.get(place)
        if voice is None or voice.generation != window.generation:
            if voice is not None:
                voice.task.cancel()
            voice = self._start_voice(turns, setup, visit, facts)
        for line in perception.new:
            voice.hearing.put(line)
        return False

    def _start_turn_if_addressed(
        self,
        turns: asyncio.TaskGroup,
        setup: TurnSetup,
        place: PlaceRef,
        window: PlaceWindow,
        facts: TokenFacts,
        perception: Perception,
        read_at: datetime,
    ) -> bool:
        """Starts the turn when a new line addresses the agent. A line starts at most one turn."""
        if not any(addressed(line, facts.name) for line in perception.new):
            return False
        lines = Lines(window.generation, frozenset(line.seq for line in perception.new), perception.missed)
        admitted = Admitted(facts, place, lines, self._mind.history, read_at)
        self._start_turn(turns, setup, place, admitted)
        return True

    def _start_turn(self, turns: asyncio.TaskGroup, setup: TurnSetup, place: PlaceRef, admitted: Admitted) -> None:
        task = turns.create_task(self._turning.take(setup, admitted, TaskRecord()))
        self._turn = ActiveTurn(task, place)
        task.add_done_callback(self._turn_over)

    def _start_voice(self, turns: asyncio.TaskGroup, setup: TurnSetup, visit: Visit, facts: TokenFacts) -> ActiveVoice:
        hearing = Hearing()
        task = turns.create_task(self._take_voice(setup, visit, facts, hearing, TaskRecord()))
        self._voices[visit.place] = voice = ActiveVoice(task, visit.generation, hearing)
        return voice

    async def _take_voice(
        self, setup: TurnSetup, visit: Visit, facts: TokenFacts, hearing: Hearing, record: TaskRecord
    ) -> None:
        """Runs a place's voice loop; whatever way it ends, a defect met in it ends the runner loud.

        The loud end is asked here, in the task's own coroutine, and the record joins a defect the session's close
        can lose from what leaves.
        """
        await ending_loud_when_cancelled(
            ending_with_recorded_defect(self._run_voice(setup, visit, facts, hearing, record), record), record.loud_end
        )

    async def _run_voice(
        self, setup: TurnSetup, visit: Visit, facts: TokenFacts, hearing: Hearing, record: TaskRecord
    ) -> None:
        """Reads the day, then hears and speaks over a session of its own until it is cancelled.

        Lines heard meanwhile wait in the hearing. A decision's tools are none, so nothing relays a call.
        """
        context = voice_context(await self._remembering.read_day(self._boundaries.clock.wall(), visit))
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
            mind = Mind(config.history_bytes)
            for app in config.apps:
                if app.listens:
                    loop = AgentLoop(agent, app, boundaries, mind, tools_only, today)
                    group.create_task(_run_loop(loop, runner))
    raise AssertionError("a loop ended")  # pragma: no cover
