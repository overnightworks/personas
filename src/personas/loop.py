"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from operator import eq
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from logline import unit_trace

from personas.decision import (
    NETWORK_FAILURE_CODE,
    NO_TURNS_CODE,
    SUSPENDED_RETRY_SECONDS,
    AppPlace,
    Joined,
    Memory,
    Reaction,
    Suspend,
    Token,
    accepts_for,
    active_view_places,
    active_window_places,
    after_refusal,
    backoff_delay,
    backoff_states_unavailable,
    decisive_failure,
    failure_code,
    failure_reason,
    forget_gone,
    is_defect,
    is_voice_app,
    left_out_for,
    lends_tools,
    perceive,
    reaction_to,
    renewal_due,
    speaks_aloud,
    sync_turned_revisions,
    turn_ending,
    turn_invalid,
    view_turn_due,
    waking_line,
)
from personas.due import kept_turns, left_standing, pause_deadline, resumed_turns
from personas.entries import Lines, ViewSeen, Visit, Waiting, WindowRead, taken_lines
from personas.failures import (
    STORE_TIMEOUT_CODE,
    CallFailure,
    StaleGenerationError,
    StoreDefect,
    StoreUnavailable,
    leaves_of,
    raise_if_cancelled,
)
from personas.mind import Mind
from personas.present import history_delta, present_of
from personas.recall import LookupBounds
from personas.remembering import LineStore, Remembering
from personas.reporting import (
    Event,
    Reporter,
    StateChange,
    StoreOutage,
    TurnTally,
    start_turn_tally,
    step_timer,
    stepping,
)
from personas.task_record import (
    TaskRecord,
    ending_loud_when_cancelled,
    unrequested_unless_cancelling,
)
from personas.tokens import facts_of
from personas.turn_states import TurnStates
from personas.turning import Admitted, Turning, TurnSetup

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from datetime import datetime

    from agents.member import PlaceRef, PlaceWindow, Standing

    from personas.app_session import AppSession, SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget, Config
    from personas.due import Due
    from personas.entries import KeptTurns
    from personas.ownership import Owned, Ownership
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
    """The running turn: its task and place, its tally, and the loop's request to end it."""

    task: asyncio.Task[None]
    place: PlaceRef
    tally: TurnTally
    cancel_requested: bool = False


class AgentLoop:
    """Keeps one stream of changes open for one agent in one app, accepts what it is invited to and answers.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next
    retry, and resumes once a read answers again. A network failure, a lost stream or a 5xx backs off,
    logged once until a read answers. In an app that binds tools, the lines of the agent's call places are read too,
    while any turn of his runs, and every new line and every view read goes into the agent's history. A line that is
    not his own and a view that waits for him make a turn due in the agent's mind, and after each read the loop starts
    the turn of the place that has waited longest when it is in this app; at most one turn of the agent runs across all
    his loops, over its own session, and a failure of the turn's calls to this app goes into the one reaction of the
    loop. The turn also holds the tools of the tools-only apps and of the other apps where the agent holds an
    active place when it is admitted, which the loops of those apps record in his mind: each request to one runs in a
    session of its own under the token for its own scope, and its failure never reaches the reaction. Each call the
    turn relays, to this app or another, enters the agent's history when its request ends. In an app
    that declares ``speak``, for an agent whose entry has a voice table, a turn at a call place is spoken: its text
    is said sentence by sentence through ``speak``.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        boundaries: Boundaries,
        mind: Mind,
        apps: tuple[AppTarget, ...],
        lookup: LookupBounds | None,
    ) -> None:
        self._agent = agent
        self._app = app
        self._boundaries = boundaries
        self._mind = mind
        self._remembering = Remembering(boundaries.lines, self._report, lookup)
        self._turning = Turning(
            agent,
            app,
            apps,
            boundaries.tokens,
            boundaries.open_session,
            mind,
            boundaries.clock,
            self._current_token,
            self._use_current_token,
            self._report,
            self._report_pause,
            self._report_tools_left_out,
            self._remembering,
        )
        self._turn_states = TurnStates(self._use_current_token, self._report, app.call_seconds)
        self._token: Token | None = None
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None
        self._backing_off = False
        self._standings: tuple[Standing, ...] = ()
        self._turned_revisions: dict[PlaceRef, int] = {}
        self._turn: ActiveTurn | None = None
        self._pause_wake: asyncio.Task[None] | None = None
        self._turn_ended = mind.wake_event()
        self._voice_app = False
        persona, backend = agent.persona, boundaries.backend
        self._turn_setup = (
            None if persona is None or app.max_tool_calls is None or backend is None else TurnSetup(persona, backend)
        )

    async def run(self) -> NoReturn:
        self._report(Event.STARTED, model=None if self._turn_setup is None else self._turn_setup.persona.model)
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
            turns.create_task(self._turn_states.state_each(session))
            self._turning.new_session()
            self._voice_app = await self._is_voice_app(session)
            while True:
                await self._use_current_token(session)
                async with session.changes() as changes, aclosing(self._wakes(changes)) as wakes:
                    await self._reconcile(session, turns)
                    async for _ in wakes:
                        await self._reconcile(session, turns)

    async def _is_voice_app(self, session: AppSession) -> bool:
        """Whether the turns at this app's call places are spoken, decided once for the lifecycle session."""
        has_voice = self._turn_setup is not None and self._agent.voice is not None
        declared = await session.declared_tools() if has_voice else ()
        return is_voice_app(has_voice, declared)

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
        """Reads the standing as it is now and acts on it; a wake says only that it may have changed.

        The views are read before the windows, so a window holds every line up to its view's revision; the lines enter
        the history before the views do.
        """
        standings = await self._take_standing(session)
        views = await self._read_views(session, standings)
        window_lost = await self._read_windows(session, standings)
        self._take_views(views)
        if window_lost:
            # A window the app lost dropped its place's checkpoint: the standing says where the place stands now.
            await self._take_standing(session)
        self._turn_states.restate_if_due()
        await self._start_due_turn(turns)
        self._served()

    async def _take_standing(self, session: AppSession) -> tuple[Standing, ...]:
        """Reads the standing, keeps the checkpoints in line with it and accepts what it invites to."""
        await self._use_current_token(session)
        standing = await session.read_standing()
        self._resume_if_unwell()
        self._memory = forget_gone(self._memory, standing.standings)
        self._mind.sync_checkpoints(self._app.mcp_url, standing.standings, self._standings)
        self._standings = standing.standings
        self._turned_revisions = sync_turned_revisions(self._turned_revisions, standing.standings)
        self._turn_states.sync_seats(standing.standings)
        self._mind.undue(left_standing(self._app.mcp_url, standing.standings))
        self._mind.stand_at(
            self._app.mcp_url, holds=lends_tools(self._app.max_tool_calls is not None, standing.standings)
        )
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
        voice = self._agent.voice
        takes_turns = self._turn_setup is not None
        self._memory, left_out = left_out_for(standings, self._memory, takes_turns=takes_turns)
        for standing in left_out:
            self._report(Event.INVITATION_LEFT_OUT, NO_TURNS_CODE, place_kind=standing.place.kind)
        for accept in accepts_for(
            standings,
            self._memory,
            takes_turns=takes_turns,
            figure=self._agent.figure,
            voices=() if voice is None else voice.names,
        ):
            await self._use_current_token(session)
            refusal = await session.accept(accept)
            if refusal is None:
                self._mind.move_checkpoint(AppPlace(self._app.mcp_url, accept.place), Joined())
                self._report(Event.JOINED)
                continue
            self._memory, is_news = after_refusal(self._memory, accept, refusal)
            if is_news:
                self._report(Event.REFUSED, refusal.value)

    async def _read_windows(self, session: AppSession, standings: tuple[Standing, ...]) -> bool:
        """Reads the active window places' windows, a turn of his running or not, and takes their lines in.

        It says whether a window was lost, which dropped its place's checkpoint.
        """
        if self._turn_setup is None:
            return False
        serving_view = frozenset(active_view_places(standings))
        lost = False
        for place in active_window_places(standings):
            await self._use_current_token(session)
            window = await session.read_place(place)
            if window is None:
                self._mind.move_checkpoint(AppPlace(self._app.mcp_url, place), None)
                lost = True
                continue
            read_at = self._boundaries.clock.wall()
            self._take_lines(place, window, read_at, serves_view=place in serving_view)
            await self._turning.hold(session, place, window)
            await self._remembering.keep(WindowRead(self._agent.client_id, self._app.mcp_url, place, window, read_at))
        return lost

    async def _read_views(
        self, session: AppSession, standings: tuple[Standing, ...]
    ) -> dict[PlaceRef, ViewSeen | None]:
        """Reads the views of the active view places in the standing's order, each stamped with the time of its read; a
        place the app gives no view of maps to none."""
        if self._turn_setup is None:
            return {}
        read: dict[PlaceRef, ViewSeen | None] = {}
        for place in active_view_places(standings):
            await self._use_current_token(session)
            view = await session.read_view(place)
            read[place] = (
                None if view is None else ViewSeen(self._app.mcp_url, place, view, self._boundaries.clock.wall())
            )
        return read

    def _take_views(self, read: dict[PlaceRef, ViewSeen | None]) -> None:
        """Takes each view read into the agent's history. A revision that waits for him and has no turn yet makes a turn
        due at its place; one that does not wait, and a view the app gives none of, take the place's due away. A
        revision waiting for him that has had its turn keeps a due that turn left behind by ending without an answer.
        No await stands in it."""
        for place, seen in read.items():
            at = AppPlace(self._app.mcp_url, place)
            if seen is None:
                self._mind.undue(partial(eq, at))
                continue
            self._mind.take((seen,))
            if view_turn_due(self._turned_revisions.get(place), seen.view):
                self._mind.due(at, Waiting(seen.view.revision, seen.view.traceparent), self._boundaries.clock.now())
            elif not seen.view.awaits_you:
                self._mind.undue(partial(eq, at))

    async def _start_due_turn(self, turns: asyncio.TaskGroup) -> None:
        """Starts the turn of the place that has waited longest when it is in this app and none of his runs. The claim
        and the turn's start share one synchronous step; the turn waits until the claim is kept, so a pod that takes
        him over before then answers the place instead of him."""
        setup = self._turn_setup
        if setup is None:
            return
        facts = facts_of((await self._current_token()).value)
        now = self._boundaries.clock.now()
        claim = self._mind.claim(self._app.mcp_url, now)
        if claim is None:
            self._wake_when_pause_over(turns, now)
            return
        at, due = claim
        claimed = self._mind.changes()
        occasion = due.occasion
        if isinstance(occasion, Waiting):
            self._turned_revisions[at.place] = occasion.revision
        aloud = speaks_aloud(self._voice_app, occasion, self._mind.checkpoint(at))
        apps = self._mind.held_apps(self._app.mcp_url)
        admitted = Admitted(facts, at.place, occasion, self._mind.present, self._boundaries.clock.wall(), aloud, apps)
        self._start_turn(turns, setup, at.place, admitted, due, claimed)

    def _wake_when_pause_over(self, turns: asyncio.TaskGroup, now: float) -> None:
        """A pause holds the turn of the place at this app that has waited longest: one waiter of this loop wakes
        every loop of his when it is over, and a waiter that ended early sees the rest of the pause again."""
        seconds = self._mind.pause_left(self._app.mcp_url, now)
        if seconds > 0 and (self._pause_wake is None or self._pause_wake.done()):
            self._pause_wake = turns.create_task(self._wake_after(seconds))

    async def _wake_after(self, seconds: float) -> None:
        await self._boundaries.clock.sleep(seconds)
        self._mind.wake()

    def _take_lines(self, place: PlaceRef, window: PlaceWindow, read_at: datetime, *, serves_view: bool) -> None:
        """Moves the place's checkpoint and takes the new lines into the agent's history; ``waking_line`` decides
        whether they make a turn due, and in whose trace. No await stands in it."""
        at = AppPlace(self._app.mcp_url, place)
        checkpoint = self._mind.checkpoint(at)
        assert checkpoint is not None  # noqa: S101 - the standing read before gave every active window place one
        perception = perceive(window, checkpoint)
        self._mind.move_checkpoint(at, perception.checkpoint)
        self._mind.take(taken_lines(Visit(self._app.mcp_url, place, window.generation), perception.new, read_at))
        waking = waking_line(perception.new, serves_view=serves_view)
        if waking is not None:
            seqs = frozenset(line.seq for line in perception.new)
            lines = Lines(window.generation, seqs, perception.missed, waking.traceparent)
            self._mind.due(at, lines, self._boundaries.clock.now())

    def _start_turn(
        self,
        turns: asyncio.TaskGroup,
        setup: TurnSetup,
        place: PlaceRef,
        admitted: Admitted,
        due: Due,
        claimed: int,
    ) -> None:
        """Starts the turn in the trace of what made it due, or in a trace of its own when that carries none or one
        logline cannot read; its task, its done callback and its step lines share that trace and the loop's own lines
        do not. The turn's wait, from ``due.since`` to now, ends here with what first kept it waiting; a pause that
        ended later than ``due.since`` says the place was due only from then."""
        with unit_trace(admitted.occasion.traceparent):
            wait = step_timer(
                "turn_wait.ended",
                self._agent.client_id,
                self._app.mcp_url,
                elapsed_seconds=max(0.0, self._boundaries.clock.now() - due.since),
            )
            wait.add(wait_reason=due.waited_on.value)
            wait.end()
            tally = start_turn_tally(self._agent.client_id, self._app.mcp_url)
            task = turns.create_task(self._take_once_kept(claimed, setup, admitted, tally))
            self._turn = ActiveTurn(task, place, tally)
            task.add_done_callback(self._turn_over)

    async def _take_once_kept(self, claimed: int, setup: TurnSetup, admitted: Admitted, tally: TurnTally) -> None:
        """Takes the turn once the ``claimed``-th change of his mind, its claim, is kept in the store."""
        with stepping(step_timer("claim_write.ended", self._agent.client_id, self._app.mcp_url)):
            await self._mind.until_written(claimed)
        await self._turn_states.turn_started(admitted.place)
        await self._turning.take(setup, admitted, TaskRecord(), tally)

    def _turn_over(self, task: asyncio.Task[None]) -> None:
        """The one writer of ``turn.ended``, whatever way the task ended."""
        turn, self._turn = self._turn, None
        assert turn is not None  # noqa: S101 - the loop clears its turn only here and once per task
        cancelled = task.cancelled()
        tally = turn.tally
        ending = turn_ending(
            cancelled=cancelled,
            error=None if cancelled else task.exception(),
            stop_code=tally.stop_code,
            answered=tally.answered,
        )
        tally.end(ending)
        self._turn_states.turn_ended(turn.place, ending)
        if cancelled and not tally.answered and not turn.cancel_requested:
            self._mind.cut()
        self._mind.release()

    def _report_pause(self, seconds: float, code: str) -> None:
        self._boundaries.report(
            StateChange(
                Event.PAUSED,
                self._agent.client_id,
                self._app.mcp_url,
                code,
                seconds=seconds,
            )
        )

    def _report_tools_left_out(self, app: AppTarget, count: int) -> None:
        self._boundaries.report(StateChange(Event.TOOLS_LEFT_OUT, self._agent.client_id, app.mcp_url, tool_count=count))

    async def _react(self, failure: CallFailure) -> None:
        match reaction_to(failure, already_renewed=self._renewed_after_401):
            case Reaction.RENEW:
                self._renewed_after_401 = True
                self._token = None
            case Suspend(code=code):
                await self._suspend(code, failure)
            case remaining:
                # pyright narrows `remaining` to the one reaction left, so a new reaction fails the type check here.
                _: Literal[Reaction.BACKOFF] = remaining
                await self._back_off(failure)

    async def _suspend(self, code: str, failure: CallFailure) -> None:
        self._park()
        if self._suspension != code:
            self._suspension = code
            self._report(Event.SUSPENDED, code, None, failure)
        self._backing_off = False
        self._token = None
        await self._wait_parked(SUSPENDED_RETRY_SECONDS, stopped_taking_turns=True)

    async def _back_off(self, failure: CallFailure) -> None:
        self._park()
        if not self._backing_off:
            self._backing_off = True
            self._report(Event.BACKOFF, failure_code(failure), None, failure)
        self._failures += 1
        await self._wait_parked(
            backoff_delay(self._failures), stopped_taking_turns=backoff_states_unavailable(self._failures)
        )

    async def _wait_parked(self, seconds: float, *, stopped_taking_turns: bool) -> None:
        """Waits before the next retry. Once the loop has stopped taking turns in the app, it states ``unavailable`` at
        his seats meanwhile, the first time until the next full pass, and retries once that has gone out too. The wait
        begins at once, in the loop's own task, whether or not something is stated beside it."""
        if not (stopped_taking_turns and self._turn_states.stopped_taking_turns()):
            await self._boundaries.clock.sleep(seconds)
            return
        stating = asyncio.create_task(self._state_on_own_session(self._turn_states.state_unavailable))
        try:
            await self._boundaries.clock.sleep(seconds)
        except asyncio.CancelledError:
            stating.cancel()
            await asyncio.gather(stating, return_exceptions=True)
            raise
        await stating

    async def state_on_stop(self, seconds: float) -> None:
        """At his stop, sends within ``seconds`` what the stop left queued, the end of a turn it cut among it, and each
        seat's last end where a statement was lost with its session; what does not go out in time is logged once."""
        self._turn_states.restate_if_due()
        if self._turn_states.nothing_queued():
            return
        try:
            async with asyncio.timeout(seconds):
                await self._state_on_own_session(self._turn_states.state_queued)
        except TimeoutError:
            self._report(Event.TURN_STATE_FAILED, NETWORK_FAILURE_CODE)

    async def _state_on_own_session(self, state: Callable[[AppSession], Awaitable[None]]) -> None:
        """States on a session of its own, outside the loop's; a failure to open it is logged as a failed statement."""
        try:
            token = await self._current_token()
            async with self._boundaries.open_session(self._app, token.value) as session:
                await state(session)
        except Exception as error:
            failure = decisive_failure(tuple(leaves_of(error)), already_renewed=False)
            if failure is None:
                raise
            self._report(Event.TURN_STATE_FAILED, failure_reason(failure), None, failure)

    def _park(self) -> None:
        """A parked app holds up no place of his other apps and lends no tools to their turns: its due places are
        dropped and its place unrecorded; it reads them again after."""
        self._mind.undue(lambda at: at.app == self._app.mcp_url)
        self._mind.stand_at(self._app.mcp_url, holds=False)

    def _report(
        self,
        event: Event,
        code: str | None = None,
        app: AppTarget | None = None,
        failure: CallFailure | None = None,
        *,
        place_kind: str | None = None,
        model: str | None = None,
    ) -> None:
        self._boundaries.report(
            StateChange(
                event,
                self._agent.client_id,
                (app or self._app).mcp_url,
                code,
                place_kind=place_kind,
                model=model,
                failure=failure,
            )
        )


async def _run_loop(loop: AgentLoop, agent_task: asyncio.Task[object]) -> None:
    """Runs a loop; one that ends cancelled while its agent's task is not being cancelled ends the runner loud.

    The agent task's own count says so: it is above zero when the ownership or the runner cancels it and while its
    group aborts after a loop's loud end. A loop task's count is not read, a cancelled lifecycle session lowers it.
    """
    return await ending_loud_when_cancelled(loop.run(), partial(unrequested_unless_cancelling, agent_task))


class _WriteThrough:
    """Writes what his present gained and how his turns changed, after each change of his mind. A write the store
    could not take is reported once per outage and retried after a backoff, with whatever changed meanwhile; a write
    the fence refused raises ``StaleGenerationError``, which stops him on this pod."""

    def __init__(self, mind: Mind, owned: Owned, boundaries: Boundaries) -> None:
        self._mind = mind
        self._owned = owned
        self._boundaries = boundaries
        self._outage = StoreOutage(
            lambda event, code: boundaries.report(StateChange(event, owned.client_id, None, code))
        )
        self._written = mind.present
        self._written_turns: KeptTurns = owned.history.turns
        self._paused = (mind.turn_state().turns.not_before, owned.history.turns.paused_until)
        self._stop_deadline: float | None = None

    async def run(self) -> NoReturn:
        failures = 0
        changed = self._mind.changed()
        while True:
            await changed.wait()
            changed.clear()
            try:
                await self.flush()
            except StoreUnavailable as unavailable:
                self._outage.failed(unavailable.code)
                failures += 1
                changed.set()
                await self._boundaries.clock.sleep(backoff_delay(failures))
                continue
            self._outage.answered()
            failures = 0

    async def flush(self) -> None:
        """Writes whatever changed since the last write that was taken; nothing when nothing did. Either way the mind
        learns that what changed up to now is kept."""
        changes = self._mind.changes()
        present = self._mind.present
        state = self._mind.turn_state()
        not_before, paused_until = self._paused
        if state.turns.not_before != not_before:
            clock = self._boundaries.clock
            paused_until = pause_deadline(state.turns.not_before, clock.now(), clock.wall())
        delta = history_delta(self._written, present)
        turns = kept_turns(state, paused_until)
        if delta is not None or turns != self._written_turns:
            await self._owned.write(delta, turns)
            self._written, self._written_turns = present, turns
            self._paused = (state.turns.not_before, paused_until)
        self._mind.written(changes)

    def stopping(self, margin_seconds: float) -> None:
        """His stop begins now; it has ``margin_seconds`` to end."""
        self._stop_deadline = self._boundaries.clock.now() + margin_seconds

    def seconds_left_to_stop(self) -> float:
        assert self._stop_deadline is not None  # noqa: S101 - a stop begins before the agent's task is cancelled
        return max(0.0, self._stop_deadline - self._boundaries.clock.now())

    async def flush_on_stop(self) -> None:
        """Writes what the stop cut off, within what is left of its margin; a write the time or the store did not take
        is reported and the stop goes on, and one the fence refused is left, since another pod serves him now."""
        try:
            async with asyncio.timeout(self.seconds_left_to_stop()):
                await self.flush()
        except TimeoutError:
            self._outage.failed(STORE_TIMEOUT_CODE)
        except StoreUnavailable as unavailable:
            self._outage.failed(unavailable.code)
        except StoreDefect as defect:
            self._outage.failed(str(defect))
        except StaleGenerationError:
            return


async def run_all(config: Config, boundaries: Boundaries, ownership: Ownership) -> NoReturn:
    """One loop for every app that listens, for each agent while this pod owns it; a loop never ends on its own."""
    memory = config.memory
    lookup = None if memory is None else LookupBounds(memory.lookup_calls, memory.lookup_bytes)
    agents = {agent.client_id: agent for agent in config.agents}

    async def serve_agent(owned: Owned) -> NoReturn:
        """Every loop of the agent, over the history he was taken over with, which is written through as it grows."""
        agent_task = asyncio.current_task()
        assert agent_task is not None  # noqa: S101 - an agent is served inside its own task
        clock = boundaries.clock
        resumed = resumed_turns(owned.history.turns, clock.now(), clock.wall())
        mind = Mind(
            config.history_bytes,
            config.history_keep_bytes,
            config.budget_pause_cap_seconds,
            present_of(owned.history),
            resumed,
        )
        agent = agents[owned.client_id]
        write_through = _WriteThrough(mind, owned, boundaries)
        loops = [AgentLoop(agent, app, boundaries, mind, config.apps, lookup) for app in config.apps if app.listens]
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(write_through.run())
                for loop in loops:
                    group.create_task(_run_loop(loop, agent_task))
                try:
                    await asyncio.Event().wait()
                finally:
                    # The agent's cancel lands here before the group cancels his loops and turns, so a turn it cuts
                    # finds his mind stopping.
                    mind.stopping()
                    write_through.stopping(config.ownership.cancel_margin_seconds)
        finally:
            if agent_task.cancelling():
                # The statements share his stop's margin with the write, beside it, so neither waits on the other; the
                # write runs outside their group, so a statement's defect does not cancel his last write.
                flush = asyncio.create_task(write_through.flush_on_stop())
                try:
                    async with asyncio.TaskGroup() as stop:
                        for loop in loops:
                            stop.create_task(loop.state_on_stop(write_through.seconds_left_to_stop()))
                finally:
                    await flush
        raise AssertionError("a loop ended")  # pragma: no cover

    await ownership.serve(list(agents), serve_agent, boundaries.ready.announce)
    raise AssertionError("the ownership ended")  # pragma: no cover
