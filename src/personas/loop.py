"""The loop of one agent in one app."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from operator import eq
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from agents.mcp_client import ToolsWake
from agents.member import DeclineInvitationRefusal
from agents.member_outcomes import CallFailure, DeclineOutcome, UnrecognisedRefusal
from logline import StepTimer, unit_trace

from personas.admission import Admitted
from personas.decision import (
    NETWORK_FAILURE_CODE,
    NO_TOKEN,
    NO_TURNS_CODE,
    SUSPENDED_RETRY_SECONDS,
    AppPlace,
    Decline,
    Joined,
    Memory,
    Reaction,
    Suspend,
    Token,
    accept_arguments,
    accepts_for,
    active_places,
    active_view_places,
    active_window_places,
    after_decline,
    after_refusal,
    backoff_delay,
    backoff_states_unavailable,
    decisive_failure,
    decline_arguments,
    declines_for,
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
    revisions_after_turn,
    sync_turned_revisions,
    turn_ending,
    turn_invalid,
    view_turn_due,
    waking_line,
)
from personas.due import left_standing, lines_of_a_hold
from personas.entries import Lines, Occasion, ViewSeen, Visit, Waiting, WindowRead, taken_lines
from personas.failures import leaves_of, raise_if_cancelled
from personas.remembering import LineStore, Remembering
from personas.reporting import (
    LOG,
    DueOccasion,
    Event,
    Reporter,
    StateChange,
    TurnTally,
    agent_app_fields,
    began_ago,
    report_fell_due,
    start_turn_tally,
    stepping,
)
from personas.task_record import TaskRecord
from personas.tokens import facts_of
from personas.turn_states import TurnStates
from personas.turning import Turning, TurnSetup
from personas.voice import Sender

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from datetime import datetime

    from agents.mcp_client import MemberSession, MemberWake
    from agents.member import PlaceRef, PlaceWindow, Standing

    from personas.app_session import SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget
    from personas.due import Due
    from personas.mind import Mind
    from personas.readiness import ReadyMarker
    from personas.recall import LookupBounds
    from personas.secret import Secret
    from personas.speech import DropReason
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
    record: TaskRecord
    revision: int | None
    """The view revision the turn was started for; none for a turn that came from lines."""
    watcher: asyncio.Task[None]
    """The task that ends the turn for a spoken place once one is due."""
    cancel_requested: bool = False
    yielded: bool = False
    """The loop ended the turn at a safe point for a spoken place."""


class AgentLoop:
    """Keeps one stream of changes open for one agent in one app, accepts what it is invited to and answers.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next retry, and resumes once
    a read answers again. A network failure, a lost stream or a 5xx backs off, logged once until a read answers. In an
    app that binds tools, the lines of the agent's call places are read too, while any turn of his runs, and every new
    line and every view read goes into the agent's history. A line that is not his own and a view that waits for him
    make a turn due in the agent's mind, and after each read the loop starts the turn of the place claimed first
    when it is in this app; at most one turn of the agent runs across all his loops, and the turn's own-app
    listing, relayed calls and speak run on this loop's session, so a failure of them goes into the one reaction of the
    loop. The turn also holds the tools of the tools-only apps and of the other apps where the agent holds an active
    place when it is admitted, which the loops of those apps record in his mind: each request to one runs in a session
    of its own under the token for its own scope, and its failure never reaches the reaction. Each call the turn relays,
    to this app or another, enters the agent's history when its request ends. In an app that declares ``speak``, for an
    agent whose entry has a voice table, a turn at a call place is spoken: the sentences of its text are released into
    the mind's record, and the place's sender, a task of this loop's serving group on its session, says them one by one
    through ``speak``, also after the turn.
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
            self._renew_token,
            self._report,
            self._report_pause,
            self._report_tools_left_out,
            self._report_dropped,
            self._remembering,
        )
        self._turn_states = TurnStates(self._renew_token, self._report, app.call_seconds)
        self._token = NO_TOKEN
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None
        self._backing_off = False
        self._standings: tuple[Standing, ...] = ()
        self._turned_revisions: dict[PlaceRef, int] = {}
        self._turn: ActiveTurn | None = None
        self._senders: dict[PlaceRef, asyncio.Task[None]] = {}
        self._pause_wake: asyncio.Task[None] | None = None
        self._turn_ended = mind.wake_event()
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
        await self._renew_token()
        async with (
            asyncio.TaskGroup() as turns,
            self._boundaries.open_session(self._app, self._bearer, self._agent.client_id) as session,
        ):
            turns.create_task(self._turn_states.state_each(session))
            try:
                self._turning.new_session()
                self._mind.voice_at(self._app.mcp_url, voice=await self._is_voice_app(session))
                while True:
                    await self._renew_token()
                    async with session.listen() as changes, aclosing(self._wakes(changes)) as wakes:
                        await self._reconcile(session, turns)
                        async for _ in wakes:
                            await self._reconcile(session, turns)
            finally:
                await self._end_turn_before_session()
                await self._end_senders()
                self._turn_states.session_ended(session)

    async def _end_turn_before_session(self) -> None:
        """A turn in flight ends cancelled while the session it shares is still open, so its place stays due and its
        request is not answered with a closed connection."""
        turn = self._turn
        if turn is not None and not turn.task.done():
            turn.task.cancel()
            await asyncio.wait({turn.task})

    async def _end_senders(self) -> None:
        """The places' senders end cancelled while their session is open, so no request meets a closed connection."""
        senders, self._senders = tuple(self._senders.values()), {}
        for sender in senders:
            sender.cancel()
        if senders:
            await asyncio.wait(senders)

    def _sender_at(self, turns: asyncio.TaskGroup, session: MemberSession, place: PlaceRef) -> None:
        if place not in self._senders:
            args = (
                session,
                self._app.mcp_url,
                place,
                self._boundaries.clock,
                self._renew_token,
            )
            sender = Sender(*args, self._report, self._report_dropped, self._mind)
            self._senders[place] = turns.create_task(sender.run())

    async def _is_voice_app(self, session: MemberSession) -> bool:
        """Whether the turns at this app's call places are spoken, decided once for the lifecycle session."""
        has_voice = self._turn_setup is not None and self._agent.voice is not None
        declared = await session.declared_tools() if has_voice else ()
        return is_voice_app(has_voice, declared)

    async def _wakes(self, changes: AsyncIterator[MemberWake]) -> AsyncGenerator[None]:
        """A wake for each change of a resource the app signals and for each wake of his mind; it ends when the stream
        does. A change of the app's tools wakes no reconcile: it changes no standing."""
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
                    wake = change.result()
                    if wake is _STREAM_ENDED:
                        return
                    change = asyncio.ensure_future(anext(changes, _STREAM_ENDED))
                    if not isinstance(wake, ToolsWake):
                        yield
        finally:
            change.cancel()
            ended.cancel()
            # A defect of the change task comes out of aclose, so it ends the runner even when it lost the race to
            # the body's error, which its traceback does not keep.
            if (defect := _defect_of(change)) is not None:
                raise defect

    async def _reconcile(self, session: MemberSession, turns: asyncio.TaskGroup) -> None:
        """Reads the standing as it is now and acts on it; a wake says only that it may have changed.

        The views are read before the windows, so a window holds every line up to its view's revision; the lines enter
        the history before the views do.
        """
        began = self._boundaries.clock.now()
        standings = await self._take_standing(session)
        views = await self._read_views(session, standings)
        window_lost = await self._read_windows(session, standings, began)
        self._take_views(views, began)
        if window_lost:
            # A window the app lost dropped its place's checkpoint: the standing says where the place stands now.
            await self._take_standing(session)
        self._turn_states.restate_after_full_pass()
        await self._start_due_turn(turns, session)
        self._served()

    async def _take_standing(self, session: MemberSession) -> tuple[Standing, ...]:
        """Reads the standing, keeps the checkpoints in line with it and accepts what it invites to."""
        await self._renew_token()
        standing = await session.read_standing()
        self._memory = forget_gone(self._memory, standing.standings)
        self._mind.sync_checkpoints(self._app.mcp_url, standing.standings, self._standings)
        self._standings = standing.standings
        self._turned_revisions = sync_turned_revisions(self._turned_revisions, standing.standings)
        self._turn_states.sync_seats(standing.standings)
        self._mind.left(left_standing(self._app.mcp_url, standing.standings))
        self._mind.stand_at(
            self._app.mcp_url, holds=lends_tools(self._app.max_tool_calls is not None, standing.standings)
        )
        self._mind.hold_places(self._app.mcp_url, active_places(standing.standings))
        self._cancel_turn_if_its_place_left(standing.standings)
        await self._accept_invitations(session, standing.standings)
        return standing.standings

    def _cancel_turn_if_its_place_left(self, standings: tuple[Standing, ...]) -> None:
        """A turn ends with its place; its slot stays taken until its task has ended."""
        turn = self._turn
        if turn is not None and not turn.cancel_requested and turn_invalid(turn.place, standings):
            self._turn = replace(turn, cancel_requested=True)
            turn.task.cancel()

    async def _renew_token(self) -> None:
        """Makes the cached token one the realm granted for now; a session reads it through ``_bearer`` at each call."""
        await self._current_token()

    def _bearer(self) -> Secret:
        """The cached token's value. Every session is opened after a renewal and the cache is cleared only after the
        sessions have closed, so no session reads the placeholder."""
        return self._token.value

    async def _current_token(self) -> Token:
        if renewal_due(self._token, self._boundaries.clock.now()):
            granted = await self._boundaries.tokens.fetch(self._agent, self._app.scope)
            expires_at = self._boundaries.clock.now() + granted.lifetime_seconds
            self._token = Token(granted.value, expires_at)
        return self._token

    def _served(self) -> None:
        """A full pass of standing, accepts and windows went through: the failure counters start over, and a
        suspension or backoff has ended."""
        self._failures = 0
        self._renewed_after_401 = False
        if self._suspension is not None or self._backing_off:
            self._suspension = None
            self._backing_off = False
            self._report(Event.RESUMED)

    async def _accept_invitations(self, session: MemberSession, standings: tuple[Standing, ...]) -> None:
        takes_turns = self._turn_setup is not None
        self._memory, left_out = left_out_for(standings, self._memory, takes_turns=takes_turns)
        for standing in left_out:
            self._report(Event.INVITATION_LEFT_OUT, NO_TURNS_CODE, place_kind=standing.place.kind)
        # No await between reading what is held elsewhere and claiming the places to accept: another app's loop that
        # decides meanwhile sees them held and declines its own invitations.
        held_elsewhere = self._mind.holds_active_place_elsewhere(self._app.mcp_url)
        accepts = accepts_for(
            standings,
            self._memory,
            takes_turns=takes_turns,
            held_elsewhere=held_elsewhere,
            figure=self._agent.figure,
            room=self._agent.room,
            voices=() if self._agent.voice is None else self._agent.voice.names,
        )
        unjoined = frozenset(accept.place for accept in accepts)
        self._mind.claim_places(self._app.mcp_url, unjoined)
        try:
            for decline in declines_for(standings, self._memory, held_elsewhere=held_elsewhere):
                await self._decline(session, decline)
            for accept in accepts:
                await self._renew_token()
                refusal = await session.accept(accept_arguments(accept))
                if refusal is None:
                    unjoined = unjoined.difference((accept.place,))
                    self._mind.move_checkpoint(AppPlace(self._app.mcp_url, accept.place), Joined())
                    self._report(Event.JOINED)
                    continue
                self._memory, is_news = after_refusal(self._memory, accept, refusal)
                if is_news:
                    self._report(Event.REFUSED, refusal.value)
        finally:
            self._mind.release_places(self._app.mcp_url, unjoined)

    async def _decline(self, session: MemberSession, decline: Decline) -> None:
        await self._renew_token()
        answer = await session.decline(decline_arguments(decline))
        self._memory = after_decline(self._memory, decline, answer)
        if isinstance(answer, DeclineInvitationRefusal | UnrecognisedRefusal):
            self._report(Event.DECLINE_REFUSED, answer.value, place_kind=decline.place.kind)
        elif answer is DeclineOutcome.SENT:
            self._report(Event.INVITATION_DECLINED, decline.reason.value, place_kind=decline.place.kind)

    async def _read_windows(self, session: MemberSession, standings: tuple[Standing, ...], began: float) -> bool:
        """Reads the active window places' windows, a turn of his running or not, and takes their lines in. ``began`` is
        when the pass began, for the due it logs.

        It says whether a window was lost, which dropped its place's checkpoint.
        """
        if self._turn_setup is None:
            return False
        serving_view = frozenset(active_view_places(standings))
        lost = False
        for place in active_window_places(standings):
            await self._renew_token()
            window = await session.read_place(place)
            if window is None:
                self._mind.move_checkpoint(AppPlace(self._app.mcp_url, place), None)
                lost = True
                continue
            read_at = self._boundaries.clock.wall()
            self._take_lines(place, window, read_at, began, serves_view=place in serving_view)
            if await self._turning.hold(session, place, window):
                at = AppPlace(self._app.mcp_url, place)
                self._report_fell_due(at, lines_of_a_hold(window.generation), DueOccasion.HOLD, began)
            await self._remembering.keep(WindowRead(self._agent.client_id, self._app.mcp_url, place, window, read_at))
        return lost

    async def _read_views(
        self, session: MemberSession, standings: tuple[Standing, ...]
    ) -> dict[PlaceRef, ViewSeen | None]:
        """Reads the views of the active view places in the standing's order, each stamped with the time of its read; a
        place the app gives no view of maps to none."""
        if self._turn_setup is None:
            return {}
        read: dict[PlaceRef, ViewSeen | None] = {}
        for place in active_view_places(standings):
            await self._renew_token()
            view = await session.read_view(place)
            read[place] = (
                None if view is None else ViewSeen(self._app.mcp_url, place, view, self._boundaries.clock.wall())
            )
        return read

    def _take_views(self, read: dict[PlaceRef, ViewSeen | None], began: float) -> None:
        """Takes each view read into the agent's history. A revision that waits for him and has no turn yet makes a turn
        due at its place; one that does not wait, and a view the app gives none of, take the place's due away. A
        revision waiting for him that has had its turn keeps a due that turn left behind by ending without an answer.
        No await stands in it."""
        for place, seen in read.items():
            at = AppPlace(self._app.mcp_url, place)
            if seen is None:
                self._mind.undue(partial(eq, at))
                continue
            self._mind.take((replace(seen, generation=self._mind.generation_at(at)),))
            if view_turn_due(self._turned_revisions.get(place), seen.view):
                waiting = Waiting(seen.view.revision, seen.view.traceparent)
                if self._mind.due(at, waiting, self._boundaries.clock.now()):
                    self._report_fell_due(at, waiting, DueOccasion.VIEW, began)
            elif not seen.view.awaits_you:
                self._mind.undue(partial(eq, at))

    async def _start_due_turn(self, turns: asyncio.TaskGroup, session: MemberSession) -> None:
        """Starts the turn of the place claimed first when it is in this app and none of his runs. The claim
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
        aloud = self._mind.speaks(at, occasion)
        apps = self._mind.held_apps(self._app.mcp_url)
        held = self._mind.hold_to_tell(at)
        went_on = self._mind.went_on_to_tell(at)
        if aloud:
            self._sender_at(turns, session, at.place)
        admitted = Admitted(
            facts, at.place, occasion, self._mind.present, self._boundaries.clock.wall(), aloud, apps, held, went_on
        )
        self._start_turn(turns, session, setup, at.place, admitted, due, claimed)

    def _wake_when_pause_over(self, turns: asyncio.TaskGroup, now: float) -> None:
        """A pause holds the turn of the place at this app that is claimed first: one waiter of this loop wakes
        every loop of his when it is over, and a waiter that ended early sees the rest of the pause again."""
        seconds = self._mind.pause_left(self._app.mcp_url, now)
        if seconds > 0 and (self._pause_wake is None or self._pause_wake.done()):
            self._pause_wake = turns.create_task(self._wake_after(seconds))

    async def _wake_after(self, seconds: float) -> None:
        await self._boundaries.clock.sleep(seconds)
        self._mind.wake()

    def _take_lines(
        self, place: PlaceRef, window: PlaceWindow, read_at: datetime, began: float, *, serves_view: bool
    ) -> None:
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
            if self._mind.due(at, lines, self._boundaries.clock.now()):
                self._report_fell_due(at, lines, DueOccasion.LINE, began)

    def _report_fell_due(self, at: AppPlace, occasion: Occasion, kind: DueOccasion, began: float) -> None:
        """Writes that the pass made ``at`` due, in the occasion's trace when it carries one; a hold has none."""
        spoken = self._mind.speaks(at, occasion)
        read_ms = round((self._boundaries.clock.now() - began) * 1000)
        with nullcontext() if occasion.traceparent is None else unit_trace(occasion.traceparent):
            report_fell_due(self._agent.client_id, self._app.mcp_url, kind, spoken=spoken, read_ms=read_ms)

    def _start_turn(
        self,
        turns: asyncio.TaskGroup,
        session: MemberSession,
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
            wait = StepTimer(
                "turn_wait.ended",
                LOG,
                started_at=began_ago(max(0.0, self._boundaries.clock.now() - due.since)),
                **agent_app_fields(self._agent.client_id, self._app.mcp_url),
            )
            wait.add(wait_reason=due.waited_on.value)
            wait.end()
            tally = start_turn_tally(self._agent.client_id, self._app.mcp_url)
            record = TaskRecord()
            task = turns.create_task(self._take_once_kept(claimed, session, setup, admitted, record, tally))
            occasion = admitted.occasion
            watcher = turns.create_task(self._yield_when_a_spoken_place_is_due(task))
            self._turn = ActiveTurn(
                task,
                place,
                tally,
                record,
                occasion.revision if isinstance(occasion, Waiting) else None,
                watcher=watcher,
            )
            task.add_done_callback(partial(self._turn_over, session))

    async def _take_once_kept(
        self,
        claimed: int,
        session: MemberSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        tally: TurnTally,
    ) -> None:
        """Takes the turn, which waits for its claim, the ``claimed``-th change of his mind, to be kept in the store
        and for its thinking to be answered before its first model request."""
        await self._turning.take(
            session,
            setup,
            admitted,
            record,
            tally,
            partial(self._claim_kept_and_thinking, claimed, admitted.place),
        )

    async def _claim_kept_and_thinking(self, claimed: int, place: PlaceRef) -> None:
        """The gate of a turn: the claim is kept, then the thinking statement is answered."""
        with stepping(
            StepTimer("claim_write.ended", LOG, **agent_app_fields(self._agent.client_id, self._app.mcp_url))
        ):
            await self._mind.until_written(claimed)
        with stepping(StepTimer("thinking.ended", LOG, **agent_app_fields(self._agent.client_id, self._app.mcp_url))):
            await self._turn_states.turn_started(place)

    async def _yield_when_a_spoken_place_is_due(self, task: asyncio.Task[None]) -> None:
        """The running written turn ends at its next safe point once a spoken place is due: at once while the model
        thinks, else once the call in flight has answered or ended and entered his history. It ends with the turn."""
        await self._mind.cut_wanted().wait()
        self._turning.at_safe_point(partial(self._yield_turn, task))

    def _yield_turn(self, task: asyncio.Task[None]) -> None:
        turn = self._turn
        if turn is not None and turn.task is task and not task.done() and not turn.cancel_requested:
            self._turn = replace(turn, yielded=True)
            task.cancel()

    def _turn_over(self, session: MemberSession, task: asyncio.Task[None]) -> None:
        """The one writer of ``turn.ended``, whatever way the task ended. A view revision whose turn read its tool list
        before a tools notice that came before the turn ended is not turned yet."""
        turn, self._turn = self._turn, None
        assert turn is not None  # noqa: S101 - the loop clears its turn only here and once per task
        turn.watcher.cancel()
        cancelled = task.cancelled()
        tally = turn.tally
        ending = turn_ending(
            cancelled=cancelled,
            error=None if cancelled else task.exception(),
            stop_code=tally.stop_code,
            answered=tally.answered,
        )
        tally.cut_for_spoken = turn.yielded and cancelled and not tally.answered
        tally.end(ending)
        self._turned_revisions = revisions_after_turn(
            self._turned_revisions, turn.place, turn.revision, turn.record.tools_notices_read, session.tools_notices()
        )
        self._turn_states.turn_ended(turn.place, ending)
        self._mind.hold_turn_ended(AppPlace(self._app.mcp_url, turn.place))
        if tally.cut_for_spoken:
            self._mind.cut_for_spoken()
        elif cancelled and not tally.answered and not turn.cancel_requested:
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

    def _report_dropped(self, reason: DropReason, count: int) -> None:
        if count:
            self._report(Event.SPEECH_DROPPED, reason.value, sentence_count=count)

    def _report_tools_left_out(self, app: AppTarget, count: int) -> None:
        self._boundaries.report(StateChange(Event.TOOLS_LEFT_OUT, self._agent.client_id, app.mcp_url, tool_count=count))

    async def _react(self, failure: CallFailure) -> None:
        match reaction_to(failure, already_renewed=self._renewed_after_401):
            case Reaction.RENEW:
                self._renewed_after_401 = True
                self._token = NO_TOKEN
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
        self._token = NO_TOKEN
        await self._wait_parked(SUSPENDED_RETRY_SECONDS, stopped_taking_turns=True)

    async def _back_off(self, failure: CallFailure) -> None:
        self._park()
        self._suspension = None
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
            await stating
        finally:
            # A wait that ends on an error or a cancel leaves no statement running beside the loop.
            stating.cancel()
            await asyncio.gather(stating, return_exceptions=True)

    async def state_on_stop(self, seconds: float) -> None:
        """At his stop, sends within ``seconds`` what the stop left queued, the end of a turn it cut among it, and each
        seat's last end where a statement was lost with its session and ``unavailable`` does not stand; what does not go
        out in time is logged once."""
        self._turn_states.restate_at_stop()
        if self._turn_states.nothing_queued():
            return
        try:
            async with asyncio.timeout(seconds):
                await self._state_on_own_session(self._turn_states.state_queued)
        except TimeoutError:
            self._report(Event.TURN_STATE_FAILED, NETWORK_FAILURE_CODE)

    async def _state_on_own_session(self, state: Callable[[MemberSession], Awaitable[None]]) -> None:
        """States on a session of its own, outside the loop's; a failure to open it is logged as a failed statement."""
        try:
            await self._renew_token()
            async with self._boundaries.open_session(self._app, self._bearer, self._agent.client_id) as session:
                await state(session)
        except Exception as error:
            failure = decisive_failure(tuple(leaves_of(error)), already_renewed=False)
            if failure is None:
                raise
            self._report(Event.TURN_STATE_FAILED, failure_reason(failure), None, failure)

    def _park(self) -> None:
        """A parked app holds up no place of his other apps, none of its places counts as held and it lends no tools to
        their turns: its due places are dropped and its place unrecorded; it reads them again after."""
        self._mind.undue(lambda at: at.app == self._app.mcp_url)
        self._mind.stand_at(self._app.mcp_url, holds=False)
        self._mind.hold_places(self._app.mcp_url, frozenset())

    def _report(
        self,
        event: Event,
        code: str | None = None,
        app: AppTarget | None = None,
        failure: CallFailure | None = None,
        *,
        place_kind: str | None = None,
        model: str | None = None,
        sentence_count: int | None = None,
        hold_id: str | None = None,
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
                sentences=sentence_count,
                hold_id=hold_id,
            )
        )
