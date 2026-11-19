"""What an agent's turns at the places of one app do, stated to the app in order."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from agents.member import SetTurnStateArguments
from agents.turn import StopOutcome

from personas.decision import answers_unknown_tool, failure_reason, sync_seat_ends, turn_end_state
from personas.failures import CallFailure, within_call_time
from personas.reporting import Event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents.member import PlaceRef, Standing

    from personas.app_session import AppSession
    from personas.config import AppTarget
    from personas.decision import TurnEnding


@dataclass(frozen=True)
class _Queued:
    """A statement with the context it was made in, so it carries the trace of its turn, and the future that is done
    once it went out."""

    statement: SetTurnStateArguments
    context: contextvars.Context
    sent: asyncio.Future[None]


class TurnStates:
    """States thinking when a turn starts and, when it ends, what its end calls for while he still holds the seat;
    each seat's last turn end is kept for the run until he no longer holds the seat.

    A turn waits until its thinking went out, so it reaches the app before the turn's first round. A turn's end is
    only queued, so no statement's wait or failure reaches how the turn ends. One worker sends them in the order
    queued, so a turn's thinking never overtakes the end of the turn before it. A refusal or a failure of a statement
    is logged and the next one goes out; one the app does not answer within ``call_seconds`` fails as a refusal does.
    A seat whose app answered that it serves no such tool gets no statement after that one line.

    Once the loop has stopped taking turns in the app, each held seat is stated ``unavailable``. Then, and after a
    session that ended with a statement failed or in flight, the next full pass states each held seat's last turn end
    again.
    """

    def __init__(
        self,
        use_current_token: Callable[[AppSession], Awaitable[None]],
        report: Callable[[Event, str | None, AppTarget | None, CallFailure | None], None],
        call_seconds: float,
    ) -> None:
        self._call_seconds = call_seconds
        self._use_current_token = use_current_token
        self._report = report
        self._seat_ends: dict[PlaceRef, StopOutcome | None] = {}
        self._queue: asyncio.Queue[_Queued] = asyncio.Queue()
        self._unserved: set[PlaceRef] = set()
        self._unavailable = False
        self._restate_due = False

    def sync_seats(self, standings: tuple[Standing, ...]) -> None:
        """Forgets the last turn end of each seat he no longer holds."""
        self._seat_ends = sync_seat_ends(self._seat_ends, standings)

    async def turn_started(self, place: PlaceRef) -> None:
        await self._queue_statement(SetTurnStateArguments(place=place, thinking=True))

    def turn_ended(self, place: PlaceRef, ending: TurnEnding) -> None:
        if place not in self._seat_ends:
            return
        stopped = turn_end_state(ending, self._seat_ends[place])
        self._seat_ends[place] = stopped
        self._queue_statement(SetTurnStateArguments(place=place, stopped=stopped))

    def stopped_taking_turns(self) -> bool:
        """The loop has stopped taking turns in the app. While he holds a seat or a statement waits, it says that
        ``unavailable`` is to be stated, once until the next full pass, which then states each seat's last end
        again."""
        something_to_state = bool(self._seat_ends) or not self._queue.empty()
        if self._unavailable or not something_to_state:
            return False
        self._unavailable = self._restate_due = True
        return True

    def restate_if_due(self) -> None:
        """Once the loop had stopped taking turns, or a session ended with a statement failed or in flight, queues
        each held seat's last turn end again, ``None`` where there was none; the loop calls it after a full pass and
        at his stop."""
        if not self._restate_due:
            return
        self._unavailable = self._restate_due = False
        for place, stopped in self._seat_ends.items():
            self._queue_statement(SetTurnStateArguments(place=place, stopped=stopped))

    def nothing_queued(self) -> bool:
        return self._queue.empty()

    def _queue_statement(self, statement: SetTurnStateArguments) -> asyncio.Future[None]:
        queued = _Queued(statement, contextvars.copy_context(), asyncio.get_running_loop().create_future())
        self._queue.put_nowait(queued)
        return queued.sent

    async def state_each(self, session: AppSession) -> NoReturn:
        """Sends each statement in the order queued on the loop's ``session``, until the session ends; what is queued
        then goes out on the next session. A statement that failed or was cut in flight may not have reached the app:
        once such a session ended, the next full pass states each seat again."""
        unsure = False
        try:
            while True:
                queued = await self._queue.get()
                unsure_before, unsure = unsure, True
                unsure = not await self._send(session, queued) or unsure_before
        finally:
            if unsure:
                self._restate_due = True

    async def state_queued(self, session: AppSession) -> None:
        """Sends what is queued now, in order, on ``session``."""
        while not self._queue.empty():
            await self._send(session, self._queue.get_nowait())

    async def state_unavailable(self, session: AppSession) -> None:
        """Sends what is queued, then ``unavailable`` at each held seat until the app does not answer one, on a session
        the loop opened while it does not take turns in the app."""
        await self.state_queued(session)
        for place in tuple(self._seat_ends):
            if not await self._state(session, SetTurnStateArguments(place=place, stopped=StopOutcome.UNAVAILABLE)):
                return

    async def _send(self, session: AppSession, queued: _Queued) -> bool:
        """States what was queued in the context it was queued in; says whether the app answered."""
        answered = await asyncio.create_task(self._state(session, queued.statement), context=queued.context)
        # A turn cancelled while it waited has cancelled the future.
        if not queued.sent.done():
            queued.sent.set_result(None)
        return answered

    async def _state(self, session: AppSession, statement: SetTurnStateArguments) -> bool:
        """States one statement; says whether the app answered it, taken or refused, or none was due at its seat."""
        if statement.place in self._unserved:
            return True
        try:
            async with within_call_time(self._call_seconds):
                await self._use_current_token(session)
                refusal = await session.set_turn_state(statement)
        except CallFailure as failure:
            if answers_unknown_tool(failure):
                self._unserved.add(statement.place)
            self._report(Event.TURN_STATE_FAILED, failure_reason(failure), None, failure)
            return False
        if refusal is not None:
            self._report(Event.TURN_STATE_FAILED, refusal.value, None, None)
        return True
