"""What an agent's turns at the places of one app do, stated to the app in order on the loop's session."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from agents.member import SetTurnStateArguments

from personas.decision import failure_code, sync_seat_ends, turn_end_state
from personas.failures import CallFailure
from personas.reporting import Event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents.member import PlaceRef, Standing
    from agents.turn import StopOutcome

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
    is logged and the next one goes out.
    """

    def __init__(
        self,
        use_current_token: Callable[[AppSession], Awaitable[None]],
        report: Callable[[Event, str | None, AppTarget | None, CallFailure | None], None],
    ) -> None:
        self._use_current_token = use_current_token
        self._report = report
        self._seat_ends: dict[PlaceRef, StopOutcome | None] = {}
        self._queue: asyncio.Queue[_Queued] = asyncio.Queue()

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

    def _queue_statement(self, statement: SetTurnStateArguments) -> asyncio.Future[None]:
        queued = _Queued(statement, contextvars.copy_context(), asyncio.get_running_loop().create_future())
        self._queue.put_nowait(queued)
        return queued.sent

    async def state_each(self, session: AppSession) -> NoReturn:
        """Sends each statement in the order queued on the loop's ``session``, until the session ends; what is queued
        then goes out on the next session."""
        while True:
            queued = await self._queue.get()
            stating = asyncio.create_task(self._state(session, queued.statement), context=queued.context)
            await stating
            # A turn cancelled while it waited has cancelled the future.
            if not queued.sent.done():
                queued.sent.set_result(None)

    async def _state(self, session: AppSession, statement: SetTurnStateArguments) -> None:
        try:
            await self._use_current_token(session)
            refusal = await session.set_turn_state(statement)
        except CallFailure as failure:
            self._report(Event.TURN_STATE_FAILED, failure_code(failure), None, failure)
            return
        if refusal is not None:
            self._report(Event.TURN_STATE_FAILED, refusal.value, None, None)
