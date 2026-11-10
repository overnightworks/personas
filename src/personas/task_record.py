"""The record a running task keeps of the defect it met, and the one rule that ends a cancelled task loud."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agents.backend import TurnStopKind, TurnStopped
from agents.turn import FailureCode

from personas.decision import is_defect
from personas.failures import Unreachable, leaves_of, task_is_being_cancelled

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class RelayFailed(TurnStopped):
    """The stop a failed relayed call gives the backend, which ends the turn at once; it holds the failure."""

    def __init__(self, failure: Exception) -> None:
        super().__init__(TurnStopKind.FAILED, FailureCode.PROVIDER_ERROR)
        self.failure = failure


def failure_held_by(leaf: BaseException) -> BaseException | None:
    """The failure a session-body leaf stands for: a relay stop holds its failure, the gateway's own stop none."""
    if isinstance(leaf, RelayFailed):
        return leaf.failure
    return None if isinstance(leaf, TurnStopped) else leaf


class UnrequestedCancellationError(Exception):
    """A task ended cancelled while the task that should have been cancelled was not."""


def unrequested_unless_cancelling(asked: asyncio.Task[object]) -> UnrequestedCancellationError | None:
    return None if asked.cancelling() else UnrequestedCancellationError()


def unreachable_unless_cancelling() -> Unreachable | None:
    """The failure of a session that ended cancelled while the running task was not: its own child ended it."""
    return None if task_is_being_cancelled() else Unreachable("the session ended")


async def ending_loud_when_cancelled[T](work: Awaitable[T], loud_end: Callable[[], Exception | None]) -> T:
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


@dataclass
class TaskRecord:
    """What one running task holds of itself: the first defect a request of it met, whatever way it then ends."""

    defect: Exception | None = None

    def note(self, error: Exception) -> None:
        """Keeps the first defect among the leaves of an error the task's session body left with.

        A cancelled task loses its errors to the cancellation; the record lets the runner end loud anyway.
        """
        held = (failure_held_by(leaf) for leaf in leaves_of(error))
        failures = (failure for each in held if each is not None for failure in leaves_of(each))
        defect = next((failure for failure in failures if isinstance(failure, Exception) and is_defect(failure)), None)
        if defect is not None and self.defect is None:
            self.defect = defect

    def joined_to(self, error: Exception | None) -> BaseException | None:
        """What leaves a task: the error, with the recorded defect among its leaves unless it holds it already."""
        leaves = list(leaves_of(error)) if error is not None else []
        if self.defect is None or any(leaf is self.defect for leaf in leaves):
            return error
        if not leaves:
            return self.defect
        return BaseExceptionGroup("the task failed", [self.defect, *leaves])

    def loud_end(self) -> Exception | None:
        """What ends the runner when the task is cancelled; asked inside the task, in its ``finally``.

        The defect comes first: the cancellation can reach a session's teardown after it and leave the defect only
        as the cancellation's context. The task's own count is above 0 on a cancel by its loop, its group or the
        runner; it is 0 when only anyio's own scope cancellation ended its session.
        """
        running = asyncio.current_task()
        assert running is not None  # noqa: S101 - a task asks its own record
        return self.defect or unrequested_unless_cancelling(running)


async def ending_with_recorded_defect(work: Awaitable[None], record: TaskRecord) -> None:
    """Runs work; what ends it, the record's defect joins unless it is among the leaves."""
    failed: Exception | None = None
    try:
        await work
    except Exception as error:
        failed = error
    if (ending := record.joined_to(failed)) is not None:
        raise ending
