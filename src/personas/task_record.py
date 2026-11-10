"""The record a running task keeps of the defect it met, and the one rule that ends a cancelled task loud."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from personas.failures import Unreachable, task_is_being_cancelled

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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

    def loud_end(self) -> Exception | None:
        """What ends the runner when the task is cancelled; asked inside the task, in its ``finally``.

        The defect comes first: the cancellation can reach a session's teardown after it and leave the defect only
        as the cancellation's context. The task's own count is above 0 on a cancel by its loop, its group or the
        runner; it is 0 when only anyio's own scope cancellation ended its session.
        """
        running = asyncio.current_task()
        assert running is not None  # noqa: S101 - a task asks its own record
        return self.defect or unrequested_unless_cancelling(running)
