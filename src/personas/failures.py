"""How a call to the realm, an app or the store fails; the loop decides what each failure means."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Final, cast

from agents.member_outcomes import StatusFailure, Unreachable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterator

STORE_UNREACHABLE_CODE: Final = "store_unreachable"
"""The code of a store that could not be reached or ended the connection."""
STORE_TIMEOUT_CODE: Final = "store_timeout"
"""The code of a write that did not finish within its bound."""
STORE_DEFECT_CODE: Final = "store_defect"
"""The code of a database error that is no outage."""


class TokenStatusFailure(StatusFailure):
    """The realm's token endpoint answered with such a status."""


class CallTimedOut(Unreachable):
    """The app did not answer within its call time."""


class CallTime:
    """The running call time of one app's request."""

    def __init__(self, call_seconds: float, bound: asyncio.Timeout) -> None:
        self._call_seconds = call_seconds
        self._bound = bound

    @contextmanager
    def paused(self) -> Generator[None]:
        """Runs its body outside the call time, for a body that bounds each of its own steps by it; the call time
        then runs again for its whole length."""
        self._bound.reschedule(None)
        try:
            yield
        finally:
            self._bound.reschedule(asyncio.get_running_loop().time() + self._call_seconds)


@asynccontextmanager
async def within_call_time(call_seconds: float) -> AsyncGenerator[CallTime]:
    """Runs its body under an app's call time; an expired bound raises ``CallTimedOut``."""
    bound = asyncio.timeout(call_seconds)
    try:
        async with bound:
            yield CallTime(call_seconds, bound)
    except TimeoutError:
        if not bound.expired():
            raise
        raise CallTimedOut("the app did not answer in time") from None


# Named for the state of the store, as Unreachable is; a suffix would repeat it.
class StoreUnavailable(Exception):  # noqa: N818
    """The store could not keep a window or be read now; ``code`` says why and the loop goes on without memory."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# Named for its kind; a suffix would repeat it.
class StoreDefect(Exception):  # noqa: N818
    """A database error that is no outage; the message holds a code and no parameter, row or secret."""


class StaleGenerationError(Exception):
    """A write of an agent's state carried a generation that is not the agent's current one: another pod has taken the
    agent over since, and the write changed nothing."""


class SchemaNotAtHeadError(Exception):
    """The database is not at the newest migration; the message names the field and the revisions, no secret."""


def leaves_of(error: BaseException) -> Iterator[BaseException]:
    """The errors an exception group holds, however deep; any other error is its own leaf."""
    if isinstance(error, BaseExceptionGroup):
        for inner in cast("BaseExceptionGroup[BaseException]", error).exceptions:
            yield from leaves_of(inner)
    else:
        yield error


def task_is_being_cancelled() -> bool:
    """Whether the running task has a cancellation requested that it has not yet taken."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def raise_if_cancelled() -> None:
    """Stops a task whose cancellation was swallowed on the way, as by a fetch or an SDK task group.

    The task still counts the request; without this the loop would go on instead of ending.
    """
    if task_is_being_cancelled():
        raise asyncio.CancelledError
