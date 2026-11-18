"""How a call to the realm, an app or the store fails; the loop decides what each failure means."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final, cast

import anyio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

INVALID_ANSWER_CODE: Final = "invalid_answer"
"""The code of an answer that arrived but is not the kind of answer asked for."""
LISTEN_NOT_HONORED_CODE: Final = "listen_not_honored"
"""The code of a stream the app acknowledged without the subscription the runner asked for."""
UNSUPPORTED_VERSION_CODE: Final = "unsupported_version"
"""The code of an app that shares no protocol revision with the runner."""
STORE_UNREACHABLE_CODE: Final = "store_unreachable"
"""The code of a store that could not be reached or ended the connection."""
STORE_TIMEOUT_CODE: Final = "store_timeout"
"""The code of a write that did not finish within its bound."""
STORE_DEFECT_CODE: Final = "store_defect"
"""The code of a database error that is no outage."""
LATE_ANSWER_ERRORS: Final = (anyio.BrokenResourceError, anyio.ClosedResourceError)
"""What the transport's task raises when an answer lands after its session ended."""


# The base of the failures below, which name their kind; a suffix would repeat it.
class CallFailure(Exception):  # noqa: N818
    """A call did not give an answer the loop can use; an app's error status says the failed MCP method and answer."""

    method: str | None = None
    answer: str | None = None


class StatusFailure(CallFailure):
    """The server answered with an HTTP status that is not success.

    An app's answer says the MCP method that failed and what it carried: a code of the JSON-RPC error, else its
    content type; never its content.
    """

    def __init__(self, status: int, method: str | None = None, answer: str | None = None) -> None:
        super().__init__(f"HTTP status {status}")
        self.status = status
        self.method = method
        self.answer = answer


class TokenStatusFailure(StatusFailure):
    """The realm's token endpoint answered with such a status."""


class Unreachable(CallFailure):
    """The server was not reached or timed out."""


@asynccontextmanager
async def within_call_time(call_seconds: float) -> AsyncGenerator[None]:
    """Runs its body under an app's call time; an expired bound raises ``Unreachable``."""
    bound = asyncio.timeout(call_seconds)
    try:
        async with bound:
            yield
    except TimeoutError:
        if not bound.expired():
            raise
        raise Unreachable("the app did not answer in time") from None


class UnusableAnswer(CallFailure):
    """The server answered, but with an error or a body the loop cannot use; ``code`` says which."""

    def __init__(self, code: str, method: str | None = None, answer: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.method = method
        self.answer = answer


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
