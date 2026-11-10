"""How a call to the realm, an app or the store fails; the loop decides what each failure means."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import anyio

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    """A call did not give an answer the loop can use."""


class StatusFailure(CallFailure):
    """The server answered with an HTTP status that is not success."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP status {status}")
        self.status = status


class TokenStatusFailure(StatusFailure):
    """The realm's token endpoint answered with such a status."""


class Unreachable(CallFailure):
    """The server was not reached or timed out."""


class UnusableAnswer(CallFailure):
    """The server answered, but with an error or a body the loop cannot use; ``code`` says which."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# Named for the state of the store, as Unreachable is; a suffix would repeat it.
class StoreUnavailable(Exception):  # noqa: N818
    """The store could not keep a window now; ``code`` says why and the loop goes on without memory."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# Named for its kind; a suffix would repeat it.
class StoreDefect(Exception):  # noqa: N818
    """A database error that is no outage; the message holds a code and no parameter, row or secret."""


def leaves_of(error: BaseException) -> Iterator[BaseException]:
    """The errors an exception group holds, however deep; any other error is its own leaf."""
    if isinstance(error, BaseExceptionGroup):
        for inner in cast("BaseExceptionGroup[BaseException]", error).exceptions:
            yield from leaves_of(inner)
    else:
        yield error
