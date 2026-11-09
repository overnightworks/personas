"""How a call to the realm or an app fails; the loop decides what each failure means."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

INVALID_ANSWER_CODE: Final = "invalid_answer"
"""The code of an answer that arrived but is not the kind of answer asked for."""
LISTEN_NOT_HONORED_CODE: Final = "listen_not_honored"
"""The code of a stream the app acknowledged without the subscription the runner asked for."""
UNSUPPORTED_VERSION_CODE: Final = "unsupported_version"
"""The code of an app that shares no protocol revision with the runner."""


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


def leaves_of(error: BaseException) -> Iterator[BaseException]:
    """The errors an exception group holds, however deep; any other error is its own leaf."""
    if isinstance(error, BaseExceptionGroup):
        for inner in cast("BaseExceptionGroup[BaseException]", error).exceptions:
            yield from leaves_of(inner)
    else:
        yield error
