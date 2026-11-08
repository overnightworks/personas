"""How a call to the realm or an app fails; the loop decides what each failure means."""

from __future__ import annotations

from typing import Final

INVALID_ANSWER_CODE: Final = "invalid_answer"
"""The code of an answer that arrived but is not the kind of answer asked for."""


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


class WaitToolError(CallFailure):
    """The ``wait`` tool answered with an error result."""


class UnusableAnswer(CallFailure):
    """The app answered, but with an error or a body the loop cannot use; ``code`` says which."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
