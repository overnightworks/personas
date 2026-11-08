"""How a call to the realm or an app fails; the loop decides what each failure means."""

from __future__ import annotations


class CallFailure(Exception):  # noqa: N818 - the base of the three failures below, which name their kind
    """A call did not give an answer the loop can use."""


class StatusFailure(CallFailure):
    """The server answered with an HTTP status that is not success."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP status {status}")
        self.status = status


class TokenStatusFailure(StatusFailure):
    """The realm's token endpoint answered with such a status."""


class Unreachable(CallFailure):
    """The server was not reached, timed out or sent something that is not an answer."""


class WaitToolError(CallFailure):
    """The ``wait`` tool answered with an error result."""
