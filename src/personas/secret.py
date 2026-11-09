"""A secret value that shows itself to nobody until asked."""

from __future__ import annotations


class Secret:
    """Holds a client secret or a token; its repr and str are fixed text."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(<hidden>)"

    __str__ = __repr__
