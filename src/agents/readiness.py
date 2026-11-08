"""The readiness marker boundary: a file whose existence tells the platform that the start worked."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class ReadinessError(Exception):
    """The marker cannot be removed or created; the message names its path."""


class ReadyMarker(Protocol):
    def clear(self) -> None:
        """Removes a marker left from an earlier run."""
        ...

    def announce(self) -> None:
        """Creates the marker; asking again changes nothing."""
        ...


class FileReadyMarker:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._announced = False

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            raise ReadinessError(f"cannot remove the ready file {self._path}") from None

    def announce(self) -> None:
        if self._announced:
            return
        try:
            self._path.touch()
        except OSError:
            raise ReadinessError(f"cannot create the ready file {self._path}") from None
        self._announced = True
