"""The places a fake app keeps for its members: who stands where, the lines and the view of each place, and the
signal of each change once it is visible."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Any

from agents.mcp_server import ResourceChanges
from agents.member import STANDING_URI, PlaceView, Standing, place_view_uri, place_window_uri
from agents.turn import Principal

from tests.support.placestore import MemoryPlaceStore

ISSUER = "https://realm.test/realms/test"
WINDOW_LINES = 100
"""How many of a place's newest lines its window holds, as hopin's chat window does."""
ACCEPTED_NAME = "Mozart"


@dataclass(frozen=True)
class Line:
    """A line of a place, as the app keeps it: who said it and what name it shows."""

    seq: int
    author: str
    name: str
    text: str
    spoken: bool = False
    cut: bool = False
    act: bool = False
    traceparent: str | None = None


@dataclass
class AppPlaces:
    """Who stands where, and what each place holds: its lines, its generation and its view."""

    standings: dict[str, list[Standing]] = field(default_factory=dict[str, list[Standing]])
    lines: dict[str, list[Line]] = field(default_factory=dict[str, list[Line]])
    generations: dict[str, str] = field(default_factory=dict[str, str])
    vanished: set[str] = field(default_factory=set[str])
    changes: ResourceChanges = field(default_factory=lambda: ResourceChanges(MemoryPlaceStore()))
    views: dict[str, PlaceView] = field(default_factory=dict[str, PlaceView])
    content: set[str] = field(default_factory=lambda: {ACCEPTED_NAME})

    def __post_init__(self) -> None:
        self._announcing: set[asyncio.Task[None]] = set()

    def _announce(self, principal: Principal, uri: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # before any loop runs nothing listens, and a loop's first sweep reads the state
        task = loop.create_task(self.changes.changed(principal, uri))
        self._announcing.add(task)
        task.add_done_callback(self._announcing.discard)

    def stand(self, subject: str, *standings: Standing) -> None:
        """Sets the subject's standings and signals the change, as an app does once the state is visible."""
        self.standings[subject] = list(standings)
        for each in standings:
            self.content.update([each.place.id, *([each.invitation.id] if each.invitation else [])])
        self._announce(Principal(iss=ISSUER, sub=subject), STANDING_URI)

    def speak(
        self,
        place_id: str,
        author: str,
        name: str,
        text: str,
        *,
        spoken: bool = False,
        cut: bool = False,
        act: bool = False,
        traceparent: str | None = None,
    ) -> int:
        """Adds a line, entered in ``traceparent``, to a place and signals its window to every member standing there,
        once it is visible."""
        place_lines = self.lines.setdefault(place_id, [])
        seq = place_lines[-1].seq + 1 if place_lines else 1
        place_lines.append(Line(seq, author, name, text, spoken, cut, act, traceparent))
        del place_lines[:-WINDOW_LINES]
        self.content.add(text)
        self._signal_window(place_id)
        return seq

    def _signal_window(self, place_id: str) -> None:
        for subject, standings in self.standings.items():
            for each in standings:
                if each.place.id == place_id:
                    self._announce(Principal(iss=ISSUER, sub=subject), place_window_uri(each.place))

    def show_view(
        self,
        place_id: str,
        revision: int,
        state: dict[str, Any],
        *,
        awaits_you: bool = True,
        traceparent: str | None = None,
    ) -> None:
        """Sets a place's view, moved in ``traceparent``, for every member standing there and signals it, as an app
        does once it is visible."""
        self.views[place_id] = PlaceView(
            revision=revision, awaits_you=awaits_you, state=dict(state), traceparent=traceparent
        )
        self.content.update(value for value in state.values() if isinstance(value, str))
        for subject, standings in self.standings.items():
            for each in standings:
                if each.place.id == place_id and each.serves_view:
                    self._announce(Principal(iss=ISSUER, sub=subject), place_view_uri(each.place))

    def join(self, place_id: str) -> None:
        """A join gives the place a new, existing window: a fresh generation, numbered from seq 1, with no old line."""
        self.generations[place_id] = f"generation-{secrets.token_hex(4)}"
        self.lines[place_id] = []
        self.vanished.discard(place_id)
