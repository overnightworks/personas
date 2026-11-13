"""What an agent took in, did and said, as the history keeps it, and how a turn is told of it: pure values with no
process, network or clock."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.member import PlaceEntry, PlaceRef, PlaceView, PlaceWindow, SpeakAnswer, SpeakRefusal, Spoken, Written

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from agents.turn import JsonObject


@dataclass(frozen=True)
class WindowRead:
    """A window as one agent read it from one app's place, and the wall time of that read."""

    client_id: str
    app: str
    place: PlaceRef
    window: PlaceWindow
    read_at: datetime


@dataclass(frozen=True)
class Visit:
    """One stay of an agent in a place: the app, the place and the window's generation, which a new join renews."""

    app: str
    place: PlaceRef
    generation: str


@dataclass(frozen=True)
class KeptLine:
    """A line the memory kept as a visit's window gave it, with the wall time it was first read."""

    visit: Visit
    seq: int
    first_read_at: datetime
    name: str = field(repr=False)
    kind: str
    text: str = field(repr=False)
    own: bool
    spoken: bool
    cut: bool
    act: bool


@dataclass(frozen=True)
class ViewSeen:
    """A revision of a place's view as an agent read it, with the wall time of that read."""

    app: str
    place: PlaceRef
    view: PlaceView = field(repr=False)
    read_at: datetime


@dataclass(frozen=True)
class Answered:
    """The app answered a call; an error result is an answer too."""

    text: str = field(repr=False)
    is_error: bool


@dataclass(frozen=True)
class Unanswered:
    """The app gave no answer, so whether the call took effect is unknown: ``code`` is the failure's, or the code of a
    call cut in flight."""

    code: str


type ActOutcome = Answered | Unanswered


@dataclass(frozen=True)
class ActDone:
    """A call relayed for an agent to the app at ``app``, with the wall time its relay ended."""

    app: str
    tool: str
    arguments: JsonObject = field(repr=False)
    outcome: ActOutcome
    at: datetime


type SpeechOutcome = SpeakAnswer | SpeakRefusal | Unanswered


@dataclass(frozen=True)
class SaidAloud:
    """A sentence an agent's turn sent to the ``speak`` of the app at ``app``, with the wall time its request ended."""

    app: str
    place: PlaceRef
    text: str = field(repr=False)
    outcome: SpeechOutcome
    at: datetime


type Entry = KeptLine | ViewSeen | ActDone | SaidAloud


def _own_time(entry: KeptLine | ActDone | SaidAloud) -> datetime:
    return entry.first_read_at if isinstance(entry, KeptLine) else entry.at


def told_bytes(told: dict[str, object]) -> int:
    return len(json.dumps(told, ensure_ascii=False).encode())


@dataclass(frozen=True)
class InputSizes:
    """The bytes a turn's input is made of: the tool declarations each model request carries, the told message and
    the tool results relayed back to the model."""

    tools_bytes: int
    told_bytes: int
    results_bytes: int


def message_bytes(role: str, text: str) -> int:
    """The bytes of one chat message of the given role, counted as ``told_bytes`` counts."""
    return told_bytes({"role": role, "content": text})


def entry_size(entry: Entry) -> int:
    """A view counts the bytes of its state's JSON; a line, an act and a spoken sentence count the bytes of the JSON
    they are told as, with their age as 0 and no mark as new, so a size does not change while the entry stays in the
    history."""
    if isinstance(entry, ViewSeen):
        return len(json.dumps(entry.view.state, ensure_ascii=False).encode())
    return told_bytes(told_entry(entry, _own_time(entry), new=False))


def newest_that_fit(sizes_newest_first: Iterable[int], size_bytes: int) -> int:
    """How many entries, whose sizes are given newest first, fit ``size_bytes``."""
    return sum(1 for total in itertools.accumulate(sizes_newest_first) if total <= size_bytes)


@dataclass(frozen=True)
class History:
    """Every line one agent took in, every view revision he read, every call relayed for him and every sentence sent to
    a ``speak`` for him since the runner started, in the order they came, within the bound of the history.

    ``left_out`` counts the entries that ever left it by the bound and ``heard`` the entries not his own that were ever
    appended. ``entries`` are the entries ever appended with the ordinals ``left_out`` up to ``taken - 1``, without a
    gap: ``history_after`` is the only writer, appends only at the newest end and removes only a prefix. A change
    that removes entries from elsewhere gives each entry its ordinal first.
    """

    entries: tuple[Entry, ...] = field(repr=False)
    left_out: int
    heard: int

    @property
    def older_dropped(self) -> bool:
        return self.left_out > 0

    @property
    def taken(self) -> int:
        """How many entries were ever appended."""
        return self.left_out + len(self.entries)


def taken_lines(visit: Visit, new: Iterable[PlaceEntry], read_at: datetime) -> tuple[KeptLine, ...]:
    """The lines a read of the visit's window took, as the history keeps them."""
    return tuple(
        KeptLine(visit, line.seq, read_at, line.name, line.kind, line.text, line.own, line.spoken, line.cut, line.act)
        for line in new
    )


def told_line(line: KeptLine) -> dict[str, object]:
    told: dict[str, object] = {
        "name": line.name,
        "kind": line.kind,
        "text": line.text,
        "own": line.own,
        "spoken": line.spoken,
    }
    if line.cut:
        told["cut"] = True
    if line.act:
        told["act"] = True
    return told


def told_place(place: PlaceRef) -> dict[str, str]:
    return {"kind": place.kind, "id": place.id}


def told_outcome(outcome: ActOutcome) -> dict[str, object]:
    if isinstance(outcome, Answered):
        return {"answered": True, "is_error": outcome.is_error, "text": outcome.text}
    return {"answered": False, "code": outcome.code}


def told_speech_outcome(outcome: SpeechOutcome) -> dict[str, object]:
    if isinstance(outcome, Written):
        return {"answered": True, "written": True}
    if isinstance(outcome, Spoken):
        return {"answered": True}
    if isinstance(outcome, SpeakRefusal):
        return {"answered": True, "refused": outcome.value}
    return {"answered": False, "code": outcome.code}


def told_entry(entry: Entry, at: datetime, *, new: bool) -> dict[str, object]:
    if isinstance(entry, SaidAloud):
        return {
            "entry": "speech",
            "app": entry.app,
            "place": told_place(entry.place),
            "seconds_ago": int((at - entry.at).total_seconds()),
            "text": entry.text,
            "outcome": told_speech_outcome(entry.outcome),
            "new": False,
        }
    if isinstance(entry, ActDone):
        return {
            "entry": "act",
            "app": entry.app,
            "seconds_ago": int((at - entry.at).total_seconds()),
            "tool": entry.tool,
            "arguments": entry.arguments,
            "outcome": told_outcome(entry.outcome),
            "new": new,
        }
    if isinstance(entry, ViewSeen):
        return {
            "entry": "view",
            "app": entry.app,
            "place": told_place(entry.place),
            "seconds_ago": int((at - entry.read_at).total_seconds()),
            "revision": entry.view.revision,
            "awaits_you": entry.view.awaits_you,
            "state": entry.view.state,
            "new": new,
        }
    return (
        {
            "entry": "line",
            "app": entry.visit.app,
            "place": told_place(entry.visit.place),
            "seconds_ago": int((at - entry.first_read_at).total_seconds()),
            "seq": entry.seq,
        }
        | told_line(entry)
        | {"new": new}
    )
