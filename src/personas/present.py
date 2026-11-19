"""What his next request holds: the history and the answered turns kept present, what entered since a mark, and the
messages they are told in; with no process, network or clock.

Within the pure chain it imports ``entries`` and ``decision`` and is imported by ``due``: ``entries`` <- ``recall`` <-
``decision`` <- ``present`` <- ``due``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agents.turn import MAX_EVENT_BYTES

from personas.entries import (
    ActDone,
    Entry,
    History,
    HistoryDelta,
    KeptHistory,
    KeptLine,
    Lines,
    SaidAloud,
    TurnTold,
    ViewSeen,
    Visit,
    entry_size,
    newest_that_fit,
    told_bytes,
    told_entry,
    told_place,
    told_time,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.member import PlaceRef



def _revision_key(view: ViewSeen) -> tuple[str, PlaceRef, int]:
    return view.app, view.place, view.view.revision


def _holds_revision(entries: Iterable[Entry], view: ViewSeen) -> bool:
    return any(isinstance(entry, ViewSeen) and _revision_key(entry) == _revision_key(view) for entry in entries)


def _not_his_own(entry: Entry) -> bool:
    return not isinstance(entry, ActDone | SaidAloud) and not (isinstance(entry, KeptLine) and entry.own)


@dataclass(frozen=True)
class Present:
    """What one agent's next request holds besides its header and its own turn: the history and the answered turns whose
    messages are still in it, oldest first. ``told_left_out`` counts the answered turns that ever left it, so the turn
    ``told[i]`` was the ``told_left_out + i``-th answered."""

    history: History
    told: tuple[TurnTold, ...] = ()
    told_left_out: int = 0

    @property
    def answered(self) -> int:
        """How many answered turns ever entered it."""
        return self.told_left_out + len(self.told)


EMPTY_PRESENT: Final = Present(History((), left_out=0, heard=0))


def _step_cut(history: History, boundaries: Iterable[int], keep_bytes: int) -> int:
    """The ordinal the entries before which leave in a step: the oldest end of a message from which on what stays fits
    ``keep_bytes``, or, when the newest message alone does not fit, the oldest of its entries that fits with the
    newer ones. The newest entry stays."""
    fitting = max(1, newest_that_fit(map(entry_size, reversed(history.entries)), keep_bytes))
    oldest_that_fits = history.taken - fitting
    return min((end for end in boundaries if oldest_that_fits <= end < history.taken), default=oldest_that_fits)


def _stays_told(turn: TurnTold, left_out: int) -> bool:
    """Whether the message of ``turn`` has an entry left to tell once the history has dropped the first ``left_out``."""
    return turn.through > left_out


def taken_into(present: Present, new: Iterable[Entry], history_bytes: int, keep_bytes: int) -> Present:
    """The present with ``new`` appended, a view revision it already holds adding nothing and an act always added. Past
    ``history_bytes`` the oldest whole messages leave in one step until what stays fits ``keep_bytes``; only the newest
    message may be cut, and its newest entry always stays. A step's cut points are the ends of the answered messages, so
    the cut can fall inside the span of a turn still in flight: when it is answered, its message renders partly cut,
    and nothing is told twice or lost."""
    entries = present.history.entries
    heard = present.history.heard
    for entry in new:
        if not (isinstance(entry, ViewSeen) and _holds_revision(entries, entry)):
            entries = (*entries, entry)
            heard += _not_his_own(entry)
    grown = History(entries, present.history.left_out, heard)
    if sum(map(entry_size, entries)) <= history_bytes:
        return Present(grown, present.told, present.told_left_out)
    cut = _step_cut(grown, (each.through for each in present.told), keep_bytes)
    history = History(entries[cut - grown.left_out :], cut, heard)
    told = tuple(each for each in present.told if _stays_told(each, cut))
    return Present(history, told, present.told_left_out + len(present.told) - len(told))


def answered_into(present: Present, turn: TurnTold) -> Present:
    """The present once ``turn`` was answered: its message is part of every request after it, while it stays. A turn
    whose entries all left in a step while it ran has no message left to tell."""
    if not _stays_told(turn, present.history.left_out):
        return present
    return Present(present.history, (*present.told, turn), present.told_left_out)


def history_delta(written: Present, current: Present) -> HistoryDelta | None:
    """What ``current`` gained and lost since ``written``, an earlier value of the same present that the store holds;
    ``None`` when nothing. What entered and left again between the two is never written."""
    first = max(written.history.taken, current.history.left_out)
    first_told = max(written.answered, current.told_left_out)
    entries = current.history.entries[first - current.history.left_out :]
    told = current.told[first_told - current.told_left_out :]
    if not entries and not told and current.history.left_out == written.history.left_out:
        return None
    return HistoryDelta(first, entries, first_told, told, current.history.left_out)


def present_of(kept: KeptHistory) -> Present:
    """The present a history the store keeps resumes as: what he heard before the store's oldest entry is not
    counted, since only a change of the count matters to what arrives next."""
    heard = sum(map(_not_his_own, kept.entries))
    return Present(History(kept.entries, kept.first, heard), kept.told, kept.first_told)


@dataclass(frozen=True)
class Arrived:
    """The entries that entered a history since an earlier value of it, and whether any of them left again by the
    bound."""

    entries: tuple[Entry, ...] = field(repr=False)
    older_dropped: bool


def _entered_since(current: History, taken: int) -> tuple[Entry, ...]:
    """The entries of ``current`` that were appended after the first ``taken`` ones that were ever appended."""
    return current.entries[max(0, taken - current.left_out) :]


def arrived_since(current: History, admitted: History) -> Arrived | None:
    """What entered ``current`` since ``admitted``, an earlier value of the same history, when any of it is not his own;
    otherwise ``None``. His own lines, his acts and his spoken sentences alone arrive as nothing."""
    if current.heard == admitted.heard:
        return None
    return Arrived(_entered_since(current, admitted.taken), current.left_out > admitted.taken)


def spoke_since(history: History, mark: int) -> bool:
    """Whether a call was relayed or a sentence sent for him since ``history`` had taken ``mark`` entries. A call of a
    tool the app hints is read-only counts too, since the hint is only the app's word. Entries the bound dropped since
    then may have been either, so a history that dropped any of them says he did."""
    return history.left_out > mark or any(
        isinstance(entry, ActDone | SaidAloud) for entry in _entered_since(history, mark)
    )


class WakeReason(StrEnum):
    LINES = "lines"
    VIEW = "view"


def held_told(arrived: Arrived) -> str:
    """The result of a call that was not sent: that it was not, and the entries that arrived. When their text would
    pass ``MAX_EVENT_BYTES`` the oldest are left out, and so is an entry that alone does not fit."""
    told = [told_entry(entry) for entry in arrived.entries]
    empty = _held_json([], older_dropped=False)
    separator = len(", ")
    sizes = reversed([told_bytes(each) + separator for each in told])
    fitting = newest_that_fit(sizes, MAX_EVENT_BYTES - len(empty.encode()) + separator)
    kept = told[len(told) - fitting :]
    return _held_json(kept, older_dropped=arrived.older_dropped or fitting < len(told))


def _held_json(entries: list[dict[str, object]], *, older_dropped: bool) -> str:
    return json.dumps(
        {"sent": False, "arrived": {"older_dropped": older_dropped, "entries": entries}}, ensure_ascii=False
    )


def _told_wake(turn: TurnTold) -> dict[str, object]:
    occasion = turn.occasion
    if isinstance(occasion, Lines):
        woke: dict[str, object] = {
            "why": WakeReason.LINES,
            "app": turn.app,
            "place": told_place(turn.place),
            "missed": occasion.missed,
            "seqs": sorted(occasion.seqs),
        }
        return woke | {"aloud": True} if turn.aloud else woke
    return {
        "why": WakeReason.VIEW,
        "app": turn.app,
        "place": told_place(turn.place),
        "revision": occasion.revision,
    }


def told_header(you: str, history: History) -> str:
    """The header message: his name and whether older entries have left the history. It holds no instruction; those
    live in the persona."""
    return json.dumps({"you": you, "older_dropped": history.left_out > 0}, ensure_ascii=False)


def told_turn(turn: TurnTold, entries: Iterable[Entry]) -> str:
    """The message of a turn: when it was admitted, what woke him and every entry taken since the message before it,
    the facts in the order things happened. A turn for lines is told when its answer is heard aloud."""
    told = {"at": told_time(turn.at), "woke": _told_wake(turn), "entries": [told_entry(entry) for entry in entries]}
    return json.dumps(told, ensure_ascii=False)


def _naming_told_lines(turn: TurnTold, history: History) -> TurnTold:
    """``turn`` with the lines its occasion names cut down to those of its visit the history still holds, since a
    message partly cut in a step no longer tells the rest; a line of another place with the same seq is no such line."""
    occasion = turn.occasion
    if not isinstance(occasion, Lines):
        return turn
    visit = Visit(turn.app, turn.place, occasion.generation)
    held = {entry.seq for entry in history.entries if isinstance(entry, KeptLine) and entry.visit == visit}
    still_told = Lines(occasion.generation, occasion.seqs & held, occasion.missed)
    return TurnTold(turn.at, turn.app, turn.place, still_told, turn.aloud, turn.through)


def turn_messages(you: str, present: Present, current: TurnTold) -> tuple[str, ...]:
    """The messages of a turn's request after the persona: the header, one for each answered turn the present holds,
    each rendered from the history as it stands, and the turn's own, which tells what the history holds from where the
    last of those ended up to ``current.through``. A message's ``woke`` names only the lines the history holds."""
    history = present.history
    messages = [told_header(you, history)]
    start = history.left_out
    for turn in (*present.told, current):
        entries = history.entries[start - history.left_out : turn.through - history.left_out]
        messages.append(told_turn(_naming_told_lines(turn, history), entries))
        start = max(start, turn.through)
    return tuple(messages)
