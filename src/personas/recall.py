"""What the store gives a turn, and when it sweeps: the lookup's declaration, the room it reads, its argument and its
result, and the period after which a kept line is swept. Pure values with no process, network or clock."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from agents.member import PlaceRef, PlaceView, Spoken
from agents.tools import ToolDeclaration, ToolEffect
from agents.turn import MAX_EVENT_BYTES

from personas.entries import (
    ActDone,
    KeptLine,
    SaidAloud,
    Unanswered,
    ViewSeen,
    Visit,
    entry_time,
    newest_that_fit,
    told_bytes,
    told_entry,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.turn import JsonObject

    from personas.entries import History, ToldEntry

SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_DAY: Final = 24 * 3600

RECALL_TOOL: Final = "recall"
TEXT_PARAMETER: Final = "text"
SCOPE_TEXT: Final = "lines, views, calls and sentences of this room, before your history"
LOOKUPS_SPENT_TEXT: Final = "no more lookups in this turn"
STORE_DID_NOT_ANSWER_TEXT: Final = "the memory did not answer"
RECALL_DECLARATION: Final = ToolDeclaration(
    name=RECALL_TOOL,
    description=(
        "Looks up what happened earlier in this room, before your history: its lines, the views you saw, your calls"
        " with their results and your sentences. Give a few words that every entry you want holds exactly as written,"
        " or an empty text for the newest entries."
    ),
    parameters={
        "type": "object",
        "properties": {TEXT_PARAMETER: {"type": "string", "description": "The words to look for."}},
        "required": [TEXT_PARAMETER],
        "additionalProperties": False,
    },
    effect=ToolEffect.READ_ONLY,
)

_SEPARATOR: Final = ", "
_LONG_AGO: Final = datetime.min.replace(tzinfo=UTC)
_SMALLEST_ROOM: Final = PlaceRef(kind="k", id="i")
# One empty entry of each kind: the least a hit can take of a result, whichever kind it is.
_SMALLEST_ENTRIES: Final[tuple[ToldEntry, ...]] = (
    KeptLine(Visit("a", _SMALLEST_ROOM, ""), 0, _LONG_AGO, "", "", "", own=False, spoken=False, cut=False, act=False),
    ViewSeen("a", _SMALLEST_ROOM, PlaceView(revision=0, awaits_you=False, state={}, traceparent=None), _LONG_AGO),
    ActDone("a", "", {}, Unanswered(""), _LONG_AGO),
    SaidAloud("a", _SMALLEST_ROOM, "", Spoken(spoken=True), _LONG_AGO),
)


@dataclass(frozen=True)
class LookupBounds:
    """What ``[memory]`` allows a turn's lookups: how many, and the bytes of one result, which are also the longest
    ``text`` a lookup accepts."""

    calls: int
    result_bytes: int


@dataclass(frozen=True)
class EntrySearch:
    """What the store is asked: the entries of this agent's room, every visit, that hold the words of ``text``, newest
    first, at most ``rows`` of them: those of the record whose ordinal is below ``first_ordinal`` and the lines first
    read before ``first_read_before`` that the record does not hold."""

    client_id: str
    app: str
    place: PlaceRef
    text: str = field(repr=False)
    first_ordinal: int
    first_read_before: datetime
    rows: int


def _result_json(told: list[dict[str, object]], *, older_dropped: bool, too_large: int) -> str:
    return json.dumps(
        {"scope": SCOPE_TEXT, "older_dropped": older_dropped, "too_large": too_large, "entries": told},
        ensure_ascii=False,
    )


# The least a result holds: its scope and flags with no entry. A smaller bound holds no result at all.
SMALLEST_RESULT_BYTES: Final = len(_result_json([], older_dropped=False, too_large=0).encode())


def _room_for_entries(result_bytes: int, *, older_dropped: bool, too_large: int) -> int:
    """What is left of a result's bytes for entries and their separators, counting the separator the first entry
    lacks."""
    envelope = len(_result_json([], older_dropped=older_dropped, too_large=too_large).encode())
    return min(result_bytes, MAX_EVENT_BYTES) - envelope + len(_SEPARATOR)


def _search_rows(result_bytes: int) -> int:
    """One more than the smallest entries that fit the result, so that no entry that fits is missed."""
    smallest = min(told_bytes(told_entry(each)) for each in _SMALLEST_ENTRIES) + len(_SEPARATOR)
    return _room_for_entries(result_bytes, older_dropped=True, too_large=0) // smallest + 1


def line_search(
    client_id: str, app: str, place: PlaceRef, history: History, text: str, result_bytes: int
) -> EntrySearch:
    """The room is the turn's own: this agent's, in the admission's app and place, whatever the model asked.

    ``result_bytes`` is at least ``SMALLEST_RESULT_BYTES``, which the configuration enforces.

    Only what the history does not tell is given: the record's entries below the first ordinal the admitted history
    still holds, and the lines first read before that entry. An admitted history always holds the entry that woke
    the turn.
    """
    return EntrySearch(
        client_id, app, place, text, history.left_out, entry_time(history.entries[0]), _search_rows(result_bytes)
    )


def search_text(arguments: JsonObject, longest_bytes: int) -> str | None:
    """The ``text`` of a lookup, or nothing when it is missing, no string or longer than ``longest_bytes``."""
    text = arguments.get(TEXT_PARAMETER)
    if not isinstance(text, str):
        return None
    try:
        size = len(text.encode())
    except UnicodeEncodeError:
        return None
    return text if size <= longest_bytes else None


def text_refused_text(longest_bytes: int) -> str:
    return f"{TEXT_PARAMETER}: give a string of at most {longest_bytes} bytes"


def lookup_result(entries_newest_first: Sequence[ToldEntry], result_bytes: int) -> str:
    """The result of a lookup: its scope, the newest entries that fit ``result_bytes`` and the wire's bound for one
    result, in the history's form, whether more matched, and how many hits are too large to be shown at all."""
    told = [told_entry(each) for each in entries_newest_first]
    alone = _room_for_entries(result_bytes, older_dropped=True, too_large=len(told))
    shown = [each for each in told if told_bytes(each) + len(_SEPARATOR) <= alone]
    too_large = len(told) - len(shown)
    sizes = [told_bytes(each) + len(_SEPARATOR) for each in shown]
    capped = len(told) >= _search_rows(result_bytes)
    if not capped and newest_that_fit(
        sizes, _room_for_entries(result_bytes, older_dropped=False, too_large=too_large)
    ) == len(shown):
        return _result_json(shown, older_dropped=False, too_large=too_large)
    fitting = newest_that_fit(sizes, _room_for_entries(result_bytes, older_dropped=True, too_large=too_large))
    kept = fitting if capped or fitting < len(shown) else len(shown) - 1
    return _result_json(shown[: max(kept, 0)], older_dropped=True, too_large=too_large)


def swept_before(now: datetime, lines_keep_days: float) -> datetime | None:
    """The time before which a line first read is swept at ``now``, or nothing when the period reaches any line."""
    earliest = datetime.min.replace(tzinfo=now.tzinfo)
    seconds = lines_keep_days * SECONDS_PER_DAY
    if seconds >= (now - earliest).total_seconds():
        return None
    return now - timedelta(seconds=seconds)
