"""What the store gives a turn, and when it sweeps: the lookup's declaration, the room it reads, its argument and its
result, and the period after which a kept line is swept. Pure values with no process, network or clock."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from agents.member import PlaceRef
from agents.tools import ToolDeclaration, ToolEffect
from agents.turn import MAX_EVENT_BYTES

from personas.entries import KeptLine, Visit, entry_time, newest_that_fit, told_bytes, told_entry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.turn import JsonObject

    from personas.entries import History

SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_DAY: Final = 24 * 3600

RECALL_TOOL: Final = "recall"
TEXT_PARAMETER: Final = "text"
SCOPE_TEXT: Final = "this room, before your history"
LOOKUPS_SPENT_TEXT: Final = "no more lookups in this turn"
STORE_DID_NOT_ANSWER_TEXT: Final = "the memory did not answer"
RECALL_DECLARATION: Final = ToolDeclaration(
    name=RECALL_TOOL,
    description=(
        "Looks up what was said earlier in this room, before your history: give a few words that every line you want"
        " holds exactly as written, or an empty text for the newest lines."
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
_SMALLEST_LINE: Final = KeptLine(
    Visit("a", PlaceRef(kind="k", id="i"), ""),
    0,
    datetime.min.replace(tzinfo=UTC),
    "",
    "",
    "",
    own=False,
    spoken=False,
    cut=False,
    act=False,
)


@dataclass(frozen=True)
class LookupBounds:
    """What ``[memory]`` allows a turn's lookups: how many, and the bytes of one result, which are also the longest
    ``text`` a lookup accepts."""

    calls: int
    result_bytes: int


@dataclass(frozen=True)
class LineSearch:
    """What the store is asked: the lines of this agent's room, every visit, that hold the words of ``text`` and were
    first read before ``first_read_before``, newest first, at most ``rows`` of them."""

    client_id: str
    app: str
    place: PlaceRef
    text: str = field(repr=False)
    first_read_before: datetime
    rows: int


def _result_json(told: list[dict[str, object]], *, older_dropped: bool) -> str:
    return json.dumps({"scope": SCOPE_TEXT, "older_dropped": older_dropped, "lines": told}, ensure_ascii=False)


# The least a result holds: its scope and flags with no line. A smaller bound holds no result at all.
SMALLEST_RESULT_BYTES: Final = len(_result_json([], older_dropped=False).encode())


def _room_for_lines(result_bytes: int) -> int:
    """What is left of a result's bytes for lines and their separators, counting the separator the first line lacks."""
    return min(result_bytes, MAX_EVENT_BYTES) - SMALLEST_RESULT_BYTES + len(_SEPARATOR)


def line_search(
    client_id: str, app: str, place: PlaceRef, history: History, text: str, result_bytes: int
) -> LineSearch:
    """The room is the turn's own: this agent's, in the admission's app and place, whatever the model asked.

    ``result_bytes`` is at least ``SMALLEST_RESULT_BYTES``, which the configuration enforces.

    Only lines first read before the first entry the admitted history still holds are given, since the history tells
    the rest. An admitted history always holds the entry that woke the turn.
    """
    smallest = told_bytes(told_entry(_SMALLEST_LINE)) + len(_SEPARATOR)
    fitting = _room_for_lines(result_bytes) // smallest
    return LineSearch(client_id, app, place, text, entry_time(history.entries[0]), fitting + 1)


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


def lookup_result(lines_newest_first: Sequence[KeptLine], result_bytes: int) -> str:
    """The result of a lookup: its scope, the newest lines that fit ``result_bytes`` and the wire's bound for one
    result, in the history's line form, and whether more matched."""
    told = [told_entry(each) for each in lines_newest_first]
    sizes = (told_bytes(each) + len(_SEPARATOR) for each in told)
    fitting = newest_that_fit(sizes, _room_for_lines(result_bytes))
    return _result_json(told[:fitting], older_dropped=fitting < len(told))


def swept_before(now: datetime, lines_keep_days: float) -> datetime | None:
    """The time before which a line first read is swept at ``now``, or nothing when the period reaches any line."""
    earliest = datetime.min.replace(tzinfo=now.tzinfo)
    seconds = lines_keep_days * SECONDS_PER_DAY
    if seconds >= (now - earliest).total_seconds():
        return None
    return now - timedelta(seconds=seconds)
