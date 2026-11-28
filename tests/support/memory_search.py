"""What the tests of the memory's search share: the agent and app they search, the windows they keep and the search they
ask."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

from agents.member import PlaceRef

from personas.entries import KeptLine, ToldEntry, Visit, WindowRead
from personas.recall import EntrySearch
from tests.support.call import PLACE_ID
from tests.support.fakes import WALL_START
from tests.support.standings import window_of

if TYPE_CHECKING:
    from agents.member import PlaceEntry

APP = "https://app.test/mcp"
AGENT = "agent-mozart"
SECOND_AGENT = "agent-ada"
LATE_ORDINAL = 1000
"""An ordinal above every entry a test records, so that the record's bound lets all of them through."""


def a_visit(generation: str) -> Visit:
    return Visit(APP, PlaceRef(kind="call", id=PLACE_ID), generation)


def a_read_in(visit: Visit, *lines: PlaceEntry, client_id: str = AGENT, hours_after_start: float = 0.0) -> WindowRead:
    read_at = WALL_START + timedelta(hours=hours_after_start)
    return WindowRead(client_id, visit.app, visit.place, window_of(*lines, generation=visit.generation), read_at)


def a_search(
    text: str = "",
    *,
    place: PlaceRef | None = None,
    first_ordinal: int = LATE_ORDINAL,
    first_read_before: datetime | None = None,
    rows: int = 50,
    client_id: str = AGENT,
    app: str = APP,
) -> EntrySearch:
    return EntrySearch(
        client_id,
        app,
        place or PlaceRef(kind="call", id=PLACE_ID),
        text,
        first_ordinal,
        first_read_before or WALL_START + timedelta(days=1),
        rows,
    )


def texts_found(found: tuple[ToldEntry, ...]) -> list[str]:
    return [cast("KeptLine", each).text for each in found]
