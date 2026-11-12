from __future__ import annotations

import secrets
from datetime import UTC, datetime

import pytest
from agents.member import PlaceRef, PlaceView
from agents.turn import Principal

from personas.decision import History, KeptLine, Lines, ViewSeen, Visit, Waiting
from personas.tokens import TokenFacts
from personas.turning import Admitted

PLACE = PlaceRef(kind="table", id="table-1")
VISIT = Visit("https://app.test/mcp", PLACE, "generation-1")
TAKEN_AT = datetime(2026, 11, 11, 20, 0, tzinfo=UTC)
FACTS = TokenFacts(Principal(iss="https://realm.example", sub="agent-subject"), frozenset(), "Ada")


def view_occasion(_marker: str) -> Waiting:
    return Waiting(revision=1)


def lines_occasion(_marker: str) -> Lines:
    return Lines(VISIT.generation, frozenset({1}), missed=False)


@pytest.mark.parametrize(
    "occasion_of",
    [view_occasion, lines_occasion],
    ids=["a view turn whose history holds a state with the marker", "the lines a window gave"],
)
def test_an_admission_repr_shows_neither_a_view_state_nor_a_line_of_the_history(occasion_of):
    marker = secrets.token_hex(8)
    view = ViewSeen(VISIT.app, PLACE, PlaceView(revision=1, awaits_you=True, state={"hand": marker}), TAKEN_AT)
    history = History((KeptLine(VISIT, 1, TAKEN_AT, marker, "message", marker, own=False), view), older_dropped=False)
    occasion = occasion_of(marker)

    admission = repr(Admitted(FACTS, PLACE, occasion, history, TAKEN_AT))

    assert "Admitted(" in admission
    assert marker not in admission
    assert marker not in repr(occasion)
