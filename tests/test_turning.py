from __future__ import annotations

from datetime import UTC, datetime

import pytest
from agents.member import PlaceRef
from agents.turn import Principal

from personas.entries import Lines, Visit, Waiting
from personas.tokens import TokenFacts
from personas.turning import Admitted
from tests.support.history import a_marked_present

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
def test_an_admission_repr_shows_neither_a_view_state_nor_a_line_nor_an_act_of_the_history(occasion_of):
    marker, present = a_marked_present()
    occasion = occasion_of(marker)

    admission = repr(Admitted(FACTS, PLACE, occasion, present, TAKEN_AT, aloud=False, apps=frozenset()))

    assert "Admitted(" in admission
    assert marker not in admission
    assert marker not in repr(occasion)
