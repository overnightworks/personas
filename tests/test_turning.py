from __future__ import annotations

import secrets

import pytest
from agents.member import PlaceEntry, PlaceRef, PlaceView, PlaceWindow
from agents.turn import Principal

from personas.tokens import TokenFacts
from personas.turning import Admitted, Lines, Waiting

PLACE = PlaceRef(kind="table", id="table-1")
FACTS = TokenFacts(Principal(iss="https://realm.example", sub="agent-subject"), frozenset(), "Ada")


def view_occasion(marker: str) -> Waiting:
    return Waiting(PlaceView(revision=1, awaits_you=True, state={"hand": marker}))


def lines_occasion(marker: str) -> Lines:
    entry = PlaceEntry(seq=1, kind="message", name="Ada", text=marker, own=False)
    window = PlaceWindow(generation="visit", starts_at=1, entries=(entry,))
    return Lines(window, (entry,), missed=False)


@pytest.mark.parametrize(
    "occasion_of",
    [view_occasion, lines_occasion],
    ids=["a view whose state holds the marker", "a window whose line holds the marker"],
)
def test_an_admission_repr_shows_neither_a_view_state_nor_a_window_line(occasion_of):
    marker = secrets.token_hex(8)

    admission = repr(Admitted(FACTS, PLACE, occasion_of(marker)))

    assert "Admitted(" in admission
    assert marker not in admission
