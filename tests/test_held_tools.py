from __future__ import annotations

from personas.held_tools import HoldBack
from tests.support.history import a_marked_present


def test_a_hold_repr_shows_neither_a_view_state_nor_a_line_nor_an_act_of_the_history():
    marker, present = a_marked_present()

    assert marker not in repr(HoldBack(present.history, None))
