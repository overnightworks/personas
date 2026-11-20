from __future__ import annotations

from personas.decision import DEFAULT_BUDGET_PAUSE_CAP_SECONDS
from personas.held_tools import HoldBack
from personas.mind import Mind
from personas.turn_bound import TurnBound
from tests.support.fakes import FakeClock
from tests.support.history import a_marked_present


def test_a_hold_repr_shows_neither_a_view_state_nor_a_line_nor_an_act_of_the_history():
    marker, present = a_marked_present()
    mind = Mind(1_000, 1_000, DEFAULT_BUDGET_PAUSE_CAP_SECONDS)
    bound = TurnBound(FakeClock(), 30, mind, lambda lines: None)

    assert marker not in repr(HoldBack(present.history, None, bound))
