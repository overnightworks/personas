from __future__ import annotations

from personas.turn_bound import TurnBound
from tests.support.fakes import FakeClock


def test_a_turn_is_past_its_bound_at_the_time_it_ends_at_and_not_a_second_before():
    clock = FakeClock()
    bound = TurnBound(clock, 30.0)

    clock.time = bound.ends_at() - 1
    assert not bound.past()
    clock.time = bound.ends_at()
    assert bound.past()
    assert bound.seconds_left() == 0
