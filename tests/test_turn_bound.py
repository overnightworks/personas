from __future__ import annotations

from personas.turn_bound import TurnBound
from tests.support.fakes import FakeClock


def test_a_turn_has_nothing_left_at_the_time_it_ends_at_and_a_second_before_it_has_one_second():
    clock = FakeClock()
    bound = TurnBound(clock, 30.0)

    clock.time = bound.ends_at() - 1
    assert bound.seconds_left() == 1
    clock.time = bound.ends_at()
    assert bound.seconds_left() == 0
