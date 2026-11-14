from __future__ import annotations

import pytest
from agents.backend import TurnStopKind, TurnStopped
from agents.turn import FailureCode

from personas.task_record import RelayFailed, failures_leaving_with


def plain_stop() -> TurnStopped:
    return TurnStopped(TurnStopKind.FAILED, FailureCode.PROVIDER_ERROR)


def test_two_failed_relayed_calls_leave_with_both_failures() -> None:
    first, second = ValueError("first"), ValueError("second")
    chosen, other = RelayFailed(first), RelayFailed(second)

    assert failures_leaving_with(chosen, [chosen, other]) == [first, second]


def test_a_plain_stop_beside_the_chosen_one_leaves_with_nothing() -> None:
    assert failures_leaving_with(plain_stop(), [plain_stop()]) == []


def test_a_late_answer_failure_leaves_beside_a_relayed_failure() -> None:
    failure, late = ValueError("relayed"), RuntimeError("late")
    chosen = RelayFailed(failure)

    assert failures_leaving_with(chosen, [late, chosen]) == [failure, late]


@pytest.mark.parametrize("beside", [[], [plain_stop()]])
def test_a_lone_relayed_failure_leaves_alone(beside: list[TurnStopped]) -> None:
    failure = ValueError("relayed")
    chosen = RelayFailed(failure)

    assert failures_leaving_with(chosen, [chosen, *beside]) == [failure]
