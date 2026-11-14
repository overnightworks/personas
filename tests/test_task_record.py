from __future__ import annotations

from personas.task_record import RelayFailed, failures_leaving_with


def test_two_failed_relayed_calls_leave_with_both_failures() -> None:
    first, second = ValueError("first"), ValueError("second")
    chosen, other = RelayFailed(first), RelayFailed(second)

    assert failures_leaving_with(chosen, first, [chosen, other]) == [first, second]
