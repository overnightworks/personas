from __future__ import annotations

import pytest

from personas.decision import waking_line
from tests.support.standings import line


@pytest.mark.parametrize(
    ("taken", "expected"),
    [
        (({"spoken": True, "continues": True},), None),
        (({"spoken": True, "continues": True}, {"spoken": True}), 2),
        (({"spoken": True}, {"spoken": True, "continues": True}), 1),
        (({"spoken": True, "continues": True}, {"own": True}), None),
    ],
    ids=[
        "a line followed by more of the same speaker's speech",
        "marked lines before the last of the run",
        "an unmarked line before a marked one",
        "a marked line, then his own",
    ],
)
def test_a_run_of_another_speakers_marked_lines_wakes_him_at_its_last_line(taken, expected):
    lines = [line(seq, "Hello everyone", **marks) for seq, marks in enumerate(taken, 1)]

    waking = waking_line(lines, serves_view=False)

    assert (None if waking is None else waking.seq) == expected
