from __future__ import annotations

import pytest
from agents.member import PlaceRef

from personas.decision import revisions_after_turn, waking_line
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


@pytest.mark.parametrize(
    ("turn_revision", "notices_read", "notices_now", "kept"),
    [
        pytest.param(2, 4, 5, False, id="a notice after the turn read its list"),
        pytest.param(2, 4, 4, True, id="no notice after the turn read its list"),
        pytest.param(2, None, 5, True, id="a turn that read no list"),
        pytest.param(None, 4, 5, True, id="a turn that came from lines"),
        pytest.param(1, 4, 5, True, id="a place already at a newer revision"),
    ],
)
def test_a_turn_a_tools_notice_overtook_does_not_count_for_its_view_revision(
    turn_revision, notices_read, notices_now, kept
):
    table = PlaceRef(kind="table", id="table-1")
    other = PlaceRef(kind="table", id="other")
    turned = {table: 2, other: 2}

    after = revisions_after_turn(turned, table, turn_revision, notices_read, notices_now)

    assert after == (turned if kept else {other: 2})
