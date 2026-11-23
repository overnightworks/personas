"""A written turn that a spoken place cuts or passes over, as the whole runner does it."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round

from tests.support.app_driving import said_and_taken, settled, waits
from tests.support.call import ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.history import (
    OTHER_CALL_ID,
    REMARK,
    TABLE,
    at_table,
    turn_order,
    url_of,
    wait_reasons,
    woke_at_table,
)
from tests.support.hopin import PLAY_TOOL, SAY_TOOL
from tests.support.logs import lines_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import in_a_voice_call_and_at_a_table

pytestmark = pytest.mark.anyio

AGAIN = "And once more."


async def a_table_turn_that_waits_for_the_model_in_a_voice_call(runner, *replies):
    """His table turn waits for the model while his call at the same app is a voice call; the call's window was read."""
    await in_a_voice_call_and_at_a_table(runner, *replies)
    running = runner.gateway.hold_request(1)
    waits(runner.app, 1)
    await runner.gateway.until_requested(1)
    return running


def turn_endings(caplog) -> list[tuple[object, object]]:
    """The outcome of each turn and whether it was cut for a spoken place, in the order the turns ended."""
    return [(line["outcome"], line.get("cut")) for line in lines_of(caplog.get_records("call"), "turn.ended")]


async def test_a_line_heard_at_a_voice_place_cuts_his_table_turn_of_that_app_that_waits_for_the_model(runner, caplog):
    """TURNS-48, LOG-38."""
    await a_table_turn_that_waits_for_the_model_in_a_voice_call(runner, completion(ANSWER), completion("table again"))

    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed(count=2)

    assert runner.gateway.arrivals == 3
    assert turn_order(runner) == [(0, "lines"), (0, "view")]
    assert runner.app.spoken == [(SUBJECT, PLACE_ID, ANSWER)]
    assert turn_endings(caplog) == [("cancelled", "spoken"), ("ok", None), ("ok", None)]
    call_turn, table_turn = runner.gateway.openings()
    assert [each["entry"] for each in call_turn.room["history"]["entries"]] == ["view", "line"]
    assert SAY_TOOL in runner.gateway.offered_tools(2)
    assert table_turn.room["woke"] == woke_at_table(url_of(runner, 0), 1)


async def test_a_line_heard_at_a_voice_call_cuts_his_table_turn_at_another_app_and_the_spoken_turn_holds_its_tools(
    runner, other, caplog
):
    """TURNS-48, TURNS-18."""
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER), completion("table again"))
    runner.gateway.hold_request(1)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    waits(other, 1)
    await runner.gateway.until_requested(1)

    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed(count=2)

    assert runner.gateway.arrivals == 3
    assert turn_order(runner) == [(0, "lines"), (1, "view")]
    assert turn_endings(caplog) == [("cancelled", "spoken"), ("ok", None), ("ok", None)]
    assert runner.app.spoken == [(SUBJECT, PLACE_ID, ANSWER)]
    assert PLAY_TOOL in runner.gateway.offered_tools(2)


async def test_a_table_turn_cut_for_a_spoken_turn_names_the_spoken_place_in_front_of_it_as_its_wait(runner, caplog):
    """LOG-32."""
    await a_table_turn_that_waits_for_the_model_in_a_voice_call(runner, completion(ANSWER), completion("table again"))

    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")
    await runner.reports.until_a_turn_completed(count=2)

    assert wait_reasons(caplog)[-1] == "older_due"


async def test_a_table_turn_with_a_call_in_flight_is_cut_once_the_call_answered_and_entered_his_history(runner, caplog):
    """TURNS-48."""
    answer = runner.app.hold_next_say()
    await in_a_voice_call_and_at_a_table(
        runner,
        tool_round(tool_call("play-1", SAY_TOOL, {"place": TABLE.id, "text": "Jack of hearts"})),
        completion(ANSWER),
        completion("table again"),
    )
    waits(runner.app, 1)
    await runner.app.until_called(1)

    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")

    assert runner.gateway.arrivals == 1
    assert runner.reports.turn_endings == []
    answer.set()
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed(count=2)
    assert turn_endings(caplog) == [("cancelled", "spoken"), ("ok", None), ("ok", None)]
    assert runner.gateway.arrivals == 3
    call_turn, _ = runner.gateway.openings()[1:]
    (act,) = call_turn.told("act")
    assert (act["tool"], act["outcome"]["answered"]) == (SAY_TOOL, True)


async def test_a_table_turn_whose_table_left_with_a_call_in_flight_ends_cancelled_without_a_spoken_cut(runner, caplog):
    """TURNS-48."""
    runner.app.hold_next_say()
    await in_a_voice_call_and_at_a_table(
        runner,
        tool_round(tool_call("play-1", SAY_TOOL, {"place": TABLE.id, "text": "Jack of hearts"})),
        completion(ANSWER),
    )
    waits(runner.app, 1)
    await runner.app.until_called(1)
    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")

    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed(count=1)

    assert turn_endings(caplog)[0] == ("cancelled", None)


async def test_a_spoken_turn_that_does_what_the_cut_table_turn_was_for_leaves_the_table_no_turn_after_it(runner):
    """TURNS-48."""
    runner.app.says_move_the_view = True
    await in_a_voice_call_and_at_a_table(
        runner,
        tool_round(tool_call("play-1", SAY_TOOL, {"place": TABLE.id, "text": "Jack of hearts"})),
        completion(ANSWER),
    )
    runner.gateway.hold_request(1)
    spoken_turn = runner.gateway.hold_request(2)
    waits(runner.app, 1)
    await runner.gateway.until_requested(1)

    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")
    await runner.gateway.until_requested(2)
    await settled(runner.app)
    spoken_turn.set()
    await runner.reports.until_a_turn_completed(count=1)
    await settled(runner.app)

    assert runner.app.said == [(SUBJECT, TABLE.id, "Jack of hearts")]
    assert runner.gateway.arrivals == 3


async def test_a_written_turn_that_gave_way_to_a_spoken_one_is_not_cut_for_the_next(runner, caplog):
    """TURNS-48."""
    await a_table_turn_that_waits_for_the_model_in_a_voice_call(
        runner, completion(ANSWER), completion("table again"), completion(AGAIN)
    )
    again = runner.gateway.hold_request(3)

    await said_and_taken(runner.app, PLACE_ID, "Anyone there?")
    await runner.gateway.until_requested(3)
    await said_and_taken(runner.app, PLACE_ID, "Hello?")
    again.set()
    await runner.app.spoken_count(2)

    assert turn_endings(caplog) == [("cancelled", "spoken"), ("ok", None), ("ok", None), ("ok", None)]
    assert turn_order(runner) == [(0, "lines"), (0, "view"), (0, "lines")]


async def test_a_due_spoken_place_is_claimed_before_a_written_place_that_became_due_before_it(runner):
    """TURNS-48."""
    runner.give_voice()
    runner.gateway = Gateway(completion(ANSWER), completion(AGAIN), completion("table"))
    running = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID), active(OTHER_CALL_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, REMARK, spoken=True)
    await runner.gateway.until_requested(1)
    runner.app.stand(SUBJECT, active(PLACE_ID), active(OTHER_CALL_ID), at_table())
    waits(runner.app, 1)
    await settled(runner.app)
    await said_and_taken(runner.app, OTHER_CALL_ID, "Anyone there?")

    running.set()
    await runner.gateway.until_turns(3)

    assert [each.room["woke"]["place"]["id"] for each in runner.gateway.openings()] == [
        PLACE_ID,
        OTHER_CALL_ID,
        TABLE.id,
    ]
