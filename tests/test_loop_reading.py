"""The whole runner, where reading starts: after a restart, a pause, a new join or a lost window."""

from __future__ import annotations

import pytest
from agent_runtime_client.member import STANDING_URI

from agents.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import WINDOW_LINES, active, invited, paused

pytestmark = pytest.mark.anyio


async def test_after_a_restart_an_addressed_line_still_in_the_window_is_not_answered_and_the_next_one_is(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    say_in_the_call(runner, f"{ADDRESSED} Before the restart.")
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_lines_said_during_a_pause_are_not_answered_after_the_resume_and_a_line_after_it_is(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    pause_seen = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.app.until_read_held()
    say_in_the_call(runner, f"{ADDRESSED} During the pause.")
    resumed_read = runner.app.hold_next_window_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    pause_seen.set()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    resumed_read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_when_the_window_moved_past_his_position_the_turn_is_told_lines_were_missed_and_none_is_answered_twice(
    runner,
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Once more."))
    first_round = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    for number in range(WINDOW_LINES + 1):
        say_in_the_call(runner, f"Chatter {number}")
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    first_round.set()
    await runner.app.said_count(2)

    first, second = runner.gateway.openings()
    assert not first.missed
    assert second.missed
    assert set(first.new_seqs).isdisjoint(second.new_seqs)


async def test_a_vanished_window_costs_one_more_standing_read_and_restarts_the_place_without_answering_old_lines(
    runner,
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    standing_reads = runner.app.reads[STANDING_URI]
    runner.app.vanished.add(PLACE_ID)
    say_in_the_call(runner, f"{ADDRESSED} Before the window vanished.")
    await runner.app.until_standing_read(standing_reads + 2)
    runner.app.vanished.discard(PLACE_ID)
    next_window_read = runner.app.hold_next_window_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_window_read_held()

    assert runner.app.window_reads(PLACE_ID) == 3
    assert runner.app.reads[STANDING_URI] == standing_reads + 3

    say_in_the_call(runner, ADDRESSED)
    next_window_read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_a_place_invited_again_when_its_window_vanished_is_joined_in_that_wake_and_read_from_seq_1(runner):
    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, ADDRESSED)

    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.on_report = addressed_once_joined
    await restarted_in_the_call(runner)
    lost_read = runner.app.hold_next_window_read()
    runner.app.vanished.add(PLACE_ID)
    say_in_the_call(runner, f"{ADDRESSED} Before the window vanished.")
    await runner.app.until_window_read_held()
    runner.app.standings[SUBJECT] = [invited(PLACE_ID, "inv-3")]
    lost_read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1]]
