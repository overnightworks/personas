"""The whole runner, the turn-rate cap: which addressed lines it holds back."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply
from agents.member import (
    place_window_uri,
)

from personas.decision import TurnCap
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import (
    Line,
    active,
    formatted,
    invited,
)

pytestmark = pytest.mark.anyio

TURN_CAP = TurnCap(max_turns=1, period_seconds=120.0)
SECOND_PLACE_ID = "room-8"
SECOND_AGENT = "agent-bach"
SECOND_SUBJECT = "agent-subject-2"


def refused_turn() -> Reply:
    """A turn that starts, asks the gateway once and ends short: it counts against the cap all the same."""
    return Reply(status=HTTPStatus.TOO_MANY_REQUESTS)


async def addressed_turn_ended(runner, text=ADDRESSED):
    say_in_the_call(runner, text)
    await runner.reports.until("turn_ended")


async def test_a_line_addressed_to_him_over_the_cap_starts_no_turn_makes_no_gateway_call_and_is_not_answered_later(
    runner, caplog
):
    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    say_in_the_call(runner, f"{ADDRESSED} Capped.")
    await runner.reports.until("capped")
    runner.app.changes.changed(runner.app_principal, place_window_uri(CALL_PLACE))
    runner.clock.time += TURN_CAP.period_seconds
    say_in_the_call(runner, f"{ADDRESSED} After the period.")
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [3]]
    assert [event for event, _ in runner.reports.events()].count("capped") == 1
    assert "capped client_id=agent-mozart" in formatted(caplog.get_records("call"))


async def test_an_agent_without_a_cap_answers_every_addressed_line_and_logs_no_capped(runner):
    runner.cap_turns(None)
    runner.gateway = Gateway(refused_turn(), refused_turn(), refused_turn())
    await restarted_in_the_call(runner)
    for round_number in range(1, 4):
        say_in_the_call(runner, f"{ADDRESSED} Round {round_number}.")
        await runner.reports.until("turn_ended", count=round_number)

    assert len(runner.gateway.openings()) == 3
    assert "capped" not in [event for event, _ in runner.reports.events()]


async def test_after_the_period_has_passed_an_addressed_line_starts_a_turn_again(runner):
    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), *says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    runner.clock.time += TURN_CAP.period_seconds
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert "capped" not in [event for event, _ in runner.reports.events()]


async def test_while_capped_an_addressed_line_in_a_second_place_is_skipped_too_and_not_answered_once_the_cap_frees(
    runner,
):
    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), *says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID), active(SECOND_PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_window_read(SECOND_PLACE_ID)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    # The read the turn's end wakes is held inside the second place, so both new lines land in the next one alone.
    first_place_read = runner.app.hold_next_window_read()
    await runner.app.until_window_read_held()
    second_place_read = runner.app.hold_next_window_read()
    first_place_read.set()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, f"{ADDRESSED} Capped.")
    quietly_said_elsewhere = f"{ADDRESSED} Capped elsewhere."
    runner.app.lines[SECOND_PLACE_ID] = [Line(1, PERSON, "Ada", quietly_said_elsewhere)]
    runner.app.content.add(quietly_said_elsewhere)
    second_place_read.set()
    await runner.reports.until("capped", count=2)
    runner.clock.time += TURN_CAP.period_seconds
    say_in_the_call(runner, f"{ADDRESSED} After the period.")
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [3]]


async def test_two_apps_of_one_agent_share_one_cap(runner):
    runner.cap_turns(TURN_CAP)
    runner.add_app()
    runner.gateway = Gateway(refused_turn(), refused_turn())
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    await runner.reports.until("capped")

    assert len(runner.gateway.openings()) == 1


async def test_leaving_and_rejoining_a_place_does_not_reset_the_cap(runner):
    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, f"{ADDRESSED} In the new window.")

    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), refused_turn())
    runner.on_report = addressed_once_joined
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until("turn_ended")
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.reports.until("capped")

    assert len(runner.gateway.openings()) == 1


async def test_two_agents_each_have_a_cap_of_their_own(runner):
    runner.cap_turns(TURN_CAP)
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.gateway = Gateway(refused_turn(), refused_turn())
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID, 2)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(2)

    assert len(runner.gateway.openings()) == 2
    assert "capped" not in [event for event, _ in runner.reports.events()]
