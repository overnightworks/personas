"""The whole runner, a turn's listing: the tools of its apps are listed while its claim is written."""

from __future__ import annotations

import asyncio
import logging

import pytest
from agents.backend.fake_gateway import completion
from agents.member_outcomes import INVALID_ANSWER_CODE
from agents.skills import implicit_skill_name

from personas.failures import leaves_of
from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, a_turn, restarted_in_the_call, say_in_the_call
from tests.support.fakes import CANCELLED_TURN
from tests.support.gateway import Gateway, says
from tests.support.history_holding import HistoriesHoldingTheAddressedLine
from tests.support.hopin import LIMIT_SECONDS, LIST_TOOLS, SAY_TOOL
from tests.support.logs import lines_of
from tests.support.openers import opener_whose_request_raises
from tests.support.realm import LIFETIME_SECONDS
from tests.support.skills import HOPIN, Shelf, a_skill_file
from tests.support.songmaker import CREATE_SONG, lists_only, songmaker_url
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.wire import hangs

pytestmark = pytest.mark.anyio


async def test_a_turn_lists_the_tools_of_its_apps_while_its_claim_is_written_and_acts_only_once_it_was_taken(
    runner, songmaker
):
    runner.histories = HistoriesHoldingTheAddressedLine()
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.histories.reached.wait()
    async with asyncio.timeout(LIMIT_SECONDS):
        await songmaker.log.until(LIST_TOOLS)
        await runner.app.log.until(LIST_TOOLS)
    await settled(runner.app)
    stated_while_the_write_waited = list(runner.app.turn_states)
    requested_while_the_write_waited = runner.gateway.arrivals
    called_while_the_write_waited = list(songmaker.called)
    runner.histories.release.set()
    await runner.app.said_count(1)

    assert (stated_while_the_write_waited, requested_while_the_write_waited) == ([], 0)
    assert called_while_the_write_waited == []
    assert runner.app.turn_states[0][1].thinking
    assert songmaker.log.count(LIST_TOOLS) == 1


async def test_a_turn_whose_place_is_left_while_its_claim_is_written_ends_cancelled_and_asks_the_model_nothing(
    runner, songmaker
):
    runner.histories = HistoriesHoldingTheAddressedLine()
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    songmaker.script(LIST_TOOLS, hangs)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.histories.reached.wait()
    async with asyncio.timeout(LIMIT_SECONDS):
        await songmaker.log.until(LIST_TOOLS)

    runner.app.stand(SUBJECT)
    await runner.reports.until_a_turn_ended()
    runner.histories.release.set()

    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert runner.gateway.arrivals == 0


async def test_a_defect_the_listing_meets_as_its_turn_is_left_while_its_claim_is_written_ends_the_runner_loud(runner):
    defect = ValueError("a defect")
    arrived = asyncio.Event()
    runner.histories = HistoriesHoldingTheAddressedLine()
    runner.open_session = opener_whose_request_raises("declared_tools", defect, arrived=arrived, when_cancelled=True)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.histories.reached.wait()
    async with asyncio.timeout(LIMIT_SECONDS):
        await arrived.wait()

    runner.app.stand(SUBJECT)

    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(runner.task, return_exceptions=True)
    assert set(leaves_of(ended)) == {defect}
    assert runner.gateway.arrivals == 0


async def test_every_turn_lists_at_an_app_whose_stream_does_not_carry_the_tools_notice(runner, caplog):
    """TURNS-18, LOG-13."""
    runner.gateway = Gateway(completion("done"), completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    say_in_the_call(runner, f"{ADDRESSED} Once more.")
    await runner.reports.until_a_turn_completed(count=2)

    assert runner.app.log.count(LIST_TOOLS) == 2
    assert len(lines_of(caplog.get_records("call"), "tool_list.ended")) == 2


def songmaker_lists_a_tool_twice(runner, songmaker) -> tuple[str, str]:
    songmaker.shelf = Shelf(declares=False)
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    songmaker.script(LIST_TOOLS, lists_only(CREATE_SONG, CREATE_SONG))
    return implicit_skill_name(songmaker_url(runner)), CREATE_SONG


def own_app_lists_a_tool_twice(runner, songmaker) -> tuple[str, str]:
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    runner.app.script(LIST_TOOLS, lists_only(SAY_TOOL, SAY_TOOL))
    return HOPIN, SAY_TOOL


@pytest.mark.parametrize(
    "arranging", [songmaker_lists_a_tool_twice, own_app_lists_a_tool_twice], ids=["another app", "own app"]
)
async def test_an_app_that_lists_a_tool_twice_gives_no_skill_line_and_no_tool_that_turn_and_the_agent_goes_on(
    runner, songmaker, arranging
):
    skill, tool = arranging(runner, songmaker)
    runner.gateway = Gateway(completion("done"), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await a_turn(runner, 2)

    assert skill not in runner.gateway.system_of(1)
    assert tool not in runner.gateway.offered_tools(1, speaking=True, reading=True)
    assert skill in runner.gateway.system_of(2)
    assert not runner.task.done()


@pytest.mark.parametrize("granted_before", range(1, 7))
async def test_a_token_renewal_that_answers_no_usable_token_suspends_the_loop_wherever_the_turn_is(
    runner, caplog, granted_before
):
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)
    runner.realm.lifetime_seconds = 1
    runner.realm.requests_before_an_answer_without_a_token = granted_before
    runner.clock.time += LIFETIME_SECONDS

    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.reported()[-1] == (Event.SUSPENDED, INVALID_ANSWER_CODE)
    ended = [record for record in caplog.get_records("call") if record.getMessage() == "tool_list.ended"]
    assert logging.WARNING not in {record.levelno for record in ended}
