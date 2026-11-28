"""The whole runner, a turn's listing: the tools of its apps are listed while its claim is written."""

from __future__ import annotations

import asyncio

import pytest
from agents.backend.fake_gateway import completion

from personas.failures import leaves_of
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import CANCELLED_TURN
from tests.support.gateway import Gateway, says
from tests.support.history_holding import HistoriesHoldingTheAddressedLine
from tests.support.hopin import LIMIT_SECONDS, LIST_TOOLS
from tests.support.logs import lines_of
from tests.support.openers import opener_whose_request_raises
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.wire import hangs

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


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
