"""The whole runner, a turn's listing: the tools of its apps are listed while its claim is written."""

from __future__ import annotations

import asyncio

import pytest

from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import CANCELLED_TURN
from tests.support.gateway import Gateway, says
from tests.support.history import HistoriesHoldingTheAddressedLine, settled
from tests.support.hopin import LIMIT_SECONDS, LIST_TOOLS
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
