"""The whole runner, the place's own app announcing its tools: the list its session keeps per stream."""

from __future__ import annotations

import asyncio

import pytest
from agents.backend.fake_gateway import completion
from agents.member import STANDING_URI, place_window_uri

from tests.support.call import ADDRESSED, CALL_PLACE, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.hopin import LIST_TOOLS, LISTEN, SAY_TOOL
from tests.support.logs import lines_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

TURNS = 4


@pytest.fixture
def announces_tools() -> bool:
    return True


def listings_written(caplog) -> int:
    return len(lines_of(caplog.get_records("call"), "tool_list.ended"))


async def a_turn_at_the_call(runner, count: int) -> None:
    """The call's addressed line makes a turn, and the ``count``-th turn has completed."""
    say_in_the_call(runner, f"{ADDRESSED} Number {count}.")
    await runner.reports.until_a_turn_completed(count=count)


@pytest.fixture
async def after_the_first_turn(runner):
    runner.gateway = Gateway(*[completion("done") for _ in range(TURNS)])
    await restarted_in_the_call(runner)
    await a_turn_at_the_call(runner, 1)
    return runner


async def test_two_turns_without_a_notice_send_one_listing_and_the_second_writes_no_listing_line(runner, caplog):
    """TURNS-18, LOG-13."""
    runner.gateway = Gateway(completion("done"), completion("done"))
    await restarted_in_the_call(runner)
    await a_turn_at_the_call(runner, 1)
    listed = runner.app.log.count(LIST_TOOLS)
    await a_turn_at_the_call(runner, 2)

    assert runner.app.log.count(LIST_TOOLS) == listed
    assert listings_written(caplog) == 1
    assert runner.gateway.offered_tools(2) == runner.gateway.offered_tools(1) == [SAY_TOOL]


async def test_a_tool_the_app_withdraws_and_announces_before_the_line_is_gone_from_the_next_turn_and_back_when_offered(
    after_the_first_turn,
):
    """TURNS-18."""
    runner = after_the_first_turn
    listed = runner.app.log.count(LIST_TOOLS)
    await runner.app.change_offer(SUBJECT, withheld={SAY_TOOL})
    await a_turn_at_the_call(runner, 2)
    await runner.app.change_offer(SUBJECT, withheld=set())
    await a_turn_at_the_call(runner, 3)

    assert [runner.gateway.offered_tools(each) for each in (1, 2, 3)] == [[SAY_TOOL], [], [SAY_TOOL]]
    assert runner.app.log.count(LIST_TOOLS) == listed + 2


async def test_a_tools_notice_alone_reads_no_standing(runner):
    """TURNS-18."""
    runner.gateway = Gateway(completion("done"))
    answered = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    standings = runner.app.standing_reads()
    await runner.app.change_offer(SUBJECT, withheld={SAY_TOOL})
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    read_for_the_line = runner.app.standing_reads() - standings
    answered.set()
    await runner.reports.until_a_turn_completed()

    assert read_for_the_line == 1


async def test_a_notice_and_the_pulse_of_one_change_behind_an_older_pulse_give_the_next_turn_the_new_list(
    after_the_first_turn,
):
    """TURNS-18."""
    runner = after_the_first_turn
    listed = runner.app.log.count(LIST_TOOLS)
    older = runner.app.hold_next_window_read()
    runner.app.speak(CALL_PLACE.id, "visitor-subject", "Ada", "Hello everyone")
    await runner.app.until_window_read_held()
    await runner.app.change_offer(SUBJECT, withheld={SAY_TOOL})
    say_in_the_call(runner, f"{ADDRESSED} Number 2.")
    older.set()
    await runner.reports.until_a_turn_completed(count=2)

    assert runner.gateway.offered_tools(2) == []
    assert runner.app.log.count(LIST_TOOLS) == listed + 1


async def test_a_listing_a_notice_arrives_during_is_not_kept_and_the_next_turn_lists_again(runner):
    """TURNS-18."""
    runner.gateway = Gateway(completion("done"), completion("done"))
    await restarted_in_the_call(runner)
    listing = runner.app.hold_next_offer()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LIST_TOOLS)
    await runner.app.change_offer(SUBJECT, withheld=set())
    say_in_the_call(runner, "Hello everyone")
    await runner.app.until_window_read(CALL_PLACE.id, 3)
    listing.set()
    await runner.reports.until_a_turn_completed()
    listed = runner.app.log.count(LIST_TOOLS)
    await a_turn_at_the_call(runner, 2)

    assert runner.app.log.count(LIST_TOOLS) == listed + 1


def honours_the_tools_notice_until_it_ends(line_said: asyncio.Event, ends: asyncio.Event):
    async def script(reply) -> None:
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI], tools=True)
        await line_said.wait()
        await reply.pulse(place_window_uri(CALL_PLACE))
        await ends.wait()
        await reply.complete()

    return script


async def test_a_turn_after_the_stream_ended_and_was_acknowledged_again_lists_again(runner, caplog):
    """TURNS-18."""
    line_said = asyncio.Event()
    ends = asyncio.Event()

    runner.gateway = Gateway(completion("done"), completion("done"))
    runner.app.script(LISTEN, honours_the_tools_notice_until_it_ends(line_said, ends))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    line_said.set()
    await runner.reports.until_a_turn_completed()
    ends.set()
    await runner.app.log.until(LISTEN, 2)
    await a_turn_at_the_call(runner, 2)

    assert listings_written(caplog) == 2


async def test_a_listing_across_the_end_and_the_acknowledgement_of_a_stream_is_not_kept_and_the_next_turn_lists_again(
    runner,
):
    """TURNS-18."""
    line_said = asyncio.Event()
    ends = asyncio.Event()
    acknowledges = asyncio.Event()
    second_line = asyncio.Event()
    stays = asyncio.Event()

    async def honours_the_tools_notice_once_let(reply) -> None:
        await reply.open_stream()
        await acknowledges.wait()
        await reply.acknowledge([STANDING_URI], tools=True)
        await reply.pulse(place_window_uri(CALL_PLACE))
        await second_line.wait()
        await reply.pulse(place_window_uri(CALL_PLACE))
        await stays.wait()
        await reply.complete()

    runner.gateway = Gateway(completion("done"), completion("done"))
    runner.app.script(
        LISTEN, honours_the_tools_notice_until_it_ends(line_said, ends), honours_the_tools_notice_once_let
    )
    await restarted_in_the_call(runner)
    listing = runner.app.hold_next_offer()
    say_in_the_call(runner, ADDRESSED)
    line_said.set()
    await runner.app.log.until(LIST_TOOLS)
    ends.set()
    await runner.app.log.until(LISTEN, 2)
    reads = runner.app.window_reads(CALL_PLACE.id)
    acknowledges.set()
    await runner.app.until_window_read(CALL_PLACE.id, reads + 1)
    listing.set()
    await runner.reports.until_a_turn_completed()
    listed = runner.app.log.count(LIST_TOOLS)
    say_in_the_call(runner, f"{ADDRESSED} Number 2.")
    second_line.set()
    await runner.reports.until_a_turn_completed(count=2)
    stays.set()

    assert runner.app.log.count(LIST_TOOLS) == listed + 1
