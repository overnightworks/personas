"""Two runner processes against one Postgres, one app, one realm and one gateway: each agent is served by one of them,
the other takes it over when its holder is killed, stopped or loses its lock connection, and resumes his history."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import tool_round

from personas.config import OwnershipTimings
from tests.support.call import ADDRESSED, PERSON, PLACE_ID
from tests.support.gateway import BUDGET_SPENT, say, says
from tests.support.hopin import AGENT_SUBJECT, active
from tests.support.pods import SECOND_AGENT, Pods, Runner, serving_pods, until
from tests.support.postgres import answered_turns_kept, lock_holders, terminate_backend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

AGENT = "agent-mozart"
TIMINGS = OwnershipTimings(
    contend_seconds=0.5, heartbeat_timeout_seconds=1.0, cancel_margin_seconds=0.5, grace_seconds=3.0
)
TAKEOVER_SECONDS = TIMINGS.contend_seconds + TIMINGS.heartbeat_timeout_seconds + TIMINGS.grace_seconds
SLACK_SECONDS = 3.0
"""What a loaded machine may add between a signal and the line the other process logs."""
FIRST_ANSWER = "the first answer"
SECOND_ANSWER = "the second answer"
LATER_LINE = "And are you still there?"
AN_ACT = "Ada plays the queen."
STALE_GENERATION = "generation_stale"
REFUSALS_BEFORE_A_LONG_PAUSE = 4
"""Turns refused for a spent budget in a row until the pause outlasts a takeover: it doubles from one second."""
EARLY_LOG_SECONDS = 0.5
"""How much later than the pause began its line may reach the test."""


@pytest.fixture
async def pods(files, database) -> AsyncIterator[Pods]:
    files.use_database(database)
    table = (
        f"[ownership]\ncontend_seconds = {TIMINGS.contend_seconds}\n"
        f"heartbeat_timeout_seconds = {TIMINGS.heartbeat_timeout_seconds}\n"
        f"cancel_margin_seconds = {TIMINGS.cancel_margin_seconds}\ngrace_seconds = {TIMINGS.grace_seconds}\n"
    )
    async with serving_pods(files, database, table) as world:
        yield world


def is_listening(runner: Runner, agent: str) -> bool:
    return bool(runner.lines("loop.started", agent))


async def holder_and_other(pods: Pods) -> tuple[Runner, Runner]:
    """The first runner, once it serves both agents, and a second that is ready beside it and serves none."""
    holder = pods.start("first")
    await until(lambda: is_listening(holder, AGENT) and is_listening(holder, SECOND_AGENT), "the first serves both")
    other = pods.start("second")
    await until(other.is_ready, "the second is ready")
    return holder, other


async def test_two_pods_turn_ready_and_each_agent_listens_from_exactly_one_of_them(pods):
    first, second = pods.start("first"), pods.start("second")

    await until(lambda: first.is_ready() and second.is_ready(), "both pods ready")
    await until(
        lambda: all(is_listening(first, each) != is_listening(second, each) for each in (AGENT, SECOND_AGENT)),
        "each agent listens from one pod",
    )

    for agent in (AGENT, SECOND_AGENT):
        assert len(first.lines("loop.started", agent)) + len(second.lines("loop.started", agent)) == 1
        assert len(first.lines("agent.taken", agent)) + len(second.lines("agent.taken", agent)) == 1


async def test_a_pod_that_holds_no_agent_is_ready_and_listens_for_none(pods):
    holder, other = await holder_and_other(pods)

    assert other.is_ready()
    assert not other.lines("loop.started")
    assert not other.lines("agent.taken")
    assert holder.is_ready()


async def test_after_a_sigkill_of_the_holder_the_other_pod_takes_the_agent_within_the_bound(pods):
    holder, other = await holder_and_other(pods)

    killed_at = await holder.kill()
    await until(lambda: is_listening(other, AGENT), "the other pod listens for the agent")

    ((listening_at, _),) = other.lines("loop.started", AGENT)
    assert listening_at - killed_at <= TAKEOVER_SECONDS + SLACK_SECONDS


async def test_after_a_sigterm_of_the_holder_it_exits_cleanly_and_the_other_pod_takes_the_agent(pods):
    holder, other = await holder_and_other(pods)

    stopped_at = time.monotonic()
    assert await holder.terminate() == 0
    await until(lambda: is_listening(other, AGENT), "the other pod listens for the agent")

    ((listening_at, _),) = other.lines("loop.started", AGENT)
    assert listening_at - stopped_at <= TAKEOVER_SECONDS + SLACK_SECONDS


async def test_a_holder_whose_lock_connection_is_ended_during_a_turn_makes_no_tool_call_after_the_takers_first(
    pods, database
):
    pods.script_gateway(tool_round(say("what the old holder would say")), *says(PLACE_ID, SECOND_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder, other = await holder_and_other(pods)
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")
    release = pods.gateway.hold_request(1)
    pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await pods.gateway.until_requested(1)

    (lock_connection,) = set(lock_holders(database))
    terminate_backend(database, lock_connection)
    await until(lambda: holder.lines("agent.lost", AGENT), "the holder stopped the agent")
    release.set()
    reads_by_the_holder = pods.app.window_reads(PLACE_ID)
    await until(lambda: is_listening(other, AGENT), "the other pod listens for the agent")
    await until(lambda: pods.app.window_reads(PLACE_ID) > reads_by_the_holder, "the other pod read the place")
    pods.app.speak(PLACE_ID, PERSON, "Ada", LATER_LINE)
    await pods.app.said_count(1)

    ((lost_at, _),) = holder.lines("agent.lost", AGENT)
    ((listening_at, _),) = other.lines("loop.started", AGENT)
    assert lost_at < listening_at
    assert pods.calls_of("first", "say") == 0
    assert pods.calls_of("second", "say") == 1


async def test_a_line_that_a_killed_holder_answered_gets_no_second_turn_from_the_pod_that_takes_over(pods):
    pods.script_gateway(*says(PLACE_ID, FIRST_ANSWER, SECOND_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder, other = await holder_and_other(pods)
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")
    first_line = pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await pods.app.said_count(1)
    await until(lambda: holder.lines("turn.ended"), "the holder's turn ended")

    await holder.kill()
    await until(lambda: is_listening(other, AGENT), "the other pod listens for the agent")
    reads_by_the_holder = pods.app.window_reads(PLACE_ID)
    await until(lambda: pods.app.window_reads(PLACE_ID) > reads_by_the_holder, "the other pod read the place")
    later_line = pods.app.speak(PLACE_ID, PERSON, "Ada", LATER_LINE)
    await pods.app.said_count(2)

    assert [turn.woke_seqs for turn in pods.gateway.turns("lines")] == [[first_line], [later_line]]
    assert (pods.calls_of("first", "say"), pods.calls_of("second", "say")) == (1, 1)


async def test_after_a_sigkill_of_the_holder_the_takers_next_turn_carries_the_earlier_answered_messages_byte_for_byte(
    pods, database
):
    pods.script_gateway(*says(PLACE_ID, FIRST_ANSWER, SECOND_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder, other = await holder_and_other(pods)
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")
    pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await pods.app.said_count(1)
    await until(lambda: answered_turns_kept(database, AGENT) == 1, "the holder kept the answered turn")

    await holder.kill()
    await until(lambda: is_listening(other, AGENT), "the other pod listens for the agent")
    reads_by_the_holder = pods.app.window_reads(PLACE_ID)
    await until(lambda: pods.app.window_reads(PLACE_ID) > reads_by_the_holder, "the other pod read the place")
    pods.app.speak(PLACE_ID, PERSON, "Ada", LATER_LINE)
    await pods.app.said_count(2)

    first, second = pods.gateway.turns("lines")
    assert second.texts[: len(first.texts)] == first.texts
    assert len(second.texts) == len(first.texts) + 1


async def test_an_erase_refuses_the_holders_next_write_so_he_stops_and_serves_the_agent_again_from_the_erased_history(
    pods, database
):
    pods.script_gateway(*says(PLACE_ID, FIRST_ANSWER, SECOND_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder = pods.start("first")
    await until(lambda: is_listening(holder, AGENT), "the holder serves the agent")
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")
    pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await pods.app.said_count(1)
    await until(lambda: answered_turns_kept(database, AGENT) == 1, "the holder kept the answered turn")

    assert await pods.command("first", "memory", "erase", "--agent", AGENT) == 0
    pods.app.speak(PLACE_ID, PERSON, "Ada", AN_ACT, act=True)
    await until(lambda: holder.lines("agent.lost", AGENT), "the holder stopped the agent")
    await until(lambda: len(holder.lines("loop.started", AGENT)) == 2, "the holder serves the agent again")
    reads_before = pods.app.window_reads(PLACE_ID)
    await until(lambda: pods.app.window_reads(PLACE_ID) > reads_before, "the holder read the place again")
    pods.app.speak(PLACE_ID, PERSON, "Ada", LATER_LINE)
    await pods.app.said_count(2)

    ((_, lost),) = holder.lines("agent.lost", AGENT)
    assert lost["reason"] == STALE_GENERATION
    first, second = pods.gateway.turns("lines")
    assert first.texts[-1] not in second.texts
    assert len(second.texts) == len(first.texts)


async def test_a_line_said_while_the_holder_is_dead_gets_exactly_one_turn_on_the_pod_that_takes_over(pods):
    pods.script_gateway(*says(PLACE_ID, FIRST_ANSWER, SECOND_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder, _ = await holder_and_other(pods)
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")

    await holder.kill()
    line = pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await pods.app.said_count(1)
    later_line = pods.app.speak(PLACE_ID, PERSON, "Ada", LATER_LINE)
    await pods.app.said_count(2)

    woken = [turn.woke_seqs for turn in pods.gateway.turns("lines")]
    assert (woken[0], len(woken), max(woken[1])) == ([line], 2, later_line)
    assert (pods.calls_of("first", "say"), pods.calls_of("second", "say")) == (0, 2)


async def test_a_line_the_killed_holders_turn_was_answering_gets_no_turn_on_the_pod_that_takes_over(pods):
    pods.script_gateway(*says(PLACE_ID, FIRST_ANSWER, SECOND_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder, other = await holder_and_other(pods)
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")
    pods.gateway.hold_request(1)
    pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await pods.gateway.until_requested(1)

    await holder.kill()
    await until(lambda: is_listening(other, AGENT), "the other pod listens for the agent")
    reads_by_the_holder = pods.app.window_reads(PLACE_ID)
    await until(lambda: pods.app.window_reads(PLACE_ID) > reads_by_the_holder, "the other pod read the place")
    later_line = pods.app.speak(PLACE_ID, PERSON, "Ada", LATER_LINE)
    await pods.app.said_count(1)

    assert [turn.woke_seqs for turn in pods.gateway.turns("lines")] == [[later_line]]
    assert (pods.calls_of("first", "say"), pods.calls_of("second", "say")) == (0, 1)


async def test_a_pause_the_killed_holder_left_holds_the_turn_on_the_pod_that_takes_over_until_it_ends(pods):
    pods.script_gateway(*[BUDGET_SPENT] * REFUSALS_BEFORE_A_LONG_PAUSE, *says(PLACE_ID, FIRST_ANSWER))
    pods.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    holder, _ = await holder_and_other(pods)
    await until(lambda: pods.app.window_reads(PLACE_ID) > 0, "the holder read the place")
    pods.app.speak(PLACE_ID, PERSON, "Ada", ADDRESSED)
    await until(
        lambda: len(holder.lines("turns.paused", AGENT)) == REFUSALS_BEFORE_A_LONG_PAUSE, "the holder paused long"
    )

    await holder.kill()
    await pods.gateway.until_requested(REFUSALS_BEFORE_A_LONG_PAUSE + 1)
    requested_at = time.monotonic()
    await pods.app.said_count(1)

    paused_at, paused = holder.lines("turns.paused", AGENT)[-1]
    assert requested_at >= paused_at + paused["pause_ms"] / 1000 - EARLY_LOG_SECONDS
    assert (pods.calls_of("first", "say"), pods.calls_of("second", "say")) == (0, 1)
