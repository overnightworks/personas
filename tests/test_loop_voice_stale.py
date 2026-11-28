"""The whole runner, a spoken turn held while he speaks, against an app that counts the worded changes of the talk-over:
a decision told fewer of them is decided again, and nothing that waited is dropped before the app took his decision."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import (
    CONTINUE_SPEAKING_TOOL,
    STOP_SPEAKING_TOOL,
    ContinueSpeakingRefusal,
    ContinueSpeakingRefused,
    StopSpeakingRefusal,
    StopSpeakingRefused,
    VoiceState,
)
from agents.tools import ToolOutcome

from personas.holds import DECISION_CALLS, GONE_TEXT
from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, PLACE_ID, say_in_the_call
from tests.support.voice import (
    FIRST,
    HOLD_ID,
    SAID,
    SECOND,
    SECOND_HOLD,
    TWO_SENTENCES,
    a_voice_call,
    calling_while_held,
    decided,
    deciding,
    held_in_the_first_sentence,
    play_out,
    sentences_spoken,
)

if TYPE_CHECKING:
    import asyncio

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

SHORT_CALL_SECONDS = 0.3
BOB_SAYS = ("Bob", "no, wait")


def deciding_times(tool: str, count: int):
    return [deciding(tool, time=time) for time in range(1, count + 1)]


def more_from_ada(*texts: str) -> list[tuple[tuple[str, str], ...]]:
    """What Ada says over him while he decides, one change for each decision that finds it."""
    return [(("Ada", text),) for text in texts]


def told_since(runner, round_request: int) -> list[list[dict[str, str]]]:
    """What each result of a decision that was not made told him was said over him since, as the round's last request
    read them."""
    return [each["talked_over"] for each in runner.gateway.held_results(round_request)]


BESIDE_THE_PARKED_RUN = pytest.mark.parametrize(
    "beside_parked_run", [True, False], ids=["round beside his parked run", "turn told the hold"]
)
DECISIONS = pytest.mark.parametrize("kind", ["continue", "stop", "line"])
WAITING_SENTENCES = {True: ["before", "waits"], False: [FIRST, SECOND]}
"""What reaches the app of what waited at the hold, by whether a round beside the parked run decides it."""
FIRST_SENTENCE_HEARD = {True: "before", False: FIRST}
LINE = "Right, one moment."


def decision(kind: str, time: int = 1, hold_id: str = HOLD_ID):
    """A reply that decides the hold with a continue, a stop or a continue that says a short line; a decision made again
    is told apart by its ``time``."""
    tool = STOP_SPEAKING_TOOL if kind == "stop" else CONTINUE_SPEAKING_TOOL
    arguments = {"id": hold_id, **({"line": LINE} if kind == "line" else {})}
    return tool_round(tool_call(f"{kind}-{hold_id}-{time}", tool, arguments))


def stale(tool: str) -> ToolOutcome:
    refusal = (
        StopSpeakingRefused(refused=StopSpeakingRefusal.STALE)
        if tool == STOP_SPEAKING_TOOL
        else ContinueSpeakingRefused(refused=ContinueSpeakingRefusal.STALE)
    )
    return ToolOutcome(refusal.model_dump_json(), is_error=True)


async def held_and_decided_by(runner, beside_parked_run: bool, *replies):
    """The member is held and ``replies`` decide the hold from the first on: in the round beside his run parked at a
    call, or in the turn told the hold once its reply is over."""
    if beside_parked_run:
        await calling_while_held(runner, *replies)
    else:
        await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), *replies)


async def held_with_the_first_request_waiting(runner, beside_parked_run: bool, *replies) -> asyncio.Event:
    """As ``held_and_decided_by``, but the request that makes the first decision waits for the returned event."""
    if beside_parked_run:
        release = await calling_while_held(runner, *replies, hold_round=True)
        await runner.gateway.until_requested(3)
    else:
        await a_voice_call(runner, completion(TWO_SENTENCES), *replies)
        release = runner.gateway.hold_request(2)
        say_in_the_call(runner, ADDRESSED, spoken=True)
        await runner.app.until_called(2)
        runner.app.hold(PLACE_ID, HOLD_ID, SAID)
        await runner.gateway.until_requested(2)
    await settled(runner.app)
    return release


def sentences_heard(runner, beside_parked_run: bool) -> list[str]:
    """What the app was sent of the turn's own sentences and of the call parked behind the hold."""
    if beside_parked_run:
        return [text for *_, text in runner.app.said]
    return sentences_spoken(runner)


async def played_out(runner, beside_parked_run: bool) -> list[str]:
    """What waited at the hold, once it has reached the app."""
    if beside_parked_run:
        await runner.app.said_count(2)
    else:
        await play_out(runner, 2)
    return sentences_heard(runner, beside_parked_run)


async def window_read_after(runner, reads: int = 1) -> None:
    await runner.app.until_window_read(PLACE_ID, runner.app.window_reads(PLACE_ID) + reads)


async def test_a_decision_told_fewer_changes_than_the_app_counted_is_decided_again_until_one_plays(runner):
    """TURNS-31, LOG-17."""
    runner.app.count_worded_changes()
    runner.app.words_before_decision.extend(more_from_ada("wait a moment please", "wait a moment please do", "stop it"))
    await held_in_the_first_sentence(
        runner, completion(TWO_SENTENCES), *deciding_times(CONTINUE_SPEAKING_TOOL, 4), completion()
    )

    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_told == [1, 2, 3, 4]
    (number,) = runner.gateway.rounds()
    assert told_since(runner, number + 4) == [
        [{"name": "Ada", "text": "wait a moment please"}],
        [{"name": "Ada", "text": "wait a moment please do"}],
        [{"name": "Ada", "text": "stop it"}],
    ]
    assert sentences_spoken(runner) == [FIRST, SECOND]
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["continued"]


@BESIDE_THE_PARKED_RUN
@DECISIONS
async def test_a_stale_decision_is_no_decision_and_the_one_made_again_plays_what_waited(
    runner, beside_parked_run, kind
):
    """TURNS-31, TURNS-46."""
    runner.app.count_worded_changes()
    runner.app.words_before_decision.extend(more_from_ada("wait a moment please"))
    await held_and_decided_by(
        runner, beside_parked_run, decision(kind), decision("continue", time=2), completion(), completion()
    )

    assert await decided(runner) == ["continued"]
    assert await played_out(runner, beside_parked_run) == WAITING_SENTENCES[beside_parked_run]
    (number,) = runner.gateway.rounds()
    await runner.gateway.until_requested(number + 2)
    assert told_since(runner, number + 2) == [[{"name": "Ada", "text": "wait a moment please"}]]
    assert runner.app.hold_told == [1, 2]


@BESIDE_THE_PARKED_RUN
@DECISIONS
async def test_decisions_told_too_few_changes_past_the_bound_end_the_hold_undecided_and_stop_him_naming_no_count(
    runner, beside_parked_run, kind
):
    """TURNS-31."""
    runner.app.count_worded_changes()
    runner.app.words_before_decision.extend(more_from_ada(*[f"wait {each}" for each in range(DECISION_CALLS)]))
    await held_and_decided_by(
        runner, beside_parked_run, *(decision(kind, time) for time in range(1, DECISION_CALLS + 2))
    )

    assert await decided(runner) == ["undecided"]
    await runner.reports.until_a_turn_ended()

    assert len(runner.app.hold_told) == DECISION_CALLS + 1
    assert runner.app.hold_told[-1] is None
    assert runner.app.hold_calls[-1] == (STOP_SPEAKING_TOOL, HOLD_ID)
    assert sentences_heard(runner, beside_parked_run) == [FIRST_SENTENCE_HEARD[beside_parked_run]]


@BESIDE_THE_PARKED_RUN
@DECISIONS
async def test_a_decision_the_app_refused_as_not_interrupted_keeps_what_waited_and_the_new_hold_is_decided_first(
    runner, beside_parked_run, kind
):
    """TURNS-31, TURNS-47."""
    runner.app.new_holds_before_decision.append((SECOND_HOLD, BOB_SAYS))
    await held_and_decided_by(
        runner,
        beside_parked_run,
        decision(kind),
        completion("[silent]"),
        decision("continue", hold_id=SECOND_HOLD),
        completion(),
        completion(),
    )

    assert await decided(runner) == ["continued"]
    assert await played_out(runner, beside_parked_run) == WAITING_SENTENCES[beside_parked_run]
    (first, _second) = runner.gateway.rounds()
    assert runner.gateway.tool_results(first + 1) == [GONE_TEXT]
    assert (CONTINUE_SPEAKING_TOOL, SECOND_HOLD) in runner.app.hold_calls
    assert Event.SPEECH_DROPPED not in [event for event, _ in runner.reports.reported()]


@BESIDE_THE_PARKED_RUN
@pytest.mark.parametrize("tool", [CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL], ids=["continue", "stop"])
async def test_a_stale_decision_whose_hold_is_gone_when_he_is_told_what_was_said_decides_nothing_for_it(
    runner, beside_parked_run, tool
):
    """TURNS-31."""
    runner.app.hold_outcomes.append(stale(tool))
    release = await held_with_the_first_request_waiting(
        runner,
        beside_parked_run,
        decision("stop" if tool == STOP_SPEAKING_TOOL else "continue"),
        completion("[silent]"),
        completion(),
    )
    runner.app.go_on(PLACE_ID)
    await window_read_after(runner)
    release.set()

    assert await played_out(runner, beside_parked_run) == WAITING_SENTENCES[beside_parked_run]
    await settled(runner.app)
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == []


@BESIDE_THE_PARKED_RUN
async def test_a_stale_decision_whose_hold_he_cannot_read_again_is_told_so_and_he_decides_again(
    runner, beside_parked_run
):
    """TURNS-31."""
    runner.app.hold_outcomes.append(stale(CONTINUE_SPEAKING_TOOL))
    release = await held_with_the_first_request_waiting(
        runner, beside_parked_run, decision("continue"), decision("continue", time=2), completion(), completion()
    )
    number = runner.gateway.arrivals
    decides_again = runner.gateway.hold_request(number + 1)
    runner.app.without_window.add(PLACE_ID)
    release.set()
    await runner.gateway.until_requested(number + 1)
    runner.app.without_window.discard(PLACE_ID)
    decides_again.set()

    assert await decided(runner) == ["continued"]
    assert await played_out(runner, beside_parked_run) == WAITING_SENTENCES[beside_parked_run]
    (failed,) = runner.gateway.tool_results(number + 1)
    assert failed.startswith("Error")


async def test_a_turn_that_does_not_speak_is_told_what_was_said_since_a_stale_decision_and_decides_again(runner):
    """TURNS-31."""
    runner.app.state_voice(PLACE_ID, VoiceState.UNAVAILABLE)
    runner.app.count_worded_changes()
    runner.app.words_before_decision.extend(more_from_ada("wait a moment please"))
    await a_voice_call(
        runner,
        completion("done"),
        decision("continue"),
        decision("continue", time=2),
        completion("[silent]"),
    )
    answer = runner.gateway.hold_request(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await window_read_after(runner)
    answer.set()

    assert await decided(runner) == ["continued"]
    assert runner.app.hold_told == [1, 2]


async def test_a_decision_that_waits_and_is_voided_by_a_new_hold_keeps_what_waited_and_the_new_hold_is_decided_first(
    runner,
):
    """TURNS-31, TURNS-47."""
    runner.app.count_worded_changes()
    runner.app.choices_wait = True
    await held_in_the_first_sentence(
        runner,
        completion(TWO_SENTENCES),
        deciding(CONTINUE_SPEAKING_TOOL),
        completion("[silent]"),
        deciding(CONTINUE_SPEAKING_TOOL, SECOND_HOLD),
        completion(),
    )
    await runner.app.until_choices_waited(1)

    runner.app.hold(PLACE_ID, SECOND_HOLD, BOB_SAYS)
    await runner.app.until_choices_waited(2)
    runner.app.talk_over_ends(PLACE_ID)

    assert await decided(runner) == ["continued"]
    await play_out(runner, 2)
    await runner.reports.until_a_turn_completed()

    assert sentences_spoken(runner) == [FIRST, SECOND]
    (first, second) = runner.gateway.rounds()
    assert runner.gateway.tool_results(first + 1) == [GONE_TEXT]
    assert runner.gateway.hold_told(second)["id"] == SECOND_HOLD
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["continued"]


async def test_a_retry_after_an_expired_call_replaces_the_decision_that_still_waits_at_the_app(runner):
    """TURNS-31."""
    runner.app.count_worded_changes()
    runner.app.choices_wait = True
    runner.bound_listening_calls_to(SHORT_CALL_SECONDS)
    await held_in_the_first_sentence(
        runner,
        completion(TWO_SENTENCES),
        deciding(CONTINUE_SPEAKING_TOOL),
        deciding(CONTINUE_SPEAKING_TOOL, time=2),
        completion(),
    )
    await runner.app.until_choices_waited(2)
    runner.app.talk_over_ends(PLACE_ID)

    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == [(CONTINUE_SPEAKING_TOOL, HOLD_ID)] * 2
    assert sentences_spoken(runner) == [FIRST, SECOND]


async def test_a_line_that_repeats_the_words_he_was_told_makes_no_turn_due_and_another_line_does(runner):
    """TURNS-31."""
    await held_in_the_first_sentence(
        runner,
        completion(TWO_SENTENCES),
        deciding(CONTINUE_SPEAKING_TOOL),
        completion("[silent]"),
        completion("Alright."),
    )
    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()
    arrivals = runner.gateway.arrivals

    say_in_the_call(runner, SAID[1], spoken=True)
    await window_read_after(runner)
    await settled(runner.app)
    assert runner.gateway.arrivals == arrivals

    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.gateway.until_requested(arrivals + 1)


async def test_an_app_that_states_no_worded_changes_is_told_no_count_and_decides_as_before(runner):
    """TURNS-31."""
    await held_in_the_first_sentence(
        runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion(), said=(SAID,)
    )

    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_told == [None]
    assert sentences_spoken(runner) == [FIRST, SECOND]
