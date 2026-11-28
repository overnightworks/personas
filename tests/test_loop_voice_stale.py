"""The whole runner, a spoken turn held while he speaks, against an app that counts the worded changes of the talk-over:
a decision told fewer of them is decided again, and nothing that waited is dropped before the app took his decision."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL

from personas.holds import DECISION_CALLS
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
    decided,
    deciding,
    held_in_the_first_sentence,
    play_out,
    sentences_spoken,
)

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


async def test_decisions_told_too_few_changes_past_the_bound_end_the_round_undecided_and_stop_him(runner):
    """TURNS-31."""
    runner.app.count_worded_changes()
    runner.app.words_before_decision.extend(more_from_ada(*[f"wait {each}" for each in range(DECISION_CALLS)]))
    await held_in_the_first_sentence(
        runner, completion(TWO_SENTENCES), *deciding_times(CONTINUE_SPEAKING_TOOL, DECISION_CALLS + 1)
    )

    assert await decided(runner) == ["undecided"]
    await runner.reports.until_a_turn_completed()

    assert len(runner.app.hold_told) == DECISION_CALLS + 1
    assert runner.app.hold_calls[-1] == (STOP_SPEAKING_TOOL, HOLD_ID)
    assert sentences_spoken(runner) == [FIRST]


async def test_a_stale_stop_keeps_what_waited_and_the_decision_made_again_plays_it(runner):
    """TURNS-31."""
    runner.app.count_worded_changes()
    runner.app.words_before_decision.extend(more_from_ada("wait a moment please"))
    await held_in_the_first_sentence(
        runner,
        completion(TWO_SENTENCES),
        deciding(STOP_SPEAKING_TOOL),
        deciding(CONTINUE_SPEAKING_TOOL, time=2),
        completion(),
    )

    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()

    assert [tool for tool, _ in runner.app.hold_calls] == [STOP_SPEAKING_TOOL, CONTINUE_SPEAKING_TOOL]
    (number,) = runner.gateway.rounds()
    assert told_since(runner, number + 2) == [[{"name": "Ada", "text": "wait a moment please"}]]
    assert sentences_spoken(runner) == [FIRST, SECOND]
    assert Event.SPEAK_REFUSED not in [event for event, _ in runner.reports.reported()]


async def test_a_stop_whose_hold_was_voided_keeps_what_waited_and_the_new_hold_is_decided_as_the_first(runner):
    """TURNS-31, TURNS-47."""
    runner.app.count_worded_changes()
    runner.app.new_holds_before_decision.append((SECOND_HOLD, BOB_SAYS))
    await held_in_the_first_sentence(
        runner,
        completion(TWO_SENTENCES),
        deciding(STOP_SPEAKING_TOOL),
        completion("[silent]"),
        deciding(CONTINUE_SPEAKING_TOOL, SECOND_HOLD),
        completion(),
    )

    assert await decided(runner) == ["continued"]
    await play_out(runner, 2)
    await runner.reports.until_a_turn_completed()

    assert sentences_spoken(runner) == [FIRST, SECOND]
    assert (CONTINUE_SPEAKING_TOOL, SECOND_HOLD) in runner.app.hold_calls
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["continued"]


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
    assert runner.gateway.tool_results(first + 1) == ['{"gone": true}']
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
        runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion("[silent]"), completion("Alright.")
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
