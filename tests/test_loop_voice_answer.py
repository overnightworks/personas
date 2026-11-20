"""The whole runner, text while his hold is undecided: it answers the hold instead of the rest he had not said."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import STOP_SPEAKING_TOOL, SpeakRefusal, SpeakRefused
from agents.tools import ToolOutcome

from personas.reporting import Event
from personas.speech import DropReason
from tests.support.call import PLACE_ID
from tests.support.history import settled
from tests.support.hopin import CALL
from tests.support.voice import (
    HOLD_ID,
    SAID,
    a_voice_call,
    deciding,
    held_in_the_first_sentence,
    sentences_spoken,
)
from tests.support.wire import refuses_with_a_server_error

pytestmark = pytest.mark.anyio

THREE_SENTENCES = "First sentence. Second sentence. Third sentence."
FIRST, SECOND = "First sentence.", "Second sentence."
ANSWER_FIRST, ANSWER_REST = "I hear you.", "Go on."
ANSWER = f"{ANSWER_FIRST} {ANSWER_REST}"


async def decided(runner, count: int = 1) -> list[str | None]:
    await runner.reports.until(Event.HOLD_DECIDED, count=count)
    return runner.reports.codes_of(Event.HOLD_DECIDED)


def refusing(refusal: SpeakRefusal) -> ToolOutcome:
    return ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True)


async def answered_in_the_round(runner, *replies):
    """A spoken turn whose first sentence plays, whose second waits for it at the app and whose third is unsent, held
    by the app; the round beside it makes the first of the replies."""
    await held_in_the_first_sentence(runner, completion(THREE_SENTENCES), *replies)


async def test_a_round_that_replies_with_text_answers_instead_and_the_rest_follows_plain(runner):
    """TURNS-27, TURNS-31, LOG-17, LOG-36."""
    await answered_in_the_round(runner, completion(ANSWER))

    assert await decided(runner) == ["answered"]
    runner.app.end_sentence()
    await runner.app.spoken_count(3)

    assert runner.app.speaks_naming_holds == [
        (FIRST, None),
        (SECOND, None),
        (ANSWER_FIRST, HOLD_ID),
        (ANSWER_REST, None),
    ]
    assert sentences_spoken(runner) == [FIRST, ANSWER_FIRST, ANSWER_REST]
    assert runner.reports.codes_of(Event.SPEECH_DROPPED) == [DropReason.REPLACED.value]
    assert [each.sentences for each in runner.reports.changes_of(Event.SPEECH_DROPPED)] == [1]
    assert runner.app.hold_calls == []
    assert Event.SPEAK_REFUSED not in [event for event, _ in runner.reports.reported()]


async def test_text_then_a_decision_call_in_one_reply_answers_instead_and_the_call_is_answered_decided(runner):
    """TURNS-27, TURNS-31."""
    stopping = tool_round(tool_call("stop-call", STOP_SPEAKING_TOOL, {"id": HOLD_ID}), content=ANSWER_FIRST)
    await answered_in_the_round(runner, stopping, completion("[silent]"))

    assert await decided(runner) == ["answered"]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == []
    assert [each for each, hold in runner.app.speaks_naming_holds if hold] == [ANSWER_FIRST]
    (result,) = runner.gateway.tool_results(3)
    assert "this hold is decided" in result


async def test_text_in_the_round_after_his_decision_says_nothing(runner):
    """TURNS-27."""
    await answered_in_the_round(runner, deciding(STOP_SPEAKING_TOOL), completion("Nothing to add."))

    assert await decided(runner) == ["stopped"]
    await runner.reports.until_a_turn_completed()

    assert all(hold is None for _, hold in runner.app.speaks_naming_holds)
    assert "Nothing to add." not in [text for text, _ in runner.app.speaks_naming_holds]


@pytest.mark.parametrize(
    ("refusal", "outcome", "stops"),
    [
        pytest.param(SpeakRefusal.TOO_LONG, "undecided", [(STOP_SPEAKING_TOOL, HOLD_ID)], id="refused"),
        pytest.param(SpeakRefusal.NOT_INTERRUPTED, "gone", [], id="not interrupted"),
    ],
)
async def test_a_first_answer_sentence_the_app_refused_sends_no_further_one_and_ends_the_hold_as_the_app_says(
    runner, refusal, outcome, stops
):
    """TURNS-27, LOG-17."""
    runner.app.speak_outcomes.extend([None, None, refusing(refusal)])
    await answered_in_the_round(runner, completion(ANSWER))

    assert await decided(runner) == [outcome]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == stops
    assert runner.reports.codes_of(Event.SPEAK_REFUSED) == [refusal.value]
    assert [text for text, _ in runner.app.speaks_naming_holds if text == ANSWER_REST] == []


async def test_a_new_hold_during_his_answer_starts_no_round_and_is_told_to_his_next_turn(runner):
    """TURNS-27, TURNS-30."""
    await a_voice_call(runner, completion(ANSWER), completion("[silent]"))
    answering = runner.app.hold_next_speak()
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.app.until_called(1)
    runner.app.hold(PLACE_ID, "hold-2", ("Bob", "no, stop"))
    await settled(runner.app)
    assert runner.gateway.arrivals == 1
    answering.set()

    assert await decided(runner, 2) == ["answered", "undecided"]

    assert runner.gateway.hold_told(2)["id"] == "hold-2"


async def test_a_turn_told_a_hold_that_replies_with_text_answers_instead_and_later_words_start_no_decision(runner):
    """TURNS-27, TURNS-30."""
    await a_voice_call(runner, completion(ANSWER))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    assert await decided(runner) == ["answered"]
    await runner.app.spoken_count(2)
    runner.app.talk_over(PLACE_ID, SAID, ("Bo", "me too"))
    await runner.app.until_window_read(PLACE_ID, runner.app.window_reads(PLACE_ID) + 1)

    assert runner.app.speaks_naming_holds == [(ANSWER_FIRST, HOLD_ID), (ANSWER_REST, None)]
    assert runner.app.hold_calls == []
    assert runner.gateway.arrivals == 1
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["answered"]


async def test_a_turn_told_a_hold_whose_first_answer_sentence_failed_without_an_answer_stops_him_and_sends_no_more(
    runner,
):
    """TURNS-27, LOG-17."""
    await a_voice_call(runner, completion(ANSWER))
    runner.app.script(CALL, refuses_with_a_server_error)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    assert await decided(runner) == ["undecided"]
    await runner.reports.until_a_turn_completed()
    assert runner.reports.codes_of(Event.HELD_APP_FAILED) != []
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]
    assert sentences_spoken(runner) == []
