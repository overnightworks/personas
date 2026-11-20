"""The whole runner, his short line before he goes on: ``continue_speaking`` with a ``line`` says the line at once, cuts
the sentence he was held in and sends his unsent sentences after it."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, SpeakRefusal

from personas.reporting import Event
from tests.support.call import PLACE_ID
from tests.support.voice import (
    FIRST,
    HOLD_ID,
    SAID,
    SECOND,
    a_voice_call,
    calling_while_held,
    held_in_the_first_sentence,
    sentences_spoken,
)
from tests.test_loop_voice_answer import THREE_SENTENCES, decided, refusing

pytestmark = pytest.mark.anyio

THIRD = "Third sentence."
LINE = "Right, one moment."


def continuing(arguments: dict[str, object]):
    return tool_round(tool_call("continue-call", CONTINUE_SPEAKING_TOOL, {"id": HOLD_ID, **arguments}))


async def held_with_a_line(runner, line: object = LINE):
    """A turn whose first sentence plays, whose second waits at the app and whose third is unsent, held by the app; the
    round that follows calls ``continue_speaking`` with the line."""
    await held_in_the_first_sentence(
        runner, completion(THREE_SENTENCES), continuing({"line": line}), completion("[silent]")
    )


async def test_a_short_line_is_said_at_once_and_his_unsent_sentences_follow_plain(runner):
    """TURNS-30, TURNS-31, LOG-17."""
    await held_with_a_line(runner)

    assert await decided(runner) == ["short_line"]
    runner.app.end_sentence()
    await runner.app.spoken_count(3)
    runner.app.end_sentence()
    await runner.app.spoken_count(4)

    assert runner.app.speaks_naming_holds == [
        (FIRST, None),
        (SECOND, None),
        (LINE, HOLD_ID),
        (SECOND, None),
        (THIRD, None),
    ]
    assert sentences_spoken(runner) == [FIRST, LINE, SECOND, THIRD]
    assert runner.app.hold_calls == []
    assert Event.SPEAK_REFUSED not in [event for event, _ in runner.reports.reported()]


@pytest.mark.parametrize(
    ("refusal", "outcome", "stops"),
    [
        pytest.param(SpeakRefusal.TOO_LONG, "undecided", [(STOP_SPEAKING_TOOL, HOLD_ID)], id="refused"),
        pytest.param(SpeakRefusal.NOT_INTERRUPTED, "gone", [], id="not interrupted"),
    ],
)
async def test_a_short_line_the_app_refused_sends_no_further_sentence_and_ends_the_hold_as_the_app_says(
    runner, refusal, outcome, stops
):
    """TURNS-30, LOG-17."""
    runner.app.speak_outcomes.extend([None, None, refusing(refusal)])
    await held_with_a_line(runner)

    assert await decided(runner) == [outcome]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == stops
    assert runner.reports.codes_of(Event.SPEAK_REFUSED) == [refusal.value]
    assert [text for text, _ in runner.app.speaks_naming_holds if text == THIRD] == []


async def test_a_line_that_is_no_string_decides_nothing(runner):
    """TURNS-30."""
    await held_with_a_line(runner, line=7)

    await runner.reports.until(Event.HOLD_DECIDED)

    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["undecided"]
    assert all(hold is None for _, hold in runner.app.speaks_naming_holds)


async def test_a_turn_told_a_hold_says_the_short_line_and_its_sentences_follow_it(runner):
    """TURNS-31, LOG-17."""
    await a_voice_call(runner, continuing({"line": LINE}), completion(f"{FIRST} {SECOND}"))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    assert await decided(runner) == ["short_line"]
    await runner.app.spoken_count(3)

    assert runner.app.speaks_naming_holds == [(LINE, HOLD_ID), (FIRST, None), (SECOND, None)]
    assert runner.app.hold_calls == []


async def test_the_short_line_beside_his_parked_run_leaves_the_run_to_go_on(runner):
    """TURNS-30, LOG-17."""
    deciding_round = await calling_while_held(
        runner, continuing({"line": LINE}), completion("[silent]"), hold_round=True
    )
    await runner.gateway.until_requested(3)
    deciding_round.set()

    assert await decided(runner) == ["short_line"]
    await runner.reports.until_a_turn_completed()

    assert [text for text, hold in runner.app.speaks_naming_holds if hold] == [LINE]
    assert [text for *_, text in runner.app.said] == ["before", "waits"]
