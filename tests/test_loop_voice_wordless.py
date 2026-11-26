"""The whole runner, a spoken turn held by a sound without words: the app ends the hold by itself."""

from __future__ import annotations

import json

import pytest
from agents.backend.fake_gateway import completion
from agents.member import CONTINUE_SPEAKING_TOOL
from agents.voice.conversation import told_went_on

from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, PLACE_ID, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.voice import (
    FIRST,
    HOLD_ID,
    SAID,
    SECOND,
    THIRD,
    THREE_SENTENCES,
    a_voice_call,
    calling_while_held,
    deciding,
    holding_sentence_ends,
    play_out,
    sentences_spoken,
)

pytestmark = pytest.mark.anyio


def said_texts(runner) -> list[str]:
    return [text for *_, text in runner.app.said]


def told_of_going_on(runner) -> list[int]:
    """The numbers (from one) of the requests that carried the line that tells him he went on."""
    line = json.dumps(told_went_on())
    return [number for number in range(1, len(runner.gateway.requests) + 1) if line in runner.gateway.told_in(number)]


async def another_turn(runner) -> None:
    """A person speaks in the call again, and the turn it wakes ends."""
    count = len(runner.reports.turn_endings) + 1
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until_a_turn_ended(count=count)


async def test_a_wordless_hold_starts_no_decision_and_his_waiting_call_waits(runner):
    """TURNS-30, TURNS-31."""
    await calling_while_held(runner, completion(), wordless=True, words=False)
    await settled(runner.app)

    assert runner.gateway.rounds() == []
    assert said_texts(runner) == ["before"]
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == []


async def test_a_hold_whose_only_partial_is_an_ellipsis_starts_no_decision(runner):
    """TURNS-30."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(THREE_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion())
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)

    assert runner.gateway.rounds() == []


async def test_at_its_going_on_his_waiting_call_goes_out_and_the_next_turn_is_told_he_went_on_once(runner):
    """TURNS-29, TURNS-31."""
    await calling_while_held(
        runner, completion("[silent]"), completion("[silent]"), completion("[silent]"), wordless=True, words=False
    )
    await settled(runner.app)

    runner.app.go_on(PLACE_ID)
    await runner.app.said_count(2)
    await runner.reports.until_a_turn_ended()
    await another_turn(runner)
    await another_turn(runner)

    assert said_texts(runner) == ["before", "waits"]
    assert runner.app.hold_calls == []
    assert len(told_of_going_on(runner)) == 1


async def test_at_its_going_on_a_sentence_past_its_bound_that_waited_plays(runner):
    """TURNS-28, TURNS-29."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(THREE_SENTENCES), completion())
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS + 1

    runner.app.go_on(PLACE_ID)
    await play_out(runner, 3)

    assert sentences_spoken(runner) == [FIRST, SECOND, THIRD]


async def test_a_word_said_over_a_wordless_hold_makes_it_decided_as_any_hold(runner):
    """TURNS-29, TURNS-30."""
    await calling_while_held(
        runner, deciding(CONTINUE_SPEAKING_TOOL), completion(), completion(), wordless=True, words=False
    )
    await settled(runner.app)
    assert runner.gateway.rounds() == []

    runner.app.talk_over(PLACE_ID, SAID)
    await runner.reports.until(Event.HOLD_DECIDED)
    await runner.app.said_count(2)

    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["continued"]
    assert runner.app.hold_calls == [(CONTINUE_SPEAKING_TOOL, HOLD_ID)]
    assert said_texts(runner) == ["before", "waits"]
