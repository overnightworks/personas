"""The whole runner, a spoken turn held by a sound without words: the app ends the hold by itself."""

from __future__ import annotations

import asyncio
import json

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL
from agents.voice.conversation import told_went_on

from personas.decision import READ_SKILL_TOOL
from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, PLACE_ID, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.gateway import say
from tests.support.hopin import CALL, LIMIT_SECONDS, SKILLS_GET
from tests.support.skills import Shelf, a_skill_file
from tests.support.songmaker import CREATE_SONG
from tests.support.voice import (
    FIRST,
    HOLD_ID,
    SAID,
    SECOND,
    SECOND_HOLD,
    THIRD,
    THREE_SENTENCES,
    a_voice_call,
    calling_while_held,
    decided,
    deciding,
    holding_sentence_ends,
    play_out,
    sentences_spoken,
)
from tests.support.wire import answers_once_released

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


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


async def test_at_its_going_on_the_result_of_his_waiting_call_tells_he_went_on_and_no_later_turn_does(runner):
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
    assert told_of_going_on(runner) == [3]


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


async def test_at_its_going_on_a_speak_the_app_holds_past_its_bound_is_not_given_up_and_plays(runner):
    """TURNS-28, TURNS-29."""
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(THREE_SENTENCES), completion())
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)
    went_on_after = PERSONA_MAX_DURATION_SECONDS // 2
    runner.clock.time += went_on_after

    runner.app.go_on(PLACE_ID)
    await settled(runner.app)
    # The held wait ends at the bound it was given, which the going on has put off.
    runner.clock.time -= went_on_after
    sleeps.let_through()
    await play_out(runner, 3)

    assert sentences_spoken(runner) == [FIRST, SECOND, THIRD]
    assert runner.reports.changes_of(Event.SPEECH_DROPPED) == []


async def test_a_round_at_a_hold_with_words_after_his_going_on_is_not_told_again_that_he_went_on(runner):
    """TURNS-29, TURNS-47."""
    await calling_while_held(
        runner,
        completion("[silent]"),
        deciding(CONTINUE_SPEAKING_TOOL, SECOND_HOLD),
        completion("[silent]"),
        wordless=True,
        words=False,
    )
    turn_goes_on = runner.gateway.hold_request(3)

    runner.app.go_on(PLACE_ID)
    await runner.gateway.until_requested(3)
    runner.app.hold(PLACE_ID, SECOND_HOLD, SAID)
    await decided(runner)
    turn_goes_on.set()

    assert told_of_going_on(runner) == []


async def test_at_its_going_on_the_next_turn_is_told_he_went_on_once_when_no_call_waited(runner):
    """TURNS-29, TURNS-31."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(THREE_SENTENCES), *[completion("[silent]")] * 3)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)

    runner.app.go_on(PLACE_ID)
    await play_out(runner, 3)
    await runner.reports.until_a_turn_ended()
    await another_turn(runner)
    await another_turn(runner)

    assert told_of_going_on(runner) == [2]


async def test_at_its_going_on_a_round_at_a_hold_with_words_is_told_he_went_on_once_when_no_call_waited(runner):
    """TURNS-29, TURNS-47."""
    holding_sentence_ends(runner)
    await a_voice_call(
        runner,
        tool_round(say("before", PLACE_ID), content=FIRST),
        tool_round(say("waits", PLACE_ID)),
        deciding(CONTINUE_SPEAKING_TOOL, SECOND_HOLD),
        *[completion("[silent]")] * 3,
    )
    turn_goes_on = runner.gateway.hold_request(2)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.said_count(1)
    await runner.app.spoken_count(1)
    runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)

    runner.app.go_on(PLACE_ID)
    await runner.gateway.until_requested(2)
    runner.app.hold(PLACE_ID, SECOND_HOLD, SAID)
    await decided(runner)
    turn_goes_on.set()
    await runner.reports.until_a_turn_ended()
    requests_before_the_next_turn = len(runner.gateway.requests)
    await another_turn(runner)

    told = told_of_going_on(runner)
    assert runner.gateway.rounds()[0] in told
    assert max(told) <= requests_before_the_next_turn


async def test_at_its_going_on_the_result_of_a_skill_read_tells_he_went_on_once_and_no_later_call_round_or_turn_does(
    runner,
):
    """TURNS-29, TURNS-30."""
    runner.app.shelf = Shelf({"hopin": a_skill_file("hopin")})
    entered, released = asyncio.Event(), asyncio.Event()

    runner.app.script(SKILLS_GET, runner.app.shelf.entry_held_until(entered, released))
    await a_voice_call(
        runner,
        tool_round(tool_call("reads", READ_SKILL_TOOL, {"skill": "hopin"})),
        tool_round(say("after", PLACE_ID)),
        completion("[silent]"),
        completion("[silent]"),
        completion("[silent]"),
    )
    say_in_the_call(runner, ADDRESSED, spoken=True)
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()
    runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)

    runner.app.go_on(PLACE_ID)
    await settled(runner.app)
    released.set()
    await runner.reports.until_a_turn_ended()
    await another_turn(runner)
    await another_turn(runner)

    turn_rounds = 3
    assert told_of_going_on(runner) == [2, turn_rounds]
    assert runner.gateway.told_in(turn_rounds).count(json.dumps(told_went_on())) == 1


async def test_at_its_going_on_the_result_of_a_waiting_call_to_another_app_tells_he_went_on_once(runner, songmaker):
    """TURNS-29, TURNS-31."""
    entered, released = asyncio.Event(), asyncio.Event()
    songmaker.script(CALL, answers_once_released(entered, released, "made"))
    await a_voice_call(
        runner,
        tool_round(tool_call("elsewhere", CREATE_SONG, {"title": "t"})),
        *[completion("[silent]")] * 3,
    )
    say_in_the_call(runner, ADDRESSED, spoken=True)
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()
    runner.app.hold_wordless(PLACE_ID, HOLD_ID, ("Ada", "…"))
    await settled(runner.app)

    runner.app.go_on(PLACE_ID)
    await settled(runner.app)
    released.set()
    await runner.reports.until_a_turn_ended()
    await another_turn(runner)

    assert told_of_going_on(runner) == [2]


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
