"""The whole runner, a spoken turn that ends while his last sentence still plays: his line for it enters after the turn,
and the next turn's calls and sentences go out at once."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, VoiceState

from personas.reporting import Event
from tests.support.app_driving import settled
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, say_in_the_call
from tests.support.gateway import says
from tests.support.hopin import SAY_TOOL
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import a_voice_call, holding_sentence_ends, sentences_spoken

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

OTHER_PLACE_ID = "room-9"
HOLD_ID = "hold-1"
SAID = ("Ada", "wait a moment")
NEXT = "Next sentence."
WRITTEN = "Written."
AGAIN = f"{ADDRESSED} Again."


def says_then_stays_silent(place_id: str, text: str):
    return [tool_round(tool_call("call-0", SAY_TOOL, {"place": place_id, "text": text})), completion("[silent]")]


def deciding(tool: str):
    return tool_round(tool_call(f"{tool}-{HOLD_ID}", tool, {"id": HOLD_ID}))


async def a_last_sentence_that_plays(runner, *replies):
    """The runner restarted in a call where his first turn spoke one sentence and ended ok while it still plays: its
    line is owed, and the app ends the sentence only when the test says so."""
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(ANSWER), *replies)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed()
    return sleeps


async def at_a_second_place(runner) -> None:
    runner.app.stand(SUBJECT, active(PLACE_ID), active(OTHER_PLACE_ID))
    await runner.app.until_window_read(OTHER_PLACE_ID)


def said_at(runner, place_id: str, text: str, *, spoken: bool = True) -> None:
    runner.app.speak(place_id, PERSON, "Ada", text, spoken=spoken)


@pytest.mark.parametrize("cut", [False, True], ids=["whole", "cut"])
async def test_a_line_said_while_his_last_sentence_plays_starts_his_next_turn_and_the_turn_after_is_told_his_own_line(
    runner, cut
):
    """TURNS-5, TURNS-25."""
    await a_last_sentence_that_plays(runner, completion("[silent]"), completion("[silent]"))
    said_at(runner, PLACE_ID, AGAIN)
    await runner.gateway.until_turns(2)

    assert len(runner.app.playing) == 1
    runner.app.end_sentence(cut=cut)
    await settled(runner.app)
    assert len(runner.gateway.turns()) == 2
    said_at(runner, PLACE_ID, f"{AGAIN} Once more.")
    await runner.gateway.until_turns(3)

    second, third = runner.gateway.turns()[1:]
    assert [each["text"] for each in second.told("speech")] == [ANSWER]
    assert [each for each in second.told("line") if each["own"]] == []
    own_line = [
        (each["entry"], each["text"], each.get("cut"))
        for each in third.told("speech", "line")
        if each["entry"] == "speech" or each["own"]
    ]
    assert own_line == [("speech", ANSWER, None), ("line", ANSWER, True if cut else None)]


@pytest.mark.parametrize(
    ("place_id", "voice"),
    [
        pytest.param(PLACE_ID, VoiceState.UNAVAILABLE, id="written at the same place"),
        pytest.param(OTHER_PLACE_ID, VoiceState.UNAVAILABLE, id="written at another place"),
        pytest.param(OTHER_PLACE_ID, None, id="spoken at another place"),
    ],
)
async def test_the_next_turns_call_goes_out_while_the_sentence_of_the_turn_before_plays(runner, place_id, voice):
    """TURNS-25."""
    await a_last_sentence_that_plays(runner, *says(place_id, WRITTEN))
    await at_a_second_place(runner)
    runner.app.state_voice(place_id, voice)
    said_at(runner, place_id, AGAIN, spoken=False)
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, place_id, WRITTEN)]
    assert len(runner.app.playing) == 1


async def test_the_next_turns_first_sentence_goes_out_while_the_one_before_plays_and_at_most_one_plays(runner):
    """TURNS-25."""
    await a_last_sentence_that_plays(runner, completion(NEXT))
    said_at(runner, PLACE_ID, AGAIN)
    await runner.app.until_called(2)
    await settled(runner.app)

    assert (sentences_spoken(runner), len(runner.app.playing)) == ([ANSWER], 1)
    runner.app.end_sentence()
    await runner.app.spoken_count(2)

    assert (sentences_spoken(runner), len(runner.app.playing)) == ([ANSWER, NEXT], 1)


@pytest.mark.parametrize(
    ("reply", "later", "dropped"),
    [
        pytest.param("New.", ["New."], [1], id="a spoken turn replaces the rest"),
        pytest.param("[silent]", ["Three."], [], id="a silent turn keeps it"),
    ],
)
async def test_a_later_spoken_turns_first_sentence_drops_the_unsent_rest_and_queues_behind_the_sent_ones(
    runner, reply, later, dropped
):
    """TURNS-25, TURNS-45."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion("One. Two. Three."), completion(reply))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    await runner.app.until_called(2)
    await runner.reports.until_a_turn_completed()
    said_at(runner, PLACE_ID, AGAIN)
    await runner.reports.until_a_turn_completed(count=2)
    await settled(runner.app)

    runner.app.end_sentence()
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.app.spoken_count(3)

    assert sentences_spoken(runner) == ["One.", "Two.", *later]
    assert [each.sentences for each in runner.reports.changes_of(Event.SPEECH_DROPPED)] == dropped
    assert runner.reports.codes_of(Event.SPEECH_DROPPED) == ["replaced"] * len(dropped)


@pytest.mark.parametrize(
    ("tool", "cut"),
    [pytest.param(CONTINUE_SPEAKING_TOOL, False, id="continue"), pytest.param(STOP_SPEAKING_TOOL, True, id="stop")],
)
async def test_a_hold_over_his_last_sentence_after_the_turn_ended_is_decided_by_the_turn_it_makes_due_at_once(
    runner, tool, cut
):
    """TURNS-30, TURNS-31."""
    await a_last_sentence_that_plays(runner, deciding(tool), completion("[silent]"))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.reports.until(Event.HOLD_DECIDED)
    await runner.reports.until_a_turn_completed(count=2)

    assert runner.app.hold_calls == [(tool, HOLD_ID)]
    assert runner.gateway.hold_told(2)["sentence"] == ANSWER
    assert [(each.text, each.cut) for each in runner.app.lines[PLACE_ID] if each.author == SUBJECT] == [(ANSWER, cut)]
