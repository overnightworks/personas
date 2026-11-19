"""The whole runner, what a spoken turn still owes when it ends: his last line enters after the turn, and a call of his
waits for it."""

from __future__ import annotations

import json

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, VoiceState
from agents.tools import ToolOutcome
from agents.turn import FailureCode

from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.fakes import COMPLETED_TURN, failed_turn
from tests.support.gateway import says, speaks_and_says
from tests.support.history import settled
from tests.support.hopin import SAY_TOOL
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import a_voice_call, holding_sentence_ends, sentences_spoken

pytestmark = pytest.mark.anyio

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
async def test_the_next_turns_call_waits_for_the_line_of_the_turn_before_in_a_written_turn_and_at_another_place(
    runner, place_id, voice
):
    """TURNS-25."""
    await a_last_sentence_that_plays(runner, *says(place_id, WRITTEN))
    await at_a_second_place(runner)
    runner.app.state_voice(place_id, voice)
    said_at(runner, place_id, AGAIN, spoken=False)
    await runner.gateway.until_turns(2)
    await settled(runner.app)

    assert runner.app.said == []
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, place_id, WRITTEN)]
    assert Event.SPEAK_LINE_MISSING not in [event for event, _ in runner.reports.reported()]


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


@pytest.mark.parametrize("place_id", [PLACE_ID, OTHER_PLACE_ID], ids=["same place", "another place"])
async def test_his_line_of_the_turn_before_entering_while_the_next_sentence_is_open_leaves_the_next_sentences_line_owed(
    runner, place_id
):
    """TURNS-25."""
    await a_last_sentence_that_plays(runner, *speaks_and_says(place_id, NEXT, WRITTEN))
    await at_a_second_place(runner)
    said_at(runner, place_id, AGAIN)
    await runner.app.until_called(2)

    runner.app.end_sentence()
    await runner.app.spoken_count(2)
    await settled(runner.app)
    assert runner.app.said == []
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, place_id, WRITTEN)]


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
    assert runner.gateway.hold_told(2)["in_sentence"] == ANSWER
    assert [(each.text, each.cut) for each in runner.app.lines[PLACE_ID] if each.author == SUBJECT] == [(ANSWER, cut)]
    assert Event.SPEAK_LINE_MISSING not in [event for event, _ in runner.reports.reported()]


async def test_a_call_elsewhere_that_would_wait_on_a_held_sentence_is_answered_not_sent_at_once_and_is_never_missing(
    runner,
):
    """TURNS-32."""
    await a_last_sentence_that_plays(runner, *says(OTHER_PLACE_ID, WRITTEN))
    await at_a_second_place(runner)
    reads = runner.app.window_reads(PLACE_ID)
    runner.app.hold(PLACE_ID, HOLD_ID, (SAID[0], ""))
    await runner.app.until_window_read(PLACE_ID, reads + 1)
    runner.clock.time += 2 * PERSONA_MAX_DURATION_SECONDS
    said_at(runner, OTHER_PLACE_ID, AGAIN, spoken=False)
    await runner.gateway.until_requested(3)

    assert json.loads(runner.gateway.tool_results(3)[0]) == {"sent": False, "talked_over": []}
    assert runner.app.said == []
    assert Event.SPEAK_LINE_MISSING not in [event for event, _ in runner.reports.reported()]


async def test_a_line_that_never_comes_leaves_the_first_turn_ok_and_the_next_call_goes_out_at_the_sending_turns_bound(
    runner,
):
    """TURNS-28."""
    sleeps = await a_last_sentence_that_plays(runner, *says_then_stays_silent(PLACE_ID, WRITTEN))
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS / 2
    said_at(runner, PLACE_ID, AGAIN)
    await runner.gateway.until_turns(2)
    await settled(runner.app)
    assert runner.app.said == []
    assert sleeps.began[-1] == PERSONA_MAX_DURATION_SECONDS / 2

    runner.clock.time += PERSONA_MAX_DURATION_SECONDS / 2
    runner.app.speak(PLACE_ID, SUBJECT, "Mozart", "Taken at that time.")
    await runner.app.said_count(1)
    await runner.reports.until_a_turn_completed(count=2)

    (missing,) = runner.reports.changes_of(Event.SPEAK_LINE_MISSING)
    assert missing.lines == 1
    assert runner.reports.turn_endings == [COMPLETED_TURN, COMPLETED_TURN]
    assert Event.PAUSED not in [event for event, _ in runner.reports.reported()]
    await runner.app.until_turn_states(4)
    assert [each.stopped for _, each in runner.app.turn_states if not each.thinking] == [None, None]
    assert runner.app.said == [(SUBJECT, PLACE_ID, WRITTEN)]


async def test_a_line_that_never_comes_lets_a_call_out_at_the_sending_turns_bound_after_the_next_turn_spoke_a_sentence(
    runner,
):
    """TURNS-28."""
    sleeps = await a_last_sentence_that_plays(runner, *speaks_and_says(PLACE_ID, NEXT, WRITTEN, after="[silent]"))
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS / 2
    said_at(runner, PLACE_ID, AGAIN)
    await runner.gateway.until_turns(2)
    await runner.app.until_called(2)
    runner.app.playing.pop(0)
    runner.app.waiting_speaks.pop(0).set_result(True)
    await runner.app.spoken_count(2)
    await settled(runner.app)
    assert runner.app.said == []
    assert sleeps.began[-1] == PERSONA_MAX_DURATION_SECONDS / 2

    runner.clock.time += PERSONA_MAX_DURATION_SECONDS / 2
    runner.app.speak(PLACE_ID, SUBJECT, "Mozart", "Taken at that time.")
    await settled(runner.app)
    runner.app.end_sentence()
    await runner.app.said_count(1)
    await runner.reports.until_a_turn_completed(count=2)

    (missing,) = runner.reports.changes_of(Event.SPEAK_LINE_MISSING)
    assert missing.lines == 1
    assert runner.reports.turn_endings == [COMPLETED_TURN, COMPLETED_TURN]
    assert Event.PAUSED not in [event for event, _ in runner.reports.reported()]
    assert runner.app.said == [(SUBJECT, PLACE_ID, WRITTEN)]


async def test_a_hold_decided_after_the_sending_turns_bound_leaves_the_line_owed_and_a_later_call_waits_for_it(runner):
    """TURNS-25, TURNS-32."""
    await a_last_sentence_that_plays(runner, deciding(CONTINUE_SPEAKING_TOOL), *says(PLACE_ID, WRITTEN))
    runner.app.hold_outcomes.append(ToolOutcome(json.dumps({"continued": True}), is_error=False))
    runner.clock.time += 2 * PERSONA_MAX_DURATION_SECONDS
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.reports.until(Event.HOLD_DECIDED)
    await runner.gateway.until_requested(3)
    await settled(runner.app)

    assert runner.app.said == []
    assert Event.SPEAK_LINE_MISSING not in [event for event, _ in runner.reports.reported()]
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, PLACE_ID, WRITTEN)]
    assert Event.SPEAK_LINE_MISSING not in [event for event, _ in runner.reports.reported()]


async def test_a_turn_that_ended_limit_exceeded_with_a_line_still_owed_leaves_it_to_the_next_turn_that_logs_it_missing(
    runner,
):
    """TURNS-28."""
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion("First sentence. Second sentence."), *says(PLACE_ID, WRITTEN))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(1)
    await runner.app.until_called(2)
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS
    sleeps.let_through()
    await runner.reports.until(Event.PAUSED)
    await sleeps.until_began(2)
    sleeps.let_through()
    say_in_the_call(runner, AGAIN)
    await runner.app.said_count(1)

    (missing,) = runner.reports.changes_of(Event.SPEAK_LINE_MISSING)
    assert missing.lines == 1
    assert runner.reports.turn_endings[0] == failed_turn(FailureCode.LIMIT_EXCEEDED)
    assert runner.app.said == [(SUBJECT, PLACE_ID, WRITTEN)]


async def test_a_written_turn_whose_own_bound_passes_while_it_waits_for_a_line_ends_limit_exceeded_and_sends_nothing(
    runner,
):
    """TURNS-28."""
    sleeps = await a_last_sentence_that_plays(runner, *says(OTHER_PLACE_ID, WRITTEN))
    await at_a_second_place(runner)
    runner.app.state_voice(OTHER_PLACE_ID, VoiceState.UNAVAILABLE)
    said_at(runner, OTHER_PLACE_ID, AGAIN, spoken=False)
    await runner.gateway.until_turns(2)
    await sleeps.until_began(2)
    await settled(runner.app)

    sleeps.let_through()
    await runner.reports.until(Event.PAUSED)

    assert runner.reports.reported()[-1] == (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value)
    assert runner.app.said == []
    assert Event.SPEAK_LINE_MISSING not in [event for event, _ in runner.reports.reported()]
