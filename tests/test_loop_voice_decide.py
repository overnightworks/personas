"""The whole runner, a hold that no spoken turn of his runs at: his next turn decides it."""

from __future__ import annotations

from dataclasses import replace

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import (
    CONTINUE_SPEAKING_TOOL,
    SPEAK_TOOL,
    STOP_SPEAKING_TOOL,
    ContinueSpeakingRefusal,
    ContinueSpeakingRefused,
    StopSpeakingRefusal,
    StopSpeakingRefused,
    VoiceState,
)
from agents.tools import ToolOutcome
from agents.turn import FailureCode

from personas.recall import RECALL_TOOL
from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.gateway import BUDGET_SPENT, say
from tests.support.hopin import CALL, SAY_TOOL
from tests.support.store import ScriptedStore
from tests.support.voice import TWO_SENTENCES, a_voice_call, gated, holding_sentence_ends, sentences_spoken
from tests.support.wire import refuses_with_a_server_error

pytestmark = pytest.mark.anyio


HOLD_ID = "hold-1"
SAID = ("Ada", "wait a moment")
FIRST, SECOND = TWO_SENTENCES.split(". ")[0] + ".", TWO_SENTENCES.split(". ")[1]
DECISIONS = [
    pytest.param(CONTINUE_SPEAKING_TOOL, "continued", id="continue"),
    pytest.param(STOP_SPEAKING_TOOL, "stopped", id="stop"),
]


def deciding(tool: str, hold_id: str = HOLD_ID):
    return tool_round(tool_call(f"{tool}-{hold_id}", tool, {"id": hold_id}))


async def decided(runner, count: int = 1) -> list[str | None]:
    await runner.reports.until(Event.HOLD_DECIDED, count=count)
    return runner.reports.codes_of(Event.HOLD_DECIDED)


async def window_read_after(runner, reads: int = 1) -> None:
    await runner.app.until_window_read(PLACE_ID, runner.app.window_reads(PLACE_ID) + reads)


def calls_of_voice(runner) -> list[str]:
    return [each for each in runner.app.calls if each in {SPEAK_TOOL, CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL}]


@pytest.mark.parametrize(("tool", "outcome"), DECISIONS)
async def test_a_hold_with_words_where_nothing_is_owed_makes_his_turn_due_and_that_turn_decides_it(
    runner, tool, outcome
):
    """TURNS-30, TURNS-31, LOG-17."""
    await a_voice_call(runner, deciding(tool), completion(TWO_SENTENCES))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    assert await decided(runner) == [outcome]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == [(tool, HOLD_ID)]
    (number,) = runner.gateway.rounds()
    assert number == 1
    assert runner.gateway.hold_told(number) == {
        "id": HOLD_ID,
        "talked_over": [{"name": SAID[0], "text": SAID[1]}],
        "in_sentence": None,
        "unsaid": [],
    }
    assert sentences_spoken(runner) == [FIRST, SECOND]
    assert calls_of_voice(runner) == [tool, SPEAK_TOOL, SPEAK_TOOL]
    assert Event.SPEAK_REFUSED not in [event for event, _ in runner.reports.reported()]


@pytest.mark.parametrize("arrival", ["words", "record gone"])
async def test_a_hold_without_words_makes_no_turn_due_until_it_has_words_or_its_record_is_gone(runner, arrival):
    await a_voice_call(runner, deciding(CONTINUE_SPEAKING_TOOL), completion("[silent]"))

    runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", ""))
    await window_read_after(runner)
    assert runner.gateway.arrivals == 0
    if arrival == "words":
        runner.app.talk_over(PLACE_ID, SAID)
    else:
        runner.app.end_record(PLACE_ID)

    assert await decided(runner) == ["continued"]
    assert runner.gateway.hold_told(1)["talked_over"] == [
        {"name": "Ada", "text": SAID[1] if arrival == "words" else ""}
    ]


async def test_a_hold_left_standing_when_his_speech_ends_is_decided_by_his_next_turn_who_is_told_the_sentence(runner):
    """TURNS-31."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(FIRST), deciding(STOP_SPEAKING_TOOL), completion("[silent]"))
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(1)
    runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", ""))
    await window_read_after(runner)
    answer.set()
    await runner.reports.until_a_turn_completed()
    assert runner.app.hold_calls == []

    runner.app.end_record(PLACE_ID)

    assert await decided(runner) == ["stopped"]
    (number,) = runner.gateway.rounds()
    assert runner.gateway.hold_told(number)["in_sentence"] == FIRST
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]


async def test_his_sentences_wait_unsent_and_every_other_call_is_answered_deciding_first_until_he_decides(runner):
    """TURNS-31."""
    replies = [
        tool_round(say("early", PLACE_ID), content=TWO_SENTENCES),
        deciding(STOP_SPEAKING_TOOL),
        deciding(CONTINUE_SPEAKING_TOOL),
        completion("[silent]"),
    ]
    await a_voice_call(runner, *replies)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.reports.until_a_turn_completed()

    results = runner.gateway.tool_results(4)
    assert [each.startswith("Error") for each in results] == [True, False, True]
    assert SAY_TOOL not in runner.app.calls
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]
    assert calls_of_voice(runner) == [STOP_SPEAKING_TOOL, SPEAK_TOOL, SPEAK_TOOL]
    assert sentences_spoken(runner) == [FIRST, SECOND]


@pytest.mark.parametrize(
    "reply",
    [completion("[silent]"), completion("Let me think."), BUDGET_SPENT],
    ids=["silent", "text only", "failed turn"],
)
async def test_a_turn_that_ends_without_a_decision_drops_what_waited_and_the_next_read_stops_him_once(runner, reply):
    """TURNS-31, LOG-17."""
    await a_voice_call(runner, reply)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    assert await decided(runner) == ["undecided"]
    arrivals = runner.gateway.arrivals
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await window_read_after(runner, 2)

    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["undecided"]
    assert sentences_spoken(runner) == []
    assert runner.gateway.arrivals == arrivals


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        pytest.param(deciding(STOP_SPEAKING_TOOL, "hold-2"), "stopped", id="stopped"),
        pytest.param(completion("[silent]"), "undecided", id="undecided"),
    ],
)
async def test_a_round_for_a_newer_hold_that_stops_him_or_ends_undecided_logs_the_older_hold_the_turn_was_told(
    runner, reply, outcome
):
    """TURNS-31, LOG-17."""
    await a_voice_call(runner, reply)
    runner.gateway.pause_request(1, "Never said. No", "r this.")

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.gateway.until_requested(1)
    runner.app.hold(PLACE_ID, "hold-2", ("Bob", "no, stop"))

    assert await decided(runner, 2) == [outcome, "undecided"]
    assert sentences_spoken(runner) == []


async def test_a_round_for_a_newer_hold_logs_no_line_for_the_older_hold_the_turn_already_decided(runner):
    """TURNS-31, LOG-17."""
    await a_voice_call(runner, deciding(CONTINUE_SPEAKING_TOOL), deciding(STOP_SPEAKING_TOOL, "hold-2"))
    resumed = runner.gateway.pause_request(2, "Still ", "thinking.")

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.gateway.until_requested(2)
    runner.app.hold(PLACE_ID, "hold-2", ("Bob", "no, stop"))
    assert await decided(runner, 2) == ["continued", "stopped"]
    resumed.release()
    await runner.reports.until_a_turn_completed()

    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["continued", "stopped"]


async def test_a_stop_the_app_refuses_because_no_such_hold_stands_ends_the_hold_as_gone_and_is_not_sent_again(runner):
    """LOG-17."""
    await a_voice_call(runner, completion("[silent]"))
    refusal = StopSpeakingRefused(refused=StopSpeakingRefusal.NOT_INTERRUPTED)
    runner.app.hold_outcomes.append(ToolOutcome(refusal.model_dump_json(), is_error=True))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    assert await decided(runner) == ["gone"]
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await window_read_after(runner)

    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["gone"]


@pytest.mark.parametrize(
    ("tool", "refusal"),
    [
        pytest.param(
            CONTINUE_SPEAKING_TOOL,
            ContinueSpeakingRefused(refused=ContinueSpeakingRefusal.NOT_INTERRUPTED),
            id="continue",
        ),
        pytest.param(STOP_SPEAKING_TOOL, StopSpeakingRefused(refused=StopSpeakingRefusal.NOT_INTERRUPTED), id="stop"),
    ],
)
async def test_a_decision_for_a_hold_that_no_longer_stands_is_logged_gone_and_he_is_told_so(runner, tool, refusal):
    """TURNS-31, LOG-17."""
    await a_voice_call(runner, deciding(tool), completion("[silent]"))
    runner.app.hold_outcomes.append(ToolOutcome(refusal.model_dump_json(), is_error=True))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    assert await decided(runner) == ["gone"]
    await runner.reports.until_a_turn_completed()
    assert runner.gateway.tool_results(2) == ['{"gone": true}']


async def test_a_stop_whose_call_failed_is_sent_again_by_a_later_read_and_decided_once(runner):
    """TURNS-31, LOG-17."""
    await a_voice_call(runner, completion("[silent]"))
    runner.app.script(CALL, refuses_with_a_server_error)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.reports.until(Event.BACKOFF)
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == []
    runner.sleeps.let_through()
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    assert await decided(runner) == ["undecided"]
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]


async def test_a_turn_told_a_hold_answers_the_lookup_that_it_decides_first_and_spends_no_lookup(runner):
    """TURNS-31."""
    store = ScriptedStore()
    runner.remember_in(store)
    await a_voice_call(
        runner,
        tool_round(tool_call("look", RECALL_TOOL, {"text": "anything"})),
        tool_round(say("early", PLACE_ID)),
        deciding(CONTINUE_SPEAKING_TOOL),
        completion("[silent]"),
    )

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await decided(runner)

    lookup, other = runner.gateway.tool_results(3)
    assert lookup.startswith("Error")
    assert lookup == other
    assert store.searches == []


async def test_a_hold_joins_the_lines_already_due_into_one_written_turn_that_is_offered_both_decisions(runner):
    runner.app.state_voice(PLACE_ID, VoiceState.UNAVAILABLE)
    await a_voice_call(runner, completion("done"), deciding(STOP_SPEAKING_TOOL), completion("[silent]"))
    answer = runner.gateway.hold_request(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await window_read_after(runner)
    say_in_the_call(runner, "Are you still there?")
    await window_read_after(runner)
    answer.set()

    assert await decided(runner) == ["stopped"]
    (number,) = runner.gateway.rounds()
    assert number == 2
    assert "Are you still there?" in runner.gateway.told_in(number)
    assert {CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL} <= set(runner.gateway.offered_tools(number))
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]


async def test_a_hold_the_turn_was_told_starts_no_round_beside_it_when_more_is_said_over_him(runner):
    await a_voice_call(runner, deciding(CONTINUE_SPEAKING_TOOL), completion("[silent]"))
    answer = runner.gateway.hold_request(1)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.gateway.until_requested(1)
    runner.app.talk_over(PLACE_ID, SAID, ("Bo", "me too"))
    await window_read_after(runner)
    answer.set()

    assert await decided(runner) == ["continued"]
    await runner.reports.until_a_turn_completed()
    assert runner.gateway.rounds() == [1]
    assert runner.app.hold_calls == [(CONTINUE_SPEAKING_TOOL, HOLD_ID)]


@pytest.mark.parametrize("hold_model", ["text/hold-low", None], ids=["set", "unset"])
async def test_the_turn_that_decides_runs_on_the_hold_model_when_one_is_set_and_on_his_own_model_otherwise(
    runner, hold_model
):
    """TURNS-30, CONFIG-17."""
    runner.config = replace(
        runner.config, agents=tuple(replace(each, hold_model=hold_model) for each in runner.config.agents)
    )
    await a_voice_call(runner, completion("Here."), deciding(CONTINUE_SPEAKING_TOOL), completion("[silent]"))
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await decided(runner)

    (number,) = runner.gateway.rounds()
    assert runner.gateway.body_of(number)["model"] == (hold_model or runner.gateway.body_of(1)["model"])


async def test_a_decision_the_app_could_not_take_is_told_to_him_as_not_made_and_he_decides_again(runner):
    await a_voice_call(runner, deciding(CONTINUE_SPEAKING_TOOL), deciding(STOP_SPEAKING_TOOL), completion("[silent]"))
    runner.app.script(CALL, refuses_with_a_server_error)

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    assert await decided(runner) == ["stopped"]
    (failed,) = runner.gateway.tool_results(2)
    assert failed.startswith("Error")
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]


async def test_a_turn_still_waiting_for_its_decision_at_its_deadline_ends_limit_exceeded_with_nothing_said(runner):
    sleeps = gated(runner)
    await a_voice_call(runner, completion("Never said. Nor this."))
    runner.gateway.pause_request(1, "Never said. No", "r this.")

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await sleeps.until_began(1)
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS
    sleeps.release.set()
    await runner.reports.until(Event.PAUSED)

    assert (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value) in runner.reports.reported()
    assert sentences_spoken(runner) == []
