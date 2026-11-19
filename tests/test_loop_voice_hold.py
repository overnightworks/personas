"""The whole runner, a spoken turn held while he speaks: his round decides to go on or stop."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from http import HTTPStatus

import anyio
import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, VoiceState
from anyio import BrokenResourceError

from personas import app_session
from personas.decision import status_code
from personas.failures import Unreachable, leaves_of
from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, say_in_the_call
from tests.support.fakes import COMPLETED_TURN
from tests.support.gateway import BUDGET_SPENT, say
from tests.support.hopin import CALL, LIMIT_SECONDS, SAY_TOOL
from tests.support.transport import opener_whose_requests_raise_together, transport_closing_with_late_answers
from tests.support.voice import TWO_SENTENCES, a_voice_call, ended_by, holding_sentence_ends, sentences_spoken
from tests.support.wire import cuts, refuses_with_a_server_error

pytestmark = pytest.mark.anyio

HOLD_ID = "hold-1"
SAID = ("Ada", "wait a moment")
FIRST, SECOND = TWO_SENTENCES.split(". ")[0] + ".", TWO_SENTENCES.split(". ")[1]


def deciding(tool: str, hold_id: str = HOLD_ID):
    return tool_round(tool_call(f"{tool}-{hold_id}", tool, {"id": hold_id}))


async def held_in_the_first_sentence(runner, *replies, said=(SAID,), hold_id: str = HOLD_ID):
    """A spoken turn whose first sentence plays and whose second waits for it, and the app then holds him."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, *replies)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.hold(PLACE_ID, hold_id, *said)


async def decided(runner, count: int = 1) -> list[str | None]:
    await runner.reports.until(Event.HOLD_DECIDED, count=count)
    return runner.reports.codes_of(Event.HOLD_DECIDED)


async def window_read_after_the_hold(runner, reads: int = 1) -> None:
    await runner.app.until_window_read(PLACE_ID, runner.app.window_reads(PLACE_ID) + reads)


async def test_a_hold_is_told_once_with_what_was_said_over_him_and_what_he_had_not_said(runner):
    """TURNS-29, TURNS-30."""
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion())

    await decided(runner)

    (number,) = runner.gateway.rounds()
    assert runner.gateway.hold_told(number) == {
        "id": HOLD_ID,
        "talked_over": [{"name": SAID[0], "text": SAID[1]}],
        "in_sentence": FIRST,
        "unsaid": [SECOND],
    }


async def test_the_round_is_asked_over_his_history_and_the_turns_tools_with_one_call_per_reply(runner):
    """TURNS-30."""
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion())

    await decided(runner)

    (number,) = runner.gateway.rounds()
    assert '"woke"' in runner.gateway.told_in(number)
    assert runner.gateway.offered_tools(number) == runner.gateway.offered_tools(1)
    assert runner.gateway.body_of(number)["parallel_tool_calls"] is False


async def test_a_continue_lets_the_rest_play_and_his_unsent_sentences_follow(runner):
    """TURNS-31."""
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion())

    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == [(CONTINUE_SPEAKING_TOOL, HOLD_ID)]
    assert sentences_spoken(runner) == [FIRST, SECOND]


async def test_a_stop_drops_the_unsent_sentences_and_tells_him_what_was_left_unsaid(runner):
    """TURNS-31."""
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(STOP_SPEAKING_TOOL), completion())

    assert await decided(runner) == ["stopped"]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]
    assert sentences_spoken(runner) == [FIRST]
    (number,) = runner.gateway.rounds()
    (result,) = [json.loads(each) for each in runner.gateway.tool_results(number + 1)]
    assert result == {"stopped": True, "unsaid": [SECOND]}
    assert Event.SPEAK_REFUSED not in [event for event, _ in runner.reports.reported()]


@pytest.mark.parametrize(
    "reply",
    [completion("[silent]"), completion("Let me think."), BUDGET_SPENT],
    ids=["silent", "text only", "failed round"],
)
async def test_a_round_that_makes_no_decision_stops_him_with_the_id_and_says_nothing(runner, reply):
    """TURNS-31."""
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), reply)

    assert await decided(runner) == ["undecided"]
    await runner.reports.until_a_turn_completed()

    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]
    assert sentences_spoken(runner) == [FIRST]


async def test_a_decision_for_a_hold_already_decided_starts_no_round_and_a_new_hold_starts_one(runner):
    await held_in_the_first_sentence(
        runner,
        completion("First sentence. Second sentence. Third sentence."),
        deciding(CONTINUE_SPEAKING_TOOL),
        completion(),
        deciding(CONTINUE_SPEAKING_TOOL, "hold-2"),
        completion(),
    )
    await decided(runner)
    arrivals = runner.gateway.arrivals

    runner.app.hold(PLACE_ID, HOLD_ID, ("Ada", "still talking"))
    await window_read_after_the_hold(runner)
    assert runner.gateway.arrivals == arrivals

    runner.app.hold(PLACE_ID, "hold-2", SAID)
    assert await decided(runner, 2) == ["continued", "continued"]
    assert len(runner.gateway.rounds()) == 2


@pytest.mark.parametrize("arrival", ["words", "record gone"])
async def test_a_hold_without_words_starts_no_round_until_it_has_words_or_its_record_is_gone(runner, arrival):
    await held_in_the_first_sentence(
        runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion(), said=(("Ada", ""),)
    )
    await window_read_after_the_hold(runner)
    assert runner.gateway.arrivals == 1

    if arrival == "words":
        runner.app.talk_over(PLACE_ID, SAID)
    else:
        runner.app.end_record(PLACE_ID)
    await decided(runner)

    (number,) = runner.gateway.rounds()
    assert runner.gateway.hold_told(number)["talked_over"] == [
        {"name": SAID[0], "text": SAID[1] if arrival == "words" else ""}
    ]


async def test_a_call_waiting_on_his_speech_is_sent_after_a_continue(runner):
    replies = [tool_round(say("written", PLACE_ID), content=TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL)]
    await held_in_the_first_sentence(runner, *replies, completion(), completion())

    await decided(runner)
    await runner.app.spoken_count(2)
    assert runner.app.said == []
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert [text for *_, text in runner.app.said] == ["written"]


async def test_words_said_over_him_after_a_continue_are_told_once_at_his_next_call_and_the_call_after_it_goes_out(
    runner,
):
    """TURNS-32."""
    more = ("Ada", "wait a moment, please")
    replies = [tool_round(say("first try", PLACE_ID), content=TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL)]
    await held_in_the_first_sentence(
        runner, *replies, completion(), tool_round(say("second try", PLACE_ID)), completion()
    )
    await decided(runner)
    await runner.gateway.until_answered(3)
    runner.app.talk_over(PLACE_ID, more)
    await window_read_after_the_hold(runner)

    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert [text for *_, text in runner.app.said] == ["second try"]
    assert [json.loads(each) for each in runner.gateway.tool_results(4)] == [
        {"sent": False, "talked_over": [{"name": more[0], "text": more[1]}]}
    ]


async def test_a_call_waiting_on_his_speech_is_never_sent_after_a_stop_and_he_is_told_so(runner):
    """TURNS-31."""
    replies = [tool_round(say("written", PLACE_ID), content=TWO_SENTENCES), deciding(STOP_SPEAKING_TOOL)]
    await held_in_the_first_sentence(runner, *replies, completion(), completion())

    await decided(runner)
    await runner.reports.until_a_turn_ended()

    assert runner.reports.turn_endings == [COMPLETED_TURN]
    assert runner.app.said == []
    assert SAY_TOOL not in runner.app.calls
    (number,) = runner.gateway.rounds()
    assert runner.gateway.hold_told(number)["call_waiting"] == {
        "tool": SAY_TOOL,
        "arguments": {"place": PLACE_ID, "text": "written"},
    }
    (result,) = [json.loads(each) for each in runner.gateway.tool_results(number + 1)]
    assert result == {"stopped": True, "unsaid": [SECOND], "call_dropped": SAY_TOOL}


async def test_a_stop_while_his_reply_is_still_streaming_ends_the_turn_and_speaks_no_more(runner):
    holding_sentence_ends(runner)
    await a_voice_call(runner, deciding(STOP_SPEAKING_TOOL), completion())
    reply = runner.gateway.pause_request(1, "First sentence. Sec", "ond sentence.")
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(1)
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    await decided(runner)
    await runner.reports.until_a_turn_ended()

    assert runner.reports.turn_endings == [COMPLETED_TURN]
    assert not reply.finished
    assert sentences_spoken(runner) == [FIRST]


@pytest.mark.parametrize("declared", [True, False], ids=["the app declares them", "an app from before the hold"])
async def test_his_own_call_to_either_decision_outside_a_round_is_answered_locally_and_never_sent(runner, declared):
    runner.app.declares_hold_tools = declared
    replies = [deciding(CONTINUE_SPEAKING_TOOL), deciding(STOP_SPEAKING_TOOL, "hold-2"), completion("done")]
    await a_voice_call(runner, *replies)
    say_in_the_call(runner, ADDRESSED, spoken=True)

    await runner.gateway.until_answered(3)

    assert runner.app.hold_calls == []
    assert {CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL} <= set(runner.gateway.offered_tools(1))
    assert len(runner.gateway.tool_results(3)) == 2


@pytest.mark.parametrize(("aloud", "asked"), [(True, False), (False, None)], ids=["spoken", "written"])
async def test_every_spoken_request_asks_for_one_call_per_reply_and_a_written_one_does_not(runner, aloud, asked):
    runner.app.state_voice(PLACE_ID, VoiceState.AVAILABLE if aloud else VoiceState.UNAVAILABLE)
    await a_voice_call(runner, completion("done"))
    say_in_the_call(runner, ADDRESSED)

    await runner.gateway.until_turns(1)

    assert runner.gateway.body_of(1).get("parallel_tool_calls") is asked


@pytest.mark.parametrize("hold_model", ["text/hold-low", None], ids=["set", "unset"])
async def test_the_round_runs_on_the_hold_model_when_one_is_set_and_on_his_own_model_otherwise(runner, hold_model):
    """TURNS-30, CONFIG-17."""
    runner.config = replace(
        runner.config, agents=tuple(replace(each, hold_model=hold_model) for each in runner.config.agents)
    )
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion())

    await decided(runner)

    (number,) = runner.gateway.rounds()
    assert runner.gateway.body_of(number)["model"] == (hold_model or runner.gateway.body_of(1)["model"])


async def test_a_decision_the_app_could_not_take_is_told_to_him_as_not_made_and_he_decides_again(runner):
    await held_in_the_first_sentence(
        runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), deciding(STOP_SPEAKING_TOOL), completion()
    )
    runner.app.script(CALL, refuses_with_a_server_error)

    assert await decided(runner) == ["stopped"]

    (number,) = runner.gateway.rounds()
    (failed,) = runner.gateway.tool_results(number + 1)
    assert failed.startswith("Error")
    assert runner.reports.reported(apps=True, started=False)[0].code == status_code(HTTPStatus.INTERNAL_SERVER_ERROR)
    assert runner.app.hold_calls == [(STOP_SPEAKING_TOOL, HOLD_ID)]


async def test_a_call_of_his_that_is_no_decision_and_a_second_decision_are_refused_in_the_round(runner):
    replies = [
        tool_round(say("early", PLACE_ID)),
        deciding(CONTINUE_SPEAKING_TOOL),
        deciding(STOP_SPEAKING_TOOL),
        completion(),
    ]
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), *replies)

    assert await decided(runner) == ["continued"]
    await runner.app.spoken_count(2)
    runner.app.end_sentence()
    await runner.reports.until_a_turn_completed()

    (number,) = runner.gateway.rounds()
    results = runner.gateway.tool_results(number + 3)
    assert [result.startswith("Error") for result in results] == [True, False, True]
    assert runner.app.hold_calls == [(CONTINUE_SPEAKING_TOOL, HOLD_ID)]
    assert runner.app.said == []


async def test_a_decision_whose_call_fails_beside_a_defect_ends_the_runner_loud_with_both(runner, monkeypatch):
    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_closing_with_late_answers(echo=RuntimeError, only_close=1)
    )
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL))
    runner.app.script(CALL, cuts)

    assert runner.task is not None
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(runner.task, return_exceptions=True)

    assert sorted(type(leaf).__name__ for leaf in leaves_of(ended)) == [RuntimeError.__name__, Unreachable.__name__]
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == []


async def test_a_decision_whose_call_fails_beside_a_speak_that_fails_ends_the_runner_loud_with_both(runner):
    released = asyncio.Event()
    speak_turns_behind_the_decision = 5
    defects = {
        "speak": (BrokenResourceError(), speak_turns_behind_the_decision),
        "continue_speaking": (KeyError("decision"), 0),
    }
    runner.open_session = opener_whose_requests_raise_together(defects, released=released)
    await a_voice_call(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.gateway.until_requested(1)
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await anyio.wait_all_tasks_blocked()
    released.set()

    ended = await ended_by(runner.task)

    assert sorted(type(leaf).__name__ for leaf in leaves_of(ended)) == [BrokenResourceError.__name__, KeyError.__name__]
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == []
