"""The whole runner, the voice loop: which places run one, what it hears and how it speaks."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion
from agents.member import SpeakRefusal, SpeakRefused, Spoken
from agents.tools import ToolOutcome
from agents.turn import Principal, person_fingerprint

from personas.decision import SPOKEN_WAIT_BOUND_SECONDS
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import BUDGET_SPENT, Gateway, says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import ISSUER, SPOKEN_DURATION_MS, Line, QuietApp, active
from tests.support.realm import LIFETIME_SECONDS
from tests.support.voice import TWO_SENTENCES, HeldSleeps, a_voice_call, sentences_spoken

pytestmark = pytest.mark.anyio

QUIET_PLACE_ID = "room-8"
SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME = "agent-bach", "agent-subject-2", "Bach"


@pytest.fixture
async def quiet(runner):
    """A second app the agent listens to, in which it is in a call, that declares no ``speak``."""
    app = QuietApp()
    async with app.serving() as url:
        runner.add_listening_app(app, url, "mcp-quiet")
        app.stand(SUBJECT, active(QUIET_PLACE_ID))
        yield app


def scripted_speak(duration_ms: int) -> ToolOutcome:
    return ToolOutcome(Spoken(duration_ms=duration_ms).model_dump_json(), is_error=False)


async def test_a_place_in_an_app_that_declares_speak_runs_the_voice_loop_and_one_in_an_app_that_does_not_the_turn(
    runner, quiet
):
    await a_voice_call(runner, *says(QUIET_PLACE_ID, "Written."), completion("Aloud."))
    await quiet.until_window_read(QUIET_PLACE_ID)
    quiet.speak(QUIET_PLACE_ID, PERSON, "Ada", ADDRESSED)
    await quiet.said_count(1)
    await runner.gateway.until_requested(2)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)

    assert quiet.said == [(SUBJECT, QUIET_PLACE_ID, "Written.")]
    assert sentences_spoken(runner) == ["Aloud."]
    assert runner.app.said == []


async def test_an_agent_without_a_voice_section_keeps_the_per_line_turn_in_an_app_that_declares_speak(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert runner.app.spoken == []


async def test_a_written_line_naming_him_in_a_voice_place_enters_the_loop_and_the_answer_is_spoken(runner):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(1)

    assert sentences_spoken(runner) == [ANSWER]
    assert runner.app.said == []
    assert runner.gateway.heard_in(1) == [
        {"name": "Ada", "kind": "message", "own": False, "spoken": False, "text": ADDRESSED}
    ]


async def test_a_spoken_question_that_does_not_name_him_is_answered(runner):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, "Anyone there?", spoken=True)
    await runner.app.spoken_count(1)

    assert sentences_spoken(runner) == [ANSWER]


async def test_a_spoken_line_naming_him_is_answered_by_one_speak_per_sentence_in_order_each_after_the_last(runner):
    sleeps = HeldSleeps(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    assert runner.app.spoken == [(SUBJECT, PLACE_ID, "First sentence.")]
    assert sleeps.began == [SPOKEN_DURATION_MS / 1000]
    sleeps.release.set()
    await runner.app.spoken_count(2)

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]


@pytest.mark.parametrize("refusal", list(SpeakRefusal))
async def test_a_refused_speak_ends_the_answer_and_its_code_is_logged(runner, refusal):
    await a_voice_call(runner, completion(TWO_SENTENCES), completion("Later."))
    runner.app.speak_outcomes.append(ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until("speak_refused")
    say_in_the_call(runner, f"{ADDRESSED} Again.", spoken=True)
    await runner.app.spoken_count(1)

    assert sentences_spoken(runner) == ["Later."]
    assert runner.reports.events() == [("started", None), ("speak_refused", refusal.value)]


async def test_lines_the_window_no_longer_holds_are_told_to_the_next_decision_as_missed(runner, monkeypatch):
    monkeypatch.setattr("tests.support.hopin.WINDOW_LINES", 1)
    sleeps = HeldSleeps(runner)
    await a_voice_call(runner, completion(ANSWER), completion("Second answer."))
    runner.app.speak_outcomes.append(scripted_speak(SPOKEN_DURATION_MS))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)
    runner.app.lines[PLACE_ID].extend(Line(seq, PERSON, "Ada", f"Line {seq}") for seq in (2, 3))
    say_in_the_call(runner, f"{ADDRESSED} Again.", spoken=True)
    sleeps.release.set()
    await runner.app.spoken_count(1)

    assert runner.gateway.heard_in(2)[-1]["missed_before"] == 2


async def test_a_budget_spent_fails_one_decision_for_each_waking_line_and_none_is_retried(runner):
    await a_voice_call(runner, BUDGET_SPENT)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until("turn_ended")

    assert len(runner.gateway.requests) == 1
    say_in_the_call(runner, f"{ADDRESSED} Again.", spoken=True)
    await runner.reports.until("turn_ended", count=2)

    assert len(runner.gateway.requests) == 2
    assert runner.reports.events() == [
        ("started", None),
        ("turn_ended", "budget_spent"),
        ("turn_ended", "budget_spent"),
    ]
    assert runner.app.spoken == []


async def test_two_agents_in_one_place_each_skip_only_their_own_spoken_lines_and_each_answers_on_its_own(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    await a_voice_call(runner, completion("Mozart answers."), completion("Bach answers."), completion("Mozart again."))
    mozart, bach = (
        person_fingerprint(runner.app_principal),
        person_fingerprint(Principal(iss=ISSUER, sub=SECOND_SUBJECT)),
    )
    say_in_the_call(runner, "Mozart, tell us about Vienna.", spoken=True)
    await runner.app.spoken_count(1)
    say_in_the_call(runner, "Bach, what do you think.", spoken=True)
    await runner.app.spoken_count(3)

    by_agent = {runner.gateway.user_of(number): number for number in (1, 2, 3)}
    assert runner.gateway.user_of(1) == mozart
    assert set(by_agent) == {mozart, bach}
    bach_heard = runner.gateway.heard_in(by_agent[bach])
    assert {"name": "Mozart", "kind": "message", "own": False, "spoken": True, "text": "Mozart answers."} in bach_heard
    assert not [heard for number in (1, 2, 3) for heard in runner.gateway.heard_in(number) if heard["own"]]


async def test_every_line_that_wakes_the_voice_gets_its_own_decision_and_none_is_refused(runner):
    await a_voice_call(runner, completion("One answer."), completion("Second answer."), completion("Third answer."))
    say_in_the_call(runner, f"{ADDRESSED} First.", spoken=True)
    await runner.app.spoken_count(1)
    say_in_the_call(runner, f"{ADDRESSED} Second.", spoken=True)
    await runner.app.spoken_count(2)
    say_in_the_call(runner, f"{ADDRESSED} Third.", spoken=True)
    await runner.app.spoken_count(3)

    assert sentences_spoken(runner) == ["One answer.", "Second answer.", "Third answer."]
    assert len(runner.gateway.requests) == 3
    assert runner.reports.events() == [("started", None)]


async def test_a_sentence_that_plays_longer_than_the_bound_is_waited_for_only_the_bound(runner):
    sleeps = HeldSleeps(runner)
    await a_voice_call(runner, completion(ANSWER))
    runner.app.speak_outcomes.append(scripted_speak(int(SPOKEN_WAIT_BOUND_SECONDS * 1000 * 10)))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    assert sleeps.began == [SPOKEN_WAIT_BOUND_SECONDS]


async def test_a_loop_that_outlives_its_token_speaks_its_next_sentence_with_a_renewed_one(runner):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    runner.clock.time += LIFETIME_SECONDS
    say_in_the_call(runner, f"{ADDRESSED} Once more.", spoken=True)
    await runner.app.spoken_count(2)

    first, second = runner.app.speak_bearers
    assert first != second
    assert second == runner.realm.issued[-1]


async def test_a_voice_loop_logs_its_start_and_no_answered_line(runner, caplog):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)

    logged = [record.getMessage() for record in caplog.get_records("call") if record.name == "personas"]
    assert [message.split()[0] for message in logged] == ["started"]
