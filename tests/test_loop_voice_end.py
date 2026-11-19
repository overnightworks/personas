"""The whole runner, the spoken turn's end: what stops it, what does not and where its failures lead."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import anyio
import pytest
from agents.backend.fake_gateway import completion
from agents.member import SPEAK_TOOL
from agents.tools import ToolOutcome
from agents.turn import FailureCode

from personas.decision import NETWORK_FAILURE_CODE, TURN_CANCELLED_CODE, status_code
from personas.failures import INVALID_ANSWER_CODE, leaves_of
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, pulses_the_window_once, say_in_the_call
from tests.support.configfiles import PERSONA_MAX_DURATION_SECONDS
from tests.support.fakes import CANCELLED_TURN, failed_turn
from tests.support.gateway import Gateway, speaks_and_says
from tests.support.hopin import CALL, LISTEN, READ, SAY_TOOL
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, paused
from tests.support.transport import opener_whose_request_raises
from tests.support.voice import (
    TWO_SENTENCES,
    a_turn_whose_speak_is_in_flight,
    a_voice_call,
    ended_by,
    gated,
    holding_sentence_ends,
    sentences_spoken,
)
from tests.support.wire import cuts

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "leaves",
    [lambda runner: runner.app.stand(SUBJECT), lambda runner: runner.app.stand(SUBJECT, paused(PLACE_ID))],
    ids=["removed", "paused"],
)
async def test_a_place_removed_or_paused_while_a_sentence_waits_cancels_the_turn_and_its_wait_and_no_more_is_spoken(
    runner, leaves
):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    leaves(runner)
    await sleeps.until_cancelled()
    await runner.reports.until_a_turn_ended()

    assert sentences_spoken(runner) == ["First sentence."]
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert not runner.task.done()


async def test_a_restart_in_a_call_answers_nothing_that_was_said_before_it(runner):
    """TURNS-3."""
    before = f"{ADDRESSED} Before the start."
    say_in_the_call(runner, before, spoken=True)
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, f"{ADDRESSED} After the start.", spoken=True)
    await runner.app.spoken_count(1)

    assert before not in runner.gateway.told_in(1)


async def test_a_resume_in_a_call_answers_nothing_that_was_said_during_the_pause(runner):
    during = f"{ADDRESSED} During the pause."
    await a_voice_call(runner, completion(ANSWER))
    pause_seen = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.app.until_read_held()
    say_in_the_call(runner, during, spoken=True)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    pause_seen.set()
    await runner.app.until_window_read(PLACE_ID, 2)
    say_in_the_call(runner, f"{ADDRESSED} After the pause.", spoken=True)
    await runner.app.spoken_count(1)

    assert during not in runner.gateway.told_in(1)


async def test_a_stream_the_app_ends_at_its_bound_while_he_speaks_lets_the_turn_speak_on(runner):
    pulse_window, end_stream = asyncio.Event(), asyncio.Event()
    sleeps = holding_sentence_ends(runner)
    runner.app.script(LISTEN, pulses_the_window_once(pulse_window, ended=end_stream))
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    pulse_window.set()
    await sleeps.until_began(1)
    end_stream.set()
    await runner.app.log.until(LISTEN, 2)
    runner.app.end_sentence()
    await runner.app.spoken_count(2)

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_lifecycle_connection_lost_while_a_sentence_is_sent_cancels_the_turn_and_enters_the_sentence_unanswered(
    runner,
):
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written.", after="Later."))
    sleeps = gated(runner)
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(1)
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = READ
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.reports.until(Event.BACKOFF)
    await sleeps.until_began(1)
    runner.app.refuse_with = None
    sleeps.let_through()
    await runner.reports.until(Event.RESUMED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)
    answer.set()

    assert runner.app.said == []
    assert runner.gateway.arrivals == 2
    (sent,) = runner.gateway.turns()[1].told("speech")
    assert (sent["text"], sent["outcome"]) == ("First sentence.", {"answered": False, "code": TURN_CANCELLED_CODE})


@pytest.mark.parametrize(
    ("break_speak", "mend_speak", "events"),
    [
        (
            lambda app: (
                setattr(app, "refuse_with", HTTPStatus.SERVICE_UNAVAILABLE)
                or setattr(app, "refused_method", CALL)
                or setattr(app, "refused_tool", SPEAK_TOOL)
            ),
            lambda app: setattr(app, "refuse_with", None),
            [(Event.STARTED, None), (Event.BACKOFF, status_code(503)), (Event.RESUMED, None)],
        ),
        (
            lambda app: app.speak_outcomes.append(ToolOutcome("not a spoken record", is_error=False)),
            lambda app: None,
            [(Event.STARTED, None), (Event.SUSPENDED, INVALID_ANSWER_CODE), (Event.RESUMED, None)],
        ),
    ],
    ids=["a call that fails", "an answer that is no Spoken"],
)
async def test_a_speak_that_fails_as_a_call_or_answers_no_spoken_ends_the_turn_into_the_reaction_and_a_new_turn_speaks(
    runner, break_speak, mend_speak, events
):
    """TURNS-27."""
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written.", after="Later."))
    sleeps = gated(runner)
    break_speak(runner.app)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(events[1][0])
    await sleeps.until_began(1)

    assert runner.app.spoken == []
    assert runner.gateway.arrivals == 1
    mend_speak(runner.app)
    sleeps.let_through()
    await runner.reports.until(Event.RESUMED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.spoken_count(1)

    assert sentences_spoken(runner) == ["Later."]
    assert runner.app.said == []
    assert runner.reports.reported() == events
    assert not runner.task.done()


async def test_a_call_to_the_own_app_that_fails_after_a_sentence_ends_the_turn_into_the_reaction_with_nothing_further(
    runner,
):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, ANSWER, "Written.", after="Never."))
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = CALL
    runner.app.refused_tool = SAY_TOOL
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(1)
    runner.app.end_sentence()
    await runner.reports.until(Event.BACKOFF)
    await sleeps.until_began(2)
    runner.app.refuse_with = None
    sleeps.let_through()
    await runner.reports.until(Event.RESUMED)

    assert sentences_spoken(runner) == [ANSWER]
    assert runner.gateway.arrivals == 1
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, status_code(503)),
        (Event.RESUMED, None),
    ]


async def test_a_gateway_that_stops_the_turn_while_a_sentence_is_sent_and_one_is_queued_answers_the_sent_one_only(
    runner,
):
    await a_voice_call(runner, completion("done"))
    sleeps = gated(runner)
    reply = runner.gateway.pause_request(1, "First sentence. Second sentence. Thi", "rd.")
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(1)
    reply.cut()
    answer.set()
    await runner.reports.until(Event.PAUSED)
    await sleeps.until_began(2)
    sleeps.let_through()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == ["First sentence."]
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.PAUSED, FailureCode.PROVIDER_ERROR.value),
    ]
    assert runner.reports.turn_endings == [failed_turn(FailureCode.PROVIDER_ERROR)]
    (sent,) = runner.gateway.turns()[1].told("speech")
    assert (sent["text"], sent["outcome"]) == ("First sentence.", {"answered": True})


async def test_a_gateway_that_stops_the_turn_while_a_sentence_is_open_waits_for_its_answer_and_enters_it_answered(
    runner,
):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion("done"))
    reply = runner.gateway.pause_request(1, "First sentence. Second sentence. Thi", "rd.")
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(2)
    reply.cut()
    runner.app.end_sentence()
    await sleeps.until_cancelled()
    await runner.reports.until(Event.PAUSED)
    await sleeps.until_began(2)
    sleeps.let_through()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.PAUSED, FailureCode.PROVIDER_ERROR.value),
    ]
    assert runner.reports.turn_endings == [failed_turn(FailureCode.PROVIDER_ERROR)]
    assert [(each["text"], each["outcome"]) for each in runner.gateway.turns()[1].told("speech")] == [
        ("First sentence.", {"answered": True}),
        ("Second sentence.", {"answered": True}),
    ]


AFTER_THE_FIRST_SENTENCE = [
    pytest.param(speaks_and_says(PLACE_ID, ANSWER, "Written."), [ANSWER], id="a call waits on the only sentence"),
    pytest.param(
        speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written."),
        ["First sentence.", "Second sentence."],
        id="a call waits on two sentences",
    ),
    pytest.param([completion("One. Two. Three.")], ["One.", "Two."], id="the speech after the final event"),
]
"""Each reply with the sentences sent before the deadline: the first is answered as spoken and the next one is open."""


async def sentences_playing_with_the_next_open(runner, sent):
    """Returns once the first sentence plays, its answer has reached the runner and, when more were sent, the next
    one's request is open."""
    await runner.app.spoken_count(1)
    await runner.app.until_called(len(sent))
    await anyio.wait_all_tasks_blocked()


@pytest.mark.parametrize(("replies", "sent"), AFTER_THE_FIRST_SENTENCE)
async def test_a_turn_past_its_deadline_when_his_line_entered_sends_no_call_or_sentence_and_ends_limit_exceeded_once(
    runner, replies, sent
):
    """TURNS-28."""
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, *replies, completion("Later."))
    say_in_the_call(runner, ADDRESSED)
    await sentences_playing_with_the_next_open(runner, sent)
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS
    runner.app.end_sentence()
    await runner.reports.until(Event.PAUSED)
    await sleeps.until_began(2)
    sleeps.let_through()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == sent
    assert runner.app.said == []
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value),
    ]
    assert runner.reports.turn_endings == [failed_turn(FailureCode.LIMIT_EXCEEDED)]
    assert [each["text"] for each in runner.gateway.turns()[1].told("speech")] == sentences_spoken(runner)
    assert not runner.task.done()


@pytest.mark.parametrize(("replies", "sent"), AFTER_THE_FIRST_SENTENCE)
async def test_a_line_or_answer_that_never_arrives_ends_the_turn_limit_exceeded_once_with_no_defect_and_no_more_speech(
    runner, replies, sent
):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, *replies, completion("Later."))
    say_in_the_call(runner, ADDRESSED)
    await sentences_playing_with_the_next_open(runner, sent)

    assert sleeps.began == [PERSONA_MAX_DURATION_SECONDS]
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS
    sleeps.release.set()
    await runner.reports.until(Event.PAUSED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == sent[:1]
    assert runner.app.said == []
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value),
    ]
    assert runner.reports.turn_endings == [failed_turn(FailureCode.LIMIT_EXCEEDED)]
    told = runner.gateway.turns()[1].told("speech")
    assert [(each["text"], each["outcome"]) for each in told] == [
        (sent[0], {"answered": True}),
        *[(text, {"answered": False, "code": FailureCode.LIMIT_EXCEEDED.value}) for text in sent[1:]],
    ]
    assert not runner.task.done()


async def test_a_turn_that_ended_limit_exceeded_with_a_line_still_owed_leaves_no_debt_to_the_next_turn_at_the_place(
    runner,
):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES), completion("Later."))
    say_in_the_call(runner, ADDRESSED)
    await sentences_playing_with_the_next_open(runner, ["First sentence.", "Second sentence."])
    runner.app.waiting_speaks.pop(0).set_result(True)
    await anyio.wait_all_tasks_blocked()
    runner.clock.time += PERSONA_MAX_DURATION_SECONDS
    sleeps.release.set()
    await runner.reports.until(Event.PAUSED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    await runner.app.until_called(3)

    assert runner.app.calls == [SPEAK_TOOL] * 3


@pytest.fixture
def speak_defect(runner):
    """A defect that ``speak`` raises, whatever the sentence."""
    defect = ValueError("a defect")
    runner.open_session = opener_whose_request_raises("speak", defect)
    return defect


async def test_a_defect_in_speaks_request_ends_the_runner_loud(runner, speak_defect):
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [speak_defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_place_left_while_a_turns_speak_is_in_flight_ends_the_turn_cancelled_and_the_next_read_works(runner):
    await a_turn_whose_speak_is_in_flight(runner)

    runner.app.stand(SUBJECT)
    await runner.reports.until_a_turn_ended()
    later_read = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_read_held()

    assert runner.reports.turn_endings == [CANCELLED_TURN]
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert not runner.task.done()
    later_read.set()


async def test_a_speak_the_app_cuts_ends_the_turn_into_the_reaction(runner):
    runner.give_voice()
    runner.gateway = Gateway(completion(TWO_SENTENCES))
    runner.app.script(CALL, cuts)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until(Event.BACKOFF)
    await runner.sleeps.until_began(1)
    runner.sleeps.let_through()
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
    ]
    assert runner.app.spoken == []
