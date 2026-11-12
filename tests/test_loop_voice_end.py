"""The whole runner, the spoken turn's end: what stops it, what does not and where its failures lead."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion
from agents.member import STANDING_URI, place_window_uri
from agents.tools import ToolOutcome
from agents.turn import FailureCode

from personas import app_session
from personas.app_session import READ_SECONDS, open_mcp_session
from personas.decision import TURN_CANCELLED_CODE
from personas.failures import leaves_of
from personas.reporting import Event
from personas.task_record import UnrequestedCancellationError
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PLACE_ID, say_in_the_call
from tests.support.gateway import Gateway, speaks_and_says
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, LIMIT_SECONDS, LISTEN, READ, SAY_TOOL, active, paused
from tests.support.transport import ClosingHold, transport_whose_session_loses_a_child
from tests.support.voice import TWO_SENTENCES, HeldSleeps, a_voice_call, ended_by, sentences_spoken
from tests.support.wire import hangs

pytestmark = pytest.mark.anyio

SHORT_SPEAK_SECONDS = 0.3
LONG_SENTENCE_MS = 20_000


@pytest.mark.parametrize(
    "leaves",
    [lambda runner: runner.app.stand(SUBJECT), lambda runner: runner.app.stand(SUBJECT, paused(PLACE_ID))],
    ids=["removed", "paused"],
)
async def test_a_place_removed_or_paused_while_a_sentence_plays_cancels_the_turn_and_no_further_sentence_is_spoken(
    runner, leaves
):
    sleeps = HeldSleeps(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    leaves(runner)
    await sleeps.until_cancelled()
    await runner.reports.until(Event.TURN_ENDED)
    sleeps.release.set()

    assert sentences_spoken(runner) == ["First sentence."]
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert not runner.task.done()


async def test_a_restart_in_a_call_answers_nothing_that_was_said_before_it(runner):
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

    async def pulses_the_window_then_ends(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        await pulse_window.wait()
        await reply.pulse(place_window_uri(CALL_PLACE))
        await end_stream.wait()
        await reply.complete()

    sleeps = HeldSleeps(runner)
    runner.app.script(LISTEN, pulses_the_window_then_ends)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    pulse_window.set()
    await sleeps.until_began(1)
    end_stream.set()
    await runner.app.log.until(LISTEN, 2)
    sleeps.release.set()
    await runner.app.spoken_count(2)

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_lifecycle_connection_lost_while_a_sentence_is_sent_cancels_the_turn_and_enters_the_sentence_unanswered(
    runner,
):
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written.", after="Later."))
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(1)
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = READ
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.reports.until(Event.BACKOFF)
    runner.app.refuse_with = None
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
                setattr(app, "refuse_with", HTTPStatus.SERVICE_UNAVAILABLE) or setattr(app, "refused_method", CALL)
            ),
            lambda app: setattr(app, "refuse_with", None),
            [(Event.STARTED, None), (Event.BACKOFF, "http_503"), (Event.RESUMED, None)],
        ),
        (
            lambda app: app.speak_outcomes.append(ToolOutcome("not a spoken record", is_error=False)),
            lambda app: None,
            [(Event.STARTED, None), (Event.SUSPENDED, "invalid_answer"), (Event.RESUMED, None)],
        ),
    ],
    ids=["a call that fails", "an answer that is no Spoken"],
)
async def test_a_speak_that_fails_as_a_call_or_answers_no_spoken_ends_the_turn_into_the_reaction_and_a_new_turn_speaks(
    runner, break_speak, mend_speak, events
):
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written.", after="Later."))
    break_speak(runner.app)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(events[1][0])

    assert runner.app.spoken == []
    assert runner.gateway.arrivals == 1
    mend_speak(runner.app)
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
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, ANSWER, "Written.", after="Never."))
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = CALL
    runner.app.refused_tool = SAY_TOOL
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.BACKOFF)
    runner.app.refuse_with = None
    await runner.reports.until(Event.RESUMED)

    assert sentences_spoken(runner) == [ANSWER]
    assert runner.gateway.arrivals == 1
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.BACKOFF, "http_503"), (Event.RESUMED, None)]


async def test_a_gateway_that_stops_the_turn_while_a_sentence_is_sent_and_one_is_queued_answers_the_sent_one_only(
    runner,
):
    await a_voice_call(runner, completion("done"))
    reply = runner.gateway.pause_request(1, "First sentence. Second sentence. Thi", "rd.")
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(1)
    reply.cut()
    answer.set()
    await runner.reports.until(Event.TURN_ENDED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == ["First sentence."]
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, FailureCode.PROVIDER_ERROR.value)]
    (sent,) = runner.gateway.turns()[1].told("speech")
    assert (sent["text"], sent["outcome"]) == ("First sentence.", {"answered": True})


@pytest.mark.parametrize(
    ("replies", "spoken"),
    [
        (speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written."), ["First sentence.", "Second sentence."]),
        ([completion("One. Two. Three.")], ["One.", "Two."]),
    ],
    ids=["during the waits before a call", "during the speech after the final event"],
)
async def test_a_turn_past_its_deadline_sends_no_call_and_no_sentence_and_ends_limit_exceeded_once(
    runner, replies, spoken
):
    await a_voice_call(runner, *replies, completion("Later."))
    runner.app.speak_duration_ms = LONG_SENTENCE_MS
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == spoken
    assert runner.app.said == []
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, FailureCode.LIMIT_EXCEEDED.value)]
    assert [each["text"] for each in runner.gateway.turns()[1].told("speech")] == spoken


@pytest.fixture
def speak_defect(monkeypatch):
    """A defect that ``speak`` raises, whatever the sentence."""
    defect = ValueError("a defect")

    async def raise_it(self, *arguments):
        raise defect

    monkeypatch.setattr("personas.app_session._McpAppSession.speak", raise_it)
    return defect


async def test_a_defect_in_speaks_request_ends_the_runner_loud(runner, speak_defect):
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [speak_defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_defect_in_speaks_request_whose_place_then_leaves_ends_the_runner_loud_without_a_cancel(
    runner, monkeypatch, speak_defect
):
    closing = ClosingHold()
    monkeypatch.setattr(app_session, "streamable_http_client", closing.transport(of_url=runner.app_url))
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.closing.wait()
        runner.app.stand(SUBJECT)
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
    finally:
        closing.release.set()

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [speak_defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_turn_task_that_ends_cancelled_while_nobody_cancelled_it_ends_the_runner_loud(runner, monkeypatch):
    failing = asyncio.Event()
    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_whose_session_loses_a_child(failing, of_url=runner.app_url)
    )
    sleeps = HeldSleeps(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    failing.set()
    ended = await ended_by(runner.task)

    (leaf,) = leaves_of(ended)
    assert isinstance(leaf, UnrequestedCancellationError)
    assert isinstance(leaf.__cause__, asyncio.CancelledError)


async def test_a_speak_that_never_answers_is_given_up_at_the_sessions_read_bound_and_the_turn_ends_into_the_reaction(
    runner,
):
    opens = 0

    def short_after_the_lifecycle(app, token):
        nonlocal opens
        opens += 1
        return open_mcp_session(app, token, stream_seconds=READ_SECONDS if opens == 1 else SHORT_SPEAK_SECONDS)

    runner.give_voice()
    runner.gateway = Gateway(completion(TWO_SENTENCES))
    runner.app.script(CALL, hangs)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start(open_session=short_after_the_lifecycle)
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.BACKOFF, "network"), (Event.RESUMED, None)]
    assert runner.app.spoken == []
