"""The whole runner, the voice loop's lifetime: what ends it, what does not and where its failures lead."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion
from agents.member import STANDING_URI, place_window_uri
from agents.tools import ToolOutcome

from personas import app_session
from personas.app_session import READ_SECONDS, open_mcp_session
from personas.failures import leaves_of
from personas.task_record import UnrequestedCancellationError
from tests.support.call import ADDRESSED, ANSWER, CALL_PLACE, PLACE_ID, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, LIMIT_SECONDS, LISTEN, active, paused
from tests.support.transport import ClosingHold, transport_whose_session_loses_a_child
from tests.support.voice import TWO_SENTENCES, HeldSleeps, a_voice_call, ended_by, sentences_spoken

pytestmark = pytest.mark.anyio

SHORT_SPEAK_SECONDS = 0.3


@pytest.mark.parametrize(
    "leaves",
    [lambda runner: runner.app.stand(SUBJECT), lambda runner: runner.app.stand(SUBJECT, paused(PLACE_ID))],
    ids=["removed", "paused"],
)
async def test_a_place_removed_or_paused_mid_answer_cancels_the_loop_and_no_further_sentence_is_spoken(runner, leaves):
    sleeps = HeldSleeps(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await sleeps.until_began(1)

    leaves(runner)
    await sleeps.until_cancelled()
    sleeps.release.set()

    assert sentences_spoken(runner) == ["First sentence."]
    assert not runner.task.done()


async def test_a_new_window_generation_ends_the_old_loop_and_starts_a_new_one_that_knows_nothing_of_the_old_window(
    runner,
):
    sleeps = HeldSleeps(runner)
    old_line, new_line = f"{ADDRESSED} In the old window.", f"{ADDRESSED} In the new window."
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, old_line, spoken=True)
    await sleeps.until_began(1)

    runner.app.join(PLACE_ID)
    say_in_the_call(runner, new_line, spoken=True)
    await sleeps.until_cancelled()
    await runner.gateway.until_requested(2)

    assert old_line in runner.gateway.told_in(1)
    assert new_line in runner.gateway.told_in(2)
    assert old_line not in runner.gateway.told_in(2)


async def test_a_stream_the_app_ends_gracefully_leaves_the_loop_and_its_conversation(runner):
    pulse_window, end_stream = asyncio.Event(), asyncio.Event()

    async def pulses_the_window_then_ends(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        await pulse_window.wait()
        await reply.pulse(place_window_uri(CALL_PLACE))
        await end_stream.wait()
        await reply.complete()

    runner.app.script(LISTEN, pulses_the_window_then_ends)
    first = f"{ADDRESSED} First."
    await a_voice_call(runner, completion(ANSWER), completion("Second."))
    say_in_the_call(runner, first, spoken=True)
    pulse_window.set()
    await runner.app.spoken_count(1)
    end_stream.set()
    await runner.app.log.until(LISTEN, 2)
    say_in_the_call(runner, f"{ADDRESSED} Second.", spoken=True)
    await runner.app.spoken_count(2)

    assert first in runner.gateway.told_in(2)


async def test_a_restart_starts_the_loop_having_heard_nothing_of_what_was_said_before(runner):
    before = f"{ADDRESSED} Before the start."
    say_in_the_call(runner, before, spoken=True)
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, f"{ADDRESSED} After the start.", spoken=True)
    await runner.app.spoken_count(1)

    assert before not in runner.gateway.told_in(1)


async def test_a_resume_starts_the_loop_having_heard_nothing_of_what_was_said_during_the_pause(runner):
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


@pytest.mark.parametrize(
    ("break_speak", "mend_speak", "events"),
    [
        (
            lambda app: (
                setattr(app, "refuse_with", HTTPStatus.SERVICE_UNAVAILABLE) or setattr(app, "refused_method", CALL)
            ),
            lambda app: setattr(app, "refuse_with", None),
            [("started", None), ("backoff", "http_503"), ("resumed", None)],
        ),
        (
            lambda app: app.speak_outcomes.append(ToolOutcome("not a spoken record", is_error=False)),
            lambda app: None,
            [("started", None), ("suspended", "invalid_answer"), ("resumed", None)],
        ),
    ],
    ids=["a call that fails", "an answer that is no Spoken"],
)
async def test_a_speak_that_fails_as_a_call_or_answers_no_spoken_ends_the_loop_into_the_reaction_and_a_new_loop_speaks(
    runner, break_speak, mend_speak, events
):
    await a_voice_call(runner, completion(TWO_SENTENCES), completion("Later."))
    break_speak(runner.app)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until(events[1][0])

    assert runner.app.spoken == []
    mend_speak(runner.app)
    await runner.reports.until("resumed")
    say_in_the_call(runner, f"{ADDRESSED} Again.", spoken=True)
    await runner.app.spoken_count(1)

    assert sentences_spoken(runner) == ["Later."]
    assert runner.reports.events() == events
    assert not runner.task.done()


async def test_a_voice_task_that_ends_cancelled_while_nobody_cancelled_it_ends_the_runner_loud(runner, monkeypatch):
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


@pytest.fixture
def speak_defect(monkeypatch):
    """A defect that ``speak`` raises, whatever the sentence."""
    defect = ValueError("a defect")

    async def raise_it(self, *arguments):
        raise defect

    monkeypatch.setattr("personas.app_session._McpAppSession.speak", raise_it)
    return defect


async def test_a_defect_in_speaks_session_ends_the_runner_loud(runner, speak_defect):
    await a_voice_call(runner, completion(TWO_SENTENCES))
    say_in_the_call(runner, ADDRESSED, spoken=True)

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [speak_defect]
    assert runner.reports.events() == [("started", None)]


async def test_a_defect_in_speaks_session_whose_place_then_leaves_ends_the_runner_loud_without_a_cancel(
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
    assert runner.reports.events() == [("started", None)]


async def test_a_speak_that_never_answers_is_given_up_at_the_sessions_read_bound_and_the_loop_ends_into_the_reaction(
    runner,
):
    opens = 0

    def short_after_the_lifecycle(app, token):
        nonlocal opens
        opens += 1
        return open_mcp_session(app, token, stream_seconds=READ_SECONDS if opens == 1 else SHORT_SPEAK_SECONDS)

    async def hangs(reply):
        await asyncio.Event().wait()

    runner.give_voice()
    runner.gateway = Gateway(completion(TWO_SENTENCES))
    runner.app.script(CALL, hangs)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start(open_session=short_after_the_lifecycle)
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]
    assert runner.app.spoken == []
