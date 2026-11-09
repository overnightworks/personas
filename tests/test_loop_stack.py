"""The whole runner: the real loop and SDK session against a realm and an app that answer for real on loopback."""

from __future__ import annotations

import asyncio
import base64
from http import HTTPStatus

import anyio
import pytest
from agent_runtime_client.backend.fake_gateway import Reply
from agent_runtime_client.member import (
    STANDING_URI,
    AcceptInvitationArguments,
    PlaceRef,
    place_window_uri,
)
from agent_runtime_client.tools import ToolOutcome
from agent_runtime_client.turn import person_fingerprint
from mcp_types import (
    INVALID_PARAMS,
    UNSUPPORTED_PROTOCOL_VERSION,
    BlobResourceContents,
    DiscoverResult,
    ListToolsResult,
    ReadResourceResult,
    ServerCapabilities,
    TextResourceContents,
)

from agents import app_session
from agents.decision import SUSPENDED_RETRY_SECONDS, TOOL_NOT_DECLARED_CODE, TurnCap, backoff_delay, status_code
from agents.failures import INVALID_ANSWER_CODE, LISTEN_NOT_HONORED_CODE, UNSUPPORTED_VERSION_CODE
from agents.reporting import Event
from tests.support.gateway import Gateway, says
from tests.support.hopin import (
    ACKNOWLEDGED,
    CALL,
    DISCOVER,
    INITIALIZE,
    LIMIT_SECONDS,
    LIST_TOOLS,
    LISTEN,
    READ,
    active,
    formatted,
    invited,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.transport import transport_closing_with_late_answers

pytestmark = pytest.mark.anyio

REFUSALS_BEFORE_THE_FIX = 3
SHORT_STREAM_SECONDS = 0.3
CALL_PLACE = PlaceRef(kind="call", id="room-7")


async def drain(queue, count):
    for _ in range(count):
        await asyncio.wait_for(queue.get(), LIMIT_SECONDS)


async def test_every_read_follows_the_acknowledgement_of_its_stream_and_an_invitation_is_accepted_once(runner):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3", epoch=2))
    runner.start()
    await runner.reports.until("joined")
    runner.app.stand(SUBJECT, active("room-7"))
    await runner.app.log.until(READ, 2)

    assert runner.app.log.reads_follow_acknowledgements()
    assert runner.app.accepted == [AcceptInvitationArguments(place=CALL_PLACE, invitation="inv-3", epoch=2)]
    assert runner.reports.events() == [("started", None), ("joined", None)]
    assert INITIALIZE not in runner.app.log.entries


async def test_a_change_during_the_first_read_leads_to_a_read_that_shows_it(runner):
    release = runner.app.hold_next_read()
    runner.start()
    async with asyncio.timeout(LIMIT_SECONDS):
        await runner.app.read_held.wait()

    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    release.set()
    await runner.reports.until("joined")

    assert len(runner.app.accepted) == 1


async def test_a_change_made_right_after_the_acknowledgement_leads_to_a_read_that_shows_it(runner):
    async def changes_what_stands_after_acknowledging(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        runner.app.standings[SUBJECT] = [invited("room-7", "inv-3")]
        await reply.pulse(STANDING_URI)
        await asyncio.Event().wait()

    runner.app.script(LISTEN, changes_what_stands_after_acknowledging)
    runner.start()
    await runner.reports.until("joined")

    assert len(runner.app.accepted) == 1


async def test_a_stream_the_app_ends_is_reopened_at_once_on_the_same_session_and_read_again(runner):
    async def acknowledges_and_completes(reply):
        await reply.open_stream()
        await reply.acknowledge([STANDING_URI])
        await reply.complete()

    runner.app.script(LISTEN, acknowledges_and_completes)
    runner.start()
    await runner.app.log.until(READ, 2)

    assert runner.app.log.count(DISCOVER) == 1
    assert runner.app.log.count(LISTEN) == 2
    assert runner.app.log.reads_follow_acknowledgements()
    assert runner.clock.slept == []
    assert runner.reports.events() == [("started", None)]


@pytest.mark.parametrize("acknowledged_first", [True, False], ids=["after the acknowledgement", "before it"])
async def test_a_stream_lost_backs_off_and_is_reopened_and_read(runner, acknowledged_first):
    async def is_lost(reply):
        await reply.open_stream()
        if acknowledged_first:
            await reply.acknowledge([STANDING_URI])
        reply.cut()

    runner.app.script(LISTEN, is_lost)
    runner.start()
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]
    assert runner.clock.slept == [backoff_delay(1)]
    assert runner.app.log.count(DISCOVER) == 2
    assert runner.app.log.reads_follow_acknowledgements()


async def test_a_stream_that_never_completes_is_cut_at_the_runners_deadline_and_reopened_after_a_backoff(runner):
    runner.app.pulses_on_read = True
    runner.start(stream_seconds=SHORT_STREAM_SECONDS)
    await runner.reports.until("backoff")
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.events()[:2] == [("started", None), ("backoff", "network")]
    assert backoff_delay(1) in runner.clock.slept


@pytest.mark.parametrize(
    "echo", [anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"]
)
async def test_an_answer_that_lands_after_the_session_closed_backs_off_and_does_not_end_the_runner(
    runner, monkeypatch, echo
):
    failing = transport_closing_with_late_answers(echo=echo, only_close=1)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    runner.app.pulses_on_read = True
    runner.start(stream_seconds=SHORT_STREAM_SECONDS)
    await runner.reports.until("backoff")
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.events()[:2] == [("started", None), ("backoff", "network")]


async def test_a_stream_with_no_acknowledgement_until_past_the_deadline_backs_off_and_is_reopened_and_read(runner):
    talked = []

    async def talks_without_acknowledging(reply):
        talked.append(reply)
        await reply.open_stream()
        await reply.keep_alives_until_closed()

    runner.app.script(LISTEN, talks_without_acknowledging)
    runner.start(stream_seconds=SHORT_STREAM_SECONDS)
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]
    assert runner.clock.slept == [backoff_delay(1)]
    assert runner.app.log.count(LISTEN) == 2
    assert runner.app.log.reads_follow_acknowledgements()
    assert talked[0].keep_alives > 1


@pytest.mark.parametrize("arrangement", ["not listed", "completed unacknowledged"])
async def test_a_stream_that_does_not_honor_the_standing_suspends_the_loop_and_leaves_no_marker(runner, arrangement):
    async def completes_unacknowledged(reply):
        await reply.open_stream()
        await reply.complete()

    if arrangement == "not listed":
        runner.app.lists_standing = False
    else:
        runner.app.script(LISTEN, completes_unacknowledged)
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", LISTEN_NOT_HONORED_CODE)]
    assert set(runner.clock.slept) == {SUSPENDED_RETRY_SECONDS}
    assert not runner.ready_file.exists()


@pytest.mark.parametrize("method", [DISCOVER, LISTEN, READ, CALL])
async def test_a_401_is_renewed_once_and_a_second_one_suspends_the_loop(runner, method):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.app.refuse_with = HTTPStatus.UNAUTHORIZED
    runner.app.refused_method = method
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", status_code(HTTPStatus.UNAUTHORIZED))]
    assert len(runner.realm.issued) > 1


@pytest.mark.parametrize("status", [HTTPStatus.FORBIDDEN, HTTPStatus.BAD_REQUEST])
@pytest.mark.parametrize("method", [DISCOVER, LISTEN, READ, CALL])
async def test_a_4xx_suspends_the_loop_with_its_status_and_it_resumes_when_the_app_answers(runner, method, status):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.app.refuse_with = status
    runner.app.refused_method = method
    runner.start()
    await runner.reports.until("suspended")
    runner.app.refuse_with = None
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("suspended", status_code(status)), ("resumed", None)]


async def test_a_5xx_backs_off_and_is_logged_once_until_a_read_answers(runner):
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = READ
    runner.start()
    await runner.reports.until("backoff")
    await drain(runner.app.refusals, REFUSALS_BEFORE_THE_FIX)
    runner.app.refuse_with = None
    await runner.reports.until("resumed")

    assert runner.reports.events() == [
        ("started", None),
        ("backoff", status_code(HTTPStatus.SERVICE_UNAVAILABLE)),
        ("resumed", None),
    ]


async def test_a_broken_connection_on_a_read_backs_off_and_is_logged_once_until_a_read_answers(runner):
    async def breaks_the_connection(reply):
        await reply.open_stream()
        reply.cut()

    runner.app.script(READ, breaks_the_connection, breaks_the_connection)
    runner.start()
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]


@pytest.mark.parametrize(
    "contents",
    [
        TextResourceContents(uri=STANDING_URI, mime_type="application/json", text="no standings here"),
        BlobResourceContents(
            uri=STANDING_URI, mime_type="application/json", blob=base64.b64encode(b"{}").decode("ascii")
        ),
    ],
    ids=["unparseable", "not text"],
)
async def test_a_read_that_is_no_standing_suspends_the_loop_until_one_answers(runner, contents):
    async def answers_wrongly(reply):
        await reply.result(ReadResourceResult(contents=[contents], ttl_ms=0, cache_scope="private"))

    runner.app.script(READ, answers_wrongly)
    runner.start()
    await runner.reports.until("resumed")

    assert runner.reports.events() == [("started", None), ("suspended", INVALID_ANSWER_CODE), ("resumed", None)]


async def test_a_read_that_asks_for_input_suspends_the_loop_and_does_not_crash_it(runner):
    async def asks_for_input(reply):
        await reply.raw_result({"resultType": "input_required", "requestState": "x"})

    runner.app.script(READ, asks_for_input)
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", INVALID_ANSWER_CODE)]
    assert not runner.task.done()


@pytest.mark.parametrize("arrangement", ["no shared version", "unsupported version error"])
async def test_an_app_without_a_shared_version_suspends_the_loop_with_its_code_and_does_not_crash_it(
    runner, arrangement
):
    async def supports_only_an_older_revision(reply):
        await reply.result(
            DiscoverResult(supported_versions=["2025-11-25"], capabilities=ServerCapabilities(), ttl_ms=0)
        )

    async def refuses_the_revision(reply):
        await reply.error(UNSUPPORTED_PROTOCOL_VERSION)

    if arrangement == "no shared version":
        runner.app.script(DISCOVER, supports_only_an_older_revision)
        code = UNSUPPORTED_VERSION_CODE
    else:
        runner.app.script(DISCOVER, refuses_the_revision)
        code = f"jsonrpc_{UNSUPPORTED_PROTOCOL_VERSION}"
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", code)]
    assert not runner.task.done()


async def test_a_pulse_for_a_place_window_is_a_wake_and_a_read_and_no_failure(runner):
    runner.app.stand(SUBJECT, active("room-7"))
    runner.start()
    await runner.app.log.until(READ)

    runner.app.changes.changed(runner.app_principal, place_window_uri(CALL_PLACE))
    await runner.app.log.until(READ, 2)

    assert runner.reports.events() == [("started", None)]
    assert not runner.task.done()


async def test_a_disabled_client_suspends_the_loop_before_any_app_call_and_a_granted_token_resumes_it(runner):
    runner.realm.refusing = True
    runner.start()
    await drain(runner.realm.refusals, REFUSALS_BEFORE_THE_FIX)

    assert runner.app.log.entries == []
    assert runner.reports.events() == [("started", None), ("suspended", "http_401")]
    assert set(runner.clock.slept) == {SUSPENDED_RETRY_SECONDS}

    runner.realm.refusing = False
    await runner.reports.until("resumed")
    await runner.app.log.until(READ)
    assert runner.app.accepted == []

    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    await runner.reports.until("joined")
    assert len(runner.app.accepted) == 1


async def test_a_token_the_app_does_not_accept_is_renewed_once_and_then_suspends_the_loop(runner):
    runner.realm.reaches_app = False
    runner.start()
    await drain(runner.app.refusals, 2 * REFUSALS_BEFORE_THE_FIX)

    assert runner.reports.events() == [("started", None), ("suspended", "http_401")]
    assert len(runner.realm.issued) > 2


async def test_an_app_without_the_accept_tool_suspends_the_loop_with_the_protocol_error_code(runner):
    async def knows_no_such_tool(reply):
        await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)

    runner.app.script(CALL, knows_no_such_tool)
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()
    await runner.reports.until("suspended")

    assert runner.reports.events() == [("started", None), ("suspended", f"jsonrpc_{INVALID_PARAMS}")]


@pytest.mark.parametrize("turns_before_stop", range(0, 40, 3))
async def test_a_runner_stopped_while_it_connects_ends(runner, turns_before_stop):
    runner.realm.reaches_app = False
    runner.start()
    for _ in range(turns_before_stop):
        await asyncio.sleep(0)

    runner.task.cancel()

    async with asyncio.timeout(LIMIT_SECONDS):
        await asyncio.gather(runner.task, return_exceptions=True)
    assert runner.task.done()


async def test_a_runner_stopped_during_a_listen_ends_and_opens_nothing_further(runner):
    runner.start()
    await runner.app.log.until(ACKNOWLEDGED)
    await runner.app.log.until(READ)

    runner.task.cancel()
    async with asyncio.timeout(LIMIT_SECONDS):
        await asyncio.gather(runner.task, return_exceptions=True)

    assert runner.task.done()
    assert runner.app.log.count(LISTEN) == 1
    assert runner.app.log.count(DISCOVER) == 1


class BugError(Exception):
    """A programming error of the runner's own code, as a callback of a test raises it."""


async def test_a_programming_error_in_a_reconciliation_ends_the_runner_loud(runner):
    def fails_when_joined(change):
        if change.event.value == "joined":
            raise BugError

    runner.on_report = fails_when_joined
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()

    with pytest.RaisesGroup(BugError, flatten_subgroups=True):
        async with asyncio.timeout(LIMIT_SECONDS):
            await runner.task


async def test_the_marker_is_absent_after_an_acknowledgement_with_no_read_and_present_after_the_first_read(runner):
    release = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()
    async with asyncio.timeout(LIMIT_SECONDS):
        await runner.app.read_held.wait()

    assert runner.app.log.count(ACKNOWLEDGED) == 1
    assert not runner.ready_file.exists()

    release.set()
    await runner.reports.until("joined")

    assert runner.ready_file.exists()


async def test_an_app_that_refuses_with_403_leaves_no_marker(runner):
    runner.app.refuse_with = HTTPStatus.FORBIDDEN
    runner.start()
    await runner.reports.until("suspended")

    assert not runner.ready_file.exists()


PLACE_ID = "room-7"
PERSON = "visitor-subject"
ADDRESSED = "Mozart, are you there?"
ANSWER = "Here I am."


def say_in_the_call(runner, text, *, author=PERSON, name="Ada"):
    return runner.app.speak(PLACE_ID, author, name, text)


async def test_an_addressed_line_gets_exactly_one_say_under_the_agents_own_fingerprint(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)

    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert len(runner.gateway.requests) == 2
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1]]
    assert runner.gateway.openings()[0].system == runner.files.persona_prompt
    assert runner.gateway.users() == {person_fingerprint(runner.app_principal)}
    assert runner.reports.events() == [("started", None)]


async def test_his_own_line_naming_him_starts_no_turn(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, "Mozart will say it himself", author=SUBJECT, name="Mozart")
    read = runner.app.hold_next_window_read()
    runner.start()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_an_unaddressed_line_makes_no_gateway_call(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, "Hello everyone")
    read = runner.app.hold_next_window_read()
    runner.start()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, ADDRESSED)
    read.set()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_after_his_own_accept_reading_starts_at_seq_1(runner):
    def becomes_active_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))

    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.on_report = becomes_active_once_joined
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1]]


async def test_a_place_he_joins_again_is_read_from_seq_1_again(runner):
    def becomes_active_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))

    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Once more."))
    runner.on_report = becomes_active_once_joined
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)

    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.app.said_count(2)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [1, 2]]


async def test_a_duplicate_pulse_or_a_re_read_starts_no_second_turn_for_a_line(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Another answer."))
    closing_round = runner.gateway.hold_request(2)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.gateway.until_requested(2)
    for uri in (STANDING_URI, place_window_uri(CALL_PLACE), STANDING_URI, place_window_uri(CALL_PLACE)):
        runner.app.changes.changed(runner.app_principal, uri)
    read = runner.app.hold_next_window_read()
    closing_round.set()
    await runner.app.until_window_read_held()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    read.set()
    await runner.app.said_count(2)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [3]]


async def test_a_line_said_while_a_turn_ran_is_answered_once_that_turn_ends_though_nothing_else_wakes_the_loop(runner):
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER))
    first_round = runner.gateway.hold_request(1)

    def lets_the_turn_end_once_a_reconciliation_has_passed_it_by(change):
        if change.event is Event.JOINED:
            first_round.set()

    runner.on_report = lets_the_turn_end_once_a_reconciliation_has_passed_it_by
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.gateway.until_requested(1)
    runner.app.standings[SUBJECT] = [active(PLACE_ID), invited("room-9", "inv-3")]
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)
    await runner.gateway.until_requested(3)

    assert runner.gateway.round_sizes() == [2, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert runner.reports.events()[:3] == [("started", None), ("joined", None), ("turn_ended", "busy")]


async def test_one_turn_runs_at_a_time_and_a_line_said_meanwhile_is_answered_after_a_turn_that_spoke(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Another answer."))
    first_round = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.gateway.until_requested(1)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    first_round.set()
    await runner.app.said_count(2)
    await runner.gateway.until_requested(4)

    assert runner.gateway.round_sizes() == [2, 4, 2, 4]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2, 3]]


async def a_turn_ended_short(runner, code):
    """Runs the runner on one addressed line until its turn reported this code."""
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("turn_ended")

    assert runner.reports.events() == [("started", None), ("turn_ended", code)]
    assert runner.app.said == []
    assert runner.clock.slept == []
    assert not runner.task.done()
    assert runner.app.log.count(LISTEN) == 1


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (Reply(status=HTTPStatus.TOO_MANY_REQUESTS, headers={"Retry-After": "3"}), "busy"),
        (Reply(status=HTTPStatus.FORBIDDEN), "provider_unavailable"),
        (Reply(status=HTTPStatus.SERVICE_UNAVAILABLE), "provider_unavailable"),
    ],
    ids=["rate limited", "refused", "failing"],
)
async def test_a_gateway_refusal_is_silent_logs_one_code_and_leaves_the_lifecycle_untouched(runner, reply, code):
    runner.gateway = Gateway(reply)

    await a_turn_ended_short(runner, code)

    assert len(runner.gateway.requests) == 1


async def test_a_key_the_gateway_cannot_use_ends_the_turn_before_any_gateway_call(runner):
    runner.files.key_path.write_text("a key with spaces\n", encoding="utf-8")

    await a_turn_ended_short(runner, "provider_unavailable")

    assert runner.gateway.requests == []


async def test_a_missing_api_group_is_refused_with_no_gateway_call(runner):
    runner.realm.groups = []

    await a_turn_ended_short(runner, "not_enabled")

    assert runner.gateway.requests == []


async def test_a_tool_that_answers_gone_shows_the_model_that_and_leaves_the_lifecycle_untouched(runner):
    gone = ToolOutcome("the place is gone", is_error=True)
    runner.app.say_outcomes.append(gone)
    runner.app.content.add(gone.content)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.gateway.until_requested(2)

    assert gone.content in runner.gateway.requests[1].content.decode()
    assert runner.app.said == []
    assert runner.reports.events() == [("started", None)]
    assert runner.clock.slept == []


async def test_a_bound_tool_the_app_declares_no_more_ends_the_turn_with_a_code(runner):
    async def lists_no_tools(reply):
        await reply.result(ListToolsResult(tools=[]))

    runner.app.script(LIST_TOOLS, lists_no_tools)

    await a_turn_ended_short(runner, TOOL_NOT_DECLARED_CODE)

    assert runner.gateway.requests == []


async def cuts(reply):
    await reply.open_stream()
    reply.cut()


async def refuses_the_token(reply):
    await reply.status(HTTPStatus.UNAUTHORIZED)


@pytest.mark.parametrize(
    ("method", "fails", "events"),
    [
        (DISCOVER, cuts, [("started", None), ("backoff", "network"), ("resumed", None)]),
        (LIST_TOOLS, cuts, [("started", None), ("backoff", "network"), ("resumed", None)]),
        (DISCOVER, refuses_the_token, [("started", None)]),
        (LIST_TOOLS, refuses_the_token, [("started", None)]),
    ],
    ids=["discovery cut", "tool listing cut", "discovery refuses the token", "tool listing refuses the token"],
)
async def test_a_turn_whose_session_fails_before_the_model_is_asked_leads_to_one_lifecycle_reaction_and_no_second_turn(
    runner, method, fails, events
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.log.until(READ)
    runner.app.script(method, fails)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert runner.reports.events() == events
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]
    assert len(runner.realm.issued) == (2 if fails is refuses_the_token else 1)


@pytest.mark.parametrize(
    "echo", [anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"]
)
async def test_a_turn_ended_short_beside_a_late_answer_is_reported_and_costs_one_backoff_and_no_second_turn(
    runner, monkeypatch, echo
):
    failing = transport_closing_with_late_answers(echo=echo, only_close=1)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("resumed")
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.events() == [
        ("started", None),
        ("turn_ended", "busy"),
        ("backoff", "network"),
        ("resumed", None),
    ]
    assert not runner.task.done()
    assert len(runner.gateway.requests) == 1


@pytest.mark.parametrize(
    "echo", [anyio.BrokenResourceError, anyio.ClosedResourceError], ids=["broken stream", "closed stream"]
)
async def test_a_turn_failure_cancelling_a_lifecycle_that_unwinds_with_a_late_answer_gives_one_reaction(
    runner, monkeypatch, echo
):
    failing = transport_closing_with_late_answers(echo=echo, only_close=2)
    monkeypatch.setattr(app_session, "streamable_http_client", failing)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.log.until(READ)
    runner.app.script(DISCOVER, cuts)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert runner.reports.events() == [("started", None), ("backoff", "network"), ("resumed", None)]
    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[2]]


async def test_a_window_the_app_lost_is_skipped_and_is_no_suspension(runner):
    runner.gateway = Gateway(*says("room-8", ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID), active("room-8"))
    runner.app.vanished.add(PLACE_ID)
    say_in_the_call(runner, ADDRESSED)
    runner.app.speak("room-8", PERSON, "Ada", ADDRESSED)
    runner.start()
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, "room-8", ANSWER)]
    assert runner.reports.events() == [("started", None)]


async def test_a_refused_and_an_answered_turn_log_a_code_and_no_secret_or_content(runner, caplog):
    runner.gateway = Gateway(Reply(status=HTTPStatus.TOO_MANY_REQUESTS), *says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("turn_ended")
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    logged = formatted(caplog.get_records("call"))
    assert "turn_ended" in logged


TURN_CAP = TurnCap(max_turns=1, period_seconds=120.0)
SECOND_AGENT = "agent-bach"
SECOND_SUBJECT = "agent-subject-2"


def refused_turn() -> Reply:
    """A turn that starts, asks the gateway once and ends short: it counts against the cap all the same."""
    return Reply(status=HTTPStatus.TOO_MANY_REQUESTS)


async def addressed_turn_ended(runner, text=ADDRESSED):
    say_in_the_call(runner, text)
    await runner.reports.until("turn_ended")


async def test_a_line_addressed_to_him_over_the_cap_starts_no_turn_makes_no_gateway_call_and_is_not_answered_later(
    runner, caplog
):
    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), *says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("turn_ended")
    say_in_the_call(runner, f"{ADDRESSED} Capped.")
    await runner.reports.until("capped")
    runner.app.changes.changed(runner.app_principal, place_window_uri(CALL_PLACE))
    runner.clock.time += TURN_CAP.period_seconds
    say_in_the_call(runner, f"{ADDRESSED} After the period.")
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [3]]
    assert [event for event, _ in runner.reports.events()].count("capped") == 1
    assert "capped client_id=agent-mozart" in formatted(caplog.get_records("call"))


async def test_after_the_period_has_passed_an_addressed_line_starts_a_turn_again(runner):
    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), *says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("turn_ended")
    runner.clock.time += TURN_CAP.period_seconds
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.said_count(1)

    assert [opening.new_seqs for opening in runner.gateway.openings()] == [[1], [2]]
    assert "capped" not in [event for event, _ in runner.reports.events()]


async def test_two_apps_of_one_agent_share_one_cap(runner):
    runner.cap_turns(TURN_CAP)
    runner.add_app()
    runner.gateway = Gateway(refused_turn(), refused_turn())
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("turn_ended")
    await runner.reports.until("capped")

    assert len(runner.gateway.openings()) == 1


async def test_leaving_and_rejoining_a_place_does_not_reset_the_cap(runner):
    def becomes_active_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))

    runner.cap_turns(TURN_CAP)
    runner.gateway = Gateway(refused_turn(), refused_turn())
    runner.on_report = becomes_active_once_joined
    runner.app.stand(SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.reports.until("turn_ended")
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.reports.until("capped")

    assert len(runner.gateway.openings()) == 1


async def test_two_agents_each_have_a_cap_of_their_own(runner):
    runner.cap_turns(TURN_CAP)
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.gateway = Gateway(refused_turn(), refused_turn())
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    say_in_the_call(runner, ADDRESSED)
    runner.start()
    await runner.gateway.until_requested(2)

    assert len(runner.gateway.openings()) == 2
    assert "capped" not in [event for event, _ in runner.reports.events()]
