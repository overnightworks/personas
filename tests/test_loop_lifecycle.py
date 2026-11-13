"""The whole runner, the lifecycle: streams, reads, failures and suspension, against a realm and an app on loopback."""

from __future__ import annotations

import asyncio
import base64
from http import HTTPStatus

import anyio
import pytest
from agents.member import (
    STANDING_URI,
    AcceptInvitationArguments,
    place_window_uri,
)
from mcp_types import (
    INVALID_PARAMS,
    UNSUPPORTED_PROTOCOL_VERSION,
    BlobResourceContents,
    DiscoverResult,
    ReadResourceResult,
    ServerCapabilities,
    TextResourceContents,
)

from personas import app_session
from personas.decision import NETWORK_FAILURE_CODE, SUSPENDED_RETRY_SECONDS, backoff_delay, status_code
from personas.failures import (
    INVALID_ANSWER_CODE,
    LISTEN_NOT_HONORED_CODE,
    UNSUPPORTED_VERSION_CODE,
    Unreachable,
    leaves_of,
)
from personas.loop import AgentLoop
from personas.reporting import Event
from tests.support.call import CALL_PLACE
from tests.support.hopin import (
    ACKNOWLEDGED,
    CALL,
    DISCOVER,
    INITIALIZE,
    LIMIT_SECONDS,
    LISTEN,
    READ,
    active,
    invited,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.transport import transport_closing_with_late_answers, transport_whose_close_loses_a_child

pytestmark = pytest.mark.anyio

REFUSALS_BEFORE_THE_FIX = 3
SHORT_STREAM_SECONDS = 0.3


async def drain(queue, count):
    for _ in range(count):
        await asyncio.wait_for(queue.get(), LIMIT_SECONDS)


async def test_every_read_follows_the_acknowledgement_of_its_stream_and_an_invitation_is_accepted_once(runner):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3", epoch=2))
    runner.start()
    await runner.reports.until(Event.JOINED)
    runner.app.stand(SUBJECT, active("room-7"))
    await runner.app.log.until(READ, 2)

    assert runner.app.log.reads_follow_acknowledgements()
    assert runner.app.accepted == [AcceptInvitationArguments(place=CALL_PLACE, invitation="inv-3", epoch=2)]
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.JOINED, None)]
    assert INITIALIZE not in runner.app.log.entries


async def test_a_change_during_the_first_read_leads_to_a_read_that_shows_it(runner):
    release = runner.app.hold_next_read()
    runner.start()
    async with asyncio.timeout(LIMIT_SECONDS):
        await runner.app.read_held.wait()

    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    release.set()
    await runner.reports.until(Event.JOINED)

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
    await runner.reports.until(Event.JOINED)

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
    assert runner.reports.reported() == [(Event.STARTED, None)]


@pytest.mark.parametrize("acknowledged_first", [True, False], ids=["after the acknowledgement", "before it"])
async def test_a_stream_lost_backs_off_and_is_reopened_and_read(runner, acknowledged_first):
    async def is_lost(reply):
        await reply.open_stream()
        if acknowledged_first:
            await reply.acknowledge([STANDING_URI])
        reply.cut()

    runner.app.script(LISTEN, is_lost)
    runner.start()
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
    ]
    assert runner.clock.slept == [backoff_delay(1)]
    assert runner.app.log.count(DISCOVER) == 2
    assert runner.app.log.reads_follow_acknowledgements()


async def test_a_stream_that_never_completes_is_cut_at_the_runners_deadline_and_reopened_after_a_backoff(runner):
    runner.app.pulses_on_read = True
    runner.start(stream_seconds=SHORT_STREAM_SECONDS)
    await runner.reports.until(Event.BACKOFF)
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.reported()[:2] == [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE)]
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
    await runner.reports.until(Event.BACKOFF)
    await runner.app.log.until(LISTEN, 2)

    assert runner.reports.reported()[:2] == [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE)]


async def test_a_stream_with_no_acknowledgement_until_past_the_deadline_backs_off_and_is_reopened_and_read(runner):
    talked = []

    async def talks_without_acknowledging(reply):
        talked.append(reply)
        await reply.open_stream()
        await reply.keep_alives_until_closed()

    runner.app.script(LISTEN, talks_without_acknowledging)
    runner.start(stream_seconds=SHORT_STREAM_SECONDS)
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
    ]
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
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, LISTEN_NOT_HONORED_CODE)]
    assert set(runner.clock.slept) == {SUSPENDED_RETRY_SECONDS}
    assert not runner.ready_file.exists()


@pytest.mark.parametrize("method", [DISCOVER, LISTEN, READ, CALL])
async def test_a_401_is_renewed_once_and_a_second_one_suspends_the_loop(runner, method):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.app.refuse_with = HTTPStatus.UNAUTHORIZED
    runner.app.refused_method = method
    runner.start()
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, status_code(HTTPStatus.UNAUTHORIZED))]
    assert len(runner.realm.issued) > 1


@pytest.mark.parametrize("status", [HTTPStatus.FORBIDDEN, HTTPStatus.BAD_REQUEST])
@pytest.mark.parametrize("method", [DISCOVER, LISTEN, READ, CALL])
async def test_a_4xx_suspends_the_loop_with_its_status_and_it_resumes_when_the_app_answers(runner, method, status):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.app.refuse_with = status
    runner.app.refused_method = method
    runner.start()
    await runner.reports.until(Event.SUSPENDED)
    runner.app.refuse_with = None
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.SUSPENDED, status_code(status)),
        (Event.RESUMED, None),
    ]


async def test_a_5xx_backs_off_and_is_logged_once_until_a_read_answers(runner):
    runner.app.refuse_with = HTTPStatus.SERVICE_UNAVAILABLE
    runner.app.refused_method = READ
    runner.start()
    await runner.reports.until(Event.BACKOFF)
    await drain(runner.app.refusals, REFUSALS_BEFORE_THE_FIX)
    runner.app.refuse_with = None
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, status_code(HTTPStatus.SERVICE_UNAVAILABLE)),
        (Event.RESUMED, None),
    ]


async def test_a_broken_connection_on_a_read_backs_off_and_is_logged_once_until_a_read_answers(runner):
    async def breaks_the_connection(reply):
        await reply.open_stream()
        reply.cut()

    runner.app.script(READ, breaks_the_connection, breaks_the_connection)
    runner.start()
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.BACKOFF, NETWORK_FAILURE_CODE),
        (Event.RESUMED, None),
    ]


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
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.SUSPENDED, INVALID_ANSWER_CODE),
        (Event.RESUMED, None),
    ]


async def test_a_read_that_asks_for_input_suspends_the_loop_and_does_not_crash_it(runner):
    async def asks_for_input(reply):
        await reply.raw_result({"resultType": "input_required", "requestState": "x"})

    runner.app.script(READ, asks_for_input)
    runner.start()
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, INVALID_ANSWER_CODE)]
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
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, code)]
    assert not runner.task.done()


async def test_a_pulse_for_a_place_window_is_a_wake_and_a_read_and_no_failure(runner):
    runner.app.stand(SUBJECT, active("room-7"))
    runner.start()
    await runner.app.log.until(READ)

    runner.app.changes.changed(runner.app_principal, place_window_uri(CALL_PLACE))
    await runner.app.log.until(READ, 2)

    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert not runner.task.done()


async def test_a_disabled_client_suspends_the_loop_before_any_app_call_and_a_granted_token_resumes_it(runner):
    runner.realm.refusing = True
    runner.start()
    await drain(runner.realm.refusals, REFUSALS_BEFORE_THE_FIX)

    assert runner.app.log.entries == []
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, status_code(401))]
    assert set(runner.clock.slept) == {SUSPENDED_RETRY_SECONDS}

    runner.realm.refusing = False
    await runner.reports.until(Event.RESUMED)
    await runner.app.log.until(READ)
    assert runner.app.accepted == []

    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    await runner.reports.until(Event.JOINED)
    assert len(runner.app.accepted) == 1


async def test_a_token_the_app_does_not_accept_is_renewed_once_and_then_suspends_the_loop(runner):
    runner.realm.reaches_app = False
    runner.start()
    await drain(runner.app.refusals, 2 * REFUSALS_BEFORE_THE_FIX)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, status_code(401))]
    assert len(runner.realm.issued) > 2


async def test_an_app_without_the_accept_tool_suspends_the_loop_with_the_protocol_error_code(runner):
    async def knows_no_such_tool(reply):
        await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)

    runner.app.script(CALL, knows_no_such_tool)
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SUSPENDED, f"jsonrpc_{INVALID_PARAMS}")]


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
        if change.event is Event.JOINED:
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
    await runner.reports.until(Event.JOINED)

    assert runner.ready_file.exists()


async def test_an_app_that_refuses_with_403_leaves_no_marker(runner):
    runner.app.refuse_with = HTTPStatus.FORBIDDEN
    runner.start()
    await runner.reports.until(Event.SUSPENDED)

    assert not runner.ready_file.exists()


async def test_a_defect_in_the_lifecycle_sessions_body_whose_close_loses_a_child_ends_the_runner_loud(
    runner, monkeypatch
):
    defect = ValueError("a defect")

    real_reconcile = AgentLoop._reconcile  # noqa: SLF001 - the lifecycle's body is where the defect arises
    raised = False

    async def raise_once(self, session, turns):
        nonlocal raised
        if raised:
            return await real_reconcile(self, session, turns)
        raised = True
        raise defect

    monkeypatch.setattr(AgentLoop, "_reconcile", raise_once)
    monkeypatch.setattr(app_session, "streamable_http_client", transport_whose_close_loses_a_child(OSError))
    runner.start()

    with pytest.RaisesGroup(ValueError, Unreachable, flatten_subgroups=True) as ended:
        async with asyncio.timeout(LIMIT_SECONDS):
            await runner.task

    assert [leaf for leaf in leaves_of(ended.value) if leaf is defect] == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]
