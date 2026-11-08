"""The loop of one agent in one app, over scripted boundaries and a clock that moves only when told."""

from __future__ import annotations

import pytest

from agents.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    SUSPENDED_RETRY_SECONDS,
    Accept,
    Refusal,
    backoff_delay,
)
from agents.failures import StatusFailure, TokenStatusFailure, Unreachable, WaitToolError
from tests.support.fakes import answer
from tests.support.hopin import active, invited

pytestmark = pytest.mark.anyio


async def test_each_wait_carries_the_cursor_of_the_answer_before(scenario):
    run = scenario([answer("c1"), answer("c2"), answer("c3")])

    await run.play()

    assert run.session.cursors == [None, "c1", "c2", "c3"]
    assert run.session.opened == 1


async def test_an_invitation_is_accepted_once_with_its_place_invitation_and_epoch(scenario):
    run = scenario([answer("c1", invited("room-7", "inv-3", epoch=2)), answer("c2", active("room-7"))])

    await run.play()

    assert run.session.accepted == [Accept("room-7", "inv-3", 2)]
    assert run.reports.events() == [("started", None), ("joined", None)]


async def test_a_stale_invitation_is_not_retried_and_is_reported_once(scenario):
    standing = invited("room-7", "inv-3")
    run = scenario(
        [answer("c1", standing), answer("c2", standing), answer("c3", standing)],
        accepts=[Refusal.STALE_INVITATION],
    )

    await run.play()

    assert len(run.session.accepted) == 1
    assert run.reports.events() == [("started", None), ("refused", "stale_invitation")]


@pytest.mark.parametrize("refusal", [Refusal.SPONSOR_AWAY, Refusal.TWO_SEATS, Refusal.BUSY])
async def test_a_transient_refusal_is_retried_at_the_next_answer_and_reported_once(scenario, refusal):
    standing = invited("room-7", "inv-3")
    run = scenario(
        [answer("c1", standing), answer("c2", standing), answer("c3", standing), answer("c4", active("room-7"))],
        accepts=[refusal, refusal, None],
    )

    await run.play()

    assert len(run.session.accepted) == 3
    assert run.reports.events() == [("started", None), ("refused", refusal.value), ("joined", None)]


async def test_a_token_is_renewed_before_it_expires_and_never_sent_expired(scenario):
    run = scenario([answer(f"c{n}") for n in range(8)], seconds_per_wait=100.0)

    await run.play()

    assert run.tokens.fetched > 1
    for bearer, sent_at in run.session.bearers_with_time:
        assert sent_at < run.tokens.expires_at[bearer] - RENEW_BEFORE_EXPIRY_SECONDS


async def test_a_refused_token_suspends_the_loop_without_an_app_call_and_logs_it_once(scenario):
    refused = [TokenStatusFailure(401)] * 3
    run = scenario([answer("c1")], tokens=refused)

    await run.play()

    assert run.session.opened == 0
    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS] * 3
    assert run.reports.events() == [("started", None), ("suspended", "http_401")]


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (StatusFailure(403), "http_403"),
        (StatusFailure(400), "http_400"),
        (WaitToolError("wait failed"), "wait_error"),
    ],
    ids=["403", "400", "wait tool error"],
)
async def test_an_app_refusal_suspends_the_loop_and_it_resumes_when_wait_answers(scenario, failure, code):
    run = scenario([failure, failure, answer("c1"), answer("c2")])

    await run.play()

    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS] * 2
    assert run.reports.events() == [("started", None), ("suspended", code), ("resumed", None)]
    assert run.tokens.fetched == 3


async def test_a_401_renews_the_token_once_and_a_second_one_suspends_the_loop(scenario):
    run = scenario([StatusFailure(401), StatusFailure(401), answer("c1")])

    await run.play()

    assert run.tokens.fetched == 3
    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert run.reports.events() == [("started", None), ("suspended", "http_401"), ("resumed", None)]


async def test_a_401_the_renewed_token_cures_suspends_nothing(scenario):
    run = scenario([StatusFailure(401), answer("c1")])

    await run.play()

    assert run.clock.slept == []
    assert run.reports.events() == [("started", None)]


async def test_a_loop_resumed_after_a_suspension_joins_only_when_invited_anew(scenario):
    run = scenario(
        [StatusFailure(403), answer("c1", active("room-7")), answer("c2", invited("room-7", "inv-9"))],
    )

    await run.play()

    assert run.reports.events() == [
        ("started", None),
        ("suspended", "http_403"),
        ("resumed", None),
        ("joined", None),
    ]
    assert run.session.accepted == [Accept("room-7", "inv-9", 0)]


async def test_a_wait_after_a_failure_starts_without_a_cursor_so_the_answer_is_the_whole_truth(scenario):
    run = scenario([answer("c1"), Unreachable("down"), answer("c2")])

    await run.play()

    assert run.session.cursors == [None, "c1", None, "c2"]


@pytest.mark.parametrize(
    "failure",
    [Unreachable("down"), StatusFailure(503)],
    ids=["network", "5xx from the app"],
)
async def test_an_app_failure_that_is_not_a_refusal_backs_off_with_a_capped_growing_interval(scenario, failure):
    failures = 40
    run = scenario([failure] * failures)

    await run.play()

    assert run.clock.slept == [backoff_delay(attempt) for attempt in range(1, failures + 1)]
    assert max(run.clock.slept) == BACKOFF_CAP_SECONDS
    assert min(run.clock.slept) > 0
    assert run.reports.events() == [("started", None)]


async def test_a_realm_that_does_not_answer_or_fails_backs_off_without_suspending(scenario):
    failures = [Unreachable("down"), TokenStatusFailure(502), Unreachable("down")]
    run = scenario([], tokens=failures)

    await run.play()

    assert run.clock.slept == [backoff_delay(1), backoff_delay(2), backoff_delay(3)]
    assert run.reports.events() == [("started", None)]


async def test_the_backoff_starts_over_after_an_answer(scenario):
    run = scenario([Unreachable("down"), Unreachable("down"), answer("c1"), Unreachable("down")])

    await run.play()

    assert run.clock.slept == [backoff_delay(1), backoff_delay(2), backoff_delay(1)]
