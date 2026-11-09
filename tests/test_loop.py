"""The loop of one agent in one app, over scripted boundaries and a clock that moves only when told."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from agent_runtime_client.member import PlaceRef

from agents.config import load_config
from agents.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    SUSPENDED_RETRY_SECONDS,
    Accept,
    Refusal,
    backoff_delay,
)
from agents.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer, leaves_of
from agents.loop import AgentLoop, Boundaries
from tests.support.fakes import (
    FakeClock,
    FakeReadyMarker,
    FakeTokens,
    Reports,
    grant,
    jwt_grant,
    line,
    read_of,
    window_of,
)
from tests.support.hopin import active, invited

pytestmark = pytest.mark.anyio


async def test_an_invitation_is_accepted_once_with_its_place_invitation_and_epoch(scenario):
    run = scenario([read_of(invited("room-7", "inv-3", epoch=2)), read_of(active("room-7"))])

    await run.play()

    assert run.session.accepted == [Accept(PlaceRef(kind="call", id="room-7"), "inv-3", 2)]
    assert run.reports.events() == [("started", None), ("joined", None)]


async def test_a_stale_invitation_is_not_retried_and_is_reported_once(scenario):
    standing = invited("room-7", "inv-3")
    run = scenario(
        [read_of(standing), read_of(standing), read_of(standing)],
        accepts=[Refusal.STALE_INVITATION],
    )

    await run.play()

    assert len(run.session.accepted) == 1
    assert run.reports.events() == [("started", None), ("refused", "stale_invitation")]


@pytest.mark.parametrize("refusal", [Refusal.SPONSOR_AWAY, Refusal.TWO_SEATS, Refusal.BUSY])
async def test_a_transient_refusal_is_retried_at_the_next_answer_and_reported_once(scenario, refusal):
    standing = invited("room-7", "inv-3")
    run = scenario(
        [read_of(standing), read_of(standing), read_of(standing), read_of(active("room-7"))],
        accepts=[refusal, refusal, None],
    )

    await run.play()

    assert len(run.session.accepted) == 3
    assert run.reports.events() == [("started", None), ("refused", refusal.value), ("joined", None)]


async def test_a_token_is_renewed_before_it_expires_and_never_sent_expired(scenario):
    run = scenario([read_of() for n in range(8)], seconds_per_read=100.0)

    await run.play()

    assert run.tokens.fetched > 1
    for bearer, sent_at in run.session.bearers_with_time:
        assert sent_at < run.tokens.expires_at[bearer] - RENEW_BEFORE_EXPIRY_SECONDS


async def test_an_accept_never_carries_a_bearer_that_expired_while_the_read_and_earlier_accepts_ran(scenario):
    invitations = [invited(f"room-{n}", f"inv-{n}") for n in range(3)]
    run = scenario([read_of(*invitations)], seconds_per_accept=150.0)

    await run.play()

    assert len(run.session.accepted) == 3
    for bearer, sent_at in run.session.bearers_with_time:
        assert sent_at < run.tokens.expires_at[bearer] - RENEW_BEFORE_EXPIRY_SECONDS


async def test_a_refused_token_suspends_the_loop_without_an_app_call_and_logs_it_once(scenario):
    refused = [TokenStatusFailure(401)] * 3
    run = scenario([read_of()], tokens=refused)

    await run.play()

    assert run.session.opened == 0
    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS] * 3
    assert run.reports.events() == [("started", None), ("suspended", "http_401")]


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (StatusFailure(403), "http_403"),
        (StatusFailure(400), "http_400"),
        (UnusableAnswer("jsonrpc_-32602"), "jsonrpc_-32602"),
        (UnusableAnswer("http_307"), "http_307"),
    ],
    ids=["403", "400", "protocol error", "redirect"],
)
async def test_an_app_refusal_suspends_the_loop_and_it_resumes_when_a_read_answers(scenario, failure, code):
    run = scenario([failure, failure, read_of(), read_of()])

    await run.play()

    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS] * 2
    assert run.reports.events() == [("started", None), ("suspended", code), ("resumed", None)]
    assert run.tokens.fetched == 3


async def test_a_401_renews_the_token_once_and_a_second_one_suspends_the_loop(scenario):
    run = scenario([StatusFailure(401), StatusFailure(401), read_of()])

    await run.play()

    assert run.tokens.fetched == 3
    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert run.reports.events() == [("started", None), ("suspended", "http_401"), ("resumed", None)]


async def test_a_401_the_renewed_token_cures_suspends_nothing(scenario):
    run = scenario([StatusFailure(401), read_of()])

    await run.play()

    assert run.clock.slept == []
    assert run.reports.events() == [("started", None)]


async def test_a_loop_resumed_after_a_suspension_joins_only_when_invited_anew(scenario):
    run = scenario(
        [StatusFailure(403), read_of(active("room-7")), read_of(invited("room-7", "inv-9"))],
    )

    await run.play()

    assert run.reports.events() == [
        ("started", None),
        ("suspended", "http_403"),
        ("resumed", None),
        ("joined", None),
    ]
    assert run.session.accepted == [Accept(PlaceRef(kind="call", id="room-7"), "inv-9", 0)]


@pytest.mark.parametrize(
    ("failure", "code"),
    [(Unreachable("down"), "network"), (StatusFailure(503), "http_503")],
    ids=["network", "5xx from the app"],
)
async def test_an_app_failure_that_is_not_a_refusal_backs_off_with_a_capped_growing_interval_and_logs_it_once(
    scenario, failure, code
):
    failures = 40
    run = scenario([failure] * failures)

    await run.play()

    assert run.clock.slept == [backoff_delay(attempt) for attempt in range(1, failures + 1)]
    assert max(run.clock.slept) == BACKOFF_CAP_SECONDS
    assert min(run.clock.slept) > 0
    assert run.reports.events() == [("started", None), ("backoff", code)]


async def test_a_realm_that_does_not_answer_or_fails_backs_off_without_suspending(scenario):
    failures = [Unreachable("down"), TokenStatusFailure(502), Unreachable("down")]
    run = scenario([], tokens=failures)

    await run.play()

    assert run.clock.slept == [backoff_delay(1), backoff_delay(2), backoff_delay(3)]
    assert run.reports.events() == [("started", None), ("backoff", "network")]


async def test_the_backoff_starts_over_after_an_answer(scenario):
    run = scenario([Unreachable("down"), Unreachable("down"), read_of(), Unreachable("down")])

    await run.play()

    assert run.clock.slept == [backoff_delay(1), backoff_delay(2), backoff_delay(1)]


async def test_an_accept_the_app_cannot_answer_suspends_the_loop_with_its_code_and_does_not_retry_at_once(scenario):
    standing = invited("room-7", "inv-3")
    failure = UnusableAnswer("jsonrpc_-32602")
    run = scenario([read_of(standing) for n in range(3)], accepts=[failure, failure, None])

    await run.play()

    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS] * 2
    assert run.reports.events() == [
        ("started", None),
        ("suspended", "jsonrpc_-32602"),
        ("resumed", None),
        ("suspended", "jsonrpc_-32602"),
        ("resumed", None),
        ("joined", None),
    ]


async def test_a_loop_resumes_once_a_read_answers_though_the_accept_after_it_fails(scenario):
    standing = invited("room-7", "inv-3")
    run = scenario(
        [UnusableAnswer("jsonrpc_-32602"), read_of(standing), read_of(standing)],
        accepts=[StatusFailure(503), None],
    )

    await run.play()

    assert run.reports.events() == [
        ("started", None),
        ("suspended", "jsonrpc_-32602"),
        ("resumed", None),
        ("backoff", "http_503"),
        ("resumed", None),
        ("joined", None),
    ]


async def test_an_accept_that_answers_401_again_after_a_renewal_suspends_the_loop_though_a_wait_succeeded_between(
    scenario,
):
    standing = invited("room-7", "inv-3")
    unauthorised = StatusFailure(401)
    run = scenario([read_of(standing) for n in range(3)], accepts=[unauthorised, unauthorised, None])

    await run.play()

    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert run.reports.events()[:2] == [("started", None), ("suspended", "http_401")]


async def test_an_accept_that_fails_on_the_network_backs_off_and_does_not_retry_every_second(scenario):
    standing = invited("room-7", "inv-3")
    run = scenario([read_of(standing) for n in range(4)], accepts=[Unreachable("down")] * 3 + [None])

    await run.play()

    assert run.clock.slept == [backoff_delay(1), backoff_delay(2), backoff_delay(3)]


async def test_the_marker_is_asked_for_when_a_read_answers_and_not_before(scenario):
    run = scenario([Unreachable("down"), StatusFailure(403), read_of(), read_of()])

    await run.play()

    assert run.ready.announced == 2


@pytest.mark.parametrize(
    "failure",
    [StatusFailure(403), StatusFailure(503), Unreachable("down")],
    ids=["refusal", "5xx", "network"],
)
async def test_no_marker_is_asked_for_while_no_read_has_answered(scenario, failure):
    run = scenario([failure, failure])

    await run.play()

    assert run.ready.announced == 0


async def test_a_failure_after_the_first_answer_does_not_take_the_marker_back(scenario):
    run = scenario([read_of(), StatusFailure(403), Unreachable("down")])

    await run.play()

    assert run.ready.announced == 1
    assert run.ready.cleared == 0


async def test_a_backoff_is_logged_once_for_its_retries_and_again_after_the_loop_resumed(scenario):
    run = scenario(
        [Unreachable("down"), StatusFailure(502), Unreachable("down"), read_of(), StatusFailure(503), read_of()]
    )

    await run.play()

    assert run.reports.events() == [
        ("started", None),
        ("backoff", "network"),
        ("resumed", None),
        ("backoff", "http_503"),
        ("resumed", None),
    ]


async def test_a_backoff_after_a_suspension_is_logged_and_one_answer_resumes_both(scenario):
    run = scenario([StatusFailure(403), Unreachable("down"), read_of()])

    await run.play()

    assert run.reports.events() == [
        ("started", None),
        ("suspended", "http_403"),
        ("backoff", "network"),
        ("resumed", None),
    ]


async def test_a_stream_reopened_after_a_renewal_carries_the_new_token_and_each_call_a_fresh_one(scenario):
    run = scenario([StatusFailure(401), read_of(invited("room-7", "inv-3"))], seconds_per_read=100.0)

    await run.play()

    assert run.session.opened == 2
    assert run.session.opened_with[0] != run.session.opened_with[1]
    for bearer, sent_at in run.session.bearers_with_time:
        assert sent_at < run.tokens.expires_at[bearer] - RENEW_BEFORE_EXPIRY_SECONDS


async def test_a_stream_reopened_after_a_graceful_end_carries_the_token_renewed_since(scenario):
    run = scenario([read_of(), read_of()], seconds_per_read=grant().lifetime_seconds, graceful_ends=1)

    await run.play()

    assert len(run.session.entered_with_time) == 2
    assert run.session.opened == 1
    for bearer, entered_at in run.session.entered_with_time:
        assert entered_at < run.tokens.expires_at[bearer] - RENEW_BEFORE_EXPIRY_SECONDS


async def test_a_place_gone_between_the_read_and_the_accept_is_the_refusal_and_no_failure(scenario):
    run = scenario([read_of(invited("room-7", "inv-3")), read_of()], accepts=[Refusal.STALE_INVITATION])

    await run.play()

    assert run.reports.events() == [("started", None), ("refused", "stale_invitation")]
    assert run.clock.slept == []


async def test_a_refusal_and_a_network_failure_leaving_one_session_together_suspend_the_loop_once(scenario):
    together = ExceptionGroup("the session failed", [Unreachable("down"), StatusFailure(403)])
    run = scenario([together, read_of()])

    await run.play()

    assert run.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert run.reports.events() == [("started", None), ("suspended", "http_403"), ("resumed", None)]


async def test_a_failure_together_with_a_programming_error_ends_the_runner_with_both_and_no_reaction(scenario):
    failure, defect = Unreachable("down"), ValueError("a defect")
    run = scenario([ExceptionGroup("the session failed", [failure, defect]), read_of()])

    with pytest.raises(ExceptionGroup) as ended:
        await run.play()

    assert list(leaves_of(ended.value)) == [failure, defect]
    assert run.clock.slept == []
    assert run.reports.events() == [("started", None)]


class _OneWake:
    """A stream that wakes once and then stays silent."""

    def __init__(self) -> None:
        self._woken = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> None:
        if self._woken:
            await asyncio.Event().wait()
        self._woken = True


class TurnAndLifecycle:
    """One loop whose turn and lifecycle both wait for one release and then fail in the same step.

    The lifecycle reads the standing once, which starts a turn on an addressed line, and waits at its second
    read. The turn waits at the opening of its session. A third session, after a reaction, waits at its first read.
    """

    def __init__(self, files, lifecycle_failure: Exception, turn_failure: Exception) -> None:
        config = load_config(files.write())
        self.release = asyncio.Event()
        self.turn_waiting = asyncio.Event()
        self.lifecycle_waiting = asyncio.Event()
        self.reopened = asyncio.Event()
        self.clock = FakeClock()
        self.reports = Reports()
        self._opened = 0
        self._reads = 0
        self._windows = [window_of(line(1, "Mozart?"))]
        self._lifecycle_failure = lifecycle_failure
        self._turn_failure = turn_failure
        tokens = FakeTokens(self.clock, [jwt_grant() for _ in range(5)])
        boundaries = Boundaries(tokens, self.open, self.clock, self.reports, FakeReadyMarker(), backend=self._no_model)
        self.loop = AgentLoop(config.agents[0], config.apps[0], boundaries)

    @asynccontextmanager
    async def open(self, app, token):
        self._opened += 1
        if self._opened == 2:
            self.turn_waiting.set()
            await self.release.wait()
            raise self._turn_failure
        yield self

    def use_token(self, token) -> None:
        pass

    @asynccontextmanager
    async def changes(self):
        yield _OneWake()

    async def read_standing(self):
        self._reads += 1
        if self._reads == 2:
            await self.turn_waiting.wait()
            self.lifecycle_waiting.set()
            await self.release.wait()
            raise self._lifecycle_failure
        if self._reads > 2:
            self.reopened.set()
            await asyncio.Event().wait()
        return read_of(active("room-7"))

    async def read_place(self, place):
        return self._windows.pop(0)

    def _no_model(self, tools):
        raise AssertionError("the turn fails before it asks the model")


@pytest.mark.parametrize(
    ("lifecycle_failure", "turn_failure"),
    [(Unreachable("down"), StatusFailure(403)), (StatusFailure(403), Unreachable("down"))],
    ids=["the turn is refused", "the lifecycle is refused"],
)
async def test_a_lifecycle_and_a_turn_failing_together_give_one_reaction_and_a_refusal_wins(
    files, lifecycle_failure, turn_failure
):
    together = TurnAndLifecycle(files, lifecycle_failure, turn_failure)
    task = asyncio.create_task(together.loop.run())
    try:
        async with asyncio.timeout(10):
            await together.lifecycle_waiting.wait()
            together.release.set()
            await together.reopened.wait()
    finally:
        task.cancel()

    assert together.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert together.reports.events() == [("started", None), ("suspended", "http_403")]


async def test_a_failure_of_a_call_and_a_defect_together_give_no_reaction_and_every_leaf_ends_the_runner(files):
    failure, defect = Unreachable("down"), ValueError("a defect")
    together = TurnAndLifecycle(files, failure, defect)
    task = asyncio.create_task(together.loop.run())
    async with asyncio.timeout(10):
        await together.lifecycle_waiting.wait()
        together.release.set()
        with pytest.raises(ExceptionGroup) as ended:
            await task

    assert sorted(leaves_of(ended.value), key=repr) == sorted([failure, defect], key=repr)
    assert together.clock.slept == []
    assert together.reports.events() == [("started", None)]
