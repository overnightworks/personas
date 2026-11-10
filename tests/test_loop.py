"""The loop of one agent in one app, over scripted boundaries and a clock that moves only when told."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import anyio
import pytest
from agents.backend import TurnStopKind, TurnStopped
from agents.backend.gateway import GatewayBackend
from agents.member import AcceptInvitationRefusal, PlaceRef
from agents.tools import ToolDeclaration, ToolEffect
from agents.turn import RejectionCode

from personas.config import load_config
from personas.decision import (
    BACKOFF_CAP_SECONDS,
    RENEW_BEFORE_EXPIRY_SECONDS,
    SUSPENDED_RETRY_SECONDS,
    Accept,
    backoff_delay,
)
from personas.failures import StatusFailure, TokenStatusFailure, Unreachable, UnusableAnswer, leaves_of
from personas.loop import AgentLoop, Boundaries, TurnCounter
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
from tests.support.gateway import Gateway, says
from tests.support.hopin import LIMIT_SECONDS, active, invited

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
        accepts=[AcceptInvitationRefusal.STALE_INVITATION],
    )

    await run.play()

    assert len(run.session.accepted) == 1
    assert run.reports.events() == [("started", None), ("refused", "stale_invitation")]


@pytest.mark.parametrize(
    "refusal", [AcceptInvitationRefusal.SPONSOR_AWAY, AcceptInvitationRefusal.TWO_SEATS, AcceptInvitationRefusal.BUSY]
)
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
    run = scenario([read_of(invited("room-7", "inv-3")), read_of()], accepts=[AcceptInvitationRefusal.STALE_INVITATION])

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


class _Wakes:
    """A stream that wakes a given number of times and then stays silent."""

    def __init__(self, count: int) -> None:
        self._left = count

    def __aiter__(self):
        return self

    async def __anext__(self) -> None:
        if not self._left:
            await asyncio.Event().wait()
        self._left -= 1


class ScriptedSessions:
    """One loop over a stand-in that is the app's session; a subclass scripts what `open` and the reads do.

    The place is invited at the first read and active at every later one, so the agent joins it and reads its window
    from the first line; two wakes follow the first read.
    """

    def __init__(self, files, backend) -> None:
        self.config = load_config(files.write())
        self.clock = FakeClock()
        self.reports = Reports()
        self._windows = [window_of(line(1, "Mozart?"))]
        tokens = FakeTokens(self.clock, [jwt_grant() for _ in range(5)])
        boundaries = Boundaries(tokens, self.open, self.clock, self.reports, FakeReadyMarker(), backend=backend)
        self.loop = AgentLoop(self.config.agents[0], self.config.apps[0], boundaries, TurnCounter())

    def use_token(self, token) -> None:
        pass

    @asynccontextmanager
    async def changes(self):
        yield _Wakes(2)

    async def accept(self, accept):
        return None

    def standing_of_a_joined_place(self, reads: int):
        return read_of(invited("room-7", "inv-3") if reads == 1 else active("room-7"))

    async def read_place(self, place):
        return self._windows.pop(0)


class TurnAndLifecycle(ScriptedSessions):
    """One loop whose turn and lifecycle both wait for one release and then fail in the same step.

    The lifecycle joins at its first read and reads the standing again, which starts a turn on an addressed line,
    and waits at its third read. The turn waits at the opening of its session. A third session, after a reaction,
    waits at its first read.
    """

    def __init__(self, files, lifecycle_failure: Exception, turn_failure: Exception) -> None:
        self.release = asyncio.Event()
        self.turn_waiting = asyncio.Event()
        self.lifecycle_waiting = asyncio.Event()
        self.reopened = asyncio.Event()
        self._opened = 0
        self._reads = 0
        self._lifecycle_failure = lifecycle_failure
        self._turn_failure = turn_failure
        super().__init__(files, self._no_model)

    @asynccontextmanager
    async def open(self, app, token):
        self._opened += 1
        if self._opened == 2:
            self.turn_waiting.set()
            await self.release.wait()
            raise self._turn_failure
        yield self

    async def read_standing(self):
        self._reads += 1
        if self._reads == 3:
            await self.turn_waiting.wait()
            self.lifecycle_waiting.set()
            await self.release.wait()
            raise self._lifecycle_failure
        if self._reads > 3:
            self.reopened.set()
            await asyncio.Event().wait()
        return self.standing_of_a_joined_place(self._reads)

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
        await asyncio.gather(task, return_exceptions=True)

    assert together.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert together.reports.events() == [("started", None), ("joined", None), ("suspended", "http_403")]


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
    assert together.reports.events() == [("started", None), ("joined", None)]


class SwallowedCancellation(TurnAndLifecycle):
    """A turn whose session opener swallows the cancellation it is given and hands back ``outcome`` instead.

    The SDK's task group can do that; ``outcome`` is raised, or the session is handed back when it is ``None``.
    """

    def __init__(self, files, outcome: Exception | None) -> None:
        super().__init__(files, Unreachable("down"), Unreachable("down"))
        self.outcome = outcome
        self.relayed: list[str] = []

    @asynccontextmanager
    async def open(self, app, token):
        self._opened += 1
        if self._opened == 2:
            self.turn_waiting.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                if self.outcome is not None:
                    raise self.outcome from None
        yield self

    async def declared_tools(self):
        self.relayed.append("declared_tools")
        return ()


@pytest.mark.parametrize(
    "outcome",
    [
        Unreachable("down"),
        TurnStopped(TurnStopKind.REFUSED, RejectionCode.BUSY),
        anyio.BrokenResourceError(),
        anyio.ClosedResourceError(),
        None,
    ],
    ids=["a connection failure", "a stop", "a broken stream", "a closed stream", "the session itself"],
)
async def test_a_cancelled_turn_whose_opener_swallowed_it_reports_nothing_calls_nothing_and_leaves_no_reaction(
    files, outcome
):
    swallowed = SwallowedCancellation(files, outcome)
    task = asyncio.create_task(swallowed.loop.run())
    async with asyncio.timeout(10):
        await swallowed.lifecycle_waiting.wait()
    task.cancel()
    async with asyncio.timeout(10):
        (ended,) = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(ended, asyncio.CancelledError)
    assert swallowed.relayed == []
    assert swallowed.clock.slept == []
    assert swallowed.reports.events() == [("started", None), ("joined", None)]


class _TurnThatEnds:
    """A backend whose turn ends at once, or after a relayed call that fails."""

    def __init__(self, tools, *, relays: bool) -> None:
        self._tools = tools
        self._relays = relays

    async def run(self, request, person):
        if self._relays:
            await self._tools[0].handler(person.principal, {})
        for event in ():
            yield event


class TurnEndingInTheStepItsPlaceLeaves(TurnAndLifecycle):
    """A turn that ends and a lifecycle whose read shows the turn's place gone, released in one step.

    The turn runs first and ends, with a stop, normally or with a failed relayed call; the lifecycle's read is
    answered before the turn's end is handled.
    """

    def __init__(self, files, ending: str) -> None:
        super().__init__(files, Unreachable("down"), Unreachable("down"))
        self._ending = ending

    @asynccontextmanager
    async def open(self, app, token):
        self._opened += 1
        if self._opened == 2:
            self.turn_waiting.set()
            await self.release.wait()
            if self._ending == "stop":
                raise TurnStopped(TurnStopKind.REFUSED, RejectionCode.BUSY)
        yield self

    async def declared_tools(self):
        return (
            ToolDeclaration(
                name="say", description="Says a line.", parameters={"type": "object"}, effect=ToolEffect.SIDE_EFFECTING
            ),
        )

    async def invoke(self, name, arguments):
        raise Unreachable("down")

    def _no_model(self, tools):
        return _TurnThatEnds(tools, relays=self._ending == "relayed failure")

    async def read_standing(self):
        self._reads += 1
        if self._reads == 3:
            await self.turn_waiting.wait()
            self.lifecycle_waiting.set()
            await self.release.wait()
            return read_of()
        if self._reads > 3:
            self.reopened.set()
            await asyncio.Event().wait()
        return self.standing_of_a_joined_place(self._reads)


@pytest.mark.parametrize(
    ("ending", "own_reports", "sleeps"),
    [
        ("stop", [("turn_ended", "busy")], []),
        ("completion", [], []),
        ("relayed failure", [], [backoff_delay(1)]),
    ],
    ids=["a stop it reported", "a completion", "a failed relayed call"],
)
async def test_a_turn_that_ends_on_its_own_in_the_step_its_place_leaves_reports_only_its_own_end(
    files, ending, own_reports, sleeps
):
    together = TurnEndingInTheStepItsPlaceLeaves(files, ending)
    task = asyncio.create_task(together.loop.run())
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await together.lifecycle_waiting.wait()
            together.release.set()
            await together.reopened.wait()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert [event for event in together.reports.events() if event[0] == "turn_ended"] == own_reports
    assert together.clock.slept == sleeps


class TokenFetchSwallowingTheCancellation:
    """A token source whose first fetch swallows the cancellation it is given and fails with a connection failure.

    The loop fetches before it enters any task group, which would otherwise ask for the cancellation again.
    """

    def __init__(self, files) -> None:
        self.fetching = asyncio.Event()
        self.fetches = 0
        config = load_config(files.write())
        boundaries = Boundaries(self, self._no_session, FakeClock(), Reports(), FakeReadyMarker())
        self.loop = AgentLoop(config.agents[0], config.apps[0], boundaries, TurnCounter())

    async def fetch(self, agent, scope):
        self.fetches += 1
        self.fetching.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if self.fetches > 1:
                raise
            raise Unreachable("down") from None

    def _no_session(self, app, token):
        raise AssertionError("no session opens before a token")


async def test_a_loop_cancelled_while_a_fetch_swallows_the_cancellation_ends_and_fetches_no_second_token(files):
    swallowed = TokenFetchSwallowingTheCancellation(files)
    task = asyncio.create_task(swallowed.loop.run())
    async with asyncio.timeout(LIMIT_SECONDS):
        await swallowed.fetching.wait()
    task.cancel()
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(ended, asyncio.CancelledError)
    assert swallowed.fetches == 1


class _WakesFailingAfterOne:
    """A stream that wakes once and then fails with ``failure`` at its next wait, after telling ``raised``."""

    def __init__(self, failure: Exception, raised: asyncio.Event) -> None:
        self._failure = failure
        self._raised = raised
        self._woken = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> None:
        if self._woken:
            self._raised.set()
            raise self._failure
        self._woken = True


class WakesFailingWhileTheReadFails(ScriptedSessions):
    """A stream that fails with ``wakes_failure`` while the second read waits, which then fails with ``read_failure``.

    A third read, after a reaction, waits and tells ``reopened``.
    """

    def __init__(self, files, wakes_failure: Exception, read_failure: Exception) -> None:
        self.raised = asyncio.Event()
        self.reopened = asyncio.Event()
        self._wakes_failure = wakes_failure
        self._read_failure = read_failure
        self._reads = 0
        super().__init__(files, None)

    @asynccontextmanager
    async def open(self, app, token):
        yield self

    @asynccontextmanager
    async def changes(self):
        yield _WakesFailingAfterOne(self._wakes_failure, self.raised)

    async def read_standing(self):
        self._reads += 1
        if self._reads == 1:
            return read_of()
        if self._reads > 2:
            self.reopened.set()
            await asyncio.Event().wait()
        await self.raised.wait()
        raise self._read_failure


async def test_a_defect_of_the_wake_stream_that_loses_the_race_to_a_failing_read_ends_the_runner_loud(files):
    defect = ValueError("a defect")
    failing = WakesFailingWhileTheReadFails(files, defect, Unreachable("down"))
    task = asyncio.create_task(failing.loop.run())
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(ended, ExceptionGroup)
    assert list(leaves_of(ended)) == [defect]
    assert failing.clock.slept == []


async def test_a_failure_of_the_wake_stream_that_loses_the_race_to_a_failing_read_leaves_the_read_to_decide(files):
    failing = WakesFailingWhileTheReadFails(files, Unreachable("down"), StatusFailure(403))
    task = asyncio.create_task(failing.loop.run())
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await failing.reopened.wait()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert failing.clock.slept == [SUSPENDED_RETRY_SECONDS]
    assert failing.reports.events() == [("started", None), ("suspended", "http_403")]


class DefectInARelayedCall(ScriptedSessions):
    """One agent whose model calls ``say`` once in the one addressed line; the relayed call raises ``defect``.

    A ``late`` failure leaves the turn's session beside what its body raised, as a late answer of its teardown does.
    """

    def __init__(self, files, defect: Exception, late: Exception | None = None) -> None:
        self.gateway = Gateway(*says("room-7", "never said"))
        self.requests_at_failure = 0
        self._defect = defect
        self._late = late
        self._reads = 0
        super().__init__(files, self._gateway_backend)

    def _gateway_backend(self, tools):
        assert self.config.gateway is not None
        return GatewayBackend(self.config.gateway, tools=tools, transport=self.gateway)

    @asynccontextmanager
    async def open(self, app, token):
        try:
            yield self
        except Exception as body:
            if self._late is None:
                raise
            raise BaseExceptionGroup("the session's teardown", [body, self._late]) from None

    async def read_standing(self):
        self._reads += 1
        if self._reads > 2:
            await asyncio.Event().wait()
        return self.standing_of_a_joined_place(self._reads)

    async def declared_tools(self):
        return (
            ToolDeclaration(
                name="say", description="Says a line.", parameters={"type": "object"}, effect=ToolEffect.SIDE_EFFECTING
            ),
        )

    async def invoke(self, name, arguments):
        self.requests_at_failure = len(self.gateway.requests)
        raise self._defect


@pytest.mark.parametrize("beside", [[], [Unreachable("late answer")]], ids=["alone", "beside a late answer"])
async def test_a_relayed_call_that_raises_a_defect_ends_the_runner_loud_with_every_leaf_and_no_second_round(
    files, library_loggers, beside
):
    defect = ValueError("a defect")
    raising = DefectInARelayedCall(files, defect, *beside)
    task = asyncio.create_task(raising.loop.run())
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(ended, ExceptionGroup)
    assert list(leaves_of(ended)) == [defect, *beside]
    assert raising.requests_at_failure == 1
    assert len(raising.gateway.requests) == 1
    assert raising.clock.slept == []
    assert raising.reports.events() == [("started", None), ("joined", None)]
