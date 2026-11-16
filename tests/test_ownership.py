"""Which pod serves an agent: only while it holds the agent's lock, only after the grace, and never after the hold is
lost; the pod's lock connection is scripted and its clock is the fake one."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from personas.config import OwnershipTimings
from personas.ownership import LockConnectionLost, LockedOwnership
from personas.reporting import Event
from tests.support.fakes import FakeClock, Reports
from tests.support.hopin import LIMIT_SECONDS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import NoReturn

pytestmark = pytest.mark.anyio

AGENT = "agent-mozart"
OTHER_AGENT = "agent-ada"
TIMINGS = OwnershipTimings(
    contend_seconds=2.0, heartbeat_timeout_seconds=1.0, cancel_margin_seconds=0.5, grace_seconds=7.0
)
LOST = LockConnectionLost()


@dataclass(frozen=True)
class After:
    """A round that answers once these agents are being served: what the pod does between two rounds."""

    served: tuple[str, ...]
    answer: set[str] | LockConnectionLost


class ScriptedLocks:
    """The lock connection as a script: each round answers with the agents held, or loses the connection.

    The round after the script blocks, which is its end. A round takes the seconds the test names on the clock, and
    each round is noted in ``trail`` when the test gives one.
    """

    def __init__(
        self,
        clock: FakeClock,
        *rounds: set[str] | LockConnectionLost | After,
        seconds_per_round: float = 0.0,
        trail: list[tuple[str, str]] | None = None,
    ) -> None:
        self._clock = clock
        self._trail = trail
        self._rounds = list(rounds)
        self._seconds_per_round = seconds_per_round
        self.spent = asyncio.Event()
        self.asked: list[Sequence[str]] = []
        self.served: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        """An event for each agent, set when it is served."""

    async def hold(self, client_ids: Sequence[str]) -> frozenset[str]:
        self.asked.append(client_ids)
        if self._trail is not None:
            self._trail.append(("hold", ",".join(client_ids)))
        self._clock.time += self._seconds_per_round
        if not self._rounds:
            self.spent.set()
            await asyncio.Event().wait()
        answer = self._rounds.pop(0)
        if isinstance(answer, After):
            for each in answer.served:
                await self.served[each].wait()
            answer = answer.answer
        if isinstance(answer, LockConnectionLost):
            raise answer
        return frozenset(answer)


class GraceGatedClock(FakeClock):
    """A fake clock whose sleep of the grace waits until the test opens it; every other sleep returns at once."""

    def __init__(self, grace_seconds: float) -> None:
        super().__init__()
        self._grace_seconds = grace_seconds
        self.opened = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        if seconds == self._grace_seconds:
            self.slept.append(seconds)
            await self.opened.wait()
            return
        await super().sleep(seconds)


def _open_gate() -> asyncio.Event:
    gate = asyncio.Event()
    gate.set()
    return gate


@dataclass
class Pod:
    """A pod's ownership over a scripted lock connection, serving agents that record when they start and end.

    An agent that is cancelled sets its ``winding_down`` event and ends once ``unwinding`` is set, which a test closes
    to hold the agents in the middle of stopping.
    """

    clock: FakeClock
    locks: ScriptedLocks
    agents: Sequence[str] = (AGENT,)
    defect_of: dict[str, Exception] = field(default_factory=dict[str, Exception])
    events: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    reports: Reports = field(default_factory=Reports)
    unwinding: asyncio.Event = field(default_factory=_open_gate)
    winding_down: defaultdict[str, asyncio.Event] = field(default_factory=lambda: defaultdict(asyncio.Event))
    contending: int = 0

    def __post_init__(self) -> None:
        self.ownership = LockedOwnership(self.locks, self.clock, TIMINGS, self.reports)

    async def _serve_agent(self, client_id: str) -> NoReturn:
        self.events.append(("started", client_id))
        self.locks.served[client_id].set()
        try:
            if client_id in self.defect_of:
                raise self.defect_of[client_id]
            await asyncio.Event().wait()
            raise AssertionError("an agent ended")  # pragma: no cover
        finally:
            self.winding_down[client_id].set()
            await self.unwinding.wait()
            self.events.append(("ended", client_id))

    def _contending(self) -> None:
        self.contending += 1

    def start(self) -> asyncio.Task[NoReturn]:
        return asyncio.create_task(self.ownership.serve(self.agents, self._serve_agent, self._contending))

    async def until_spent(self, *served: str) -> None:
        """Returns once the script is spent and these agents are served."""
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.locks.spent.wait()
            for each in served:
                await self.locks.served[each].wait()

    async def stopped(self, task: asyncio.Task[NoReturn]) -> BaseException:
        task.cancel()
        async with asyncio.timeout(LIMIT_SECONDS):
            (ended,) = await asyncio.gather(task, return_exceptions=True)
        return ended


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


async def test_an_agent_whose_lock_is_held_is_served_and_one_whose_lock_another_pod_holds_is_not(clock):
    pod = Pod(clock, ScriptedLocks(clock, {AGENT}), agents=(AGENT, OTHER_AGENT))
    task = pod.start()
    await pod.until_spent(AGENT)

    await pod.stopped(task)

    assert pod.events == [("started", AGENT), ("ended", AGENT)]
    assert list(pod.locks.asked[0]) == [AGENT, OTHER_AGENT]


async def test_an_agent_is_not_served_before_the_grace_has_passed_after_its_lock_was_taken():
    clock = GraceGatedClock(TIMINGS.grace_seconds)
    pod = Pod(clock, ScriptedLocks(clock, {AGENT}, {AGENT}, {AGENT}))
    task = pod.start()
    await pod.until_spent()

    before_the_grace = list(pod.events)
    clock.opened.set()
    async with asyncio.timeout(LIMIT_SECONDS):
        await pod.locks.served[AGENT].wait()
    await pod.stopped(task)

    assert before_the_grace == []


def rounds_in(events: list[tuple[str, str]]) -> list[int]:
    """The positions in the events at which a round of the lock connection began."""
    return [position for position, (what, _) in enumerate(events) if what == "hold"]


async def test_an_agent_whose_lock_is_no_longer_held_is_stopped_and_has_ended_before_the_next_round(clock):
    events: list[tuple[str, str]] = []
    unwinding = asyncio.Event()
    locks = ScriptedLocks(clock, {AGENT}, After((AGENT,), set()), trail=events)
    pod = Pod(clock, locks, events=events, unwinding=unwinding)
    task = pod.start()
    async with asyncio.timeout(LIMIT_SECONDS):
        await pod.winding_down[AGENT].wait()
    reported_while_stopping = pod.reports.reported()
    unwinding.set()
    await pod.until_spent()

    await pod.stopped(task)

    assert reported_while_stopping == [(Event.TAKEN, None)]
    assert ("ended", AGENT) in events[: rounds_in(events)[2]]
    assert pod.reports.reported() == [(Event.TAKEN, None), (Event.LOST, "hold_lost")]


async def test_a_lost_lock_connection_stops_every_agent_and_the_pod_takes_them_again_when_it_holds_them(clock):
    events: list[tuple[str, str]] = []
    unwinding = asyncio.Event()
    locks = ScriptedLocks(clock, {AGENT, OTHER_AGENT}, After((AGENT, OTHER_AGENT), LOST), {AGENT}, trail=events)
    pod = Pod(clock, locks, agents=(AGENT, OTHER_AGENT), events=events, unwinding=unwinding)
    task = pod.start()
    async with asyncio.timeout(LIMIT_SECONDS):
        await pod.winding_down[AGENT].wait()
        await pod.winding_down[OTHER_AGENT].wait()
    unwinding.set()
    await pod.until_spent()

    await pod.stopped(task)

    before_the_third_round = events[: rounds_in(events)[2]]
    assert ("ended", AGENT) in before_the_third_round
    assert ("ended", OTHER_AGENT) in before_the_third_round
    assert [each for each in events if each[0] != "hold"] == [
        ("started", AGENT),
        ("started", OTHER_AGENT),
        ("ended", AGENT),
        ("ended", OTHER_AGENT),
        ("started", AGENT),
        ("ended", AGENT),
    ]


async def test_a_round_starts_one_contend_interval_after_the_start_of_the_one_before(clock):
    pod = Pod(clock, ScriptedLocks(clock, set(), set(), seconds_per_round=0.75))
    task = pod.start()
    await pod.until_spent()

    await pod.stopped(task)

    assert clock.slept == [TIMINGS.contend_seconds - 0.75, TIMINGS.contend_seconds - 0.75]


async def test_a_pod_whose_lock_connection_never_answers_is_not_contending(clock):
    pod = Pod(clock, ScriptedLocks(clock, LOST, LOST))
    task = pod.start()
    await pod.until_spent()

    await pod.stopped(task)

    assert pod.contending == 0


async def test_an_outage_of_the_lock_connection_is_reported_once_and_so_is_its_end(clock):
    pod = Pod(clock, ScriptedLocks(clock, LOST, LOST, set(), set()))
    task = pod.start()
    await pod.until_spent()

    await pod.stopped(task)

    assert pod.reports.reported() == [(Event.STORE_FAILED, "store_unreachable"), (Event.STORE_RESUMED, None)]


async def test_a_defect_of_one_agent_ends_the_pod_and_stops_the_other_agents(clock):
    defect = ValueError("a defect")
    pod = Pod(clock, ScriptedLocks(clock, {AGENT, OTHER_AGENT}), agents=(AGENT, OTHER_AGENT), defect_of={AGENT: defect})
    task = pod.start()

    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(ended, ExceptionGroup)
    assert ended.exceptions == (defect,)
    assert ("ended", OTHER_AGENT) in pod.events


async def test_a_cancelled_pod_ends_cancelled_after_every_agent_has_ended(clock):
    pod = Pod(clock, ScriptedLocks(clock, {AGENT, OTHER_AGENT}), agents=(AGENT, OTHER_AGENT))
    task = pod.start()
    await pod.until_spent(AGENT, OTHER_AGENT)

    ended = await pod.stopped(task)

    assert isinstance(ended, asyncio.CancelledError)
    assert set(pod.events[2:]) == {("ended", AGENT), ("ended", OTHER_AGENT)}
