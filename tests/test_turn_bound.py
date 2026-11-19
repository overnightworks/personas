from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import anyio
import pytest
from agents.backend import TurnStopped
from agents.member import Interrupted, PartialSpeech, PlaceRef, PlaceWindow, TalkedOver
from agents.turn import FailureCode

from personas.decision import DEFAULT_BUDGET_PAUSE_CAP_SECONDS, AppPlace
from personas.entries import KeptLine, Visit
from personas.mind import Mind
from personas.speech import Held
from personas.turn_bound import TurnBound
from tests.support.fakes import FakeClock

pytestmark = pytest.mark.anyio

AT = AppPlace("https://app.test/mcp", PlaceRef(kind="call", id="room-7"))
SENDING_TURN_OVER_AT = 50.0
OWN_BOUND_SECONDS = 100.0


class SleepsOfLengths(FakeClock):
    """A clock whose sleep of one length returns only once the test lets that length through."""

    def __init__(self) -> None:
        super().__init__()
        self.gates: dict[float, asyncio.Event] = {}

    async def sleep(self, seconds: float) -> None:
        await self.gates.setdefault(seconds, asyncio.Event()).wait()
        self.time += seconds


class Scenario:
    """A turn that started at time zero with ``OWN_BOUND_SECONDS`` to run, and a sentence sent by a turn that is over at
    ``SENDING_TURN_OVER_AT``, answered as spoken."""

    def __init__(self, sent_by: float = SENDING_TURN_OVER_AT) -> None:
        self.clock = SleepsOfLengths()
        self.clock.time = 0.0
        self.mind = Mind(1_000, 1_000, DEFAULT_BUDGET_PAUSE_CAP_SECONDS)
        self.missing: list[int] = []
        self.bound = TurnBound(self.clock, OWN_BOUND_SECONDS, self.mind, self.missing.append)
        self.mind.speech_sent(AT, sent_by)
        self.mind.speech_answered(AT, "One.", spoken=True)
        self.mind.speech_ended(AT)

    async def waiting(self, **held) -> asyncio.Task[object]:
        waiting = asyncio.create_task(self.bound.until_lines(lambda owed: owed == 0, **held))
        await anyio.wait_all_tasks_blocked()
        return waiting


async def test_a_wait_for_lines_ends_when_the_turn_that_sent_them_is_over_and_reports_them_missing_once():
    scenario = Scenario()
    waiting = await scenario.waiting()

    scenario.clock.gates[SENDING_TURN_OVER_AT].set()
    owed = await waiting

    assert (owed.count, owed.dropped, scenario.missing) == (0, 1, [1])
    assert scenario.bound.past() is False


async def test_a_wait_ends_the_turn_limit_exceeded_at_its_own_bound_leaving_the_lines_to_the_next_waiter():
    scenario = Scenario(sent_by=OWN_BOUND_SECONDS)
    waiting = await scenario.waiting()

    scenario.clock.gates[OWN_BOUND_SECONDS].set()

    with pytest.raises(TurnStopped) as stopped:
        await waiting
    assert stopped.value.code is FailureCode.LIMIT_EXCEEDED
    assert scenario.missing == []
    assert scenario.mind.lines_owed(0.0, 2 * OWN_BOUND_SECONDS, AT).count == 1


async def test_a_wait_that_asks_for_it_ends_at_a_hold_over_a_line_owed_which_is_no_loss_however_late_it_is():
    scenario = Scenario()
    held = Held("h1", (PartialSpeech(name="Ada", text="wait"),), record_gone=False)
    scenario.mind.hold_read(AT, _window_holding(held))
    scenario.clock.time = 2 * OWN_BOUND_SECONDS

    owed = await scenario.bound.until_lines(lambda count: count == 0, held_ends=True)

    assert (owed.count, scenario.missing) == (1, [])
    assert owed.held is not None
    assert owed.held.id == held.id


async def test_a_wait_that_does_not_ask_for_a_hold_waits_on_until_the_turns_bound():
    scenario = Scenario()
    scenario.mind.hold_read(AT, _window_holding(Held("h1", (), record_gone=True)))
    waiting = await scenario.waiting()

    assert not waiting.done()
    scenario.clock.gates[OWN_BOUND_SECONDS].set()
    with pytest.raises(TurnStopped):
        await waiting


async def test_a_take_keeps_the_wait_sleeping_for_the_same_sending_turn_and_swaps_it_for_an_earlier_one():
    scenario = Scenario()
    waiting = await scenario.waiting()

    scenario.mind.take((_a_line_of_his_own_unspoken(),))
    await anyio.wait_all_tasks_blocked()
    await anyio.wait_all_tasks_blocked()
    assert not waiting.done()

    earlier = SENDING_TURN_OVER_AT - 10
    for length in (earlier, SENDING_TURN_OVER_AT - earlier):
        scenario.clock.gates[length] = asyncio.Event()
        scenario.clock.gates[length].set()
    other = AppPlace(AT.app, PlaceRef(kind="call", id="room-8"))
    scenario.mind.speech_sent(other, earlier)
    scenario.mind.speech_answered(other, "Two.", spoken=True)
    scenario.mind.speech_ended(other)
    scenario.mind.take((_a_line_of_his_own_unspoken(),))
    await anyio.wait_all_tasks_blocked()
    owed = await waiting

    assert (owed.count, scenario.missing) == (0, [1, 1])


def _a_line_of_his_own_unspoken() -> KeptLine:
    visit = Visit(AT.app, AT.place, "g1")
    return KeptLine(
        visit,
        1,
        datetime(2026, 1, 1, tzinfo=UTC),
        "Mozart",
        "message",
        "x",
        own=True,
        spoken=False,
        cut=False,
        act=False,
    )


def _window_holding(held: Held) -> PlaceWindow:
    talked_over = None if held.record_gone else TalkedOver(id="talk", playing=True, speech=held.speech)
    return PlaceWindow(
        generation="g", starts_at=1, entries=(), talked_over=talked_over, interrupted=Interrupted(id=held.id)
    )
