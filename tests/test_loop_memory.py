"""The whole runner, the memory: a window read is handed to the store, the store's failure costs no answer, and a turn
looks up what the store kept of its place."""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.tools import ToolEffect
from agents.turn import FailureCode
from mcp_types import ListToolsResult

from personas.decision import NETWORK_FAILURE_CODE
from personas.failures import (
    STORE_DEFECT_CODE,
    STORE_TIMEOUT_CODE,
    STORE_UNREACHABLE_CODE,
    SchemaNotAtHeadError,
    StoreDefect,
    StoreUnavailable,
    leaves_of,
)
from personas.recall import (
    LOOKUPS_SPENT_TEXT,
    RECALL_DECLARATION,
    RECALL_TOOL,
    SCOPE_TEXT,
    SMALLEST_RESULT_BYTES,
    STORE_DID_NOT_ANSWER_TEXT,
    text_refused_text,
)
from personas.reporting import Event
from tests.support.call import (
    ADDRESSED,
    ANSWER,
    CALL_PLACE,
    PLACE_ID,
    pulses_the_window_once,
    restarted_in_the_call,
    say_in_the_call,
)
from tests.support.fakes import CANCELLED_TURN, failed_turn, memory_target
from tests.support.gateway import Gateway, play, say, says
from tests.support.history import (
    MORNING,
    TABLE,
    a_state,
    at_table,
    entry,
    said,
    said_and_taken,
    settled,
    url_of,
    waits,
)
from tests.support.hopin import LIMIT_SECONDS, LIST_TOOLS, LISTEN, PLAY_TOOL, SAY_TOOL
from tests.support.logs import formatted, lines_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from personas.entries import KeptLine, WindowRead
    from personas.recall import LineSearch

pytestmark = pytest.mark.anyio

OUTAGE_EVENTS = {Event.BACKOFF, Event.SUSPENDED}
SHORT_STREAM_SECONDS = 0.5
"""The runner's deadline on a listen; the scripted app ends each listen once it has pulsed, so only a held write
reaches it."""


class ScriptedStore:
    """Keeps what it is handed, or fails with the failure it is told to, or waits for a release first; it answers a
    search with the lines it is told to give, or fails with the failure it is told to."""

    def __init__(self) -> None:
        self.kept: list[WindowRead] = []
        self.failure: StoreUnavailable | None = None
        self.searches: list[LineSearch] = []
        self.found: tuple[KeptLine, ...] = ()
        self.search_failure: Exception | None = None
        self.on_search_failure: Callable[[], None] = lambda: None
        self.waiting = asyncio.Event()
        self.release: asyncio.Event | None = None
        self.attempts = 0
        self._attempted = asyncio.Event()

    async def until_waiting(self) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            await self.waiting.wait()

    async def until_attempts(self, count: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while self.attempts < count:
                self._attempted.clear()
                await self._attempted.wait()

    async def keep(self, read: WindowRead) -> None:
        self.attempts += 1
        self._attempted.set()
        if self.release is not None:
            self.waiting.set()
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        self.kept.append(read)

    async def search(self, search: LineSearch) -> tuple[KeptLine, ...]:
        self.searches.append(search)
        if self.search_failure is not None:
            self.on_search_failure()
            raise self.search_failure
        return self.found

    async def erase_before(self, first_read_before: datetime) -> int:
        raise AssertionError("a test about the loop swept")

    def seqs_kept(self) -> list[int]:
        return [line.seq for read in self.kept for line in read.window.entries]


async def test_a_store_that_is_down_is_reported_once_and_costs_no_answer_and_the_lines_are_kept_when_it_is_back(runner):
    store = ScriptedStore()
    store.failure = StoreUnavailable(STORE_UNREACHABLE_CODE)
    runner.lines = store
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await runner.gateway.until_requested(2)
    await store.until_attempts(2)
    store.failure = None
    say_in_the_call(runner, "A line once it is back")
    await runner.reports.until(Event.STORE_RESUMED)

    assert runner.app.said == [(runner.app_principal.sub, PLACE_ID, ANSWER)]
    assert runner.reports.reported() == [
        (Event.STARTED, None),
        (Event.STORE_FAILED, STORE_UNREACHABLE_CODE),
        (Event.STORE_RESUMED, None),
    ]
    assert store.seqs_kept()[-3:] == [1, 2, 3]


async def test_the_window_of_a_view_place_is_handed_to_the_store_under_its_kind_and_id_with_the_act_marks(runner):
    store = ScriptedStore()
    runner.lines = store
    runner.app.stand(SUBJECT, at_table())
    runner.app.speak(TABLE.id, "visitor", "Ada", "took a card", act=True)
    runner.app.show_view(TABLE.id, 1, a_state(), awaits_you=False)
    runner.start()
    await store.until_attempts(1)

    (read, *_) = store.kept
    assert read.place == TABLE
    assert [(each.text, each.act) for each in read.window.entries] == [("took a card", True)]


async def test_a_write_that_ends_in_a_timeout_is_reported_by_its_code_and_the_turn_it_held_up_follows(runner):
    store = ScriptedStore()
    store.release = asyncio.Event()
    store.failure = StoreUnavailable(STORE_TIMEOUT_CODE)
    runner.lines = store
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await store.until_waiting()
    assert runner.gateway.requests == []
    store.release.set()
    await runner.app.said_count(1)
    await runner.reports.until(Event.STORE_FAILED)

    assert (Event.STORE_FAILED, STORE_TIMEOUT_CODE) in runner.reports.reported()
    assert not OUTAGE_EVENTS & {event for event, _ in runner.reports.reported()}


async def test_a_write_held_past_the_listens_deadline_cancels_the_running_turn_with_no_store_failed(runner):
    """CONFIG-25."""
    addressed_said, held_line_said = asyncio.Event(), asyncio.Event()
    runner.app.script(LISTEN, pulses_the_window_once(addressed_said), pulses_the_window_once(held_line_said))
    store = ScriptedStore()
    runner.lines = store
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.start(stream_seconds=SHORT_STREAM_SECONDS)
    await runner.app.until_window_read(PLACE_ID)
    say_in_the_call(runner, ADDRESSED)
    addressed_said.set()
    await runner.gateway.until_requested(1)
    store.release = asyncio.Event()
    say_in_the_call(runner, "A line the dark store holds")
    held_line_said.set()
    await store.until_waiting()
    await runner.reports.until_a_turn_ended()
    await runner.reports.until(Event.BACKOFF)

    assert runner.reports.turn_endings == [CANCELLED_TURN]
    # The turn's end statement may be cut by the session's close, which logs it as lost.
    reported = [report for report in runner.reports.reported() if report[0] is not Event.TURN_STATE_LOST]
    assert reported[:2] == [(Event.STARTED, None), (Event.BACKOFF, NETWORK_FAILURE_CODE)]
    assert Event.STORE_FAILED not in {event for event, _ in reported}


LATER_TOLD = "2026-11-10T18:00:00Z"


def stored_in(runner, store: ScriptedStore, **memory_fields) -> None:
    """The runner keeps and looks up through the scripted store, as a configuration with a memory table does."""
    runner.lines = store
    runner.config = replace(runner.config, memory=replace(memory_target(), **memory_fields))


def recalling(text: str = "", **more):
    """A call of the lookup."""
    return tool_call(secrets.token_hex(4), RECALL_TOOL, {"text": text, **more})


def told_said(seq: int, text: str) -> dict[str, object]:
    return entry(MORNING.app, MORNING.place.id, text, seq=seq, at=LATER_TOLD, name="Bo")


def result_of(runner, request: int, index: int = 0) -> dict[str, object]:
    return json.loads(runner.gateway.tool_results(request)[index])


async def looking_in_the_call(runner, store: ScriptedStore, *replies, **memory_fields) -> None:
    """The runner restarted in the call with a memory, and the first line addressed to him."""
    stored_in(runner, store, **memory_fields)
    runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)


async def looking_at_a_table(runner, store: ScriptedStore, *replies, **memory_fields) -> None:
    """The runner with a memory, standing in the call and at a table, whose view waits for him."""
    stored_in(runner, store, **memory_fields)
    runner.gateway = Gateway(*replies)
    await restarted_in_the_call(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")


async def lists_nothing(reply) -> None:
    await reply.result(ListToolsResult(tools=[]))


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_a_written_turn_with_a_memory_is_handed_the_lookup_first_declared_read_only_before_the_apps_tools(runner):
    """TURNS-39."""
    await looking_in_the_call(runner, ScriptedStore(), completion("done"))
    await runner.gateway.until_requested(1)

    (handed,) = runner.handed
    assert runner.gateway.offered_tools() == [RECALL_TOOL, SAY_TOOL]
    assert (handed[0].declaration.name, handed[0].declaration.effect) == (RECALL_TOOL, ToolEffect.READ_ONLY)


async def test_a_turn_at_a_view_with_a_memory_is_handed_the_lookup_first_declared_read_only_before_the_apps_tools(
    runner,
):
    await looking_at_a_table(runner, ScriptedStore(), completion("done"))

    (handed,) = runner.handed
    assert runner.gateway.offered_tools() == [RECALL_TOOL, SAY_TOOL]
    assert (handed[0].declaration.name, handed[0].declaration.effect) == (RECALL_TOOL, ToolEffect.READ_ONLY)


@pytest.mark.parametrize("lists_tools", [True, False], ids=["his app holds a tool", "his apps hold no tool"])
async def test_a_turns_call_total_is_his_apps_bounds_plus_his_lookups_and_the_one_a_held_call_counts(
    runner, lists_tools
):
    """TURNS-22, CONFIG-36."""
    if not lists_tools:
        # The loop's session lists the tools first, to check the answer of the turn's thinking.
        runner.app.script(LIST_TOOLS, lists_nothing, lists_nothing)
    stored_in(runner, ScriptedStore())
    lookups = runner.config.memory.lookup_calls
    own_calls = runner.config.apps[0].max_tool_calls if lists_tools else 0
    total = [
        *(tool_round(recalling()) for _ in range(lookups + 1)),
        *(tool_round(say(secrets.token_hex(4))) for _ in range(own_calls)),
    ]
    runner.gateway = Gateway(*total, tool_round(recalling()), completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.PAUSED)

    assert runner.gateway.arrivals == len(total) + 1
    await runner.reports.until_a_turn_ended()
    assert runner.reports.reported(started=False) == [
        (Event.PAUSED, FailureCode.LIMIT_EXCEEDED.value),
    ]
    assert runner.reports.turn_endings == [failed_turn(FailureCode.LIMIT_EXCEEDED)]


@pytest.mark.parametrize("at_a_view", [False, True], ids=["written turn", "turn at a view"])
async def test_a_lookup_asks_for_the_turns_own_room_whatever_arguments_the_model_gave(runner, at_a_view):
    """TURNS-39."""
    elsewhere = {"place": "elsewhere", "app": "https://elsewhere.test", "client_id": "another", "kind": "call"}
    replies = (tool_round(recalling("some words", **elsewhere)), completion("done"))
    store = ScriptedStore()
    if at_a_view:
        await looking_at_a_table(runner, store, *replies)
    else:
        await looking_in_the_call(runner, store, *replies)
    await runner.gateway.until_requested(2)

    (search,) = store.searches
    assert (search.client_id, search.app, search.place) == (
        runner.agent.client_id,
        url_of(runner, 0),
        TABLE if at_a_view else CALL_PLACE,
    )
    assert search.text == "some words"
    assert search.first_read_before == runner.clock.wall()


async def test_a_lookup_gives_the_model_its_scope_and_the_lines_found_in_the_historys_form_newest_first(runner):
    store = ScriptedStore()
    store.found = (said(2, "Newer"), said(1, "Older"))
    await looking_in_the_call(runner, store, tool_round(recalling("words")), completion("done"))
    await runner.gateway.until_requested(2)

    assert result_of(runner, 2) == {
        "scope": SCOPE_TEXT,
        "older_dropped": False,
        "lines": [told_said(2, "Newer"), told_said(1, "Older")],
    }


async def test_a_result_holds_the_newest_lines_within_lookup_bytes_and_says_when_more_matched(runner):
    result_bytes = 1000
    store = ScriptedStore()
    store.found = tuple(said(seq, "x" * 200) for seq in range(5, 0, -1))
    await looking_in_the_call(runner, store, tool_round(recalling()), completion("done"), lookup_bytes=result_bytes)
    await runner.gateway.until_requested(2)

    result = result_of(runner, 2)
    assert len(runner.gateway.tool_results(2)[0].encode()) <= result_bytes
    assert result["lines"] == [told_said(seq, "x" * 200) for seq in range(5, 5 - len(result["lines"]), -1)]
    assert 0 < len(result["lines"]) < 5
    assert result["older_dropped"]


async def test_a_lookup_past_lookup_calls_is_refused_with_a_text_of_its_own_and_reads_nothing(runner):
    """CONFIG-36."""
    store = ScriptedStore()
    await looking_in_the_call(
        runner, store, tool_round(recalling(), recalling(), recalling()), completion("done"), lookup_calls=2
    )
    await runner.gateway.until_requested(2)

    first, second, third = runner.gateway.tool_results(2)
    assert len(store.searches) == 2
    assert json.loads(first) == json.loads(second)
    assert LOOKUPS_SPENT_TEXT in third


async def test_a_lookup_is_not_held_back_by_what_arrived_and_does_not_use_up_the_hold(runner):
    """TURNS-41."""
    store = ScriptedStore()
    stored_in(runner, store)
    runner.gateway = Gateway(tool_round(recalling()), tool_round(say(ANSWER)), completion("done"))
    running = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, "News that arrived meanwhile")
    running.set()
    await runner.gateway.until_requested(3)

    assert "scope" in result_of(runner, 2)
    assert len(runner.gateway.held_results(3)) == 1
    assert runner.app.said == []


@pytest.mark.parametrize(
    "arguments",
    [{}, {"text": 5}, {"text": None}, {"text": "x" * (SMALLEST_RESULT_BYTES + 1)}],
    ids=["missing", "no string", "null", "longer than lookup bytes"],
)
async def test_a_text_that_is_missing_no_string_or_too_long_is_an_error_result_and_the_store_is_not_read(
    runner, arguments
):
    store = ScriptedStore()
    await looking_in_the_call(
        runner,
        store,
        tool_round(tool_call("call-0", RECALL_TOOL, arguments)),
        completion("done"),
        lookup_bytes=SMALLEST_RESULT_BYTES,
    )
    await runner.gateway.until_requested(2)

    assert store.searches == []
    assert text_refused_text(SMALLEST_RESULT_BYTES) in runner.gateway.tool_results(2)[0]


async def test_a_store_that_cannot_answer_gives_an_error_result_and_store_failed_once_and_his_answer_goes_out(runner):
    """TURNS-41, LOG-20."""
    store = ScriptedStore()
    store.search_failure = StoreUnavailable(STORE_UNREACHABLE_CODE)
    await looking_in_the_call(
        runner, store, tool_round(recalling(), recalling()), tool_round(say(ANSWER)), completion("done")
    )
    await runner.app.said_count(1)

    first, second = runner.gateway.tool_results(2)
    assert STORE_DID_NOT_ANSWER_TEXT in first
    assert STORE_DID_NOT_ANSWER_TEXT in second
    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert runner.reports.codes_of(Event.STORE_FAILED) == [STORE_UNREACHABLE_CODE]


@pytest.mark.parametrize(
    "failure", [StoreDefect(STORE_DEFECT_CODE), SchemaNotAtHeadError("memory: behind")], ids=["a defect", "a schema"]
)
@pytest.mark.parametrize("place_leaves", [False, True], ids=["place stays", "place leaves meanwhile"])
async def test_a_defect_or_a_schema_behind_its_head_met_in_a_lookup_ends_the_runner_with_it_and_no_answer_is_sent(
    runner, failure, place_leaves
):
    store = ScriptedStore()
    store.search_failure = failure
    if place_leaves:
        store.on_search_failure = lambda: runner.app.stand(SUBJECT)
    await looking_in_the_call(runner, store, tool_round(recalling()), tool_round(say(ANSWER)), completion("done"))

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [failure]
    assert len(store.searches) == 1
    assert runner.app.said == []


async def test_a_turn_that_looked_leaves_nothing_of_the_lookup_in_his_history_and_the_next_turn_repeats_every_message(
    runner,
):
    """TURNS-41."""
    hidden = "Said " + secrets.token_hex(6)
    store = ScriptedStore()
    store.found = (said(1, hidden),)
    await looking_in_the_call(
        runner,
        store,
        tool_round(recalling("said")),
        tool_round(say(ANSWER)),
        completion("done"),
        tool_round(say(ANSWER)),
        completion("done"),
    )
    await runner.reports.until_a_turn_completed()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.reports.until_a_turn_completed(count=2)

    earlier, later = runner.gateway.openings()
    assert runner.app.said[0] == (SUBJECT, PLACE_ID, ANSWER)
    assert later.texts[: len(earlier.texts)] == earlier.texts
    assert [each["tool"] for each in later.told("act")] == [SAY_TOOL]
    assert not [each for each in later.texts if hidden in each or RECALL_TOOL in each]


async def test_after_a_lookup_that_gave_him_a_line_a_call_to_another_apps_tool_goes_out_as_if_he_had_not_looked(
    runner, other
):
    played = secrets.token_hex(4)
    store = ScriptedStore()
    store.found = (said(1, "Said before"),)
    stored_in(runner, store)
    runner.gateway = Gateway(tool_round(recalling()), tool_round(play(played)), completion("done"))
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    await other.until_standing_read()
    await settled(other)
    say_in_the_call(runner, ADDRESSED)
    await other.said_count(1)

    assert other.said == [(SUBJECT, TABLE.id, played)]
    assert runner.gateway.offered_tools() == [RECALL_TOOL, SAY_TOOL, PLAY_TOOL]
    assert len(store.searches) == 1


@pytest.mark.parametrize(
    ("found", "failure", "stored"),
    [
        ((said(1, "Said " + secrets.token_hex(6)),), None, []),
        ((), None, []),
        ((), StoreUnavailable(STORE_UNREACHABLE_CODE), [(Event.STORE_FAILED, STORE_UNREACHABLE_CODE)]),
    ],
    ids=["found a line", "found nothing", "the read failed"],
)
async def test_a_lookup_adds_no_event_and_no_content_to_the_log_and_turn_ended_counts_its_sizes(
    runner, caplog, found, failure, stored
):
    needle = secrets.token_hex(6)
    store = ScriptedStore()
    store.found = found
    store.search_failure = failure
    await looking_in_the_call(runner, store, tool_round(recalling(needle)), completion("done"))
    await runner.reports.until_a_turn_completed()

    logged = formatted(caplog.get_records("call"))
    (ended,) = lines_of(caplog.get_records("call"), "turn.ended")
    results = ended["results_bytes"]
    assert runner.reports.reported() == [(Event.STARTED, None), *stored]
    assert needle not in logged
    assert not [line for line in found if line.text in logged]
    assert results >= len(runner.gateway.tool_results(2)[0].encode())


async def test_a_turn_counts_the_lookups_declaration_in_its_tools_bytes(runner, caplog):
    await looking_in_the_call(runner, ScriptedStore(), completion("done"))
    await runner.reports.until_a_turn_completed()
    runner.task.cancel()
    await asyncio.gather(runner.task, return_exceptions=True)
    runner.config = replace(runner.config, memory=None)
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)
    await settled(runner.app)
    say_in_the_call(runner, f"{ADDRESSED} Once more.")
    await runner.reports.until_a_turn_completed(count=2)

    with_lookup, without = (line["tools_bytes"] for line in lines_of(caplog.get_records("call"), "turn.ended"))
    assert with_lookup - without >= len(RECALL_DECLARATION.description) + len(json.dumps(RECALL_DECLARATION.parameters))
