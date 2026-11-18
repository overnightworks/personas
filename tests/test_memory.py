"""The memory against a real Postgres: which lines are kept, what a failed write is, and what leaks nowhere."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import secrets
import time
import traceback
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import PlaceEntry, PlaceRef, PlaceWindow
from sqlalchemy.exc import IntegrityError, ProgrammingError

from personas.entries import KeptLine, Visit, WindowRead
from personas.failures import STORE_TIMEOUT_CODE, STORE_UNREACHABLE_CODE, StoreDefect, StoreUnavailable
from personas.memory import PostgresLineStore, SchemaNotAtHeadError
from personas.recall import RECALL_TOOL, LineSearch
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.fakes import WALL_START
from tests.support.gateway import Gateway, says
from tests.support.history import (
    act_entry,
    after_start,
    answered,
    entry,
    history_of,
    url_of,
)
from tests.support.hopin import LIMIT_SECONDS
from tests.support.logs import formatted
from tests.support.postgres import (
    Forwarder,
    at_an_unknown_revision,
    kept_lines,
    lines_table_locked,
    migrated_to,
    run_sql,
    waiting_statements,
)
from tests.support.realm import AGENT_NAME
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, invited, line, paused, window_of

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from personas.config import DatabaseTarget

pytestmark = [pytest.mark.anyio, pytest.mark.postgres]

APP = "https://app.test/mcp"
AGENT = "agent-mozart"
SECOND_AGENT = "agent-ada"
SECOND_SUBJECT = "ada-subject"
BOUND_SECONDS = 0.5
MARGIN_SECONDS = 1.0
CLOSED_PORT = 1
DARK_WRITE_SECONDS = 1.0
RESET_AFTER_CANCEL_SECONDS = 0.5
BEYOND_64_BITS = 2**63
FRACTIONAL_SEQ_ROW = (
    "INSERT INTO lines VALUES ('agent', 'app', 'call', 'room', 'visit', 1.5, now(), 'Ada', 'message', 'text',"
    " false, false, false, false)"
)
REFUSE_EVERY_NEW_ROW = "ALTER TABLE lines ADD CONSTRAINT refuses_every_row CHECK (false) NOT VALID"
"""Its error names the failing row, so only the adapter's own wording keeps the row out of a traceback."""


class ObservedStore(PostgresLineStore):
    """The real store, which tells how many writes have finished, in all and per agent, whatever way they ended."""

    def __init__(self, target: DatabaseTarget) -> None:
        super().__init__(target)
        self.finished = 0
        self._finished_by_client_id = Counter[str]()
        self._finishing = asyncio.Event()

    async def keep(self, read: WindowRead) -> None:
        try:
            await super().keep(read)
        finally:
            self.finished += 1
            self._finished_by_client_id[read.client_id] += 1
            self._finishing.set()

    async def until_finished(self, count: int) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while self.finished < count:
                self._finishing.clear()
                await self._finishing.wait()

    async def until_finished_for(self, client_id: str) -> None:
        async with asyncio.timeout(LIMIT_SECONDS):
            while not self._finished_by_client_id[client_id]:
                self._finishing.clear()
                await self._finishing.wait()


@pytest.fixture
async def store(database: DatabaseTarget) -> AsyncIterator[ObservedStore]:
    kept = ObservedStore(database)
    yield kept
    await kept.close()


def a_read(*lines: PlaceEntry) -> WindowRead:
    return WindowRead(AGENT, APP, PlaceRef(kind="call", id=PLACE_ID), window_of(*lines), WALL_START)


def a_visit(generation: str) -> Visit:
    return Visit(APP, PlaceRef(kind="call", id=PLACE_ID), generation)


def a_read_in(visit: Visit, *lines: PlaceEntry, client_id: str = AGENT, hours_after_start: float = 0.0) -> WindowRead:
    read_at = WALL_START + timedelta(hours=hours_after_start)
    return WindowRead(client_id, visit.app, visit.place, window_of(*lines, generation=visit.generation), read_at)


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def test_a_window_read_again_and_after_a_restart_keeps_each_line_once_with_the_time_of_its_first_read(
    runner, store, database
):
    runner.lines = store
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "First line")
    await store.until_finished(1)
    first_read_at = runner.clock.wall()
    runner.clock.time += 3600
    say_in_the_call(runner, "Second line")
    await store.until_finished(2)
    second_read_at = runner.clock.wall()
    runner.task.cancel()
    await ended_by(runner.task)
    runner.clock.time += 3600
    runner.start()
    await store.until_finished(3)

    assert kept_lines(database, "seq, text, first_read_at") == [
        (1, "First line", first_read_at),
        (2, "Second line", second_read_at),
    ]


async def test_his_own_line_a_spoken_line_and_one_said_during_a_pause_are_kept_as_the_window_gave_them(
    runner, store, database
):
    runner.lines = store
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "Said aloud", spoken=True)
    await store.until_finished(1)
    runner.app.speak(PLACE_ID, SUBJECT, "Mozart", "My own words")
    await store.until_finished(2)
    pause_seen = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.app.until_read_held()
    say_in_the_call(runner, "During the pause")
    runner.app.stand(SUBJECT, active(PLACE_ID))
    pause_seen.set()
    await store.until_finished(3)

    assert kept_lines(database, "seq, speaker, text, own, spoken") == [
        (1, "Ada", "Said aloud", False, True),
        (2, "Mozart", "My own words", True, False),
        (3, "Ada", "During the pause", False, False),
    ]


async def test_two_agents_in_one_place_keep_each_their_own_rows_under_their_own_client_id(runner, store, database):
    runner.lines = store
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID, 2)
    say_in_the_call(runner, "Hello both")
    await store.until_finished_for(runner.agent.client_id)
    await store.until_finished_for(SECOND_AGENT)

    assert kept_lines(database, "client_id, app, seq, text") == [
        (SECOND_AGENT, runner.app_url, 1, "Hello both"),
        (runner.agent.client_id, runner.app_url, 1, "Hello both"),
    ]


async def test_a_line_with_u0000_is_kept_with_the_character_replaced_and_the_others_are_kept(store, database):
    entries = (
        PlaceEntry(seq=1, kind="mess\x00age", name="Ad\x00a", text="before\x00after", own=False),
        line(2, "plain"),
    )
    window = PlaceWindow(generation="visit\x00", starts_at=1, entries=entries)
    place = PlaceRef(kind="ca\x00ll", id="room\x00")

    await store.keep(WindowRead(AGENT, APP, place, window, WALL_START))

    assert kept_lines(database, "place_kind, place_id, generation, seq, kind, speaker, text") == [
        ("ca�ll", "room�", "visit�", 1, "mess�age", "Ad�a", "before�after"),
        ("ca�ll", "room�", "visit�", 2, "message", "Ada", "plain"),
    ]


async def test_a_write_waiting_on_a_lock_ends_within_its_bound_releases_its_statement_and_the_next_write_succeeds(
    store, database
):
    await store.keep(a_read(line(1, "Before the lock")))
    bounded = PostgresLineStore(replace(database, write_seconds=BOUND_SECONDS))
    try:
        with lines_table_locked(database):
            started = time.monotonic()
            during_the_lock = a_read(line(2, "During the lock"))
            async with asyncio.timeout(LIMIT_SECONDS):
                with pytest.raises(StoreUnavailable) as cut:
                    await bounded.keep(during_the_lock)
            elapsed = time.monotonic() - started
            still_waiting = waiting_statements(database)

        await store.keep(a_read(line(3, "After the lock")))
    finally:
        await bounded.close()

    assert cut.value.code == STORE_TIMEOUT_CODE
    assert still_waiting == 0
    assert BOUND_SECONDS <= elapsed < BOUND_SECONDS + MARGIN_SECONDS
    assert kept_lines(database, "seq") == [(1,), (3,)]


@pytest.mark.parametrize("reset_after_seconds", [None, RESET_AFTER_CANCEL_SECONDS], ids=["dark", "then_reset"])
async def test_a_write_cancelled_while_its_connection_answers_nothing_ends_as_the_cancellation(
    database, reset_after_seconds
):
    forwarder = Forwarder(replace(database, write_seconds=DARK_WRITE_SECONDS))
    async with forwarder.serving() as through_the_forwarder:
        store = PostgresLineStore(through_the_forwarder)
        try:
            await store.keep(a_read(line(1, "Before the dark")))
            forwarder.go_dark()
            write = asyncio.create_task(store.keep(a_read(line(2, "Into the dark"))))
            async with asyncio.timeout(LIMIT_SECONDS):
                await forwarder.dropped.wait()
            write.cancel()
            if reset_after_seconds is not None:
                asyncio.get_running_loop().call_later(reset_after_seconds, forwarder.reset_dark)
            ended = await ended_by(write)
            await store.keep(a_read(line(3, "After the dark")))
        finally:
            await store.close()

    assert isinstance(ended, asyncio.CancelledError)
    assert kept_lines(database, "seq") == [(1,), (3,)]


async def test_a_line_whose_seq_is_beyond_64_bits_is_kept_with_the_other_lines_of_its_window(store, database):
    await store.keep(a_read(line(1, "Ordinary"), line(BEYOND_64_BITS, "Far along")))

    assert kept_lines(database, "seq, text") == [(1, "Ordinary"), (BEYOND_64_BITS, "Far along")]


def test_a_fraction_for_seq_is_refused_by_the_database(database):
    with pytest.raises(IntegrityError):
        run_sql(database, FRACTIONAL_SEQ_ROW)


async def test_a_store_without_a_write_bound_keeps_a_window(database):
    store = PostgresLineStore(replace(database, write_seconds=math.inf))

    await store.keep(a_read(line(1, "Unbounded")))
    await store.close()

    assert kept_lines(database, "seq") == [(1,)]


async def test_a_store_that_cannot_be_reached_is_unavailable_by_its_code(database):
    store = PostgresLineStore(replace(database, port=CLOSED_PORT))

    nowhere_to_go = a_read(line(1, "Nowhere to go"))

    with pytest.raises(StoreUnavailable) as unreachable:
        await store.keep(nowhere_to_go)
    await store.close()

    assert unreachable.value.code == STORE_UNREACHABLE_CODE


async def test_a_schema_behind_its_head_ends_the_write_naming_the_field_and_no_secret(unmigrated_database):
    store = PostgresLineStore(unmigrated_database)

    no_table_yet = a_read(line(1, "No table yet"))

    with pytest.raises(SchemaNotAtHeadError) as behind:
        await store.keep(no_table_yet)
    await store.close()

    assert "memory" in str(behind.value)
    assert unmigrated_database.password.reveal() not in str(behind.value)


async def test_a_schema_at_a_known_revision_behind_its_head_ends_the_write(unmigrated_database):
    migrated_to(unmigrated_database, "0001")
    store = PostgresLineStore(unmigrated_database)
    behind_the_head = a_read(line(1, "Behind"))

    try:
        with pytest.raises(SchemaNotAtHeadError):
            await store.keep(behind_the_head)
    finally:
        await store.close()


async def test_a_schema_check_on_a_database_at_its_head_answers(store):
    await store.require_head()


async def test_a_database_at_a_revision_the_scripts_do_not_know_is_served(store, database):
    at_an_unknown_revision(database)

    await store.keep(a_read(line(1, "Kept on a newer schema")))

    assert kept_lines(database, "text") == [("Kept on a newer schema",)]


async def test_a_database_error_that_is_no_outage_raises_a_code_that_holds_no_line_name_or_password(store, database):
    await store.keep(a_read(line(1, "Kept before")))
    run_sql(database, REFUSE_EVERY_NEW_ROW)
    secret_line = "text-" + secrets.token_hex(8)
    secret_speaker = "speaker-" + secrets.token_hex(8)

    secret_read = a_read(PlaceEntry(seq=2, kind="message", name=secret_speaker, text=secret_line, own=False))

    with pytest.raises(StoreDefect) as defect:
        await store.keep(secret_read)

    shown = "".join(traceback.format_exception(defect.value))
    for held in (secret_line, secret_speaker, database.password.reveal()):
        assert held not in shown
        assert held not in repr(defect.value)


async def test_a_runner_whose_store_cannot_be_reached_starts_and_reports_the_failure_once(runner, database):
    runner.lines = ObservedStore(replace(database, port=CLOSED_PORT))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "First line")
    await runner.lines.until_finished(1)
    say_in_the_call(runner, "Second line")
    await runner.lines.until_finished(2)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.STORE_FAILED, STORE_UNREACHABLE_CODE)]
    await runner.lines.close()


async def test_a_runner_with_a_schema_behind_its_head_ends_loud_with_nothing_but_the_field_in_its_error(
    runner, unmigrated_database
):
    store = ObservedStore(unmigrated_database)
    runner.lines = store
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    ended = await ended_by(runner.task)

    assert "memory" in "".join(traceback.format_exception(ended))
    assert unmigrated_database.password.reveal() not in "".join(traceback.format_exception(ended))
    await store.close()


async def test_a_runner_met_by_a_database_error_that_is_no_outage_ends_with_a_code_and_logs_no_line(
    runner, database, caplog
):
    store = ObservedStore(database)
    runner.lines = store
    await store.keep(a_read(line(1, "Kept before")))
    run_sql(database, REFUSE_EVERY_NEW_ROW)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, "A line to be lost")

    ended = await ended_by(runner.task)

    shown = "".join(traceback.format_exception(ended)) + formatted(caplog.get_records("call"))
    assert "A line to be lost" not in shown
    assert database.password.reveal() not in shown
    await store.close()


async def test_back_four_hours_after_a_visit_his_first_turn_is_told_its_history_and_nothing_the_memory_kept(
    runner, store, database
):
    welcome_back = f"{ADDRESSED} Back again?"

    def addressed_once_joined(change):
        if change.event is Event.JOINED:
            runner.app.stand(SUBJECT, active(PLACE_ID))
            say_in_the_call(runner, welcome_back)

    runner.remember_in(store)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, "Yes."))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await store.until_finished(2)
    runner.app.stand(SUBJECT)
    runner.clock.time += 4 * 3600
    runner.on_report = addressed_once_joined
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    await runner.app.said_count(2)

    _first_visit, back_again = runner.gateway.openings()
    assert back_again.room["history"] == history_of(
        [
            entry(url_of(runner, 0), PLACE_ID, ADDRESSED),
            act_entry(
                url_of(runner, 0),
                "say",
                {"place": PLACE_ID, "text": ANSWER},
                answered("said"),
            ),
            entry(url_of(runner, 0), PLACE_ID, ANSWER, seq=2, name=AGENT_NAME, own=True),
            entry(url_of(runner, 0), PLACE_ID, welcome_back, at=after_start(4 * 3600)),
        ]
    )


@pytest.mark.parametrize(
    "insert",
    [
        pytest.param(
            "INSERT INTO lines (client_id, app, place_kind, place_id, generation, seq, first_read_at, speaker, kind,"
            " text, own, spoken, act) VALUES ('agent', 'app', 'call', 'room', 'visit', 1, now(), 'Ada', 'message',"
            " 'text', false, false, false)",
            id="forgets cut",
        ),
        pytest.param(
            "INSERT INTO lines (client_id, app, place_kind, place_id, generation, seq, first_read_at, speaker, kind,"
            " text, own, spoken, cut) VALUES ('agent', 'app', 'call', 'room', 'visit', 1, now(), 'Ada', 'message',"
            " 'text', false, false, false)",
            id="forgets act",
        ),
    ],
)
def test_a_writer_that_forgets_a_mark_of_a_line_is_refused_by_the_database(database, insert):
    with pytest.raises(IntegrityError):
        run_sql(database, insert)


async def test_a_write_slower_than_its_bound_is_cut_as_a_timeout_and_the_turn_answers(runner, database):
    bounded = replace(database, write_seconds=BOUND_SECONDS)
    store = ObservedStore(bounded)
    runner.remember_in(store)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    try:
        with lines_table_locked(database):
            say_in_the_call(runner, ADDRESSED)
            await runner.app.said_count(1)
    finally:
        await store.close()

    assert runner.reports.reported().count((Event.STORE_FAILED, STORE_TIMEOUT_CODE)) == 1


async def test_every_line_first_read_before_the_time_is_erased_of_every_agent_whether_or_not_it_is_configured(
    store, database
):
    cutoff = WALL_START + timedelta(days=1)
    visit = a_visit("sweep")
    await store.keep(a_read_in(visit, line(1, "Old, configured agent"), hours_after_start=0))
    await store.keep(a_read_in(visit, line(1, "Old, agent no longer configured"), client_id="agent-gone"))
    await store.keep(a_read_in(visit, line(2, "Exactly at the time"), hours_after_start=24))
    await store.keep(a_read_in(visit, line(3, "Younger"), hours_after_start=30, client_id=SECOND_AGENT))

    erased = await store.erase_before(cutoff)

    assert erased == 2
    assert kept_lines(database, "text") == [("Younger",), ("Exactly at the time",)]


def a_search(
    text: str = "",
    *,
    place: PlaceRef | None = None,
    first_read_before: datetime | None = None,
    rows: int = 50,
    client_id: str = AGENT,
    app: str = APP,
) -> LineSearch:
    return LineSearch(
        client_id,
        app,
        place or PlaceRef(kind="call", id=PLACE_ID),
        text,
        first_read_before or WALL_START + timedelta(days=1),
        rows,
    )


def texts_found(found: tuple[KeptLine, ...]) -> list[str]:
    return [each.text for each in found]


async def test_a_search_gives_the_lines_of_this_agents_room_across_its_visits_and_the_line_as_it_was_kept(store):
    """TURNS-39."""
    await store.keep(a_read_in(a_visit("morning"), line(1, "From the morning"), hours_after_start=1))
    await store.keep(a_read_in(a_visit("noon"), line(1, "From the noon"), hours_after_start=2))

    found = await store.search(a_search())

    assert found == (
        KeptLine(
            a_visit("noon"),
            1,
            WALL_START + timedelta(hours=2),
            "Ada",
            "message",
            "From the noon",
            own=False,
            spoken=False,
            cut=False,
            act=False,
        ),
        KeptLine(
            a_visit("morning"),
            1,
            WALL_START + timedelta(hours=1),
            "Ada",
            "message",
            "From the morning",
            own=False,
            spoken=False,
            cut=False,
            act=False,
        ),
    )


async def test_a_search_gives_no_line_of_another_place_kind_app_or_agent_whatever_its_text_holds(store):
    room = PlaceRef(kind="call", id=PLACE_ID)
    await store.keep(a_read_in(a_visit("here"), line(1, "Said here"), hours_after_start=1))
    await store.keep(a_read_in(Visit(APP, PlaceRef(kind="call", id="room-9"), "g"), line(1, "Said elsewhere")))
    await store.keep(a_read_in(Visit(APP, PlaceRef(kind="table", id=PLACE_ID), "g"), line(1, "Said at a table")))
    await store.keep(a_read_in(Visit("https://other.test/mcp", room, "g"), line(1, "Said in another app")))
    await store.keep(a_read_in(a_visit("g"), line(1, "Said to another agent"), client_id=SECOND_AGENT))

    everything = await store.search(a_search())
    by_their_words = await store.search(a_search("Said"))
    other_agent = await store.search(a_search(client_id=SECOND_AGENT))

    assert texts_found(everything) == texts_found(by_their_words) == ["Said here"]
    assert texts_found(other_agent) == ["Said to another agent"]


@pytest.mark.parametrize(
    ("words", "found"),
    [
        ("red fox", ["the Fox is red", "The red fox jumps"]),
        ("FOX", ["the Fox is red", "The red fox jumps"]),
        ("red dog fox", []),
        ("foxes", []),
        ("", ["a red dog", "the Fox is red", "The red fox jumps"]),
        ("?! ...", ["a red dog", "the Fox is red", "The red fox jumps"]),
    ],
    ids=["every word", "any case", "a word missing", "no stemming", "no word", "only marks"],
)
async def test_a_line_is_found_when_it_holds_every_word_and_a_text_without_a_word_gives_every_line_newest_first(
    store, words, found
):
    """TURNS-40."""
    await store.keep(a_read_in(a_visit("v"), line(1, "The red fox jumps"), hours_after_start=1))
    await store.keep(a_read_in(a_visit("v"), line(2, "the Fox is red"), hours_after_start=2))
    await store.keep(a_read_in(a_visit("v"), line(3, "a red dog"), hours_after_start=3))

    assert texts_found(await store.search(a_search(words))) == found


async def test_a_line_first_read_at_or_after_the_time_is_not_found_and_the_newest_rows_come_first_up_to_the_bound(
    store,
):
    """TURNS-40."""
    await store.keep(a_read_in(a_visit("v"), line(1, "one"), line(2, "two"), hours_after_start=1))
    await store.keep(a_read_in(a_visit("v"), line(3, "three"), hours_after_start=2))
    await store.keep(a_read_in(a_visit("v"), line(4, "four"), hours_after_start=3))
    two_hours = WALL_START + timedelta(hours=2)

    before_two_hours = await store.search(a_search(first_read_before=two_hours))
    newest_two = await store.search(a_search(rows=2))

    assert texts_found(before_two_hours) == ["two", "one"]
    assert texts_found(newest_two) == ["four", "three"]


async def test_a_u0000_in_the_text_or_in_the_places_key_is_replaced_as_a_write_replaces_it_and_the_lines_are_found(
    store,
):
    place = PlaceRef(kind="ca\x00ll", id="room\x00")
    window = PlaceWindow(generation="visit", starts_at=1, entries=(line(1, "before\x00after"),))
    await store.keep(WindowRead(AGENT, APP, place, window, WALL_START))

    by_the_words = await store.search(a_search("before\x00after", place=place))
    by_nothing_but_the_character = await store.search(a_search("\x00", place=place))

    assert texts_found(by_the_words) == texts_found(by_nothing_but_the_character) == ["before�after"]


async def test_a_search_on_a_schema_behind_its_head_raises_what_a_write_raises(unmigrated_database):
    store = PostgresLineStore(unmigrated_database)
    search = a_search()

    try:
        with pytest.raises(SchemaNotAtHeadError) as behind:
            await store.search(search)
    finally:
        await store.close()

    assert unmigrated_database.password.reveal() not in str(behind.value)


async def test_a_search_met_by_a_database_error_that_is_no_outage_raises_a_defect_that_holds_no_search_text(
    store, database
):
    await store.keep(a_read(line(1, "Kept before")))
    run_sql(database, "ALTER TABLE lines RENAME COLUMN text TO body")
    secret_words = "words-" + secrets.token_hex(8)
    search = a_search(secret_words)

    with pytest.raises(StoreDefect) as defect:
        await store.search(search)

    shown = "".join(traceback.format_exception(defect.value))
    for held in (secret_words, database.password.reveal()):
        assert held not in shown
        assert held not in repr(defect.value)


class _DriverAnsweringACancellationWithADefect:
    """An engine whose connection takes the cancellation of its task as a driver does on its cancel request, then
    fails with a database error that is no outage; the task still counts the cancellation."""

    def begin(self) -> _DriverAnsweringACancellationWithADefect:
        return self

    async def __aenter__(self) -> None:
        asyncio.current_task().cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        raise ProgrammingError("statement", {}, Exception("not an outage"))

    async def __aexit__(self, *_exception) -> None:
        return None

    async def dispose(self) -> None:
        return None


async def test_a_search_met_by_a_defect_while_its_task_is_being_cancelled_raises_the_defect(database):
    """The engine is a fake that never connects: it models a driver defect that asyncpg's real cancel path does not
    raise, so the test pins how the store treats one."""
    store = PostgresLineStore(database, lambda *_arguments, **_options: _DriverAnsweringACancellationWithADefect())

    try:
        searching = asyncio.create_task(store.search(a_search()))
        ended = await ended_by(searching)
    finally:
        await store.close()

    assert isinstance(ended, StoreDefect)


async def test_a_search_waiting_on_a_lock_ends_within_the_write_bound_as_a_timeout(store, database):
    await store.keep(a_read(line(1, "Before the lock")))
    bounded = PostgresLineStore(replace(database, write_seconds=BOUND_SECONDS))
    searching_during_the_lock = a_search()
    try:
        with lines_table_locked(database):
            started = time.monotonic()
            async with asyncio.timeout(LIMIT_SECONDS):
                with pytest.raises(StoreUnavailable) as cut:
                    await bounded.search(searching_during_the_lock)
            elapsed = time.monotonic() - started
            still_waiting = waiting_statements(database)

        after_the_lock = await store.search(a_search())
    finally:
        await bounded.close()

    assert cut.value.code == STORE_TIMEOUT_CODE
    assert still_waiting == 0
    assert BOUND_SECONDS <= elapsed < BOUND_SECONDS + MARGIN_SECONDS
    assert texts_found(after_the_lock) == ["Before the lock"]


RESTART_LINE = "Said before the restart."
PAUSE_LINE = "Said during the pause."


async def until_rows(store: ObservedStore, database: DatabaseTarget, count: int) -> None:
    """The store holds this many lines, once the writes in flight have finished."""
    while len(kept_lines(database, "seq")) < count:
        await store.until_finished(store.finished + 1)


async def said_in_a_pause(runner, store: ObservedStore, database: DatabaseTarget, text: str, rows: int) -> None:
    """A line said while his place was paused, read when it is active again, and kept."""
    pause_seen = runner.app.hold_next_read()
    runner.app.stand(SUBJECT, paused(PLACE_ID))
    await runner.app.until_read_held()
    say_in_the_call(runner, text)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    pause_seen.set()
    await until_rows(store, database, rows)


def told_lines(opening) -> list[str]:
    return [each["text"] for each in opening.told("line")]


async def test_a_line_a_window_held_at_a_restart_and_one_said_in_a_pause_are_told_to_no_turn_and_a_lookup_gives_them(
    runner, store, database
):
    """TURNS-17."""
    runner.remember_in(store)
    runner.gateway = Gateway(tool_round(tool_call("call-0", RECALL_TOOL, {"text": ""})), completion("done"))
    runner.app.speak(PLACE_ID, PERSON, "Ada", RESTART_LINE)
    await restarted_in_the_call(runner)
    await until_rows(store, database, 1)
    runner.clock.time += 60
    await said_in_a_pause(runner, store, database, PAUSE_LINE, 2)
    runner.clock.time += 3600
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()

    (opening,) = runner.gateway.openings()
    found = json.loads(runner.gateway.tool_results(2)[0])
    assert [each["text"] for each in found["lines"]] == [PAUSE_LINE, RESTART_LINE]
    assert told_lines(opening) == [ADDRESSED]


async def test_a_lookup_gives_none_of_the_lines_kept_in_a_pause_while_the_first_entry_of_his_history_is_older(
    runner, store, database
):
    runner.remember_in(store)
    runner.gateway = Gateway(
        *says(PLACE_ID, ANSWER), tool_round(tool_call("call-1", RECALL_TOOL, {"text": ""})), completion("done")
    )
    runner.app.speak(PLACE_ID, PERSON, "Ada", RESTART_LINE)
    await restarted_in_the_call(runner)
    await until_rows(store, database, 1)
    runner.clock.time += 3600
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    await until_rows(store, database, 3)
    runner.clock.time += 3600
    await said_in_a_pause(runner, store, database, PAUSE_LINE, 4)
    runner.clock.time += 3600
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.reports.until_a_turn_completed(count=2)

    _first, second = runner.gateway.openings()
    found = json.loads(runner.gateway.tool_results(4)[0])
    assert [each["text"] for each in found["lines"]] == [RESTART_LINE]
    assert PAUSE_LINE not in told_lines(second)
