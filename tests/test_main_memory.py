"""The ``memory`` commands: what each agent keeps, listed per visit without its content, and erased by agent or all."""

from __future__ import annotations

import asyncio
import csv
import io
import secrets
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from agents.member import PlaceEntry, PlaceRef

from personas.config import CONFIG_ENVIRONMENT_VARIABLE
from personas.decision import Visit, WindowRead
from personas.failures import STORE_DEFECT_CODE, STORE_UNREACHABLE_CODE
from personas.main import CONFIGURATION_ERROR_EXIT_CODE, STORE_ERROR_EXIT_CODE, main
from personas.memory import PostgresLineStore
from tests.support.fakes import WALL_START, window_of
from tests.support.postgres import kept_lines, run_sql

if TYPE_CHECKING:
    from personas.config import MemoryTarget
    from tests.support.configfiles import Files

pytestmark = pytest.mark.usefixtures("library_loggers")

APP = "https://app.test/mcp"
OTHER_APP = "https://other.test/mcp"
AGENT = "agent-mozart"
OTHER_AGENT = "agent-ada"
REMOVED_AGENT = "agent-gone"
CLOSED_PORT = 1
FIRST_VISIT = Visit(APP, PlaceRef(kind="call", id="room-1"), "generation-1")
SECOND_VISIT = Visit(APP, PlaceRef(kind="call", id="room-1"), "generation-2")
OTHER_VISIT = Visit(OTHER_APP, PlaceRef(kind="call", id="room-9"), "generation-1")
LIST = ("memory", "list")
ERASE_ALL = ("memory", "erase", "--all")
RENAME_CLIENT_ID = "ALTER TABLE lines RENAME COLUMN client_id TO renamed"
RENAME_CLIENT_ID_BACK = "ALTER TABLE lines RENAME COLUMN renamed TO client_id"
SCHEMA_BEHIND = "UPDATE alembic_version SET version_num = 'behind'"


class Spoken:
    """The made-up lines of a scenario; their text and their speakers' names must reach no output."""

    def __init__(self) -> None:
        self.held: list[str] = []

    def lines(self, first_seq: int, count: int) -> tuple[PlaceEntry, ...]:
        entries = []
        for seq in range(first_seq, first_seq + count):
            name, text = f"speaker-{secrets.token_hex(8)}", f"text-{secrets.token_hex(8)}"
            self.held += [name, text]
            entries.append(PlaceEntry(seq=seq, kind="message", name=name, text=text, own=False))
        return tuple(entries)


def a_read(client_id: str, visit: Visit, lines: tuple[PlaceEntry, ...], hours_after_start: float) -> WindowRead:
    window = window_of(*lines, generation=visit.generation)
    return WindowRead(client_id, visit.app, visit.place, window, WALL_START + timedelta(hours=hours_after_start))


def kept_by_the_store(target: MemoryTarget, *reads: WindowRead) -> None:
    async def keep_each() -> None:
        store = PostgresLineStore(target)
        try:
            for read in reads:
                await store.keep(read)
        finally:
            await store.close()

    asyncio.run(keep_each())


def three_visits_of_two_agents(target: MemoryTarget, spoken: Spoken) -> None:
    """The first visit is read twice, an hour apart, with a new line the second time."""
    first_lines = spoken.lines(1, 2)
    kept_by_the_store(
        target,
        a_read(AGENT, FIRST_VISIT, first_lines, 0),
        a_read(AGENT, FIRST_VISIT, (*first_lines, *spoken.lines(3, 1)), 1),
        a_read(AGENT, SECOND_VISIT, spoken.lines(1, 1), 2),
        a_read(OTHER_AGENT, OTHER_VISIT, spoken.lines(1, 2), 3),
    )


def configured_for(files: Files, target: MemoryTarget, monkeypatch: pytest.MonkeyPatch) -> None:
    files.text += files.memory_table(
        host=f'"{target.host}"', port=str(target.port), dbname=f'"{target.dbname}"', user=f'"{target.user}"'
    )
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))


def listed(output: str) -> set[tuple[str, str, str, str, str, int, datetime, datetime]]:
    """The visits a list printed, read as tab-separated records below the header."""
    rows = set()
    for record in list(csv.reader(io.StringIO(output), dialect="excel-tab"))[1:]:
        agent, app, kind, place_id, generation, count, first, last = record
        rows.add(
            (
                agent,
                app,
                kind,
                place_id,
                generation,
                int(count),
                datetime.fromisoformat(first),
                datetime.fromisoformat(last),
            )
        )
    return rows


def a_row(agent: str, visit: Visit, count: int, first_hours: float, last_hours: float):
    return (
        agent,
        visit.app,
        visit.place.kind,
        visit.place.id,
        visit.generation,
        count,
        WALL_START + timedelta(hours=first_hours),
        WALL_START + timedelta(hours=last_hours),
    )


def agents_kept(target: MemoryTarget) -> list[object]:
    return [row[0] for row in kept_lines(target, "client_id")]


def no_content_in(shown: str, spoken: Spoken, files: Files) -> None:
    for held in (*spoken.held, files.password):
        assert held not in shown


@pytest.mark.postgres
def test_the_list_shows_one_row_per_visit_with_its_count_and_first_and_last_time_and_no_content(
    files, database, monkeypatch, capsys, caplog
):
    spoken = Spoken()
    three_visits_of_two_agents(database, spoken)
    configured_for(files, database, monkeypatch)

    main(arguments=list(LIST))

    output = capsys.readouterr()
    assert listed(output.out) == {
        a_row(AGENT, FIRST_VISIT, 3, 0, 1),
        a_row(AGENT, SECOND_VISIT, 1, 2, 2),
        a_row(OTHER_AGENT, OTHER_VISIT, 2, 3, 3),
    }
    no_content_in(output.out + output.err + caplog.text, spoken, files)


@pytest.mark.postgres
def test_the_list_for_one_agent_shows_only_that_agents_visits(files, database, monkeypatch, capsys):
    three_visits_of_two_agents(database, Spoken())
    configured_for(files, database, monkeypatch)

    main(arguments=[*LIST, "--agent", OTHER_AGENT])

    assert listed(capsys.readouterr().out) == {a_row(OTHER_AGENT, OTHER_VISIT, 2, 3, 3)}


@pytest.mark.postgres
def test_the_list_shows_a_visit_whose_place_id_holds_a_tab_and_a_line_break_as_one_record(
    files, database, monkeypatch, capsys
):
    visit = Visit(APP, PlaceRef(kind="call", id="room\t1\nforged"), "generation-1")
    kept_by_the_store(database, a_read(AGENT, visit, Spoken().lines(1, 1), 0))
    configured_for(files, database, monkeypatch)

    main(arguments=list(LIST))

    assert listed(capsys.readouterr().out) == {a_row(AGENT, visit, 1, 0, 0)}


@pytest.mark.postgres
def test_the_list_of_an_empty_memory_shows_no_visit(files, database, monkeypatch, capsys):
    configured_for(files, database, monkeypatch)

    main(arguments=list(LIST))

    assert listed(capsys.readouterr().out) == set()


@pytest.mark.postgres
def test_erasing_one_agent_removes_its_lines_keeps_the_others_and_prints_the_number(
    files, database, monkeypatch, capsys, caplog
):
    spoken = Spoken()
    three_visits_of_two_agents(database, spoken)
    configured_for(files, database, monkeypatch)

    main(arguments=["memory", "erase", "--agent", AGENT])

    output = capsys.readouterr()
    assert output.out == "4 lines erased\n"
    assert agents_kept(database) == [OTHER_AGENT, OTHER_AGENT]
    no_content_in(output.out + output.err + caplog.text, spoken, files)


@pytest.mark.postgres
def test_erasing_all_removes_every_line_and_a_later_write_through_the_store_is_kept(
    files, database, monkeypatch, capsys
):
    spoken = Spoken()
    three_visits_of_two_agents(database, spoken)
    configured_for(files, database, monkeypatch)

    main(arguments=list(ERASE_ALL))

    assert capsys.readouterr().out == "6 lines erased\n"
    assert agents_kept(database) == []
    kept_by_the_store(database, a_read(AGENT, FIRST_VISIT, spoken.lines(1, 1), 4))
    assert agents_kept(database) == [AGENT]


@pytest.mark.postgres
def test_erasing_an_agent_without_lines_prints_zero_also_when_the_configuration_does_not_name_it(
    files, database, monkeypatch, capsys
):
    three_visits_of_two_agents(database, Spoken())
    configured_for(files, database, monkeypatch)

    main(arguments=["memory", "erase", "--agent", REMOVED_AGENT])

    assert capsys.readouterr().out == "0 lines erased\n"
    assert len(agents_kept(database)) == 6


@pytest.mark.postgres
@pytest.mark.parametrize(
    "arguments",
    [
        ["memory"],
        ["memory", "erase"],
        ["memory", "erase", "--agent", AGENT, "--all"],
        ["memory", "erase", "--all", "--everything"],
        ["memory", "list", "everything"],
        ["memory", "forget"],
        ["memory", "erase", "--agent", AGENT, "--agent", OTHER_AGENT],
        ["memory", "list", "--agent", AGENT, "--agent", OTHER_AGENT],
        ["memory", "erase", "--al"],
    ],
    ids=[
        "memory-alone",
        "no-selector",
        "both-selectors",
        "unknown-option",
        "unknown-argument",
        "unknown-command",
        "repeated-agent",
        "repeated-agent-in-list",
        "abbreviated-option",
    ],
)
def test_a_memory_command_used_wrongly_is_refused_with_its_usage_and_every_line_remains(
    files, database, monkeypatch, capsys, arguments
):
    three_visits_of_two_agents(database, Spoken())
    configured_for(files, database, monkeypatch)

    with pytest.raises(SystemExit) as refused:
        main(arguments=arguments)

    assert refused.value.code == CONFIGURATION_ERROR_EXIT_CODE
    assert "usage: personas memory" in capsys.readouterr().err
    assert len(agents_kept(database)) == 6


@pytest.mark.parametrize("arguments", [LIST, ERASE_ALL], ids=["list", "erase"])
def test_a_memory_command_without_a_memory_table_stops_and_names_the_table(files, monkeypatch, capsys, arguments):
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))

    with pytest.raises(SystemExit) as stopped:
        main(arguments=list(arguments))

    assert stopped.value.code == CONFIGURATION_ERROR_EXIT_CODE
    assert "personas: memory: the table is required" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [LIST, ERASE_ALL], ids=["list", "erase"])
def test_a_memory_command_against_a_store_that_cannot_be_reached_stops_with_its_code_and_no_traceback(
    files, monkeypatch, capsys, arguments
):
    files.text += files.memory_table(host='"127.0.0.1"', port=str(CLOSED_PORT))
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))

    with pytest.raises(SystemExit) as stopped:
        main(arguments=list(arguments))

    assert stopped.value.code == STORE_ERROR_EXIT_CODE
    assert stopped.value.__cause__ is None
    assert capsys.readouterr().err == f"personas: memory: {STORE_UNREACHABLE_CODE}\n"


@pytest.mark.postgres
@pytest.mark.parametrize("arguments", [LIST, ("memory", "erase", "--agent", AGENT)], ids=["list", "erase"])
def test_a_memory_command_met_by_a_database_error_stops_with_its_code_and_deletes_nothing(
    files, database, monkeypatch, capsys, caplog, arguments
):
    spoken = Spoken()
    three_visits_of_two_agents(database, spoken)
    run_sql(database, RENAME_CLIENT_ID)
    configured_for(files, database, monkeypatch)

    with pytest.raises(SystemExit) as stopped:
        main(arguments=list(arguments))

    assert stopped.value.code == STORE_ERROR_EXIT_CODE
    assert stopped.value.__cause__ is None
    output = capsys.readouterr()
    assert output.err == f"personas: memory: {STORE_DEFECT_CODE}\n"
    run_sql(database, RENAME_CLIENT_ID_BACK)
    assert len(agents_kept(database)) == 6
    no_content_in(output.out + output.err + caplog.text, spoken, files)


@pytest.mark.postgres
@pytest.mark.parametrize("arguments", [LIST, ERASE_ALL], ids=["list", "erase"])
def test_a_memory_command_on_a_schema_not_at_its_head_refuses_naming_the_migration_and_deletes_nothing(
    files, database, monkeypatch, capsys, arguments
):
    three_visits_of_two_agents(database, Spoken())
    run_sql(database, SCHEMA_BEHIND)
    configured_for(files, database, monkeypatch)

    with pytest.raises(SystemExit) as stopped:
        main(arguments=list(arguments))

    assert stopped.value.code == STORE_ERROR_EXIT_CODE
    assert stopped.value.__cause__ is None
    error = capsys.readouterr().err
    assert error.startswith("personas: memory:")
    assert "personas migrate" in error
    assert len(agents_kept(database)) == 6


@pytest.mark.postgres
def test_a_window_read_again_after_its_agent_was_erased_is_kept_again(files, database, monkeypatch, capsys):
    spoken = Spoken()
    in_the_call = a_read(AGENT, FIRST_VISIT, spoken.lines(1, 2), 0)
    kept_by_the_store(database, in_the_call)
    configured_for(files, database, monkeypatch)
    main(arguments=["memory", "erase", "--agent", AGENT])

    kept_by_the_store(database, in_the_call)

    assert agents_kept(database) == [AGENT, AGENT]
    assert capsys.readouterr().out == "2 lines erased\n"
