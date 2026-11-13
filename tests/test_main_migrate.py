"""The ``migrate`` command: it brings the memory's schema to its head, and stops with the field named without one."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.exc import IntegrityError

from personas.config import CONFIG_ENVIRONMENT_VARIABLE
from personas.failures import STORE_UNREACHABLE_CODE
from personas.main import CONFIGURATION_ERROR_EXIT_CODE, MIGRATE_COMMAND, STORE_ERROR_EXIT_CODE, command_line, main
from tests.support.postgres import kept_lines, migrated_to, run_sql, tables_of

if TYPE_CHECKING:
    from tests.support.configfiles import Files

CLOSED_PORT = 1


def migrate(files: Files, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))
    main(arguments=[MIGRATE_COMMAND])


@pytest.mark.postgres
def test_migrating_an_empty_database_creates_the_schema_and_migrating_again_changes_nothing(
    files, unmigrated_database, monkeypatch, library_loggers
):
    target = unmigrated_database
    files.text += files.memory_table(
        host=f'"{target.host}"', port=str(target.port), dbname=f'"{target.dbname}"', user=f'"{target.user}"'
    )
    migrate(files, monkeypatch)
    run_sql(
        unmigrated_database,
        "INSERT INTO lines VALUES ('a','b','c','d','e',1,now(),'f','g','h',false,false,false,false)",
    )
    after_the_first = (tables_of(unmigrated_database), kept_lines(unmigrated_database, "client_id, seq, first_read_at"))

    migrate(files, monkeypatch)

    after_the_second = (
        tables_of(unmigrated_database),
        kept_lines(unmigrated_database, "client_id, seq, first_read_at"),
    )

    assert "lines" in after_the_first[0]
    assert after_the_second == after_the_first


@pytest.mark.postgres
def test_migrating_a_database_at_the_first_revision_keeps_its_lines_and_reads_them_back_as_not_cut(
    files, unmigrated_database, monkeypatch, library_loggers
):
    target = unmigrated_database
    files.text += files.memory_table(
        host=f'"{target.host}"', port=str(target.port), dbname=f'"{target.dbname}"', user=f'"{target.user}"'
    )
    migrated_to(target, "0001")
    run_sql(target, "INSERT INTO lines VALUES ('a','b','c','d','e',1,now(),'f','g','h',false,true)")

    migrate(files, monkeypatch)

    assert kept_lines(target, "client_id, spoken, cut") == [("a", True, False)]


@pytest.mark.postgres
def test_migrating_a_database_at_the_second_revision_keeps_its_lines_and_reads_them_back_as_no_act(
    files, unmigrated_database, monkeypatch, library_loggers
):
    target = unmigrated_database
    files.text += files.memory_table(
        host=f'"{target.host}"', port=str(target.port), dbname=f'"{target.dbname}"', user=f'"{target.user}"'
    )
    migrated_to(target, "0002")
    run_sql(target, "INSERT INTO lines VALUES ('a','b','c','d','e',1,now(),'f','g','h',false,true,true)")

    migrate(files, monkeypatch)

    assert kept_lines(target, "client_id, cut, act") == [("a", True, False)]
    with pytest.raises(IntegrityError):
        run_sql(target, "INSERT INTO lines VALUES ('a','b','c','d','e',2,now(),'f','g','h',false,true,true,NULL)")
    with pytest.raises(IntegrityError):
        run_sql(
            target,
            "INSERT INTO lines (client_id, app, place_kind, place_id, generation, seq, first_read_at, speaker, kind,"
            " text, own, spoken, cut) VALUES ('a','b','c','d','e',3,now(),'f','g','h',false,true,true)",
        )


def test_migrating_without_a_memory_table_stops_and_names_the_field(files, monkeypatch, capsys, library_loggers):
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))

    with pytest.raises(SystemExit) as stopped:
        main(arguments=[MIGRATE_COMMAND])

    assert stopped.value.code == CONFIGURATION_ERROR_EXIT_CODE
    assert "memory" in capsys.readouterr().err


def test_migrating_a_store_that_cannot_be_reached_stops_with_its_code_and_no_traceback(
    files, monkeypatch, capsys, library_loggers
):
    files.text += files.memory_table(host='"127.0.0.1"', port=str(CLOSED_PORT))

    with pytest.raises(SystemExit) as stopped:
        migrate(files, monkeypatch)

    assert stopped.value.code == STORE_ERROR_EXIT_CODE
    assert stopped.value.__cause__ is None
    error = capsys.readouterr().err
    assert error.startswith("personas: memory:")
    assert STORE_UNREACHABLE_CODE in error


def test_an_unknown_command_of_the_console_script_stops_before_the_configuration_is_read(monkeypatch, capsys):
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(sys, "argv", ["personas", "serve"])

    with pytest.raises(SystemExit) as stopped:
        command_line()

    assert stopped.value.code == CONFIGURATION_ERROR_EXIT_CODE
    assert "unknown command" in capsys.readouterr().err
