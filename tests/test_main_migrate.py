"""The ``migrate`` command: it brings the memory's schema to its head, and stops with the field named without one."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from personas.config import CONFIG_ENVIRONMENT_VARIABLE
from personas.failures import STORE_UNREACHABLE_CODE
from personas.main import CONFIGURATION_ERROR_EXIT_CODE, MIGRATE_COMMAND, STORE_ERROR_EXIT_CODE, command_line, main
from tests.support.postgres import kept_lines, run_sql, tables_of

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
    run_sql(unmigrated_database, "INSERT INTO lines VALUES ('a','b','c','d','e',1,now(),'f','g','h',false,false)")
    after_the_first = (tables_of(unmigrated_database), kept_lines(unmigrated_database, "client_id, seq, first_read_at"))

    migrate(files, monkeypatch)

    assert "lines" in after_the_first[0]
    assert (tables_of(unmigrated_database), kept_lines(unmigrated_database, "client_id, seq, first_read_at")) == (
        after_the_first
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
