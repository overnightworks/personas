"""The ``migrate`` command: it brings the schema to its head and leaves one a newer release migrated as it is."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.exc import IntegrityError

import personas
from personas.config import CONFIG_ENVIRONMENT_VARIABLE
from personas.failures import STORE_UNREACHABLE_CODE
from personas.main import (
    CONFIGURATION_ERROR_EXIT_CODE,
    MEMORY_COMMAND,
    MIGRATE_COMMAND,
    STORE_ERROR_EXIT_CODE,
    command_line,
    main,
)
from tests.support.pods import ENTRY_POINT_CALL
from tests.support.postgres import at_an_unknown_revision, kept_lines, migrated_to, revision_of, run_sql, tables_of

if TYPE_CHECKING:
    from tests.support.configfiles import Files

CLOSED_PORT = 1


def migrate(files: Files, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]))
    main(arguments=[MIGRATE_COMMAND])


@pytest.mark.postgres
def test_migrating_an_empty_database_creates_the_schema_and_migrating_again_changes_nothing(
    files, unmigrated_database, monkeypatch, restored_logging
):
    target = unmigrated_database
    files.use_database(target)
    migrate(files, monkeypatch)
    run_sql(target, "INSERT INTO lines VALUES ('a','b','c','d','e',1,now(),'f','g','h',false,false,false,false)")
    after_the_first = (tables_of(target), revision_of(target), kept_lines(target, "client_id, seq, first_read_at"))

    migrate(files, monkeypatch)

    after_the_second = (tables_of(target), revision_of(target), kept_lines(target, "client_id, seq, first_read_at"))

    tables, revision, _ = after_the_first
    assert {"alembic_version", "lines"} <= set(tables)
    assert revision is not None
    assert after_the_second == after_the_first


@pytest.mark.postgres
def test_migrating_a_database_at_the_first_revision_keeps_its_lines_and_reads_them_back_as_not_cut(
    files, unmigrated_database, monkeypatch, restored_logging
):
    target = unmigrated_database
    files.use_database(target)
    migrated_to(target, "0001")
    run_sql(target, "INSERT INTO lines VALUES ('a','b','c','d','e',1,now(),'f','g','h',false,true)")

    migrate(files, monkeypatch)

    assert kept_lines(target, "client_id, spoken, cut") == [("a", True, False)]


@pytest.mark.postgres
def test_migrating_a_database_at_the_second_revision_keeps_its_lines_and_reads_them_back_as_no_act(
    files, unmigrated_database, monkeypatch, restored_logging
):
    target = unmigrated_database
    files.use_database(target)
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


@pytest.mark.postgres
def test_migrating_a_database_at_a_revision_it_does_not_know_changes_nothing_and_stops_with_no_error(
    files, database, monkeypatch, restored_logging
):
    files.use_database(database)
    newer = at_an_unknown_revision(database)

    migrate(files, monkeypatch)

    assert revision_of(database) == newer


@pytest.mark.postgres
def test_the_console_script_prints_nothing_when_it_migrates_an_empty_or_a_migrated_database(files, unmigrated_database):
    files.use_database(unmigrated_database)
    environment = {
        **os.environ,
        CONFIG_ENVIRONMENT_VARIABLE: str(files.write()[CONFIG_ENVIRONMENT_VARIABLE]),
        "PYTHONPATH": str(Path(personas.__file__).parent.parent),
    }

    first, second = (
        subprocess.run(
            [sys.executable, "-c", ENTRY_POINT_CALL, MIGRATE_COMMAND],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        for _ in range(2)
    )

    assert [(each.returncode, each.stdout, each.stderr) for each in (first, second)] == [(0, "", "")] * 2


def test_migrating_a_store_that_cannot_be_reached_stops_with_its_code_and_no_traceback(
    files, monkeypatch, capsys, restored_logging
):
    files.set_database(host='"127.0.0.1"', port=str(CLOSED_PORT))

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
    assert "invalid choice: 'serve'" in capsys.readouterr().err


def test_the_help_of_the_console_script_names_each_command_on_one_line_and_exits_zero(monkeypatch, capsys):
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(sys, "argv", ["personas", "--help"])

    with pytest.raises(SystemExit) as stopped:
        command_line()

    assert stopped.value.code == 0
    lines = [each.split() for each in capsys.readouterr().out.splitlines()]
    for command in (MIGRATE_COMMAND, MEMORY_COMMAND):
        described = [words for words in lines if words[:1] == [command]]
        assert len(described) == 1
        assert len(described[0]) > 1
