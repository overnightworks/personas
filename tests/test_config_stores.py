"""The [database], [ownership] and [memory] tables, read through the entry point."""

from __future__ import annotations

import math

import pytest

from personas.config import (
    DEFAULT_CANCEL_MARGIN_SECONDS,
    DEFAULT_CONTEND_SECONDS,
    DEFAULT_GRACE_SECONDS,
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_LINES_KEEP_DAYS,
    DEFAULT_LOOKUP_BYTES,
    DEFAULT_LOOKUP_CALLS,
    DEFAULT_STOP_SECONDS,
    DEFAULT_SWEEP_MINUTES,
    DEFAULT_WRITE_SECONDS,
    OwnershipTimings,
    load_config,
)
from personas.recall import SMALLEST_RESULT_BYTES
from tests.support.configfiles import fails_at_start

pytestmark = pytest.mark.usefixtures("restored_logging")


def test_a_database_table_gives_the_runner_its_database_its_password_and_a_default_write_bound(files):
    """CONFIG-22, CONFIG-23, CONFIG-24."""
    database = load_config(files.write()).database

    assert (database.host, database.port, database.dbname, database.user) == ("db.test", 5432, "personas", "mozart")
    assert database.password.reveal() == files.password
    assert database.write_seconds == DEFAULT_WRITE_SECONDS


@pytest.mark.parametrize(("written", "bound"), [("0.5", 0.5), ("inf", math.inf)], ids=["seconds", "no bound"])
def test_a_database_table_may_set_the_write_bound_or_leave_a_write_unbounded(files, written, bound):
    """CONFIG-24."""
    files.set_database(write_seconds=written)

    assert load_config(files.write()).database.write_seconds == bound


def test_a_file_without_a_database_table_stops_the_start_and_names_it(files, monkeypatch, capsys):
    """CONFIG-22."""
    files.text = files.text.replace(files.database_table(), "")

    assert "database" in fails_at_start(files.write(), monkeypatch, capsys)


def test_a_file_without_an_ownership_table_shares_the_agents_by_the_defaults(files):
    """CONFIG-26, CONFIG-27, CONFIG-28, CONFIG-29, CONFIG-30."""
    assert load_config(files.write()).ownership == OwnershipTimings(
        DEFAULT_CONTEND_SECONDS,
        DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        DEFAULT_CANCEL_MARGIN_SECONDS,
        DEFAULT_GRACE_SECONDS,
        DEFAULT_STOP_SECONDS,
    )


def test_an_ownership_table_sets_each_of_its_times(files):
    """CONFIG-26, CONFIG-27, CONFIG-28, CONFIG-29, CONFIG-30."""
    files.text += (
        "[ownership]\ncontend_seconds = 1.5\nheartbeat_timeout_seconds = 2.5\n"
        "cancel_margin_seconds = 0.5\ngrace_seconds = 9\nstop_seconds = 0.75\n"
    )

    assert load_config(files.write()).ownership == OwnershipTimings(1.5, 2.5, 0.5, 9, 0.75)


@pytest.mark.parametrize(
    ("outlasted", "field", "bound"),
    [
        ("contend_seconds = 2.0\nheartbeat_timeout_seconds = 1.5\ncancel_margin_seconds = 1.0\n", "grace_seconds", 4.5),
        ("cancel_margin_seconds = 1.0\n", "stop_seconds", 1.0),
    ],
    ids=["grace", "stop"],
)
@pytest.mark.parametrize(
    ("over", "starts"), [(-0.5, False), (0.0, False), (0.1, True)], ids=["below", "equal", "above"]
)
def test_a_grace_or_a_stop_at_or_below_the_times_it_must_outlast_is_refused(
    files, monkeypatch, capsys, outlasted, field, bound, over, starts
):
    """CONFIG-31."""
    files.text += f"[ownership]\n{outlasted}{field} = {bound + over}\n"

    if starts:
        assert getattr(load_config(files.write()).ownership, field) == bound + over
    else:
        assert field in fails_at_start(files.write(), monkeypatch, capsys)


@pytest.mark.parametrize(
    "field", ["contend_seconds", "heartbeat_timeout_seconds", "cancel_margin_seconds", "grace_seconds", "stop_seconds"]
)
@pytest.mark.parametrize("written", ["0", "-1", '"5"', "inf", "nan"])
def test_an_ownership_time_that_is_no_positive_finite_number_stops_the_start_and_names_the_field(
    files, monkeypatch, capsys, field, written
):
    """CONFIG-31."""
    files.text += f"[ownership]\n{field} = {written}\n"

    assert f"ownership.{field}" in fails_at_start(files.write(), monkeypatch, capsys)


@pytest.mark.parametrize(
    ("written", "days", "minutes"),
    [
        ({}, DEFAULT_LINES_KEEP_DAYS, DEFAULT_SWEEP_MINUTES),
        ({"lines_keep_days": "inf", "sweep_minutes": "inf"}, math.inf, math.inf),
    ],
    ids=["defaults", "no limit"],
)
def test_a_memory_table_keeps_and_sweeps_by_default_unless_it_sets_either_or_leaves_it_infinite(
    files, written, days, minutes
):
    """CONFIG-34, CONFIG-35."""
    files.text += files.memory_table(**written)

    memory = load_config(files.write()).memory

    assert memory is not None
    assert memory.lines_keep_days == days
    assert memory.sweep_minutes == minutes


@pytest.mark.parametrize(
    ("written", "calls", "result_bytes"),
    [
        ({}, DEFAULT_LOOKUP_CALLS, DEFAULT_LOOKUP_BYTES),
        ({"lookup_calls": "5", "lookup_bytes": "300"}, 5, 300),
        ({"lookup_calls": "1", "lookup_bytes": str(SMALLEST_RESULT_BYTES)}, 1, SMALLEST_RESULT_BYTES),
    ],
    ids=["defaults", "set", "the least"],
)
def test_a_memory_table_lets_a_turn_look_up_twice_within_8192_bytes_unless_it_sets_either(
    files, written, calls, result_bytes
):
    """CONFIG-36, CONFIG-37."""
    files.text += files.memory_table(**written)

    memory = load_config(files.write()).memory

    assert memory is not None
    assert (memory.lookup_calls, memory.lookup_bytes) == (calls, result_bytes)


@pytest.mark.parametrize("field", ["lookup_calls", "lookup_bytes"])
@pytest.mark.parametrize("written", ["0", "-1", "1.5", "2.0", '"2"', "true", "inf", "nan"])
def test_a_lookup_bound_that_is_no_positive_integer_stops_the_start_and_names_the_field(
    files, monkeypatch, capsys, field, written
):
    """CONFIG-36, CONFIG-37."""
    files.text += files.memory_table(**{field: written})

    assert f"memory.{field}" in fails_at_start(files.write(), monkeypatch, capsys)


def test_a_result_bound_too_small_for_a_result_without_a_line_stops_the_start_and_names_the_field(
    files, monkeypatch, capsys
):
    """CONFIG-37."""
    files.text += files.memory_table(lookup_bytes=str(SMALLEST_RESULT_BYTES - 1))

    assert "memory.lookup_bytes" in fails_at_start(files.write(), monkeypatch, capsys)


def test_a_file_without_a_memory_table_opens_no_store(files):
    """CONFIG-33."""
    assert load_config(files.write()).memory is None


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"host": None}, "database.host"),
        ({"host": '""'}, "database.host"),
        ({"port": None}, "database.port"),
        ({"port": '"5432"'}, "database.port"),
        ({"port": "70000"}, "database.port"),
        ({"dbname": None}, "database.dbname"),
        ({"user": None}, "database.user"),
        ({"password_file": None}, "database.password_file"),
        ({"write_seconds": "0"}, "database.write_seconds"),
        ({"write_seconds": '"2"'}, "database.write_seconds"),
        ({"write_seconds": "nan"}, "database.write_seconds"),
        ({"password": '"in-the-file"'}, "database.password"),
    ],
    ids=[
        "host missing",
        "host empty",
        "port missing",
        "port a string",
        "port out of range",
        "database missing",
        "user missing",
        "password file missing",
        "write bound zero",
        "write bound a string",
        "write bound no number",
        "password written into the file",
    ],
)
def test_a_missing_or_malformed_database_field_stops_the_start_and_names_it(
    files, monkeypatch, capsys, overrides, named
):
    """CONFIG-22, CONFIG-23, CONFIG-24."""
    files.set_database(**overrides)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert "in-the-file" not in error


@pytest.mark.parametrize("field", ["host", "port", "dbname", "user", "password_file", "write_seconds"])
def test_a_connection_field_in_the_memory_table_stops_the_start_and_names_it(files, monkeypatch, capsys, field):
    files.text += f"[memory]\n{field} = 1\n"

    assert f"memory.{field}" in fails_at_start(files.write(), monkeypatch, capsys)


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"lines_keep_days": "0"}, "memory.lines_keep_days"),
        ({"lines_keep_days": "-3"}, "memory.lines_keep_days"),
        ({"lines_keep_days": '"30"'}, "memory.lines_keep_days"),
        ({"lines_keep_days": "nan"}, "memory.lines_keep_days"),
        ({"sweep_minutes": "0"}, "memory.sweep_minutes"),
        ({"sweep_minutes": "-3"}, "memory.sweep_minutes"),
        ({"sweep_minutes": '"60"'}, "memory.sweep_minutes"),
        ({"sweep_minutes": "nan"}, "memory.sweep_minutes"),
    ],
    ids=[
        "keep period of no days",
        "keep period negative",
        "keep period a string",
        "keep period no number",
        "sweep period of no minutes",
        "sweep period negative",
        "sweep period a string",
        "sweep period no number",
    ],
)
def test_a_malformed_memory_field_stops_the_start_and_names_it(files, monkeypatch, capsys, overrides, named):
    """CONFIG-34, CONFIG-35, CONFIG-36, CONFIG-37."""
    files.text += files.memory_table(**overrides)

    assert named in fails_at_start(files.write(), monkeypatch, capsys)


@pytest.mark.parametrize(
    "arrange",
    [
        lambda files: files.password_path.unlink(),
        lambda files: files.password_path.write_text("  \n", encoding="utf-8"),
        lambda files: setattr(files, "text", files.text.replace(str(files.password_path), "memory.password")),
    ],
    ids=["missing", "empty", "relative path"],
)
def test_a_password_file_that_is_unusable_stops_the_start_and_names_the_field(files, monkeypatch, capsys, arrange):
    """CONFIG-23."""
    arrange(files)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "database.password_file" in error
