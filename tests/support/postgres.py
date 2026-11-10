"""The tests' Postgres: one migrated template database per session and one database per test copied from it."""

from __future__ import annotations

import os
import secrets
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from personas.config import MemoryTarget
from personas.memory import migrate
from personas.secret import Secret

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine
    from sqlalchemy.engine import URL

URL_ENVIRONMENT_VARIABLE = "PERSONAS_TEST_POSTGRES_URL"
WRAPPER = "scripts/test_postgres.sh"
DATABASE_NAME_BYTES = 8


def server_url() -> URL:
    """The throwaway server's URL; the wrapper and CI's service provide it."""
    url = os.environ.get(URL_ENVIRONMENT_VARIABLE)
    if not url:
        pytest.fail(f"{URL_ENVIRONMENT_VARIABLE} names no Postgres; run the tests through {WRAPPER}")
    return make_url(url)


def engine_of(database: str) -> Engine:
    """A synchronous engine on one database of the server, for the tests' own plain SQL."""
    return create_engine(server_url().set(database=database), isolation_level="AUTOCOMMIT")


def target_of(database: str, *, write_seconds: float = 2.0) -> MemoryTarget:
    """What the runner's configuration holds for this database; the password is the test's own."""
    url = server_url()
    assert url.host is not None
    assert url.port is not None
    assert url.username is not None
    return MemoryTarget(url.host, url.port, database, url.username, Secret(secrets.token_hex(8)), write_seconds)


@contextmanager
def created_database(template: str | None = None) -> Iterator[str]:
    """The name of a new database, a copy of ``template`` when one is named; it is dropped afterwards."""
    name = f"personas_{secrets.token_hex(DATABASE_NAME_BYTES)}"
    server = engine_of(server_url().database or "postgres")
    with server.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"' + ("" if template is None else f' TEMPLATE "{template}"')))
    try:
        yield name
    finally:
        with server.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        server.dispose()


@contextmanager
def migrated_template() -> Iterator[str]:
    """A database migrated once through the command's own code, which ``created_database`` copies for each test."""
    with created_database() as name:
        migrate(target_of(name))
        yield name


def kept_lines(target: MemoryTarget, columns: str) -> list[tuple[object, ...]]:
    """The rows of the lines table, read with plain SQL in the order of agent and ``seq``."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT {columns} FROM lines ORDER BY client_id, seq"))  # noqa: S608 - test-chosen column names
            return [tuple(row) for row in result]
    finally:
        engine.dispose()


def run_sql(target: MemoryTarget, statement: str) -> None:
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            connection.execute(text(statement))
    finally:
        engine.dispose()


def waiting_statements(target: MemoryTarget) -> int:
    """How many statements of this database wait for a lock right now."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            return connection.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE datname = :name AND wait_event_type = 'Lock'"),
                {"name": target.dbname},
            ).scalar_one()
    finally:
        engine.dispose()


@contextmanager
def lines_table_locked(target: MemoryTarget) -> Iterator[None]:
    """Another session holds the lines table, so that a write to it waits."""
    engine = create_engine(server_url().set(database=target.dbname))
    with engine.connect() as connection:
        connection.execute(text("LOCK TABLE lines IN ACCESS EXCLUSIVE MODE"))
        try:
            yield
        finally:
            connection.rollback()
    engine.dispose()


def tables_of(target: MemoryTarget) -> list[str]:
    """The names of the tables in the database's public schema."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY 1")
            )
            return [row[0] for row in result]
    finally:
        engine.dispose()
