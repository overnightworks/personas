"""The tests' Postgres: one migrated template database per session and one database per test copied from it."""

from __future__ import annotations

import asyncio
import os
import secrets
import string
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from personas.config import DatabaseTarget
from personas.memory import MIGRATIONS_DIRECTORY, migrate
from personas.secret import Secret

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Iterator

    from sqlalchemy import Engine
    from sqlalchemy.engine import URL

URL_ENVIRONMENT_VARIABLE = "PERSONAS_TEST_POSTGRES_URL"
WRAPPER = "scripts/test_postgres.sh"
DATABASE_NAME_BYTES = 8
LOOPBACK = "127.0.0.1"
FORWARD_CHUNK_BYTES = 65536


def server_url() -> URL:
    """The throwaway server's URL; the wrapper and CI's service provide it."""
    url = os.environ.get(URL_ENVIRONMENT_VARIABLE)
    if not url:
        pytest.fail(f"{URL_ENVIRONMENT_VARIABLE} names no Postgres; run the tests through {WRAPPER}")
    return make_url(url)


def engine_of(database: str) -> Engine:
    """A synchronous engine on one database of the server, for the tests' own plain SQL."""
    return create_engine(server_url().set(database=database), isolation_level="AUTOCOMMIT")


def target_of(database: str) -> DatabaseTarget:
    """What the runner's configuration holds for this database; the password is the test's own."""
    url = server_url()
    assert url.host is not None
    assert url.port is not None
    assert url.username is not None
    return DatabaseTarget(url.host, url.port, database, url.username, Secret(secrets.token_hex(8)))


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


def migrated_to(target: DatabaseTarget, revision: str) -> None:
    """Brings the target's database up to one revision, as a deployment that has not yet taken the newer ones."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIRECTORY))
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, revision)
    finally:
        engine.dispose()


@contextmanager
def migrated_template() -> Iterator[str]:
    """A database migrated once through the command's own code, which ``created_database`` copies for each test."""
    with created_database() as name:
        migrate(target_of(name))
        yield name


def kept_lines(target: DatabaseTarget, columns: str) -> list[tuple[object, ...]]:
    """The rows of the lines table, read with plain SQL in the order of agent and ``seq``."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT {columns} FROM lines ORDER BY client_id, seq"))  # noqa: S608 - test-chosen column names
            return [tuple(row) for row in result]
    finally:
        engine.dispose()


def revision_of(target: DatabaseTarget) -> str | None:
    """The migration revision the database holds."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            return connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
    finally:
        engine.dispose()


def run_sql(target: DatabaseTarget, statement: str) -> None:
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            connection.execute(text(statement))
    finally:
        engine.dispose()


def at_an_unknown_revision(target: DatabaseTarget) -> str:
    """Sets the database to a revision the migration scripts do not know and returns it."""
    unknown = "".join(secrets.choice(string.ascii_lowercase) for _ in range(12))
    run_sql(target, f"UPDATE alembic_version SET version_num = '{unknown}'")  # noqa: S608 - a test-generated name
    return unknown


def waiting_statements(target: DatabaseTarget) -> int:
    """How many statements of this database wait for a lock right now."""
    engine = engine_of(target.dbname)
    try:
        return _waiting_statements_through(engine, target.dbname)
    finally:
        engine.dispose()


def _waiting_statements_through(engine: Engine, database: str) -> int:
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT count(*) FROM pg_stat_activity WHERE datname = :name AND wait_event_type = 'Lock'"),
            {"name": database},
        ).scalar_one()


@contextmanager
def lines_table_locked(target: DatabaseTarget) -> Iterator[None]:
    """Another session holds the lines table, so that a write to it waits."""
    engine = create_engine(server_url().set(database=target.dbname))
    with engine.connect() as connection:
        connection.execute(text("LOCK TABLE lines IN ACCESS EXCLUSIVE MODE"))
        try:
            yield
        finally:
            connection.rollback()
    engine.dispose()


@contextmanager
def uncommitted_behind_a_lock(
    target: DatabaseTarget, table: str, statements: Iterable[str]
) -> Iterator[Callable[[], None]]:
    """Another session holds ``table`` and has run ``statements`` without committing them; calling what this yields
    commits them and lets go of the table, and leaving rolls back what was not committed."""
    engine = create_engine(server_url().set(database=target.dbname))
    with engine.connect() as connection:
        connection.execute(text(f"LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE"))
        for statement in statements:
            connection.execute(text(statement))
        try:
            yield connection.commit
        finally:
            connection.rollback()
    engine.dispose()


async def until_a_statement_waits(target: DatabaseTarget) -> None:
    """Returns once a statement of this database waits for a lock; one engine asks on every pass."""
    engine = engine_of(target.dbname)
    try:
        while not await asyncio.to_thread(_waiting_statements_through, engine, target.dbname):
            continue
    finally:
        engine.dispose()


class Forwarder:
    """A TCP forwarder in front of the server whose open connections can go dark: their bytes are then dropped."""

    def __init__(self, target: DatabaseTarget) -> None:
        self._target = target
        self._dark: set[asyncio.StreamWriter] = set()
        self._open: set[asyncio.StreamWriter] = set()
        self.dropped = asyncio.Event()

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[DatabaseTarget]:
        """The target as seen through the forwarder; every connection is closed at the end."""
        server = await asyncio.start_server(self._forward, LOOPBACK, 0)
        port = server.sockets[0].getsockname()[1]
        try:
            yield replace(self._target, host=LOOPBACK, port=port)
        finally:
            server.close()
            for writer in self._open:
                writer.transport.abort()
            await server.wait_closed()

    def go_dark(self) -> None:
        """Every connection open now stops answering, in both directions; new ones are forwarded."""
        self._dark |= self._open

    def reset_dark(self) -> None:
        """Every dark connection is reset, as a peer that is gone answers a later packet."""
        for writer in self._dark:
            writer.transport.abort()

    async def _forward(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        server_reader, server_writer = await asyncio.open_connection(self._target.host, self._target.port)
        self._open |= {client_writer, server_writer}
        await asyncio.gather(
            self._pipe(client_reader, server_writer),
            self._pipe(server_reader, client_writer),
            return_exceptions=True,
        )

    async def _pipe(self, source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
        try:
            while data := await source.read(FORWARD_CHUNK_BYTES):
                if sink in self._dark:
                    self.dropped.set()
                    continue
                sink.write(data)
                await sink.drain()
        finally:
            sink.transport.abort()
            self._open.discard(sink)


def tables_of(target: DatabaseTarget) -> list[str]:
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


def lock_holders(target: DatabaseTarget) -> list[int]:
    """The backend of each session holding an advisory lock of the two-integer form in the target's database.

    A session appears as often as it holds a lock; those of other databases on the server are not seen."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            return [
                pid
                for (pid,) in connection.execute(
                    text(
                        "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND objsubid = 2 AND granted "
                        "AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
                    )
                )
            ]
    finally:
        engine.dispose()


def terminate_backend(target: DatabaseTarget, pid: int) -> None:
    """Ends the session of this backend, as an operator or a failover does."""
    engine = engine_of(target.dbname)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
    finally:
        engine.dispose()
