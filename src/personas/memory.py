"""The memory: a Postgres table of the lines every agent read, verbatim, and the migration that owns its schema.

The runner writes through ``PostgresLineStore``; ``migrate`` is the only code that changes the schema. A line is
keyed by the agent, the app, the place, the visit and its ``seq``, so reading a window again changes nothing and the
time of the first read stands. No database error reaches a log or a traceback with a parameter or a row in it.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    URL,
    Boolean,
    Column,
    DateTime,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from personas.failures import (
    STORE_DEFECT_CODE,
    STORE_TIMEOUT_CODE,
    STORE_UNREACHABLE_CODE,
    StoreDefect,
    StoreUnavailable,
    task_is_being_cancelled,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from personas.config import MemoryTarget
    from personas.decision import WindowRead

MIGRATIONS_DIRECTORY: Final = Path(__file__).parent / "migrations"
# Serialises concurrent `personas migrate` runs; any constant shared by all of them does.
MIGRATION_LOCK_KEY: Final = 5_201_907_364
POSTGRES_DRIVER: Final = "postgresql+psycopg"
MINIMUM_CONNECT_SECONDS: Final = 2
"""The least connect timeout the driver honours; it raises a smaller one to this and truncates a
fraction, so the bound is rounded up."""
NUL: Final = "\x00"
REPLACEMENT: Final = "�"

_LINES: Final = Table(
    "lines",
    MetaData(),
    Column("client_id", Text),
    Column("app", Text),
    Column("place_kind", Text),
    Column("place_id", Text),
    Column("generation", Text),
    # The wire bounds a line's seq only from below; the migration's check refuses a fraction.
    Column("seq", Numeric()),
    Column("first_read_at", DateTime(timezone=True)),
    Column("speaker", Text),
    Column("kind", Text),
    Column("text", Text),
    Column("own", Boolean),
    Column("spoken", Boolean),
    PrimaryKeyConstraint("client_id", "app", "place_kind", "place_id", "generation", "seq"),
)


class SchemaNotAtHeadError(Exception):
    """The database is not at the newest migration; the message names the field and the revisions, no secret."""


def _url(target: MemoryTarget) -> URL:
    return URL.create(
        POSTGRES_DRIVER,
        username=target.user,
        password=target.password.reveal(),
        host=target.host,
        port=target.port,
        database=target.dbname,
    )


def _write_bound(target: MemoryTarget) -> float | None:
    """The seconds one write may take; an infinite ``write_seconds`` is no bound."""
    return None if math.isinf(target.write_seconds) else target.write_seconds


def _connect_arguments(target: MemoryTarget) -> dict[str, int]:
    bound = _write_bound(target)
    return {} if bound is None else {"connect_timeout": max(MINIMUM_CONNECT_SECONDS, math.ceil(bound))}


def _alembic_config() -> AlembicConfig:
    config = AlembicConfig()
    config.set_main_option("script_location", str(MIGRATIONS_DIRECTORY))
    return config


def _head() -> str | None:
    return ScriptDirectory.from_config(_alembic_config()).get_current_head()


def _not_at_head(current: str | None) -> SchemaNotAtHeadError:
    return SchemaNotAtHeadError(
        f"memory: the database is at revision {current or 'none'} but personas needs {_head()}; run personas migrate"
    )


@contextmanager
def _as_store_failure() -> Generator[None]:
    """A database error leaves as a code alone: an outage is the store's, any other is a defect."""
    try:
        yield
    except OperationalError, InterfaceError:
        raise StoreUnavailable(STORE_UNREACHABLE_CODE) from None
    except SQLAlchemyError:
        raise StoreDefect(STORE_DEFECT_CODE) from None


def migrate(target: MemoryTarget) -> None:
    """Brings the database to head; concurrent runs wait for one another on a Postgres advisory lock.

    It is synchronous and bounded by the driver's connect timeout alone.
    """
    engine = create_engine(_url(target), hide_parameters=True, connect_args=_connect_arguments(target))
    try:
        with _as_store_failure(), engine.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
            connection.commit()
            try:
                config = _alembic_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
            finally:
                connection.rollback()
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": MIGRATION_LOCK_KEY})
                connection.commit()
    finally:
        engine.dispose()


def _without_nul(value: str) -> str:
    """Postgres refuses U+0000 in text, and a line of a call may carry it."""
    return value.replace(NUL, REPLACEMENT)


def _rows_of(read: WindowRead) -> list[dict[str, object]]:
    return [
        {
            "client_id": read.client_id,
            "app": read.app,
            "place_kind": _without_nul(read.place.kind),
            "place_id": _without_nul(read.place.id),
            "generation": _without_nul(read.window.generation),
            "seq": line.seq,
            "first_read_at": read.read_at,
            "speaker": _without_nul(line.name),
            "kind": _without_nul(line.kind),
            "text": _without_nul(line.text),
            "own": line.own,
            "spoken": line.spoken,
        }
        for line in read.window.entries
    ]


class PostgresLineStore:
    """Keeps the lines of the windows the loops read in the database the configuration names.

    It opens no connection until the first write. That write checks the schema is at its head, and a schema that
    is not ends the runner. A pooled connection is not checked before its use: a stale one fails its write, and the
    pool takes another for the next.
    """

    def __init__(self, target: MemoryTarget) -> None:
        self._write_bound = _write_bound(target)
        self._engine: AsyncEngine = create_async_engine(
            _url(target), hide_parameters=True, connect_args=_connect_arguments(target)
        )
        self._schema_checked = False

    async def keep(self, read: WindowRead) -> None:
        """Keeps the window's lines, or raises ``StoreUnavailable`` or ``StoreDefect`` with a code.

        ``write_seconds`` bounds taking a connection, connecting and the statement against a store that answers. A
        connection that goes dark in the middle of the statement is given up only after the driver's own cancel
        wait, about ten seconds beyond the bound.
        """
        statement = insert(_LINES).values(_rows_of(read)).on_conflict_do_nothing()
        try:
            async with asyncio.timeout(self._write_bound):
                with _as_store_failure():
                    async with self._engine.begin() as connection:
                        await self._require_head(connection)
                        await connection.execute(statement)
        except TimeoutError:
            raise StoreUnavailable(STORE_TIMEOUT_CODE) from None
        except StoreUnavailable, StoreDefect:
            # The driver answers a cancellation with its own cancel request and raises a break of the connection
            # during that wait instead of the cancellation; the time bound has taken back its own request by now.
            if task_is_being_cancelled():
                raise asyncio.CancelledError from None
            raise

    async def _require_head(self, connection: AsyncConnection) -> None:
        if self._schema_checked:
            return
        current = await connection.run_sync(lambda each: MigrationContext.configure(each).get_current_revision())
        if current != _head():
            raise _not_at_head(current)
        self._schema_checked = True

    async def close(self) -> None:
        await self._engine.dispose()
