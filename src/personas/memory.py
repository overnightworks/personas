"""The memory: personas' Postgres, which holds the lines every agent read, verbatim, and each agent's history, and the
migration that owns its schema.

The runner writes, searches and sweeps through ``PostgresLineStore``; ``PostgresHistoryStore`` keeps each agent's
history for the pod that owns him; the ``memory`` commands list through both and erase through ``PostgresMemory``;
``migrate`` is the only code that changes the schema. A line is keyed by the agent, the app, the place, the visit and
its ``seq``, so reading a window again changes nothing and the time of the first read stands. No database error
reaches a log or a traceback with a parameter or a row in it.
"""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from agents.member import PlaceRef
from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    URL,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    Table,
    Text,
    create_engine,
    delete,
    func,
    literal,
    literal_column,
    or_,
    select,
    text,
    union_all,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from personas.entries import (
    KeptHistory,
    KeptLine,
    Visit,
    entry_from_stored,
    entry_time,
    stored_entry,
    stored_turn,
    turn_from_stored,
)
from personas.failures import (
    STORE_DEFECT_CODE,
    STORE_TIMEOUT_CODE,
    STORE_UNREACHABLE_CODE,
    SchemaNotAtHeadError,
    StaleGenerationError,
    StoreDefect,
    StoreUnavailable,
    task_is_being_cancelled,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator
    from datetime import datetime

    from sqlalchemy import Delete

    from personas.config import DatabaseTarget
    from personas.entries import HistoryDelta, WindowRead
    from personas.recall import LineSearch

MIGRATIONS_DIRECTORY: Final = Path(__file__).parent / "migrations"
# Serialises concurrent `personas migrate` runs; any constant shared by all of them does.
MIGRATION_LOCK_KEY: Final = 5_201_907_364
POSTGRES_DRIVER: Final = "postgresql+psycopg"
MINIMUM_CONNECT_SECONDS: Final = 2
"""The least connect timeout the driver honours; it raises a smaller one to this and truncates a
fraction, so the bound is rounded up."""
NUL: Final = "\x00"
REPLACEMENT: Final = "�"
SIMPLE_CONFIGURATION: Final = literal_column("'simple'::regconfig", Text)
"""Postgres full text without stemming or stop words: a word is what the parser makes of the text, lower case."""

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
    Column("cut", Boolean),
    Column("act", Boolean),
    PrimaryKeyConstraint("client_id", "app", "place_kind", "place_id", "generation", "seq"),
)

_HISTORY_ENTRIES: Final = Table(
    "history_entries",
    MetaData(),
    Column("client_id", Text),
    Column("ordinal", BigInteger),
    Column("at", DateTime(timezone=True)),
    Column("entry", Text),
    PrimaryKeyConstraint("client_id", "ordinal"),
)
_ANSWERED_TURNS: Final = Table(
    "answered_turns",
    MetaData(),
    Column("client_id", Text),
    Column("ordinal", BigInteger),
    Column("through", BigInteger),
    Column("turn", Text),
    PrimaryKeyConstraint("client_id", "ordinal"),
)
_AGENT_OWNER: Final = Table(
    "agent_owner",
    MetaData(),
    Column("client_id", Text, primary_key=True),
    Column("generation", BigInteger),
)


@dataclass(frozen=True)
class KeptVisit:
    """What one agent keeps of one visit: how many lines, and the first and the last time one of them was first read."""

    client_id: str
    visit: Visit
    lines: int
    first_read_at: datetime
    last_read_at: datetime


@dataclass(frozen=True)
class KeptAgentHistory:
    """What one agent's history holds: how many entries and answered turns, and the time of its first and last entry."""

    client_id: str
    entries: int
    answered_turns: int
    first_at: datetime
    last_at: datetime


@dataclass(frozen=True)
class MemoryErased:
    lines: int
    entries: int
    answered_turns: int


def _url(target: DatabaseTarget) -> URL:
    return URL.create(
        POSTGRES_DRIVER,
        username=target.user,
        password=target.password.reveal(),
        host=target.host,
        port=target.port,
        database=target.dbname,
    )


def _write_bound(target: DatabaseTarget) -> float | None:
    """The seconds one write may take; an infinite ``write_seconds`` is no bound."""
    return None if math.isinf(target.write_seconds) else target.write_seconds


def _connect_arguments(target: DatabaseTarget) -> dict[str, int]:
    bound = _write_bound(target)
    return {} if bound is None else {"connect_timeout": max(MINIMUM_CONNECT_SECONDS, math.ceil(bound))}


def _alembic_config() -> AlembicConfig:
    config = AlembicConfig()
    config.set_main_option("script_location", str(MIGRATIONS_DIRECTORY))
    return config


def _head() -> str | None:
    return ScriptDirectory.from_config(_alembic_config()).get_current_head()


def _newer_than_this_release(current: str | None) -> bool:
    """Whether a newer release migrated the database: its revision is none of this release's scripts.

    Releases that run side by side change the schema only by adding to it, so the older one still works on it.
    """
    known = {each.revision for each in ScriptDirectory.from_config(_alembic_config()).walk_revisions()}
    return current is not None and current not in known


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


def migrate(target: DatabaseTarget) -> None:
    """Brings the database to head; concurrent runs wait for one another on a Postgres advisory lock. A database that a
    newer release migrated is left as it is.

    It is synchronous and bounded by the driver's connect timeout alone.
    """
    engine = create_engine(_url(target), hide_parameters=True, connect_args=_connect_arguments(target))
    try:
        with _as_store_failure(), engine.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
            connection.commit()
            try:
                current = MigrationContext.configure(connection).get_current_revision()
                # The read began a transaction, which would swallow the upgrade's own and be rolled back below.
                connection.rollback()
                if not _newer_than_this_release(current):
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
            "cut": line.cut,
            "act": line.act,
        }
        for line in read.window.entries
    ]


class _PostgresStore:
    """A store in the database the configuration names, with no connection until its first statement, which checks
    the schema is at its head; a schema that is not ends the runner. A pooled connection is not checked before its use:
    a stale one fails its statement, and the pool takes another for the next. ``engine_factory`` makes the engine; a
    test hands it a driver of its own.
    """

    def __init__(
        self, target: DatabaseTarget, engine_factory: Callable[..., AsyncEngine] = create_async_engine
    ) -> None:
        self._write_bound = _write_bound(target)
        self._engine: AsyncEngine = engine_factory(
            _url(target), hide_parameters=True, connect_args=_connect_arguments(target)
        )
        self._schema_checked = False

    @asynccontextmanager
    async def _bounded_transaction(self) -> AsyncGenerator[AsyncConnection]:
        """A transaction on a schema at its head, whose failure leaves as ``StoreUnavailable`` or ``StoreDefect``.

        ``write_seconds`` bounds taking a connection, connecting and the statement against a store that answers. A
        connection that goes dark in the middle of the statement is given up only after the driver's own cancel
        wait, about ten seconds beyond the bound.
        """
        try:
            async with asyncio.timeout(self._write_bound):
                with _as_store_failure():
                    async with self._engine.begin() as connection:
                        await self._require_head(connection)
                        yield connection
        except TimeoutError:
            raise StoreUnavailable(STORE_TIMEOUT_CODE) from None
        except StoreUnavailable:
            # The driver answers a cancellation with its own cancel request and raises a break of the connection
            # during that wait instead of the cancellation; the time bound has taken back its own request by now.
            # A defect is no break of the connection: it leaves as itself, so that the turn records it.
            if task_is_being_cancelled():
                raise asyncio.CancelledError from None
            raise

    async def require_head(self) -> None:
        """Raises ``SchemaNotAtHeadError`` when the schema is behind this release's head, or ``StoreUnavailable`` or
        ``StoreDefect`` with a code when it cannot tell now."""
        async with self._bounded_transaction():
            pass

    async def _require_head(self, connection: AsyncConnection) -> None:
        if self._schema_checked:
            return
        current = await connection.run_sync(lambda each: MigrationContext.configure(each).get_current_revision())
        if current != _head() and not _newer_than_this_release(current):
            raise _not_at_head(current)
        self._schema_checked = True

    async def close(self) -> None:
        await self._engine.dispose()


class PostgresLineStore(_PostgresStore):
    """Keeps the lines of the windows the loops read."""

    async def keep(self, read: WindowRead) -> None:
        """Keeps the window's lines, or raises ``StoreUnavailable`` or ``StoreDefect`` with a code."""
        statement = insert(_LINES).values(_rows_of(read)).on_conflict_do_nothing()
        async with self._bounded_transaction() as connection:
            await connection.execute(statement)

    async def search(self, search: LineSearch) -> tuple[KeptLine, ...]:
        """The lines of the agent's place that hold every word of the text and were first read before the time, newest
        first, at most the rows asked for; a text without a word matches every line.

        It raises as a write does: ``StoreUnavailable`` or ``StoreDefect`` with a code, never the text searched for.
        """
        words = func.plainto_tsquery(SIMPLE_CONFIGURATION, literal(_without_nul(search.text), Text))
        holds_every_word = func.to_tsvector(SIMPLE_CONFIGURATION, _LINES.c.text).bool_op("@@")(words)
        statement = (
            select(
                _LINES.c.app,
                _LINES.c.place_kind,
                _LINES.c.place_id,
                _LINES.c.generation,
                _LINES.c.seq,
                _LINES.c.first_read_at,
                _LINES.c.speaker,
                _LINES.c.kind,
                _LINES.c.text,
                _LINES.c.own,
                _LINES.c.spoken,
                _LINES.c.cut,
                _LINES.c.act,
            )
            .where(
                _LINES.c.client_id == search.client_id,
                _LINES.c.app == search.app,
                _LINES.c.place_kind == _without_nul(search.place.kind),
                _LINES.c.place_id == _without_nul(search.place.id),
                _LINES.c.first_read_at < search.first_read_before,
                or_(func.numnode(words) == 0, holds_every_word),
            )
            .order_by(_LINES.c.first_read_at.desc(), _LINES.c.seq.desc())
            .limit(search.rows)
        )
        async with self._bounded_transaction() as connection:
            rows = (await connection.execute(statement)).all()
        return tuple(
            KeptLine(
                Visit(row.app, PlaceRef(kind=row.place_kind, id=row.place_id), row.generation),
                int(row.seq),
                row.first_read_at,
                row.speaker,
                row.kind,
                row.text,
                own=row.own,
                spoken=row.spoken,
                cut=row.cut,
                act=row.act,
            )
            for row in rows
        )

    async def kept_visits(self, client_id: str | None) -> tuple[KeptVisit, ...]:
        """Every agent's visits, or those of one agent, ordered by agent and first read; no line's content."""
        visit = (_LINES.c.client_id, _LINES.c.app, _LINES.c.place_kind, _LINES.c.place_id, _LINES.c.generation)
        first_read_at = func.min(_LINES.c.first_read_at)
        statement = (
            select(*visit, func.count(), first_read_at, func.max(_LINES.c.first_read_at))
            .group_by(*visit)
            .order_by(_LINES.c.client_id, first_read_at, *visit[1:])
        )
        if client_id is not None:
            statement = statement.where(_LINES.c.client_id == client_id)
        async with self._bounded_transaction() as connection:
            rows = (await connection.execute(statement)).all()
        return tuple(
            KeptVisit(agent, Visit(app, PlaceRef(kind=kind, id=place_id), generation), count, first, last)
            for agent, app, kind, place_id, generation, count, first, last in rows
        )

    async def erase_before(self, first_read_before: datetime) -> int:
        """Deletes every line of every agent first read before the time, in one statement, and returns how many."""
        return await self._erase(delete(_LINES).where(_LINES.c.first_read_at < first_read_before))

    async def _erase(self, statement: Delete) -> int:
        async with self._bounded_transaction() as connection:
            return (await connection.execute(statement)).rowcount


def _entry_rows(client_id: str, delta: HistoryDelta) -> list[dict[str, object]]:
    return [
        {"client_id": client_id, "ordinal": ordinal, "at": entry_time(entry), "entry": json.dumps(stored_entry(entry))}
        for ordinal, entry in enumerate(delta.entries, start=delta.first)
    ]


def _turn_rows(client_id: str, delta: HistoryDelta) -> list[dict[str, object]]:
    return [
        {"client_id": client_id, "ordinal": ordinal, "through": turn.through, "turn": json.dumps(stored_turn(turn))}
        for ordinal, turn in enumerate(delta.told, start=delta.first_told)
    ]


class PostgresHistoryStore(_PostgresStore):
    """Keeps each agent's history: his entries by ordinal and his answered turns, written by the pod that owns him.

    A pod that takes an agent over renews his generation, and every write carries the writer's; one whose generation
    is no longer current is refused and changes nothing, so a pod that lost an agent cannot write over his new owner.
    """

    async def take_over(self, client_id: str) -> int:
        """Renews the agent's generation and returns it: every write with an older one is refused from now on."""
        statement = (
            insert(_AGENT_OWNER)
            .values(client_id=client_id, generation=1)
            .on_conflict_do_update(
                index_elements=[_AGENT_OWNER.c.client_id], set_={"generation": _AGENT_OWNER.c.generation + 1}
            )
            .returning(_AGENT_OWNER.c.generation)
        )
        async with self._bounded_transaction() as connection:
            return (await connection.execute(statement)).scalar_one()

    async def generation(self, client_id: str) -> int | None:
        """The agent's current generation; none before his first owner took him."""
        statement = select(_AGENT_OWNER.c.generation).where(_AGENT_OWNER.c.client_id == client_id)
        async with self._bounded_transaction() as connection:
            return (await connection.execute(statement)).scalar_one_or_none()

    async def append(self, client_id: str, generation: int, delta: HistoryDelta) -> None:
        """Writes the delta in one transaction while ``generation`` is the agent's current one, or raises
        ``StaleGenerationError`` and writes nothing. Writing a delta again changes nothing."""
        async with self._bounded_transaction() as connection:
            # The shared row lock holds a take-over back until this write has committed.
            current = (
                select(_AGENT_OWNER.c.generation)
                .where(_AGENT_OWNER.c.client_id == client_id, _AGENT_OWNER.c.generation == generation)
                .with_for_update(read=True)
            )
            if (await connection.execute(current)).first() is None:
                raise StaleGenerationError
            if delta.entries:
                await connection.execute(
                    insert(_HISTORY_ENTRIES).values(_entry_rows(client_id, delta)).on_conflict_do_nothing()
                )
            if delta.told:
                await connection.execute(
                    insert(_ANSWERED_TURNS).values(_turn_rows(client_id, delta)).on_conflict_do_nothing()
                )
            await connection.execute(
                delete(_HISTORY_ENTRIES).where(
                    _HISTORY_ENTRIES.c.client_id == client_id, _HISTORY_ENTRIES.c.ordinal < delta.left_out
                )
            )
            await connection.execute(
                delete(_ANSWERED_TURNS).where(
                    _ANSWERED_TURNS.c.client_id == client_id, _ANSWERED_TURNS.c.through <= delta.left_out
                )
            )

    async def load(self, client_id: str) -> KeptHistory:
        """The agent's history as one committed write left it; a stored value that is no entry or turn is a
        ``StoreDefect``."""
        # One statement reads both tables in one snapshot: two would let a write commit between them.
        stored = union_all(
            select(literal(False).label("is_turn"), _HISTORY_ENTRIES.c.ordinal, _HISTORY_ENTRIES.c.entry).where(
                _HISTORY_ENTRIES.c.client_id == client_id
            ),
            select(literal(True), _ANSWERED_TURNS.c.ordinal, _ANSWERED_TURNS.c.turn).where(
                _ANSWERED_TURNS.c.client_id == client_id
            ),
        ).subquery()
        statement = select(stored).order_by(stored.c.is_turn, stored.c.ordinal)
        async with self._bounded_transaction() as connection:
            rows = (await connection.execute(statement)).all()
        entry_rows = [(ordinal, value) for is_turn, ordinal, value in rows if not is_turn]
        turn_rows = [(ordinal, value) for is_turn, ordinal, value in rows if is_turn]
        try:
            return KeptHistory(
                entry_rows[0][0] if entry_rows else 0,
                tuple(entry_from_stored(json.loads(value)) for _, value in entry_rows),
                turn_rows[0][0] if turn_rows else 0,
                tuple(turn_from_stored(json.loads(value)) for _, value in turn_rows),
            )
        except ValueError:
            raise StoreDefect(STORE_DEFECT_CODE) from None

    async def kept_histories(self, client_id: str | None) -> tuple[KeptAgentHistory, ...]:
        """Every agent's history, or that of one agent, ordered by agent; no entry's content."""
        turns = (
            select(_ANSWERED_TURNS.c.client_id, func.count().label("answered"))
            .group_by(_ANSWERED_TURNS.c.client_id)
            .subquery()
        )
        statement = (
            select(
                _HISTORY_ENTRIES.c.client_id,
                func.count(),
                func.coalesce(func.max(turns.c.answered), 0),
                func.min(_HISTORY_ENTRIES.c.at),
                func.max(_HISTORY_ENTRIES.c.at),
            )
            .outerjoin(turns, turns.c.client_id == _HISTORY_ENTRIES.c.client_id)
            .group_by(_HISTORY_ENTRIES.c.client_id)
            .order_by(_HISTORY_ENTRIES.c.client_id)
        )
        if client_id is not None:
            statement = statement.where(_HISTORY_ENTRIES.c.client_id == client_id)
        async with self._bounded_transaction() as connection:
            rows = (await connection.execute(statement)).all()
        return tuple(KeptAgentHistory(*row) for row in rows)


class PostgresMemory(_PostgresStore):
    """Everything the agents keep in the database, as ``personas memory erase`` deletes it: their lines, history
    entries and answered turns, all of them or none."""

    async def erase_agent(self, client_id: str) -> MemoryErased:
        """Deletes the agent's lines and history in one transaction and returns how much; his generation stays, so a
        pod that lost him still cannot write."""
        return await self._erase(
            delete(_LINES).where(_LINES.c.client_id == client_id),
            delete(_HISTORY_ENTRIES).where(_HISTORY_ENTRIES.c.client_id == client_id),
            delete(_ANSWERED_TURNS).where(_ANSWERED_TURNS.c.client_id == client_id),
        )

    async def erase_all(self) -> MemoryErased:
        """Deletes every agent's lines and history in one transaction and returns how much; the schema and the
        generations stay."""
        return await self._erase(delete(_LINES), delete(_HISTORY_ENTRIES), delete(_ANSWERED_TURNS))

    async def _erase(self, lines: Delete, entries: Delete, answered_turns: Delete) -> MemoryErased:
        async with self._bounded_transaction() as connection:
            erased_lines = (await connection.execute(lines)).rowcount
            erased_entries = (await connection.execute(entries)).rowcount
            return MemoryErased(erased_lines, erased_entries, (await connection.execute(answered_turns)).rowcount)
