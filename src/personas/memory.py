"""The memory: personas' Postgres, which holds the lines every agent read, verbatim, each agent's history and the record
of every entry his history took, and the migration that owns its schema.

The runner writes, searches and sweeps through ``PostgresLineStore``; ``PostgresHistoryStore`` keeps each agent's
history and record for the pod that owns him; the ``memory`` commands list through both and erase through
``PostgresMemory``; ``migrate`` is the only code that changes the schema. A line is keyed by the agent, the app, the
place, the visit and its ``seq``, so reading a window again changes nothing and the time of the first read stands. No
database error reaches a log or a traceback with a parameter or a row in it.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

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
    exists,
    func,
    literal,
    or_,
    select,
    text,
    true,
    union_all,
    update,
)
from sqlalchemy import cast as sql_cast
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from personas.entries import (
    NO_TURNS_KEPT,
    KeptHistory,
    KeptLine,
    Visit,
    entry_from_stored,
    entry_time,
    record_rows,
    stored_entry,
    stored_turn,
    stored_turns,
    turn_from_stored,
    turns_from_stored,
)
from personas.failures import (
    STORE_DEFECT_CODE,
    STORE_TIMEOUT_CODE,
    STORE_UNREACHABLE_CODE,
    OrdinalReusedError,
    SchemaNotAtHeadError,
    StaleGenerationError,
    StoreDefect,
    StoreUnavailable,
    task_is_being_cancelled,
)
from personas.record_words import LINE_KIND, SIMPLE_CONFIGURATION, entry_words
from personas.sweep import Swept

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator
    from datetime import datetime

    from pydantic import JsonValue
    from sqlalchemy import ColumnElement, Delete, Row, Select, Update

    from personas.config import DatabaseTarget
    from personas.entries import HistoryDelta, KeptTurns, RecordRow, ToldEntry, WindowRead
    from personas.recall import EntrySearch

MIGRATIONS_DIRECTORY: Final = Path(__file__).parent / "migrations"
# Serialises concurrent `personas migrate` runs; any constant shared by all of them does.
MIGRATION_LOCK_KEY: Final = 5_201_907_364
POSTGRES_DRIVER: Final = "postgresql+psycopg"
MINIMUM_CONNECT_SECONDS: Final = 2
"""The least connect timeout the driver honours; it raises a smaller one to this and truncates a
fraction, so the bound is rounded up."""
RECORD_ROWS_PER_STATEMENT: Final = 1000
"""The most rows of the record one statement inserts, which keeps its parameters below the driver's limit."""
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
_KEPT_ENTRIES: Final = Table(
    "kept_entries",
    MetaData(),
    Column("id", BigInteger, primary_key=True),
    Column("client_id", Text),
    Column("ordinal", BigInteger),
    Column("at", DateTime(timezone=True)),
    Column("kind", Text),
    Column("app", Text),
    Column("place_kind", Text),
    Column("place_id", Text),
    Column("generation", Text),
    Column("own", Boolean),
    Column("speaker", Text),
    Column("entry", JSONB),
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
_AGENT_TURNS: Final = Table(
    "agent_turns",
    MetaData(),
    Column("client_id", Text, primary_key=True),
    Column("turns", Text),
    Column("generation", BigInteger),
)
_AGENT_OWNER: Final = Table(
    "agent_owner",
    MetaData(),
    Column("client_id", Text, primary_key=True),
    Column("generation", BigInteger),
)
_ERASED_TABLES: Final = (_LINES, _HISTORY_ENTRIES, _KEPT_ENTRIES, _ANSWERED_TURNS)
"""The tables an erase empties, in the order ``MemoryErased`` counts them."""
_RENEWED_GENERATION: Final = _AGENT_OWNER.c.generation + 1
"""An agent's next generation, which a takeover and an erase both set: every write under the one before is refused."""


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
class KeptRecord:
    """What the record holds of one agent's entries of one kind: how many, and the time of the first and the last."""

    client_id: str
    kind: str
    entries: int
    first_at: datetime
    last_at: datetime


@dataclass(frozen=True)
class MemoryErased:
    lines: int
    entries: int
    kept_entries: int
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


_UNSTORABLE: Final = re.compile("[\x00\ud800-\udfff]")
"""What no jsonb value holds: U+0000 and a lone surrogate, which JSON writes as an escape Postgres refuses."""


def _storable(value: JsonValue) -> JsonValue:
    """The JSON value with every string, a key included, free of what jsonb refuses."""
    if isinstance(value, str):
        return _UNSTORABLE.sub(REPLACEMENT, value)
    if isinstance(value, list):
        return [_storable(each) for each in value]
    if isinstance(value, dict):
        return {_UNSTORABLE.sub(REPLACEMENT, key): _storable(each) for key, each in value.items()}
    return value


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
            # Opening the transaction checks the schema; the check is the whole work.
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


def _recorded_entries(search: EntrySearch, words: ColumnElement[object]) -> Select[tuple[object]]:
    """The JSON of the record's entries in the search's room below its ordinal that hold the words, newest first."""
    return (
        select(_KEPT_ENTRIES.c.entry)
        .where(
            _KEPT_ENTRIES.c.client_id == search.client_id,
            _KEPT_ENTRIES.c.app == search.app,
            _KEPT_ENTRIES.c.place_kind == _without_nul(search.place.kind),
            _KEPT_ENTRIES.c.place_id == _without_nul(search.place.id),
            _KEPT_ENTRIES.c.ordinal < search.first_ordinal,
            or_(
                func.numnode(words) == 0, entry_words(_KEPT_ENTRIES.c.kind, _KEPT_ENTRIES.c.entry).bool_op("@@")(words)
            ),
        )
        .order_by(_KEPT_ENTRIES.c.at.desc(), _KEPT_ENTRIES.c.ordinal.desc())
        .limit(search.rows)
    )


def _lines_the_record_lacks(search: EntrySearch, words: ColumnElement[object]) -> Select[tuple[object, ...]]:
    """The lines of the search's room first read before its time that hold the words and that the record does not hold
    under the same visit and ``seq``, newest first."""
    in_the_record = exists().where(
        _KEPT_ENTRIES.c.client_id == _LINES.c.client_id,
        _KEPT_ENTRIES.c.kind == LINE_KIND,
        _KEPT_ENTRIES.c.app == _LINES.c.app,
        _KEPT_ENTRIES.c.place_kind == _LINES.c.place_kind,
        _KEPT_ENTRIES.c.place_id == _LINES.c.place_id,
        _KEPT_ENTRIES.c.generation == _LINES.c.generation,
        sql_cast(_KEPT_ENTRIES.c.entry["seq"].astext, Numeric) == _LINES.c.seq,
    )
    holds_every_word = func.to_tsvector(SIMPLE_CONFIGURATION, _LINES.c.text).bool_op("@@")(words)
    return (
        select(_LINES)
        .where(
            _LINES.c.client_id == search.client_id,
            _LINES.c.app == search.app,
            _LINES.c.place_kind == _without_nul(search.place.kind),
            _LINES.c.place_id == _without_nul(search.place.id),
            _LINES.c.first_read_at < search.first_read_before,
            ~in_the_record,
            or_(func.numnode(words) == 0, holds_every_word),
        )
        .order_by(_LINES.c.first_read_at.desc(), _LINES.c.seq.desc())
        .limit(search.rows)
    )


def _kept_line(row: Row[Any]) -> KeptLine:
    return KeptLine(
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


class PostgresLineStore(_PostgresStore):
    """Keeps the lines of the windows the loops read."""

    async def keep(self, read: WindowRead) -> None:
        """Keeps the window's lines, or raises ``StoreUnavailable`` or ``StoreDefect`` with a code."""
        statement = insert(_LINES).values(_rows_of(read)).on_conflict_do_nothing()
        async with self._bounded_transaction() as connection:
            await connection.execute(statement)

    async def search(self, search: EntrySearch) -> tuple[ToldEntry, ...]:
        """The entries of the agent's place that hold every word of the text, newest first, at most the rows asked
        for: those of the record below the ordinal, of every kind, and the lines first read before the time that the
        record does not hold. A text without a word matches every entry.

        It raises as a write does: ``StoreUnavailable`` or ``StoreDefect`` with a code, never the text searched for.
        """
        words = func.plainto_tsquery(SIMPLE_CONFIGURATION, literal(_without_nul(search.text), Text))
        async with self._bounded_transaction() as connection:
            stored = (await connection.execute(_recorded_entries(search, words))).scalars().all()
            kept_lines = (await connection.execute(_lines_the_record_lacks(search, words))).all()
        try:
            found = [
                *(cast("ToldEntry", entry_from_stored(each)) for each in stored),
                *(_kept_line(row) for row in kept_lines),
            ]
        except ValueError:
            raise StoreDefect(STORE_DEFECT_CODE) from None
        # Python's sort is stable, so of two entries read at the same time the record's comes first.
        return tuple(sorted(found, key=entry_time, reverse=True)[: search.rows])

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

    async def erase_before(self, first_read_before: datetime) -> Swept:
        """Deletes, in one transaction, every line of every agent first read before the time and every entry of the
        record that entered before it and whose client and ordinal his history holds no entry for, and returns how many
        of each."""
        held_by_history = exists().where(
            _HISTORY_ENTRIES.c.client_id == _KEPT_ENTRIES.c.client_id,
            _HISTORY_ENTRIES.c.ordinal == _KEPT_ENTRIES.c.ordinal,
        )
        lines = delete(_LINES).where(_LINES.c.first_read_at < first_read_before)
        entries = delete(_KEPT_ENTRIES).where(_KEPT_ENTRIES.c.at < first_read_before, ~held_by_history)
        async with self._bounded_transaction() as connection:
            return Swept((await connection.execute(lines)).rowcount, (await connection.execute(entries)).rowcount)


def _entry_rows(client_id: str, delta: HistoryDelta) -> list[dict[str, object]]:
    return [
        {"client_id": client_id, "ordinal": ordinal, "at": entry_time(entry), "entry": json.dumps(stored_entry(entry))}
        for ordinal, entry in enumerate(delta.entries, start=delta.first)
    ]


def _nullable_text(value: str | None) -> str | None:
    return None if value is None else _without_nul(value)


def _record_values(client_id: str, row: RecordRow) -> dict[str, object]:
    return {
        "client_id": client_id,
        "ordinal": row.ordinal,
        "at": row.at,
        "kind": row.kind,
        "app": _without_nul(row.app),
        "place_kind": None if row.place is None else _without_nul(row.place.kind),
        "place_id": None if row.place is None else _without_nul(row.place.id),
        "generation": _nullable_text(row.generation),
        "own": row.own,
        "speaker": _nullable_text(row.speaker),
        "entry": _storable(row.entry),
    }


def _turn_rows(client_id: str, delta: HistoryDelta) -> list[dict[str, object]]:
    return [
        {"client_id": client_id, "ordinal": ordinal, "through": turn.through, "turn": json.dumps(stored_turn(turn))}
        for ordinal, turn in enumerate(delta.told, start=delta.first_told)
    ]


class PostgresHistoryStore(_PostgresStore):
    """Keeps each agent's history and turns: his entries by ordinal, his answered turns and his kept turns, written by
    the pod that owns him.

    A pod that takes an agent over renews his generation, and every write carries the writer's; one whose generation
    is no longer current is refused and changes nothing, so a pod that lost an agent cannot write over his new owner.
    """

    async def take_over(self, client_id: str) -> int:
        """Renews the agent's generation and returns it: every write with an older one is refused from now on. His kept
        turns come along when they are of the generation before; older ones are dropped, so he resumes without any."""
        statement = (
            insert(_AGENT_OWNER)
            .values(client_id=client_id, generation=1)
            .on_conflict_do_update(index_elements=[_AGENT_OWNER.c.client_id], set_={"generation": _RENEWED_GENERATION})
            .returning(_AGENT_OWNER.c.generation)
        )
        async with self._bounded_transaction() as connection:
            generation = (await connection.execute(statement)).scalar_one()
            await connection.execute(_turns_carried(_AGENT_OWNER.c.client_id == client_id))
            await connection.execute(
                delete(_AGENT_TURNS).where(
                    _AGENT_TURNS.c.client_id == client_id, _AGENT_TURNS.c.generation != generation
                )
            )
            return generation

    async def generation(self, client_id: str) -> int | None:
        """The agent's current generation; none before his first owner took him."""
        statement = select(_AGENT_OWNER.c.generation).where(_AGENT_OWNER.c.client_id == client_id)
        async with self._bounded_transaction() as connection:
            return (await connection.execute(statement)).scalar_one_or_none()

    async def write(self, client_id: str, generation: int, delta: HistoryDelta | None, turns: KeptTurns) -> None:
        """Writes the delta, if any, and the turns in one transaction while ``generation`` is the agent's current one,
        or raises ``StaleGenerationError`` and writes nothing. Writing a delta again changes nothing."""
        async with self._bounded_transaction() as connection:
            # The shared row lock holds a take-over back until this write has committed.
            current = (
                select(_AGENT_OWNER.c.generation)
                .where(_AGENT_OWNER.c.client_id == client_id, _AGENT_OWNER.c.generation == generation)
                .with_for_update(read=True)
            )
            if (await connection.execute(current)).first() is None:
                raise StaleGenerationError
            stored = json.dumps(stored_turns(turns), ensure_ascii=False)
            await connection.execute(
                insert(_AGENT_TURNS)
                .values(client_id=client_id, turns=stored, generation=generation)
                .on_conflict_do_update(
                    index_elements=[_AGENT_TURNS.c.client_id], set_={"turns": stored, "generation": generation}
                )
            )
            if delta is not None:
                await self._append(connection, client_id, delta)

    @staticmethod
    async def _keep(connection: AsyncConnection, client_id: str, delta: HistoryDelta) -> None:
        """Keeps the entries the delta hands over that the record lacks. An ordinal the record holds with another time
        or kind is no entry written again, but one an erase by an older release left behind: it raises
        ``OrdinalReusedError``, and the transaction takes nothing."""
        rows = record_rows(delta.kept_first, delta.kept)
        if not rows:
            return
        held = select(_KEPT_ENTRIES.c.ordinal, _KEPT_ENTRIES.c.at, _KEPT_ENTRIES.c.kind).where(
            _KEPT_ENTRIES.c.client_id == client_id,
            _KEPT_ENTRIES.c.ordinal.between(rows[0].ordinal, rows[-1].ordinal),
        )
        held_as = {ordinal: (at, kind) for ordinal, at, kind in await connection.execute(held)}
        if any(held_as.get(row.ordinal, (row.at, row.kind)) != (row.at, row.kind) for row in rows):
            raise OrdinalReusedError(STORE_DEFECT_CODE)
        for batch in batched(rows, RECORD_ROWS_PER_STATEMENT, strict=False):
            await connection.execute(
                insert(_KEPT_ENTRIES)
                .values([_record_values(client_id, row) for row in batch])
                .on_conflict_do_nothing(index_elements=[_KEPT_ENTRIES.c.client_id, _KEPT_ENTRIES.c.ordinal])
            )

    @classmethod
    async def _append(cls, connection: AsyncConnection, client_id: str, delta: HistoryDelta) -> None:
        await cls._keep(connection, client_id, delta)
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
        """The agent's history and turns as one committed write left them; a stored value that is no entry, answered
        turn or kept turns is a ``StoreDefect``."""
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
        kept_turns = select(_AGENT_TURNS.c.turns).where(_AGENT_TURNS.c.client_id == client_id)
        async with self._bounded_transaction() as connection:
            rows = (await connection.execute(statement)).all()
            # The fence leaves no writer of his but this pod after its takeover, so a second statement reads what the
            # first did.
            turns = (await connection.execute(kept_turns)).scalar_one_or_none()
        entry_rows = [(ordinal, value) for is_turn, ordinal, value in rows if not is_turn]
        turn_rows = [(ordinal, value) for is_turn, ordinal, value in rows if is_turn]
        try:
            return KeptHistory(
                entry_rows[0][0] if entry_rows else 0,
                tuple(entry_from_stored(json.loads(value)) for _, value in entry_rows),
                turn_rows[0][0] if turn_rows else 0,
                tuple(turn_from_stored(json.loads(value)) for _, value in turn_rows),
                NO_TURNS_KEPT if turns is None else turns_from_stored(json.loads(turns)),
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

    async def kept_records(self, client_id: str | None) -> tuple[KeptRecord, ...]:
        """What the record holds, per agent and kind, or of one agent, ordered by agent and kind; no entry's content."""
        statement = (
            select(
                _KEPT_ENTRIES.c.client_id,
                _KEPT_ENTRIES.c.kind,
                func.count(),
                func.min(_KEPT_ENTRIES.c.at),
                func.max(_KEPT_ENTRIES.c.at),
            )
            .group_by(_KEPT_ENTRIES.c.client_id, _KEPT_ENTRIES.c.kind)
            .order_by(_KEPT_ENTRIES.c.client_id, _KEPT_ENTRIES.c.kind)
        )
        if client_id is not None:
            statement = statement.where(_KEPT_ENTRIES.c.client_id == client_id)
        async with self._bounded_transaction() as connection:
            rows = (await connection.execute(statement)).all()
        return tuple(KeptRecord(*row) for row in rows)


class PostgresMemory(_PostgresStore):
    """Everything the agents keep in the database, as ``personas memory erase`` deletes it: their lines, history
    entries, record and answered turns, all of them or none.

    An erase renews the generation of each agent it erases in the same transaction, so his owner's next write is
    refused: the owner stops him and takes him over again from the erased history, instead of writing back what it
    still holds in memory. His kept turns stay and come along to the renewed generation, so where reading his places
    stands holds across the erase."""

    async def erase_agent(self, client_id: str) -> MemoryErased:
        """Deletes the agent's lines, history and record in one transaction, renewing his generation, and returns how
        much."""
        return await self._erase(
            _AGENT_OWNER.c.client_id == client_id,
            [delete(table).where(table.c.client_id == client_id) for table in _ERASED_TABLES],
        )

    async def erase_all(self) -> MemoryErased:
        """Deletes every agent's lines, history and record in one transaction, renewing every generation, and returns
        how much; the schema stays."""
        return await self._erase(true(), [delete(table) for table in _ERASED_TABLES])

    async def _erase(self, owners: ColumnElement[bool], deletes: list[Delete]) -> MemoryErased:
        async with self._bounded_transaction() as connection:
            await connection.execute(_renewed(owners))
            await connection.execute(_turns_carried(owners))
            return MemoryErased(*[(await connection.execute(each)).rowcount for each in deletes])


def _renewed(owners: ColumnElement[bool]) -> Update:
    """Renews the generation of each agent ``owners`` selects; an agent never taken has none to renew."""
    return update(_AGENT_OWNER).where(owners).values(generation=_RENEWED_GENERATION)


def _turns_carried(owners: ColumnElement[bool]) -> Update:
    """Carries the kept turns of each agent ``owners`` selects to his renewed generation, when they are of the one just
    before it.

    A release that keeps no turns renews the generation without carrying them and writes his history past them, so
    turns of any older generation no longer say where reading his places stands.
    """
    return (
        update(_AGENT_TURNS)
        .where(
            _AGENT_TURNS.c.client_id == _AGENT_OWNER.c.client_id,
            _AGENT_TURNS.c.generation == _AGENT_OWNER.c.generation - 1,
            owners,
        )
        .values(generation=_AGENT_OWNER.c.generation)
    )
