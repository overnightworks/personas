"""The pod's lock connection: one Postgres session that holds a session advisory lock for each agent the pod serves.

The locks use the two-integer form in a namespace of their own, apart from the migration's lock, and a lock lives
exactly as long as the session. Every round is bounded; a round that fails drops the connection, whose locks the
server frees as soon as it notices the session is gone: at once for a closed socket, after its TCP keepalive for a
silent loss.
"""

from __future__ import annotations

import asyncio
import math
from hashlib import blake2b
from typing import TYPE_CHECKING, Final

import psycopg

from personas.ownership import LockConnectionLost

if TYPE_CHECKING:
    from collections.abc import Sequence

    from personas.config import DatabaseTarget, OwnershipTimings

AGENT_LOCK_NAMESPACE: Final = 1_885_487_713
"""The first integer of every agent lock; below 2**31, so that Postgres reads it as the same oid."""
KEY_BYTES: Final = 4
KEY_MASK: Final = 0x7FFF_FFFF
HELD_KEYS: Final = """
SELECT objid::int8 FROM pg_locks
WHERE locktype = 'advisory' AND objsubid = 2 AND classid = %s AND granted AND pid = pg_backend_pid()
"""
TAKE_FREE_KEYS: Final = "SELECT key, pg_try_advisory_lock(%s, key) FROM unnest(%s::int4[]) AS key"


type Round = asyncio.Task[set[int]]
"""One round on the lock connection: the keys held after it."""


def key_of(client_id: str) -> int:
    """The second integer of the agent's lock: a digest of the client id, in the range both integer and oid hold."""
    return int.from_bytes(blake2b(client_id.encode(), digest_size=KEY_BYTES).digest()) & KEY_MASK


class PostgresAgentLocks:
    """Holds the agents' locks on one connection that it opens when it has none and drops when a round fails.

    The connection is the object's own from the moment it opens, never a round's result, so that ``close`` finds it
    however a round ended. At most one connection exists: no round starts while an abandoned one still unwinds.
    """

    def __init__(self, database: DatabaseTarget, timings: OwnershipTimings) -> None:
        self._database = database
        self._timeout = timings.heartbeat_timeout_seconds
        self._connection: psycopg.AsyncConnection | None = None
        self._abandoned: set[Round] = set()

    async def hold(self, client_ids: Sequence[str]) -> frozenset[str]:
        if any(not each.done() for each in self._abandoned):
            raise LockConnectionLost
        by_key = {key_of(each): each for each in client_ids}
        round_ = asyncio.create_task(self._round(by_key))
        try:
            await asyncio.wait({round_}, timeout=self._timeout)
            if not round_.done():
                raise LockConnectionLost
            held = round_.result()
        except psycopg.OperationalError, psycopg.InterfaceError:
            raise LockConnectionLost from None
        finally:
            if not round_.done():
                self._abandon(round_)
        return frozenset(by_key[key] for key in held if key in by_key)

    async def close(self) -> None:
        """Ends the session, which frees every lock it holds, once the rounds abandoned before have unwound."""
        if self._abandoned:
            await asyncio.wait(self._abandoned)
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    def _abandon(self, round_: Round) -> None:
        """Lets a round that overran its bound or lost its caller unwind on its own: psycopg may take seconds to cancel.

        The caller does not wait for it, so the bound holds however long the unwinding takes; the round closes the
        connection it used when it ends, and ``hold`` starts no round until then.
        """
        self._abandoned.add(round_)
        round_.add_done_callback(self._forget)
        round_.cancel()

    def _forget(self, round_: Round) -> None:
        self._abandoned.discard(round_)
        if not round_.cancelled():
            round_.exception()

    async def _round(self, by_key: dict[int, str]) -> set[int]:
        """Confirms the held locks and takes the free ones; a round that fails closes the connection."""
        connection = self._connection or await self._open()
        self._connection = connection
        try:
            held = set(await self._keys_held(connection))
            free = [key for key in by_key if key not in held]
            if free:
                held |= await self._take(connection, free)
        except BaseException:
            self._connection = None
            await connection.close()
            raise
        return held

    async def _open(self) -> psycopg.AsyncConnection:
        return await psycopg.AsyncConnection.connect(
            host=self._database.host,
            port=self._database.port,
            dbname=self._database.dbname,
            user=self._database.user,
            password=self._database.password.reveal(),
            connect_timeout=math.ceil(self._timeout),
            autocommit=True,
        )

    @staticmethod
    async def _keys_held(connection: psycopg.AsyncConnection) -> list[int]:
        cursor = await connection.execute(HELD_KEYS, (AGENT_LOCK_NAMESPACE,))
        return [key for (key,) in await cursor.fetchall()]

    @staticmethod
    async def _take(connection: psycopg.AsyncConnection, keys: list[int]) -> set[int]:
        cursor = await connection.execute(TAKE_FREE_KEYS, (AGENT_LOCK_NAMESPACE, keys))
        return {key for key, taken in await cursor.fetchall() if taken}
