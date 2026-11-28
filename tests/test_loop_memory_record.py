"""The whole runner against a real Postgres: a turn finds by their words the entries his history's cut left in the
record."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round

from personas.memory import PostgresHistoryStore, PostgresLineStore
from personas.recall import RECALL_TOOL
from tests.support.app_driving import shows, waits
from tests.support.fakes import memory_target
from tests.support.gateway import Gateway, say
from tests.support.history import TABLE, at_table, bounded_history, url_of
from tests.support.ownership import KeptHistories
from tests.support.standings import AGENT_SUBJECT as SUBJECT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from personas.config import DatabaseTarget
    from personas.entries import HistoryDelta, KeptTurns

pytestmark = [pytest.mark.anyio, pytest.mark.postgres, pytest.mark.integration]


@dataclass
class HistoriesInPostgres(KeptHistories):
    """The histories a runner test holds, which every write also reaches in the real store, record included."""

    postgres: PostgresHistoryStore | None = None
    owned: dict[str, int] = field(default_factory=dict[str, int])

    async def take_over(self, client_id: str) -> int:
        assert self.postgres is not None
        self.owned[client_id] = await self.postgres.take_over(client_id)
        return await super().take_over(client_id)

    async def write(self, client_id: str, generation: int, delta: HistoryDelta | None, turns: KeptTurns) -> None:
        assert self.postgres is not None
        await self.postgres.write(client_id, self.owned[client_id], delta, turns)
        await super().write(client_id, generation, delta, turns)


@pytest.fixture
async def histories(database: DatabaseTarget) -> AsyncIterator[HistoriesInPostgres]:
    postgres = PostgresHistoryStore(database)
    yield HistoriesInPostgres(postgres=postgres)
    await postgres.close()


@pytest.fixture
async def lines(database: DatabaseTarget) -> AsyncIterator[PostgresLineStore]:
    store = PostgresLineStore(database)
    yield store
    await store.close()


async def test_a_turn_finds_by_their_words_a_view_revision_and_a_call_the_cut_dropped_from_his_history(
    runner, histories, lines
):
    """TURNS-39, TURNS-40."""
    runner.histories = histories
    runner.remember_in(lines, memory_target())
    runner.config = bounded_history(runner.config, 1)
    call_word = secrets.token_hex(4)
    runner.app.stand(SUBJECT, at_table())
    view_word = waits(runner.app, 1)["hand"]
    recalling = [tool_call(secrets.token_hex(4), RECALL_TOOL, {"text": word}) for word in (view_word, call_word)]
    runner.gateway = Gateway(
        tool_round(say(call_word, TABLE.id)), completion("done"), tool_round(*recalling), completion("done")
    )
    runner.start()
    await runner.gateway.until_turns(1, why="view")
    await runner.reports.until_a_turn_completed()
    await shows(runner.app, 2, {"hand": secrets.token_hex(4)}, awaits_you=True)
    await runner.gateway.until_requested(4)

    by_view_word, by_call_word = (json.loads(each)["entries"] for each in runner.gateway.tool_results(4))
    (view,) = by_view_word
    (call,) = (each for each in by_call_word if each["entry"] == "act")
    assert (view["entry"], view["app"], view["revision"], view["state"]) == (
        "view",
        url_of(runner, 0),
        1,
        {"hand": view_word},
    )
    assert (call["tool"], call["arguments"]["text"]) == ("say", call_word)
