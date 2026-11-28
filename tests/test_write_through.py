"""What the write of an agent's mind hands the store beside his history: the entries taken and not yet kept."""

from __future__ import annotations

import pytest

from personas.entries import KeptHistory, KeptLine
from personas.failures import STORE_UNREACHABLE_CODE, StoreUnavailable
from personas.reporting import Event
from tests.support.app_driving import said_and_taken
from tests.support.call import PLACE_ID, restarted_in_the_call
from tests.support.history import REMARK, said
from tests.support.ownership import KeptHistories
from tests.support.store import ScriptedStore

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


class HistoriesDownOnce(KeptHistories):
    """Histories whose first write fails as a store that does not answer."""

    failed = False

    async def write(self, client_id, generation, delta, turns) -> None:
        if not self.failed and delta is not None and delta.entries:
            self.failed = True
            raise StoreUnavailable(STORE_UNREACHABLE_CODE)
        await super().write(client_id, generation, delta, turns)


def texts(recorded) -> list[str]:
    return [entry.text for _, entry in recorded if isinstance(entry, KeptLine)]


async def test_with_a_memory_every_entry_taken_is_handed_the_store_to_keep_once(runner):
    runner.remember_in(ScriptedStore())
    await restarted_in_the_call(runner)

    await said_and_taken(runner.app, PLACE_ID, REMARK)
    await said_and_taken(runner.app, PLACE_ID, "Again.")

    recorded = runner.histories.recorded()
    assert texts(recorded) == [REMARK, "Again."]
    assert [ordinal for ordinal, _ in recorded] == list(range(len(recorded)))


async def test_without_a_memory_no_entry_is_handed_the_store_to_keep(runner):
    await restarted_in_the_call(runner)

    await said_and_taken(runner.app, PLACE_ID, REMARK)

    assert runner.histories.entries()
    assert runner.histories.recorded() == []


async def test_an_entry_the_store_did_not_take_is_handed_again_with_the_next_write(runner):
    runner.histories = HistoriesDownOnce()
    runner.remember_in(ScriptedStore())
    await restarted_in_the_call(runner)

    await said_and_taken(runner.app, PLACE_ID, REMARK)
    await runner.reports.until(Event.STORE_RESUMED)

    assert texts(runner.histories.recorded()) == [REMARK]


async def test_the_first_write_after_a_load_hands_the_store_every_entry_the_loaded_history_holds_from_its_first(runner):
    held = (said(8, "kept before"), said(9, "also"))
    agent = runner.config.agents[0].client_id
    runner.histories.kept[agent] = KeptHistory(4, held, 0, ())
    runner.remember_in(ScriptedStore())

    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)

    assert runner.histories.recorded()[:2] == [(4, held[0]), (5, held[1])]
    assert texts(runner.histories.recorded()) == ["kept before", "also", REMARK]
