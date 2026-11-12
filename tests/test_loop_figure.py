"""The whole runner, the figure: an agent brings the figure his entry names to every accept, and nothing else."""

from __future__ import annotations

import secrets
from dataclasses import replace

import pytest
from agents.member import FigureRef

from personas.config import load_config
from tests.support.history import TABLE
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import invited

pytestmark = pytest.mark.anyio

SECOND_AGENT, SECOND_SUBJECT = "agent-bach", "agent-subject-2"


def a_figure() -> FigureRef:
    return FigureRef(url=f"https://figures.test/{secrets.token_hex(6)}.glb", sha256=secrets.token_hex(32))


def brings(runner, figure: FigureRef) -> None:
    """The first agent's entry names this figure; the configuration is read again from the file."""
    runner.files.text += f'[agents.figure]\nurl = "{figure.url}"\nsha256 = "{figure.sha256}"\n'
    runner.config = load_config(runner.files.write())


def the_figure_sent(app, subject: str = SUBJECT) -> dict | None:
    [wire] = [arguments for sender, arguments in app.accepted_on_the_wire if sender == subject]
    return wire.get("figure")


async def test_an_agent_with_a_figure_brings_it_to_a_call(runner):
    figure = a_figure()
    brings(runner, figure)
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()
    await runner.reports.until("joined")

    assert the_figure_sent(runner.app) == {"url": figure.url, "sha256": figure.sha256}


async def test_the_same_agent_brings_his_figure_to_a_table_of_a_second_app(runner, other):
    figure = a_figure()
    brings(runner, figure)
    other.stand(SUBJECT, invited(TABLE.id, "inv-4", kind=TABLE.kind, serves_view=True))
    runner.start()
    await runner.reports.until("joined")

    assert the_figure_sent(other) == {"url": figure.url, "sha256": figure.sha256}


async def test_an_agent_without_a_figure_sends_no_figure_key(runner):
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.start()
    await runner.reports.until("joined")

    [(_, arguments)] = runner.app.accepted_on_the_wire
    assert "figure" not in arguments


async def test_two_agents_each_bring_their_own_figure_or_none(runner):
    figure = a_figure()
    brings(runner, figure)
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    runner.config = replace(
        runner.config, agents=(runner.config.agents[0], replace(runner.config.agents[1], figure=None))
    )
    runner.app.stand(SUBJECT, invited("room-7", "inv-3"))
    runner.app.stand(SECOND_SUBJECT, invited("room-7", "inv-4"))
    runner.start()
    await runner.reports.until("joined", count=2)

    assert the_figure_sent(runner.app) == {"url": figure.url, "sha256": figure.sha256}
    assert the_figure_sent(runner.app, SECOND_SUBJECT) is None
