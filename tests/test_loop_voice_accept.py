"""The whole runner, the voice: an agent whose entry has a voice table brings its names to every accept, in order."""

from __future__ import annotations

import secrets
from dataclasses import replace

import pytest
from agents.backend.fake_gateway import completion
from agents.member import FigureRef

from personas.config import Voice
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.history import TABLE
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, invited

pytestmark = pytest.mark.anyio

SECOND_AGENT, SECOND_SUBJECT = "agent-bach", "agent-subject-2"
VOICE_NAMES = ("Rachel_V2.en-GB", "de_DE-thorsten-medium", "Amy.en-US")


def the_wire_sent(app, subject: str = SUBJECT) -> dict:
    [wire] = [arguments for sender, arguments in app.accepted_on_the_wire if sender == subject]
    return wire


def the_voices_sent(app, subject: str = SUBJECT) -> list[str] | None:
    return the_wire_sent(app, subject).get("voices")


async def test_an_agent_with_named_voices_brings_them_in_order_beside_his_figure_to_a_call_and_a_second_apps_table(
    runner, other
):
    """CONFIG-18, CONFIG-19."""
    figure = FigureRef(url=f"https://figures.test/{secrets.token_hex(6)}.glb", sha256=secrets.token_hex(32))
    runner.config = replace(
        runner.config, agents=(replace(runner.config.agents[0], figure=figure, voice=Voice(VOICE_NAMES)),)
    )
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    other.stand(SUBJECT, invited(TABLE.id, "inv-4", kind=TABLE.kind, serves_view=True))
    runner.start()
    await runner.reports.until(Event.JOINED, count=2)

    for app in (runner.app, other):
        wire = the_wire_sent(app)
        assert wire["voices"] == list(VOICE_NAMES)
        assert "voice" not in wire
        assert wire["figure"] == {"url": figure.url, "sha256": figure.sha256}


@pytest.mark.parametrize("voice", [Voice(()), None], ids=["a table without names", "no table"])
async def test_an_agent_without_named_voices_sends_neither_voices_nor_voice(runner, voice):
    runner.config = replace(runner.config, agents=(replace(runner.config.agents[0], voice=voice),))
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    runner.start()
    await runner.reports.until(Event.JOINED)

    [(_, arguments)] = runner.app.accepted_on_the_wire
    assert "voices" not in arguments
    assert "voice" not in arguments


async def test_two_agents_each_bring_their_own_voices_or_none(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    first, second = runner.config.agents
    runner.config = replace(
        runner.config, agents=(replace(first, voice=Voice(VOICE_NAMES)), replace(second, voice=None))
    )
    runner.app.stand(SUBJECT, invited(PLACE_ID, "inv-3"))
    runner.app.stand(SECOND_SUBJECT, invited(PLACE_ID, "inv-4"))
    runner.start()
    await runner.reports.until(Event.JOINED, count=2)

    assert the_voices_sent(runner.app) == list(VOICE_NAMES)
    assert the_voices_sent(runner.app, SECOND_SUBJECT) is None


async def test_only_the_turn_of_an_agent_with_a_voice_is_told_aloud(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT)
    first, second = runner.config.agents
    runner.config = replace(runner.config, agents=(replace(first, voice=Voice(())), replace(second, voice=None)))
    runner.gateway = Gateway(completion(ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID, 2)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(2)

    assert sorted(turn.aloud for turn in runner.gateway.turns()) == [False, True]
