"""An optional argument an app's tool does not declare is left out of the call and logged once per session."""

from __future__ import annotations

import secrets

import pytest
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    SET_TURN_STATE_TOOL,
    SPEAK_TOOL,
    FigureRef,
    SetTurnStateArguments,
)

from personas.decision import Accept, accept_arguments
from personas.secret import Secret
from tests.support.app_session import opened, served
from tests.support.entry_logs import configure_logging_into
from tests.support.hopin import FakeApp
from tests.support.logs import fields_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

AGENT_ID = "agent-one"


@pytest.fixture(autouse=True)
def logging_captured(restored_logging, caplog):
    configure_logging_into(caplog)


def left_out_events(caplog):
    return [fields_of(record) for record in caplog.get_records("call") if record.message == "argument.left_out"]


@pytest.fixture
async def app_that_predates_the_optional_arguments():
    fake = FakeApp(undeclared_arguments=frozenset({"figure", "voices", "style", "interrupted", "thinking"}))
    async with served(fake):
        yield fake


async def test_optional_arguments_an_app_does_not_declare_are_left_out_and_each_is_logged_once_per_session(
    app_that_predates_the_optional_arguments, place, caplog
):
    """LOG-39."""
    app = app_that_predates_the_optional_arguments
    bearer = Secret(secrets.token_urlsafe(24))
    app.let_in(bearer.reveal(), SUBJECT)
    figure = FigureRef(url="https://figures.test/figure.glb", sha256=secrets.token_hex(32))
    accept = accept_arguments(Accept(place, "inv-3", 2, figure, ("amy",)))

    async with opened(app, bearer) as session:
        assert await session.accept(accept) is None
        assert await session.accept(accept) is None
        await session.speak(place, "hello", style="calm")
        await session.speak(place, "again", interrupted="hold-1")
        await session.speak(place, "once more", interrupted="hold-2")
        await session.set_turn_state(SetTurnStateArguments(place=place, thinking=True))

    assert [sent for _, sent in app.accepted_on_the_wire] == [
        {"place": {"kind": "call", "id": "room-7"}, "invitation": "inv-3", "epoch": 2}
    ] * 2
    assert app.speak_styles == [None] * 3
    assert app.speaks_naming_holds == [("hello", None), ("again", None), ("once more", None)]
    assert [(left["tool"], left["argument"]) for left in left_out_events(caplog)] == [
        (ACCEPT_INVITATION_TOOL, "figure"),
        (ACCEPT_INVITATION_TOOL, "voices"),
        (SPEAK_TOOL, "style"),
        (SPEAK_TOOL, "interrupted"),
        (SET_TURN_STATE_TOOL, "thinking"),
    ]
    assert {(left["agent_id"], left["app_id"]) for left in left_out_events(caplog)} == {(AGENT_ID, "127.0.0.1")}
