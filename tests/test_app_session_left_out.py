"""An optional argument an app's tool does not declare is left out of the call and logged once per session."""

from __future__ import annotations

import secrets
from http import HTTPStatus

import pytest
from agents.member import (
    ACCEPT_INVITATION_TOOL,
    SET_TURN_STATE_TOOL,
    SPEAK_TOOL,
    FigureRef,
    HttpsFileRef,
    SetTurnStateArguments,
)
from agents.member_outcomes import UnusableAnswer, jsonrpc_code
from mcp_types import INTERNAL_ERROR

from personas.decision import Accept
from personas.secret import Secret
from tests.support.app_session import opened, served
from tests.support.entry_logs import configure_logging_into
from tests.support.hopin import LIST_TOOLS, FakeApp
from tests.support.logs import fields_of
from tests.support.standings import AGENT_SUBJECT as SUBJECT

pytestmark = pytest.mark.anyio

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
    accept = Accept(place, "inv-3", 2, figure, ("amy",))

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


async def test_optional_arguments_an_app_declares_are_sent_and_logged_as_none(app, bearer, place, caplog):
    figure = FigureRef(url="https://figures.test/figure.glb", sha256=secrets.token_hex(32))

    async with opened(app, bearer) as session:
        await session.accept(Accept(place, "inv-3", 2, figure, ("amy",)))
        await session.speak(place, "hello", style="calm")

    [(_, sent)] = app.accepted_on_the_wire
    assert sent["voices"] == ["amy"]
    assert sent["figure"]["url"] == figure.url
    assert app.speak_styles == ["calm"]
    assert left_out_events(caplog) == []


def a_figure_with_a_room() -> tuple[FigureRef, HttpsFileRef]:
    figure = FigureRef(url="https://figures.test/figure.vrm", sha256=secrets.token_hex(32))
    return figure, HttpsFileRef(url="https://figures.test/figure.room.glb", sha256=secrets.token_hex(32))


async def test_an_app_that_declares_the_room_of_a_look_gets_the_look_with_it(app, bearer, place, caplog):
    figure, room = a_figure_with_a_room()

    async with opened(app, bearer) as session:
        await session.accept(Accept(place, "inv-3", 2, figure, room=room))

    [(_, sent)] = app.accepted_on_the_wire
    assert sent["looks"] == [
        {"kind": "figure", "url": figure.url, "sha256": figure.sha256, "room": {"url": room.url, "sha256": room.sha256}}
    ]
    assert left_out_events(caplog) == []


@pytest.mark.parametrize(
    ("undeclared", "looks_kept"),
    [("looks", False), ("looks[].room", True)],
)
async def test_an_app_that_does_not_declare_looks_or_the_room_of_a_look_gets_the_accept_without_it_and_it_is_logged(
    place, caplog, undeclared, looks_kept
):
    figure, room = a_figure_with_a_room()
    fake = FakeApp(undeclared_arguments=frozenset({undeclared}))
    bearer = Secret(secrets.token_urlsafe(24))
    async with served(fake):
        fake.let_in(bearer.reveal(), SUBJECT)
        async with opened(fake, bearer) as session:
            assert await session.accept(Accept(place, "inv-3", 2, figure, room=room)) is None
            assert await session.accept(Accept(place, "inv-4", 2, figure, room=room)) is None

    for _, sent in fake.accepted_on_the_wire:
        assert sent["figure"] == {"url": figure.url, "sha256": figure.sha256}
        assert sent.get("looks") == (
            [{"kind": "figure", "url": figure.url, "sha256": figure.sha256}] if looks_kept else None
        )
    assert [(left["tool"], left["argument"]) for left in left_out_events(caplog)] == [
        (ACCEPT_INVITATION_TOOL, undeclared)
    ]


async def test_an_accept_on_a_cold_session_fails_with_the_failure_of_the_apps_tools_list(app, bearer, place):
    async def refuses_the_list(reply):
        await reply.error(INTERNAL_ERROR, status=HTTPStatus.OK)

    app.script(LIST_TOOLS, refuses_the_list)
    figure = FigureRef(url="https://figures.test/figure.glb", sha256=secrets.token_hex(32))

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await session.accept(Accept(place, "inv-3", 2, figure, ()))

    assert failure.value.code == jsonrpc_code(INTERNAL_ERROR)
    assert app.accepted_on_the_wire == []
