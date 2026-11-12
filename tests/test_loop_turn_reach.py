"""The whole runner, a turn that reaches a tools-only app: its tools, its token, its failures and its bounds."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import anyio
import httpx2
import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.tools import ToolOutcome
from mcp_types import ListToolsResult, Tool

from personas import app_session
from personas.config import load_config
from personas.decision import TOOL_NOT_DECLARED_CODE, TURN_CANCELLED_CODE
from personas.failures import Unreachable, leaves_of
from personas.reporting import Event
from personas.tokens import HttpTokenSource
from personas.turning import Turning
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, DISCOVER, LIMIT_SECONDS, LIST_TOOLS, LISTEN, SAY_TOOL
from tests.support.songmaker import CALLS, CREATE_SONG, GENERATE_SONG, SCOPE
from tests.support.transport import (
    ClosingHold,
    transport_by_url,
    transport_closing_with_late_answers,
    transport_closing_with_leaves,
    transport_whose_close_loses_a_child,
    transport_whose_session_loses_a_child,
)
from tests.support.wire import cuts, hangs, hangs_once_entered, refuses_with_a_server_error

pytestmark = pytest.mark.anyio

SHORT_CALL_SECONDS = 0.3


def songmaker_url(runner) -> str:
    (url,) = (each.mcp_url for each in runner.config.apps if not each.listens)
    return url


def calling(*names: str, prefix: str = "call") -> object:
    return tool_round(*(tool_call(f"{prefix}-{index}", name, {"title": "t"}) for index, name in enumerate(names)))


def saying(text: str, *, prefix: str = "say") -> object:
    return tool_round(tool_call(f"{prefix}-0", SAY_TOOL, {"place": PLACE_ID, "text": text}))


async def ended_by(task):
    async with asyncio.timeout(LIMIT_SECONDS):
        (ended,) = await asyncio.gather(task, return_exceptions=True)
    return ended


async def addressed_in_the_call(runner):
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)


def failed_apps(runner):
    return [change.code for change in runner.reports.changes if change.event is Event.TOOLS_ONLY_APP_FAILED]


async def a_song_is_made_and_announced(runner, songmaker):
    last = songmaker.results[GENERATE_SONG]
    runner.gateway = Gateway(calling(CREATE_SONG, GENERATE_SONG), saying(last), completion("done"))
    await addressed_in_the_call(runner)
    await runner.app.said_count(1)


async def test_an_addressed_line_runs_the_bound_tools_of_a_tools_only_app_in_order_and_the_last_result_reaches_say(
    runner, songmaker
):
    await a_song_is_made_and_announced(runner, songmaker)

    assert songmaker.called == [(CREATE_SONG, SUBJECT), (GENERATE_SONG, SUBJECT)]
    assert runner.gateway.tool_results(2) == [songmaker.results[CREATE_SONG], songmaker.results[GENERATE_SONG]]
    assert runner.app.said == [(SUBJECT, PLACE_ID, songmaker.results[GENERATE_SONG])]
    assert sorted(runner.gateway.offered_tools()) == sorted([SAY_TOOL, CREATE_SONG, GENERATE_SONG])
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_each_app_sees_only_the_bearers_issued_for_its_own_scope(runner, songmaker):
    await a_song_is_made_and_announced(runner, songmaker)

    songmaker_bearers = songmaker.presented - {""}
    hopin_bearers = runner.app.presented - {""}
    assert songmaker_bearers
    assert songmaker_bearers <= set(runner.realm.issued_for[SCOPE])
    assert hopin_bearers <= set(runner.realm.issued_for[runner.config.apps[0].scope])
    assert songmaker_bearers.isdisjoint(hopin_bearers)


@pytest.mark.parametrize(
    ("arrange", "code", "call_seconds"),
    [
        (lambda app: setattr(app, "refuse_with", HTTPStatus.FORBIDDEN), "http_403", None),
        (lambda app: setattr(app, "refuse_with", HTTPStatus.INTERNAL_SERVER_ERROR), "http_500", None),
        (lambda app: app.script(LIST_TOOLS, cuts), "network", None),
        (lambda app: app.script(LIST_TOOLS, hangs), "network", SHORT_CALL_SECONDS),
    ],
    ids=["refused", "failing", "cut", "hanging"],
)
async def test_a_tools_only_app_whose_listing_fails_at_turn_start_is_absent_and_the_turn_answers_in_its_place(
    runner, songmaker, arrange, code, call_seconds
):
    arrange(songmaker)
    songmaker.refused_method = LIST_TOOLS
    if call_seconds is not None:
        runner.bound_tools_only_calls_to(call_seconds)
    runner.gateway = Gateway(saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL]
    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, code, songmaker_url(runner))
    ]
    assert runner.clock.slept == []
    assert runner.app.log.count(LISTEN) == 1
    assert not runner.task.done()


async def test_a_tools_only_app_that_declares_one_bound_tool_no_more_is_absent_as_a_whole(runner, songmaker):
    async def lists_one_of_the_two(reply):
        await reply.result(
            ListToolsResult(
                tools=[Tool(name=CREATE_SONG, description="Creates a song.", input_schema={"type": "object"})]
            )
        )

    songmaker.script(LIST_TOOLS, lists_one_of_the_two)
    runner.gateway = Gateway(saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL]
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, TOOL_NOT_DECLARED_CODE, songmaker_url(runner))
    ]


async def a_call_that_fails_and_a_turn_that_goes_on(runner, songmaker, code, *, call_seconds=None):
    """The model calls one tool of the tools-only app, whose call is scripted to fail, and then says."""
    if call_seconds is not None:
        runner.bound_tools_only_calls_to(call_seconds)
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    second_round = runner.gateway.hold_request(2)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(2)

    assert runner.gateway.cancelling_at_arrival[1] == 0
    second_round.set()
    await runner.app.said_count(1)

    (shown,) = runner.gateway.tool_results(2)
    assert shown.startswith("Error")
    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, code, songmaker_url(runner))
    ]
    assert runner.clock.slept == []
    assert runner.app.log.count(LISTEN) == 1
    assert not runner.task.done()


@pytest.mark.parametrize(
    ("handler", "code", "call_seconds"),
    [
        (refuses_with_a_server_error, "http_500", None),
        (cuts, "network", None),
        (hangs, "network", SHORT_CALL_SECONDS),
    ],
    ids=["a server error", "a cut connection", "a hang past the call time"],
)
async def test_a_tools_only_call_that_fails_shows_the_model_an_error_and_the_turn_goes_on_and_answers(
    runner, songmaker, handler, code, call_seconds
):
    songmaker.script(CALL, handler)

    await a_call_that_fails_and_a_turn_that_goes_on(runner, songmaker, code, call_seconds=call_seconds)


async def test_a_session_of_a_tools_only_call_whose_sdk_child_is_lost_shows_the_model_an_error_and_the_runner_goes_on(
    runner, songmaker, monkeypatch
):
    failing = asyncio.Event()
    monkeypatch.setattr(
        app_session,
        "streamable_http_client",
        transport_whose_session_loses_a_child(failing, nth=2, of_url=songmaker_url(runner)),
    )

    async def loses_its_child(reply):
        failing.set()
        await asyncio.Event().wait()

    songmaker.script(CALL, loses_its_child)

    await a_call_that_fails_and_a_turn_that_goes_on(runner, songmaker, "network")


async def test_a_tools_only_app_whose_call_failed_takes_no_more_calls_in_the_turn_and_is_reported_once(
    runner, songmaker
):
    songmaker.script(CALL, refuses_with_a_server_error)
    runner.gateway = Gateway(calling(CREATE_SONG, GENERATE_SONG), saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    failed, closed = runner.gateway.tool_results(2)
    assert failed.startswith("Error")
    assert closed.startswith("Error")
    assert closed != failed
    assert songmaker.log.count(CALL) == 1
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, "http_500", songmaker_url(runner))
    ]


async def a_loss_while_the_app_session_opens_and_a_turn_that_answers(runner, songmaker, monkeypatch, transport):
    monkeypatch.setattr(app_session, "streamable_http_client", transport)
    runner.gateway = Gateway(saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL]
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, "network", songmaker_url(runner))
    ]
    assert not runner.task.done()


async def test_a_child_lost_while_the_listing_session_discovers_leaves_the_app_absent_and_the_runner_goes_on(
    runner, songmaker, monkeypatch
):
    failing = asyncio.Event()

    async def loses_its_child(reply):
        failing.set()
        await asyncio.Event().wait()

    songmaker.script(DISCOVER, loses_its_child)
    transport = transport_whose_session_loses_a_child(failing, nth=1, of_url=songmaker_url(runner))

    await a_loss_while_the_app_session_opens_and_a_turn_that_answers(runner, songmaker, monkeypatch, transport)


async def test_a_child_lost_while_a_calls_session_discovers_shows_the_model_an_error_and_the_runner_goes_on(
    runner, songmaker, monkeypatch
):
    failing = asyncio.Event()
    monkeypatch.setattr(
        app_session,
        "streamable_http_client",
        transport_whose_session_loses_a_child(failing, nth=2, of_url=songmaker_url(runner)),
    )
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    first_round = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)

    async def loses_its_child(reply):
        failing.set()
        await asyncio.Event().wait()

    songmaker.script(DISCOVER, loses_its_child)
    first_round.set()
    await runner.app.said_count(1)

    assert runner.gateway.tool_results(2)[0].startswith("Error")
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, "network", songmaker_url(runner))
    ]
    assert not runner.task.done()


@pytest.mark.parametrize("hung", [DISCOVER, LIST_TOOLS], ids=["discovery", "listing"])
async def test_a_hang_past_the_call_time_whose_close_gives_late_answers_leaves_the_app_absent_and_the_turn_answers(
    runner, songmaker, monkeypatch, hung
):
    songmaker.script(hung, hangs)
    runner.bound_tools_only_calls_to(SHORT_CALL_SECONDS)
    transport = transport_closing_with_late_answers(echo=anyio.BrokenResourceError, only_close=1)

    await a_loss_while_the_app_session_opens_and_a_turn_that_answers(runner, songmaker, monkeypatch, transport)


async def test_a_token_the_realm_refuses_for_the_tools_only_scope_leaves_the_app_absent_without_a_reaction(
    runner, songmaker
):
    runner.realm.refused_scopes.add(SCOPE)
    runner.gateway = Gateway(saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL]
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, "http_401", songmaker_url(runner))
    ]
    assert runner.clock.slept == []
    assert runner.app.log.count(LISTEN) == 1


async def test_a_token_the_realm_refuses_in_mid_turn_for_the_tools_only_scope_is_an_error_and_no_reaction(
    runner, songmaker
):
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    first_round = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)
    runner.realm.refused_scopes.add(SCOPE)
    first_round.set()
    await runner.app.said_count(1)

    assert songmaker.called == []
    assert runner.gateway.tool_results(2)[0].startswith("Error")
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, "http_401", songmaker_url(runner))
    ]
    assert runner.clock.slept == []
    assert runner.app.log.count(LISTEN) == 1


async def a_turn_inside_a_tools_only_call(runner, songmaker):
    entered = asyncio.Event()
    songmaker.script(CALL, hangs_once_entered(entered))
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)
    async with asyncio.timeout(LIMIT_SECONDS):
        await entered.wait()


async def test_a_turn_whose_place_left_during_a_tools_only_call_stops_with_no_further_round_or_say(runner, songmaker):
    await a_turn_inside_a_tools_only_call(runner, songmaker)

    runner.app.stand(SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert len(runner.gateway.requests) == 1
    assert runner.app.said == []
    assert not runner.task.done()


async def test_a_defect_in_a_tools_only_request_whose_place_then_leaves_while_it_closes_ends_the_runner_loud(
    runner, songmaker, monkeypatch
):
    defect = ValueError("a defect")

    async def raise_it(self, name, arguments):
        raise defect

    closing = ClosingHold()
    monkeypatch.setattr("personas.app_session._McpAppSession.invoke", raise_it)
    monkeypatch.setattr(app_session, "streamable_http_client", closing.transport(nth=2, of_url=songmaker_url(runner)))
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.closing.wait()
        runner.app.stand(SUBJECT)
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
    finally:
        closing.release.set()

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]


def raising_in_the_tools_only_request(runner, monkeypatch, where, defect):
    """The listing session or the first call of the tools-only app raises ``defect`` once."""
    session_class = app_session._McpAppSession  # noqa: SLF001 - the session is where the app's request raises
    real_invoke = session_class.invoke
    real_declared = session_class.declared_tools
    raised = False

    async def invoke(self, name, arguments):
        nonlocal raised
        if where == "call" and name == CREATE_SONG and not raised:
            raised = True
            raise defect
        return await real_invoke(self, name, arguments)

    async def declared_tools(self):
        nonlocal raised
        if where == "listing" and self.auth.token.reveal() in runner.realm.issued_for[SCOPE] and not raised:
            raised = True
            raise defect
        return await real_declared(self)

    monkeypatch.setattr(session_class, "invoke", invoke)
    monkeypatch.setattr(session_class, "declared_tools", declared_tools)


@pytest.mark.parametrize("where", ["listing", "call"])
async def test_a_defect_in_a_tools_only_request_whose_close_outlasts_the_apps_call_time_ends_the_runner_loud(
    runner, songmaker, monkeypatch, where
):
    defect = ValueError("a defect")
    raising_in_the_tools_only_request(runner, monkeypatch, where, defect)
    runner.bound_tools_only_calls_to(SHORT_CALL_SECONDS)
    closing = ClosingHold()
    nth = 1 if where == "listing" else 2
    monkeypatch.setattr(app_session, "streamable_http_client", closing.transport(nth=nth, of_url=songmaker_url(runner)))
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
    finally:
        closing.release.set()

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert len(runner.gateway.requests) == (0 if where == "listing" else 1)
    assert runner.app.said == []


@pytest.mark.parametrize("armed_at", ["listing", "call"])
async def test_a_timeout_the_token_source_raises_itself_before_the_call_time_is_up_is_a_defect_ending_the_runner(
    runner, songmaker, monkeypatch, armed_at
):
    armed = asyncio.Event()
    if armed_at == "listing":
        armed.set()
    real_fetch = HttpTokenSource.fetch
    defect = TimeoutError("not the call time of the app")

    async def fetching(self, agent, scope):
        if scope == SCOPE and armed.is_set():
            raise defect
        return await real_fetch(self, agent, scope)

    monkeypatch.setattr(HttpTokenSource, "fetch", fetching)
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    first_round = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    if armed_at == "call":
        await runner.gateway.until_requested(1)
        armed.set()
        first_round.set()

    ended = await ended_by(runner.task)

    assert defect in list(leaves_of(ended))
    assert not any(isinstance(leaf, Unreachable) for leaf in leaves_of(ended))
    assert failed_apps(runner) == []


async def test_a_defect_in_a_tools_only_call_whose_close_outlasts_the_turns_duration_ends_the_runner_loud(
    runner, songmaker, monkeypatch
):
    runner.files.persona_path.write_text(runner.files.persona_text(max_duration_seconds=1), encoding="utf-8")
    runner.config = load_config(runner.files.write())
    defect = ValueError("a defect")
    raising_in_the_tools_only_request(runner, monkeypatch, "call", defect)
    closing = ClosingHold()
    monkeypatch.setattr(app_session, "streamable_http_client", closing.transport(nth=2, of_url=songmaker_url(runner)))
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
    finally:
        closing.release.set()

    ended = await ended_by(runner.task)

    assert list(leaves_of(ended)) == [defect]
    assert runner.reports.reported() == [(Event.STARTED, None)]
    assert len(runner.gateway.requests) == 1
    assert runner.app.said == []


async def test_a_tools_only_call_whose_cancelled_session_leaves_as_a_failure_still_stops_the_turn(
    runner, songmaker, monkeypatch
):
    monkeypatch.setattr(app_session, "streamable_http_client", transport_closing_with_leaves(OSError(), only_close=1))
    await a_turn_inside_a_tools_only_call(runner, songmaker)

    runner.app.stand(SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)

    assert runner.reports.reported() == [(Event.STARTED, None), (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert len(runner.gateway.requests) == 1
    assert runner.app.said == []


async def test_a_runner_cancelled_during_a_tools_only_call_ends_cancelled_with_no_further_round_or_say(
    runner, songmaker
):
    await a_turn_inside_a_tools_only_call(runner, songmaker)

    runner.task.cancel()

    assert isinstance(await ended_by(runner.task), asyncio.CancelledError)
    assert len(runner.gateway.requests) == 1
    assert runner.app.said == []


async def test_calls_past_an_apps_own_bound_are_errors_and_the_other_apps_calls_still_run(runner, songmaker):
    over_the_bound = CALLS + 1
    runner.gateway = Gateway(calling(*[CREATE_SONG] * over_the_bound), saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    shown = runner.gateway.tool_results(2)
    assert [each.startswith("Error") for each in shown] == [False] * CALLS + [True]
    assert [name for name, _ in songmaker.called] == [CREATE_SONG] * CALLS
    assert runner.app.said == [(SUBJECT, PLACE_ID, ANSWER)]


async def test_calls_past_the_places_own_bound_are_errors_while_the_tools_only_app_has_calls_left(runner, songmaker):
    own_bound = runner.config.apps[0].binding.max_tool_calls
    over_the_bound = own_bound + 1
    says = tool_round(
        *(tool_call(f"say-{index}", SAY_TOOL, {"place": PLACE_ID, "text": ANSWER}) for index in range(over_the_bound))
    )
    runner.gateway = Gateway(says, completion("done"))

    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(2)

    assert len(runner.app.said) == own_bound
    assert runner.gateway.tool_results(2)[-1].startswith("Error")


@pytest.mark.parametrize("armed_at", ["listing", "call"])
async def test_a_leaf_that_is_no_call_failure_ends_the_runner_though_a_tools_only_app_raised_it(
    runner, songmaker, monkeypatch, armed_at
):
    armed = asyncio.Event()
    if armed_at == "listing":
        armed.set()
    real_fetch = HttpTokenSource.fetch
    defect = ValueError("a defect")

    async def fetching(self, agent, scope):
        if scope == SCOPE and armed.is_set():
            raise defect
        return await real_fetch(self, agent, scope)

    monkeypatch.setattr(HttpTokenSource, "fetch", fetching)
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    first_round = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    if armed_at == "call":
        await runner.gateway.until_requested(1)
        armed.set()
        first_round.set()

    ended = await ended_by(runner.task)

    assert defect in list(leaves_of(ended))


async def test_a_tools_only_outcome_that_is_an_error_shows_the_model_its_text(runner, songmaker):
    refused = ToolOutcome("the song limit of the account is reached", is_error=True)
    songmaker.outcomes.append(refused)
    songmaker.content.add(refused.content)
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))

    await addressed_in_the_call(runner)
    await runner.app.said_count(1)

    assert refused.content in runner.gateway.tool_results(2)[0]
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def ends_with_the_defect_beside(runner, defect, *others):
    """The runner ended with ``defect`` once among its leaves, beside exactly the ``others``, reporting no turn."""
    ended = await ended_by(runner.task)
    leaves = list(leaves_of(ended))

    assert [leaf for leaf in leaves if leaf is defect] == [defect]
    assert sorted(type(leaf).__name__ for leaf in leaves if leaf is not defect) == sorted(
        each.__name__ for each in others
    )
    assert runner.reports.reported() == [(Event.STARTED, None)]


CHILD_LOST = pytest.mark.parametrize(
    "child", [lambda: OSError("connection reset"), lambda: httpx2.ReadError("cut")], ids=["os error", "read error"]
)
REQUEST_AT = pytest.mark.parametrize("where", ["listing", "call"])


@REQUEST_AT
@CHILD_LOST
async def test_a_defect_in_a_tools_only_request_whose_turn_session_loses_a_child_while_it_closes_ends_the_runner_loud(
    runner, songmaker, monkeypatch, where, child
):
    defect = ValueError("a defect")
    raising_in_the_tools_only_request(runner, monkeypatch, where, defect)
    failing = asyncio.Event()
    closing = ClosingHold()
    held = closing.transport(nth=1 if where == "listing" else 2, of_url=songmaker_url(runner))
    lost = transport_whose_session_loses_a_child(failing, nth=2, of_url=runner.app_url, error=child)
    monkeypatch.setattr(
        app_session,
        "streamable_http_client",
        transport_by_url({songmaker_url(runner): held, runner.app_url: lost}),
    )
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)
    try:
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.closing.wait()
        failing.set()
        async with asyncio.timeout(LIMIT_SECONDS):
            await closing.cancelled.wait()
    finally:
        closing.release.set()

    await ends_with_the_defect_beside(runner, defect, Unreachable)


async def test_a_second_defect_of_the_turns_session_body_leaves_beside_the_first_one_a_tools_only_call_recorded(
    runner, songmaker, monkeypatch
):
    first, second = ValueError("first"), KeyError("second")
    raising_in_the_tools_only_request(runner, monkeypatch, "call", first)
    real_body = Turning._run_session_body  # noqa: SLF001 - the turn's body is where the second defect arises

    async def body_raising_the_second(self, *arguments):
        try:
            await real_body(self, *arguments)
        except Exception:
            raise second from None

    monkeypatch.setattr(Turning, "_run_session_body", body_raising_the_second)
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)

    await ends_with_the_defect_beside(runner, first, KeyError)

    assert [leaf for leaf in leaves_of(runner.task.exception()) if leaf is second] == [second]


async def test_a_tools_only_call_that_fails_whose_close_adds_a_defect_ends_the_runner_with_each_leaf_once(
    runner, songmaker, monkeypatch
):
    songmaker.script(CALL, refuses_with_a_server_error)
    monkeypatch.setattr(
        app_session, "streamable_http_client", transport_closing_with_late_answers(echo=RuntimeError, only_close=1)
    )
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)

    ended = await ended_by(runner.task)

    assert sorted(type(leaf).__name__ for leaf in leaves_of(ended)) == ["RuntimeError", "StatusFailure"]
    assert runner.reports.reported() == [(Event.STARTED, None)]


@REQUEST_AT
@pytest.mark.parametrize(
    ("lost_child", "then"),
    [
        (OSError, "say answers"),
        (OSError, "say fails"),
        (anyio.BrokenResourceError, "say answers"),
    ],
    ids=["a lost child, say answers", "a lost child, say fails", "a late answer"],
)
async def test_a_defect_in_a_tools_only_request_whose_own_close_loses_a_child_ends_the_runner_loud_whatever_follows(
    runner, songmaker, monkeypatch, where, lost_child, then
):
    defect = ValueError("a defect")
    raising_in_the_tools_only_request(runner, monkeypatch, where, defect)
    lost = transport_whose_close_loses_a_child(
        lost_child, nth=1 if where == "listing" else 2, of_url=songmaker_url(runner)
    )
    monkeypatch.setattr(app_session, "streamable_http_client", lost)
    if then == "say fails":
        runner.app.script(CALL, refuses_with_a_server_error)
    runner.gateway = Gateway(calling(CREATE_SONG), saying(ANSWER), completion("done"))
    await addressed_in_the_call(runner)

    await ends_with_the_defect_beside(runner, defect, Unreachable)

    assert failed_apps(runner) == []
    assert runner.app.said == []
