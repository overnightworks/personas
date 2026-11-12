"""The whole runner, the acts: every call relayed for an agent is told to his next turns with the app's answer."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from dataclasses import replace
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.tools import ToolOutcome
from agents.turn import Principal, person_fingerprint
from mcp_types import INVALID_PARAMS, CallToolResult, TextContent

from personas.app_session import READ_SECONDS, open_mcp_session
from personas.failures import Unreachable
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says_after_a_hold
from tests.support.history import (
    OTHER_CALL_ID,
    REMARK,
    SPEAKER,
    act_entry,
    answered,
    at_table,
    entry,
    history_of,
    said_and_taken,
    settled,
    told_line_bytes,
    unanswered,
    url_of,
    view_entry,
    view_turns,
    waits,
    written_turns,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, ISSUER, LISTEN, READ, SAY_TOOL, active
from tests.support.realm import AGENT_NAME
from tests.support.songmaker import CALLS, CREATE_SONG, ToolsOnlyApp

pytestmark = pytest.mark.anyio

AGAIN = f"{ADDRESSED} Again."
SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME = "agent-bach", "agent-subject-2", "Bach"


@pytest.fixture
async def songmaker(runner):
    app = ToolsOnlyApp()
    async with app.serving() as url:
        runner.add_tools_only_app(app, url)
        yield app


def say_round(*texts: str) -> object:
    return tool_round(
        *(tool_call(f"say-{index}", SAY_TOOL, {"place": PLACE_ID, "text": text}) for index, text in enumerate(texts))
    )


def song_round(count: int) -> object:
    return tool_round(*(tool_call(f"song-{index}", CREATE_SONG, {"title": "t"}) for index in range(count)))


def a_say(runner, text: str, outcome: dict[str, object]) -> dict[str, object]:
    return act_entry(url_of(runner, 0), "say", {"place": PLACE_ID, "text": text}, outcome)


def a_song(runner, songmaker_url: str, outcome: dict[str, object]) -> dict[str, object]:
    return act_entry(songmaker_url, CREATE_SONG, {"title": "t"}, outcome)


def songmaker_url(runner) -> str:
    (url,) = (each.mcp_url for each in runner.config.apps if not each.listens)
    return url


async def second_written_turn(runner, text: str = AGAIN):
    """A line that names him is taken once the first turn went through its last round; the turn it starts."""
    await runner.gateway.until_requested(2)
    await settled(runner.app)
    await said_and_taken(runner.app, PLACE_ID, text)
    await runner.gateway.until_turns(2)
    return written_turns(runner)[1]


async def cuts(reply):
    await reply.open_stream()
    reply.cut()


async def refuses_the_token(reply):
    await reply.status(HTTPStatus.UNAUTHORIZED)


async def answers_wrongly(reply):
    await reply.error(INVALID_PARAMS, status=HTTPStatus.OK)


async def refuses_with_a_server_error(reply):
    await reply.status(HTTPStatus.INTERNAL_SERVER_ERROR)


async def hangs(reply):
    await asyncio.Event().wait()


def hangs_once_entered(entered):
    async def hang(reply):
        entered.set()
        await asyncio.Event().wait()

    return hang


async def test_a_say_the_app_answered_is_told_to_his_next_turn_before_his_own_line_read_back_and_no_closing_text(
    runner,
):
    closing = secrets.token_hex(8)
    runner.gateway = Gateway(say_round(ANSWER), completion(closing), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    second = await second_written_turn(runner)

    app = url_of(runner, 0)
    assert second.room["history"] == history_of(
        [
            entry(app, PLACE_ID, ADDRESSED),
            a_say(runner, ANSWER, answered("said")),
            entry(app, PLACE_ID, ANSWER, seq=2, name=AGENT_NAME, own=True),
            entry(app, PLACE_ID, AGAIN, seq=3, new=True),
        ]
    )
    assert second.new_seqs == [3]
    assert closing not in second.user


async def test_an_error_result_the_app_answers_is_told_as_an_answer_that_is_an_error_and_the_turn_goes_on(runner):
    runner.app.say_outcomes.append(ToolOutcome("gone", is_error=True))
    runner.gateway = Gateway(say_round(ANSWER), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    second = await second_written_turn(runner)

    assert second.room["history"]["entries"][1] == a_say(runner, ANSWER, answered("gone", is_error=True))
    assert len(runner.gateway.requests) == 3
    assert runner.app.said == []


@pytest.mark.parametrize(
    ("fails", "code"),
    [(cuts, "network"), (refuses_the_token, "http_401"), (answers_wrongly, f"jsonrpc_{INVALID_PARAMS}")],
    ids=["transport", "401", "protocol error"],
)
async def test_a_relayed_say_that_fails_is_told_as_not_answered_with_the_failures_code_and_the_turn_ends_as_before(
    runner, fails, code
):
    runner.gateway = Gateway(say_round(ANSWER), say_round(ANSWER), completion("done"))
    await restarted_in_the_call(runner)
    runner.app.script(CALL, fails)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.log.until(LISTEN, 2)
    await runner.app.log.until(READ, 3)
    say_in_the_call(runner, AGAIN)
    await runner.gateway.until_requested(3)
    await runner.gateway.until_turns(2)

    app = url_of(runner, 0)
    waited = int(sum(runner.clock.slept))
    assert written_turns(runner)[1].room["history"] == history_of(
        [
            entry(app, PLACE_ID, ADDRESSED, seconds_ago=waited),
            act_entry(app, "say", {"place": PLACE_ID, "text": ANSWER}, unanswered(code), seconds_ago=waited),
            entry(app, PLACE_ID, AGAIN, seq=2, new=True),
        ]
    )
    assert runner.gateway.round_sizes() == [2, 2, 4]


async def test_calls_past_the_places_own_bound_enter_no_act(runner, songmaker):
    own_bound = runner.config.apps[0].binding.max_tool_calls
    runner.gateway = Gateway(say_round(*[ANSWER] * (own_bound + 1)), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    second = await second_written_turn(runner)

    acts = [each for each in second.room["history"]["entries"] if each["entry"] == "act"]
    assert acts == [a_say(runner, ANSWER, answered("said"))] * own_bound


async def test_a_tools_only_call_and_the_say_after_it_are_acts_in_the_order_made(runner, songmaker):
    runner.gateway = Gateway(song_round(1), say_round(ANSWER), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(3)

    second = await second_written_turn(runner)

    acts = [each for each in second.room["history"]["entries"] if each["entry"] == "act"]
    assert acts == [
        a_song(runner, songmaker_url(runner), answered(songmaker.results[CREATE_SONG])),
        a_say(runner, ANSWER, answered("said")),
    ]


async def test_calls_past_a_tools_only_apps_own_bound_enter_no_act(runner, songmaker):
    runner.gateway = Gateway(song_round(CALLS + 1), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    second = await second_written_turn(runner)

    acts = [each for each in second.room["history"]["entries"] if each["entry"] == "act"]
    assert acts == [a_song(runner, songmaker_url(runner), answered(songmaker.results[CREATE_SONG]))] * CALLS


@pytest.mark.parametrize(
    ("handler", "code", "call_seconds"),
    [(refuses_with_a_server_error, "http_500", None), (hangs, "network", 0.3)],
    ids=["an error of the app", "a hang past the call time"],
)
async def test_a_tools_only_call_that_fails_is_told_as_not_answered_with_the_failures_code_never_cancelled(
    runner, songmaker, handler, code, call_seconds
):
    songmaker.script(CALL, handler)
    if call_seconds is not None:
        runner.bound_tools_only_calls_to(call_seconds)
    runner.gateway = Gateway(song_round(1), say_round(ANSWER), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    second = await second_written_turn(runner)

    acts = [each for each in second.room["history"]["entries"] if each["entry"] == "act"]
    assert acts[0] == a_song(runner, songmaker_url(runner), unanswered(code))
    assert runner.reports.events() == [("started", None), ("tools_only_app_failed", code)]
    assert runner.gateway.tool_results(2)[0].startswith("Error")


CUT_IN_FLIGHT = pytest.mark.parametrize("tools_only", [False, True], ids=["the places app", "a tools-only app"])


async def a_call_waiting_for_its_app_while_its_place_leaves(runner, songmaker, *, tools_only, open_session=None):
    """The call of the first turn hangs at its app, the place leaves, and a line names him at another place.

    The act the call made, as the turn at the other place is told it.
    """
    entered = asyncio.Event()
    if tools_only:
        songmaker.script(CALL, hangs_once_entered(entered))
        runner.gateway = Gateway(song_round(1), completion("done"))
        act = a_song(runner, songmaker_url(runner), unanswered("cancelled"))
    else:
        runner.app.script(CALL, hangs_once_entered(entered))
        runner.gateway = Gateway(say_round(ANSWER), completion("done"))
        act = a_say(runner, ANSWER, unanswered("cancelled"))
    runner.app.stand(SUBJECT, active(PLACE_ID), active(OTHER_CALL_ID))
    runner.start(open_session=open_session)
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_window_read(OTHER_CALL_ID)
    say_in_the_call(runner, ADDRESSED)
    async with asyncio.timeout(10):
        await entered.wait()
    runner.app.stand(SUBJECT, active(OTHER_CALL_ID))
    await runner.reports.until("turn_ended")
    runner.app.speak(OTHER_CALL_ID, PERSON, SPEAKER, ADDRESSED)
    await runner.gateway.until_turns(2)
    return act, written_turns(runner)[1]


@CUT_IN_FLIGHT
async def test_a_call_cut_in_flight_because_its_place_left_is_told_cancelled_to_a_turn_at_another_place(
    runner, songmaker, tools_only
):
    act, second = await a_call_waiting_for_its_app_while_its_place_leaves(runner, songmaker, tools_only=tools_only)

    assert second.room["history"] == history_of(
        [
            entry(url_of(runner, 0), PLACE_ID, ADDRESSED),
            act,
            entry(url_of(runner, 0), OTHER_CALL_ID, ADDRESSED, new=True),
        ]
    )
    assert runner.reports.events() == [("started", None), ("turn_ended", "cancelled")]


async def test_a_call_cut_in_flight_whose_request_ends_as_a_transport_failure_is_told_cancelled_not_network(
    runner, songmaker
):
    class EndsACutCallAsAFailure:
        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        async def invoke(self, name, arguments):
            with contextlib.suppress(asyncio.CancelledError):
                return await self._session.invoke(name, arguments)
            raise Unreachable("cut")

    @contextlib.asynccontextmanager
    async def opened(app, token):
        async with open_mcp_session(app, token, stream_seconds=READ_SECONDS) as session:
            yield EndsACutCallAsAFailure(session)

    act, second = await a_call_waiting_for_its_app_while_its_place_leaves(
        runner, songmaker, tools_only=False, open_session=opened
    )

    assert second.room["history"]["entries"][1] == act
    assert runner.reports.events() == [("started", None), ("turn_ended", "cancelled")]


async def test_a_say_whose_only_effect_is_an_act_starts_no_further_turn_once_his_own_line_is_read(runner):
    runner.gateway = Gateway(say_round(ANSWER), completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(2)
    await said_and_taken(runner.app, PLACE_ID, REMARK, author=SUBJECT, name=AGENT_NAME)

    assert len(runner.gateway.openings()) == 1
    assert len(runner.gateway.requests) == 2


@pytest.mark.parametrize(
    ("bytes_over_the_three", "kept_kinds"),
    [
        pytest.param(0, ["act", "line", "line"], id="the bound is the act's, his line's and the new line's size"),
        pytest.param(-1, ["line", "line"], id="the bound lies one byte below them"),
    ],
)
async def test_the_bound_counts_an_act_by_the_utf8_bytes_of_its_told_json(runner, bytes_over_the_three, kept_kinds):
    text = "Grüße ♠"
    app = url_of(runner, 0)
    told_act = (
        f'{{"entry": "act", "app": "{app}", "seconds_ago": 0, "tool": "say",'
        f' "arguments": {{"place": "{PLACE_ID}", "text": "{text}"}},'
        ' "outcome": {"answered": true, "is_error": false, "text": "said"}, "new": false}'
    )
    act_bytes = len(told_act.encode())
    own_line_bytes = told_line_bytes(app, PLACE_ID, text, seq=2, name=AGENT_NAME, own=True)
    new_line_bytes = told_line_bytes(app, PLACE_ID, AGAIN, seq=3)
    runner.config = replace(
        runner.config, history_bytes=act_bytes + own_line_bytes + new_line_bytes + bytes_over_the_three
    )
    runner.gateway = Gateway(say_round(text), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)

    second = await second_written_turn(runner)

    history = second.room["history"]
    assert [each["entry"] for each in history["entries"]] == kept_kinds
    assert history["older_dropped"] is True


async def test_each_agent_is_told_only_his_own_acts(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    first, second = (person_fingerprint(Principal(iss=ISSUER, sub=subject)) for subject in (SUBJECT, SECOND_SUBJECT))
    runner.gateway = Gateway(completion("done"))
    runner.gateway.script_for(first, say_round("one"), completion("done"))
    runner.gateway.script_for(second, completion("done"), say_round("two"), completion("done"))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await said_and_taken(runner.app, PLACE_ID, ADDRESSED)
    await runner.app.said_count(1)
    await settled(runner.app, SECOND_SUBJECT)
    await said_and_taken(runner.app, PLACE_ID, f"{SECOND_NAME}, are you there?")
    await runner.app.said_count(2)
    await said_and_taken(runner.app, PLACE_ID, AGAIN)

    told = {
        user: [
            each
            for each in (await runner.gateway.until_written_turn(PLACE_ID, 5, user)).room["history"]["entries"]
            if each["entry"] == "act"
        ]
        for user in (first, second)
    }
    assert told == {
        first: [a_say(runner, "one", answered("said"))],
        second: [a_say(runner, "two", answered("said"))],
    }


async def test_an_act_is_told_after_the_line_taken_while_its_turn_ran_and_ages_from_the_moment_its_answer_came(
    runner, other
):
    async def answers_after_a_while(reply):
        runner.clock.time += 30
        await reply.result(CallToolResult(content=[TextContent(text="said")], is_error=False))

    runner.gateway = Gateway(*says_after_a_hold(OTHER_CALL_ID, ANSWER))
    running = runner.gateway.hold_request(1)
    other.script(CALL, answers_after_a_while)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    first = waits(other, 1)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await runner.gateway.until_requested(3)
    second = waits(other, 2)
    await runner.gateway.until_turns(2, why="view")

    table = url_of(runner, 1)
    assert view_turns(runner)[1].room["history"] == history_of(
        [
            view_entry(table, first, revision=1, seconds_ago=30),
            entry(url_of(runner, 0), PLACE_ID, REMARK, seconds_ago=30),
            act_entry(table, SAY_TOOL, {"place": OTHER_CALL_ID, "text": ANSWER}, answered("said")),
            view_entry(table, second, revision=2, new=True),
        ]
    )
