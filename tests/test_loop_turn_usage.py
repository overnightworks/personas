"""The whole runner, a turn that ends with its final event: what the line of its end says of the tokens it used."""

from __future__ import annotations

import re
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply, completion, tool_round
from agents.tools import ToolOutcome
from mcp_types import ListToolsResult

from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, say
from tests.support.hopin import LIST_TOOLS, formatted
from tests.support.voice import a_voice_call

pytestmark = pytest.mark.anyio


def counted(prompt: int, cached: int, written: int, completed: int) -> dict[str, object]:
    """A usage chunk as the gateway sends it."""
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completed,
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
    }


_SIZES = r" tools_bytes=(\d+) told_bytes=(\d+) results_bytes=(\d+)"
_COUNTS = r"(?P<counts>(?: input=\d+ cache_read=\d+ cache_write=\d+ output=\d+)?)"
_LINE = re.compile(rf"turn_ended client_id=\S+ app=\S+ code=(?P<code>\S+)(?:{_SIZES})?{_COUNTS}")


def the_turn_ended_line(caplog) -> re.Match[str]:
    """The one `turn_ended` line, which holds nothing but the fields the log declares."""
    ended = [
        line for line in formatted(caplog.get_records("call")).splitlines() if line.startswith(Event.TURN_ENDED.value)
    ]
    assert len(ended) == 1
    matched = _LINE.fullmatch(ended[0])
    assert matched is not None, ended[0]
    return matched


@pytest.mark.parametrize(
    ("rounds", "counts"),
    [
        pytest.param(
            (tool_round(say(ANSWER), usage=counted(100, 40, 10, 5)), completion("done", usage=counted(150, 100, 0, 3))),
            " input=250 cache_read=140 cache_write=10 output=8",
            id="two rounds with usage give the library's sum once",
        ),
        pytest.param(
            (tool_round(say(ANSWER)), completion("done")),
            "",
            id="rounds without usage give none of the four fields",
        ),
        pytest.param(
            (tool_round(say(ANSWER), usage=counted(100, 40, 10, 5)), completion("done")),
            "",
            id="a round without usage gives no partial sum",
        ),
    ],
)
@pytest.mark.parametrize("aloud", [False, True], ids=["written", "aloud"])
async def test_a_turn_that_ends_with_its_final_event_logs_the_tokens_it_used_or_none(
    runner, caplog, rounds, counts, aloud
):
    if aloud:
        await a_voice_call(runner, *rounds)
    else:
        runner.gateway = Gateway(*rounds)
        await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=aloud)
    await runner.reports.until_a_turn_completed()

    assert the_turn_ended_line(caplog).group("counts") == counts


async def the_sizes_of_a_turn(runner, caplog, *rounds) -> tuple[int, int, int]:
    runner.gateway = Gateway(*rounds)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    sizes = the_turn_ended_line(caplog).groups()[1:4]
    assert None not in sizes
    return tuple(int(each) for each in sizes)


async def test_a_turn_with_no_call_results_logs_its_sizes_and_zero_results(runner, caplog):
    tools, told, results = await the_sizes_of_a_turn(runner, caplog, completion("done"))

    assert tools > 0
    assert told > 0
    assert results == 0


@pytest.mark.parametrize("calls", [1, 2, 3])
async def test_a_turn_logs_the_sum_of_its_call_results(runner, caplog, calls):
    answer = "x" * 100
    runner.app.say_outcomes.extend(ToolOutcome(answer, is_error=False) for _ in range(calls))
    runner.app.content.add(answer)

    _, _, results = await the_sizes_of_a_turn(
        runner, caplog, tool_round(*(say(ANSWER) for _ in range(calls))), completion("done")
    )

    assert results % calls == 0
    assert results // calls >= len(answer)
    assert results // calls < 2 * len(answer)


async def test_a_turn_that_fails_after_its_request_logs_its_sizes_and_no_usage(runner, caplog):
    runner.gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)

    ended = the_turn_ended_line(caplog)
    assert ended.group(2) is not None
    assert ended.group("counts") == ""


async def test_a_turn_that_ends_before_any_request_logs_none_of_the_sizes(runner, caplog):
    async def lists_no_tools(reply):
        await reply.result(ListToolsResult(tools=[]))

    runner.app.script(LIST_TOOLS, lists_no_tools)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)

    ended = the_turn_ended_line(caplog)
    assert ended.group(2) is None
    assert ended.group("counts") == ""
