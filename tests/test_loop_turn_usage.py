"""The whole runner, a turn that ends with its final event: what the line of its end says of the tokens it used."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply, completion, tool_round
from agents.tools import ToolOutcome
from agents.turn import TOOL_IMAGE_MEDIA_TYPE

from personas.app_session import open_mcp_session
from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, say, words
from tests.support.history import REMARK, said_and_taken
from tests.support.hopin import AGENT_SUBJECT, LIMIT_SECONDS, a_picture, active, base64_of, formatted
from tests.support.voice import a_voice_call

pytestmark = pytest.mark.anyio


def counted(prompt: int, cached: int, written: int, completed: int) -> dict[str, object]:
    """A usage chunk as the gateway sends it."""
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completed,
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
    }


_SIZES = r" tools_bytes=(\d+) told_bytes=(\d+) results_bytes=(\d+) images_bytes=(\d+)"
_COUNTS = r"(?P<counts>(?: input=\d+ cache_read=\d+ cache_write=\d+ output=\d+ requests=\d+)?)"
_REQUESTED = re.compile(r"model_requested client_id=\S+ app=\S+ code=- duration_ms=(?P<duration_ms>\d+)")
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
            " input=250 cache_read=140 cache_write=10 output=8 requests=2",
            id="a turn with one tool round gives the library's sum once and two requests",
        ),
        pytest.param(
            (tool_round(say(ANSWER)), completion("done")),
            "",
            id="rounds without usage give none of the five fields",
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


async def the_sizes_of_a_turn(runner, caplog, *rounds) -> tuple[int, int, int, int]:
    runner.gateway = Gateway(*rounds)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    sizes = the_turn_ended_line(caplog).groups()[1:5]
    assert None not in sizes
    return tuple(int(each) for each in sizes)


async def test_a_turn_with_no_call_results_logs_its_sizes_and_zero_results(runner, caplog):
    tools, told, results, images = await the_sizes_of_a_turn(runner, caplog, completion("done"))

    assert tools > 0
    assert told > 0
    assert results == 0
    assert images == 0


@pytest.mark.parametrize("calls", [1, 2, 3])
async def test_a_turn_logs_the_sum_of_its_call_results(runner, caplog, calls):
    answer = "x" * 100
    runner.app.say_outcomes.extend(ToolOutcome(answer, is_error=False) for _ in range(calls))
    runner.app.content.add(answer)

    _, _, results, images = await the_sizes_of_a_turn(
        runner, caplog, tool_round(*(say(ANSWER) for _ in range(calls))), completion("done")
    )

    assert images == 0

    assert results % calls == 0
    assert results // calls >= len(answer)
    assert results // calls < 2 * len(answer)


@pytest.mark.parametrize("pictures", [1, 2], ids=["one picture", "two pictures"])
async def test_a_turn_logs_the_decoded_bytes_of_its_pictures_apart_from_its_results(runner, caplog, pictures):
    answer = "x" * 100
    shown = [a_picture(size) for size in (5000, 7000)[:pictures]]
    runner.app.say_outcomes.extend(ToolOutcome(answer, is_error=False, image=each) for each in shown)
    runner.app.content.add(answer)

    _, _, results, images = await the_sizes_of_a_turn(
        runner, caplog, tool_round(*(say(ANSWER) for _ in shown)), completion("done")
    )

    assert images == sum(len(each.content) for each in shown)
    assert results < pictures * 2 * len(answer)


async def test_an_aloud_turn_logs_its_picture_and_sends_it_to_the_model_like_a_written_one(runner, caplog):
    picture = a_picture()
    runner.app.say_outcomes.append(ToolOutcome("", is_error=False, image=picture))
    await a_voice_call(runner, tool_round(say(ANSWER)), completion("done"))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until_a_turn_completed()

    assert the_turn_ended_line(caplog).group(5) == str(len(picture.content))
    assert runner.gateway.pictures_in(2) == [f"data:{TOOL_IMAGE_MEDIA_TYPE};base64,{base64_of(picture)}"]


async def test_a_turn_counts_the_result_of_a_held_call_among_its_results(runner, caplog):
    first, held_text = words(2)
    runner.gateway = Gateway(tool_round(say(first)), tool_round(say(held_text)), completion("done"))
    second = runner.gateway.hold_request(2)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    second.set()
    await runner.reports.until_a_turn_completed()

    (held,) = runner.gateway.held_result_texts(3)
    _, _, results, _ = (int(each) for each in the_turn_ended_line(caplog).groups()[1:5])
    assert results >= len(held)


async def test_a_turn_that_fails_after_its_request_logs_its_sizes_and_no_usage(runner, caplog):
    runner.gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)

    ended = the_turn_ended_line(caplog)
    assert ended.group(2) is not None
    assert ended.group("counts") == ""


OPENING_SECONDS = 2.5
OPENING_MILLISECONDS = 2500


class SlowOpening:
    """Session openers whose opening takes the fake clock this long, the turn's own session, after ``arm``, waiting
    until ``release`` is set when ``held``."""

    def __init__(self, runner, *, held: bool = False) -> None:
        self._runner = runner
        self._armed = False
        self._held = held
        self.arrived = asyncio.Event()
        self.release = asyncio.Event()

    def arm(self) -> None:
        self._armed = True

    @asynccontextmanager
    async def __call__(self, app, token):
        if self._armed:
            self._runner.clock.time += OPENING_SECONDS
            self.arrived.set()
            if self._held:
                await self.release.wait()
        async with open_mcp_session(app, token) as session:
            yield session


async def a_turn_whose_opening_takes_time(runner, *rounds, held: bool = False) -> SlowOpening:
    """The runner in the call over a gateway of these rounds, its next session's opening, the turn's, slow."""
    runner.gateway = Gateway(*rounds)
    opening = SlowOpening(runner, held=held)
    runner.app.stand(AGENT_SUBJECT, active(PLACE_ID))
    runner.start(open_session=opening)
    await runner.app.until_window_read(PLACE_ID)
    opening.arm()
    say_in_the_call(runner, ADDRESSED)
    return opening


def requested_durations(caplog) -> list[int]:
    """The `duration_ms` of each `model_requested` line, which holds nothing but the fields the log declares."""
    requested = [
        line
        for line in formatted(caplog.get_records("call")).splitlines()
        if line.startswith(Event.MODEL_REQUESTED.value)
    ]
    matches = [_REQUESTED.fullmatch(line) for line in requested]
    assert all(matches), requested
    return [int(matched.group("duration_ms")) for matched in matches if matched is not None]


@pytest.mark.parametrize(
    "rounds",
    [
        pytest.param((completion("done"),), id="a turn of one request"),
        pytest.param((tool_round(say(ANSWER)), completion("done")), id="a turn with one tool round"),
    ],
)
async def test_a_turn_logs_one_model_requested_line_with_the_milliseconds_to_its_first_request(runner, caplog, rounds):
    await a_turn_whose_opening_takes_time(runner, *rounds)
    first = runner.gateway.hold_request(1)
    await runner.gateway.until_requested(1)
    runner.clock.time += 100
    first.set()
    await runner.reports.until_a_turn_completed()

    assert requested_durations(caplog) == [OPENING_MILLISECONDS]


async def test_a_turn_that_fails_after_its_request_logs_the_milliseconds_to_the_request(runner, caplog):
    await a_turn_whose_opening_takes_time(runner, Reply(status=HTTPStatus.SERVICE_UNAVAILABLE))
    await runner.reports.until(Event.TURN_ENDED)

    assert requested_durations(caplog) == [OPENING_MILLISECONDS]


async def test_a_turn_cancelled_after_its_request_logs_the_milliseconds_to_the_request(runner, caplog):
    await a_turn_whose_opening_takes_time(runner, completion("done"))
    runner.gateway.hold_request(1)
    await runner.gateway.until_requested(1)
    runner.app.stand(AGENT_SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)

    assert requested_durations(caplog) == [OPENING_MILLISECONDS]


async def test_a_turn_cancelled_before_its_request_logs_no_model_requested_line(runner, caplog):
    opening = await a_turn_whose_opening_takes_time(runner, completion("done"), held=True)
    async with asyncio.timeout(LIMIT_SECONDS):
        await opening.arrived.wait()
    runner.app.stand(AGENT_SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)

    assert requested_durations(caplog) == []
