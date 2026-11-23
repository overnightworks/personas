"""The whole runner, a turn that ends with its final event: what the line of its end says of the tokens it used."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import Reply, completion, tool_round
from agents.tools import ToolOutcome
from agents.turn import TOOL_IMAGE_MEDIA_TYPE

from tests.support.app_driving import said_and_taken
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, say, words
from tests.support.history import REMARK
from tests.support.logs import lines_of
from tests.support.pictures import a_picture, base64_of
from tests.support.voice import a_voice_call

pytestmark = pytest.mark.anyio


def counted(prompt: int, cached: int, written: int, completed: int) -> dict[str, object]:
    """A usage chunk as the gateway sends it."""
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completed,
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
    }


_SIZE_FIELDS = ("tools_bytes", "told_bytes", "results_bytes", "images_bytes")
_COUNT_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens", "model_request_count")


def the_turn_ended_line(caplog) -> dict[str, object]:
    """The one `turn.ended` line."""
    (ended,) = lines_of(caplog.get_records("call"), "turn.ended")
    return ended


@pytest.mark.parametrize(
    ("rounds", "counts"),
    [
        pytest.param(
            (tool_round(say(ANSWER), usage=counted(100, 40, 10, 5)), completion("done", usage=counted(150, 100, 0, 3))),
            dict(zip(_COUNT_FIELDS, (250, 140, 10, 8, 2), strict=True)),
            id="a turn with one tool round gives the library's sum once and two requests",
        ),
        pytest.param(
            (tool_round(say(ANSWER)), completion("done")),
            {},
            id="rounds without usage give none of the five fields",
        ),
        pytest.param(
            (tool_round(say(ANSWER), usage=counted(100, 40, 10, 5)), completion("done")),
            {},
            id="a round without usage gives no partial sum",
        ),
    ],
)
@pytest.mark.parametrize("aloud", [False, True], ids=["written", "aloud"])
async def test_a_turn_that_ends_with_its_final_event_logs_the_tokens_it_used_or_none(
    runner, caplog, rounds, counts, aloud
):
    """LOG-31."""
    if aloud:
        await a_voice_call(runner, *rounds)
    else:
        runner.gateway = Gateway(*rounds)
        await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED, spoken=aloud)
    await runner.reports.until_a_turn_completed()

    ended = the_turn_ended_line(caplog)
    assert {key: ended[key] for key in _COUNT_FIELDS if key in ended} == counts


async def the_sizes_of_a_turn(runner, caplog, *rounds) -> tuple[int, int, int, int]:
    runner.gateway = Gateway(*rounds)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_completed()
    ended = the_turn_ended_line(caplog)
    return tuple(ended[field] for field in _SIZE_FIELDS)


async def test_a_turn_with_no_call_results_logs_its_sizes_and_zero_results(runner, caplog):
    """LOG-30."""
    tools, told, results, images = await the_sizes_of_a_turn(runner, caplog, completion("done"))

    assert tools > 0
    assert told > 0
    assert results == 0
    assert images == 0


@pytest.mark.parametrize("calls", [1, 2, 3])
async def test_a_turn_logs_the_sum_of_its_call_results(runner, caplog, calls):
    """LOG-30."""
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
    """LOG-30."""
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

    assert the_turn_ended_line(caplog)["images_bytes"] == len(picture.content)
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
    results = the_turn_ended_line(caplog)["results_bytes"]
    assert results >= len(held)


async def test_a_turn_that_fails_after_its_request_logs_its_sizes_and_no_usage(runner, caplog):
    """LOG-30, LOG-31."""
    runner.gateway = Gateway(Reply(status=HTTPStatus.SERVICE_UNAVAILABLE))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until_a_turn_ended()

    ended = the_turn_ended_line(caplog)
    assert set(_SIZE_FIELDS) <= ended.keys()
    assert not set(_COUNT_FIELDS) & ended.keys()
