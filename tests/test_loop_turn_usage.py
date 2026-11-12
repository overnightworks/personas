"""The whole runner, a turn that ends with its final event: what the line of its end says of the tokens it used."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import completion, tool_round

from tests.support.call import ADDRESSED, ANSWER, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, say
from tests.support.hopin import formatted
from tests.support.voice import a_voice_call

pytestmark = pytest.mark.anyio


def counted(prompt: int, cached: int, written: int, completed: int) -> dict[str, object]:
    """A usage chunk as the gateway sends it."""
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completed,
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
    }


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

    ended = [line for line in formatted(caplog.get_records("call")).splitlines() if line.startswith("turn_ended")]
    assert len(ended) == 1
    assert ended[0].endswith(f" code=-{counts}")
