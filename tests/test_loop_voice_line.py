"""The whole runner, his short line before he goes on: ``continue_speaking`` with a ``line`` says the line at once, cuts
the sentence he was held in and sends his unsent sentences after it."""

from __future__ import annotations

import pytest
from agents.backend.fake_gateway import Reply, completion
from agents.member import STOP_SPEAKING_TOOL, SpeakRefusal, VoiceState

from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, say_in_the_call
from tests.support.voice import (
    FIRST,
    HOLD_ID,
    REFUSALS,
    SAID,
    SECOND,
    SECOND_HOLD,
    SECOND_HOLD_DECISIONS,
    THIRD,
    THREE_SENTENCES,
    a_voice_call,
    calling_while_held,
    continuing,
    continuing_hold,
    decided,
    heard_after_the_second_hold,
    held_in_the_first_sentence,
    play_out,
    refused_ends_the_hold_as_the_app_says,
    sentences_spoken,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

LINE = "Right, one moment."


STYLE = "louder and angrier"

ARRANGEMENTS = pytest.mark.parametrize("told", [False, True], ids=["round", "told"])


async def held_deciding(
    runner, told: bool, arguments: dict[str, object], then: str = f"{FIRST} {SECOND}", *after: Reply
):
    """A hold decided with ``continue_speaking``: in the round beside his parked run, a turn whose first sentence plays,
    whose second waits at the app and whose third is unsent, held by the app; or a turn told the hold at its start,
    which says ``then`` once the call is answered."""
    if told:
        await a_voice_call(runner, continuing(arguments), completion(then), *after)
        runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    else:
        await held_in_the_first_sentence(
            runner, completion(THREE_SENTENCES), continuing(arguments), completion("[silent]"), *after
        )


async def held_with_a_line(runner, line: object = LINE):
    await held_deciding(runner, False, {"line": line})


async def test_a_short_line_is_said_at_once_and_his_unsent_sentences_follow_plain(runner):
    """TURNS-30, TURNS-46, LOG-17."""
    await held_with_a_line(runner)

    assert await decided(runner) == ["short_line"]
    runner.app.end_sentence()
    await runner.app.spoken_count(3)
    runner.app.end_sentence()
    await runner.app.spoken_count(4)

    assert runner.app.speaks_naming_holds == [
        (FIRST, None),
        (SECOND, None),
        (LINE, HOLD_ID),
        (SECOND, None),
        (THIRD, None),
    ]
    assert sentences_spoken(runner) == [FIRST, LINE, SECOND, THIRD]
    assert runner.app.hold_calls == []
    assert Event.SPEAK_REFUSED not in [event for event, _ in runner.reports.reported()]


async def test_a_styled_short_line_reaches_speak_with_its_style_and_his_unsent_sentences_follow_plain(runner):
    """TURNS-46."""
    await held_deciding(runner, False, {"line": LINE, "style": STYLE})

    assert await decided(runner) == ["short_line"]
    runner.app.end_sentence()
    await runner.app.spoken_count(3)
    runner.app.end_sentence()
    await runner.app.spoken_count(4)

    assert runner.app.speaks_naming_holds[2] == (LINE, HOLD_ID)
    assert runner.app.speak_styles == [None, None, STYLE, None, None]


@ARRANGEMENTS
@REFUSALS
async def test_a_short_line_the_app_refused_sends_no_further_sentence_and_ends_the_hold_as_the_app_says(
    runner, told, refusal, outcome, stops
):
    """TURNS-30, TURNS-46, LOG-17."""
    await refused_ends_the_hold_as_the_app_says(
        runner,
        lambda: held_deciding(runner, told, {"line": LINE}),
        refusal,
        outcome,
        stops,
        unsent=FIRST if told else THIRD,
        taken_before=0 if told else 2,
        cut=None if told else SECOND,
    )


@ARRANGEMENTS
async def test_a_style_the_app_refused_sends_no_further_sentence_stops_him_and_ends_the_hold_undecided(runner, told):
    """TURNS-46, LOG-17."""
    await refused_ends_the_hold_as_the_app_says(
        runner,
        lambda: held_deciding(runner, told, {"line": LINE, "style": STYLE}),
        SpeakRefusal.STYLE_UNAVAILABLE,
        "undecided",
        [(STOP_SPEAKING_TOOL, HOLD_ID)],
        unsent=FIRST if told else THIRD,
        taken_before=0 if told else 2,
        cut=None if told else SECOND,
    )


@ARRANGEMENTS
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"line": 7}, id="line is no string"),
        pytest.param({"line": ""}, id="line is empty"),
        pytest.param({"line": LINE, "style": ""}, id="style is empty"),
        pytest.param({"line": LINE, "style": 7}, id="style is no string"),
        pytest.param({"style": STYLE}, id="style without a line"),
    ],
)
async def test_a_line_or_style_that_is_no_non_empty_string_decides_nothing(runner, told, arguments):
    """TURNS-46."""
    await held_deciding(runner, told, arguments, then="[silent]")

    assert await decided(runner) == ["undecided"]
    assert all(hold is None for _, hold in runner.app.speaks_naming_holds)


async def test_a_turn_told_a_hold_says_the_short_line_and_its_sentences_follow_it(runner):
    """TURNS-46, LOG-17."""
    await a_voice_call(runner, continuing({"line": LINE}), completion(f"{FIRST} {SECOND}"))

    runner.app.hold(PLACE_ID, HOLD_ID, SAID)

    assert await decided(runner) == ["short_line"]
    await runner.app.spoken_count(3)

    assert runner.app.speaks_naming_holds == [(LINE, HOLD_ID), (FIRST, None), (SECOND, None)]
    assert runner.app.hold_calls == []


async def test_a_turn_told_a_hold_says_the_styled_line_and_its_sentences_follow_plain(runner):
    """TURNS-46."""
    await held_deciding(runner, True, {"line": LINE, "style": STYLE})

    assert await decided(runner) == ["short_line"]
    await runner.app.spoken_count(3)

    assert runner.app.speaks_naming_holds[0] == (LINE, HOLD_ID)
    assert runner.app.speak_styles == [STYLE, None, None]


async def test_the_short_line_beside_his_parked_run_leaves_the_run_to_go_on(runner):
    """TURNS-30, LOG-17."""
    deciding_round = await calling_while_held(
        runner, continuing({"line": LINE}), completion("[silent]"), hold_round=True
    )
    await runner.gateway.until_requested(3)
    deciding_round.set()

    assert await decided(runner) == ["short_line"]
    await runner.reports.until_a_turn_completed()

    assert [text for text, hold in runner.app.speaks_naming_holds if hold] == [LINE]
    assert [text for *_, text in runner.app.said] == ["before", "waits"]


async def test_a_new_hold_voiding_the_one_his_short_line_names_decides_nothing_for_it_and_starts_its_own_round(runner):
    """TURNS-46."""
    deciding_round = await calling_while_held(
        runner,
        continuing({"line": LINE}),
        completion("[silent]"),
        continuing_hold("hold-2"),
        completion("[silent]"),
        hold_round=True,
    )
    await runner.gateway.until_requested(3)
    runner.app.hold(PLACE_ID, "hold-2", ("Bob", "no, stop"))
    deciding_round.set()

    assert await decided(runner) == ["continued"]
    assert runner.gateway.hold_told(5)["id"] == "hold-2"


async def test_a_line_in_a_turn_that_does_not_speak_is_refused_and_the_hold_stays_undecided(runner):
    """TURNS-46."""
    runner.app.state_voice(PLACE_ID, VoiceState.UNAVAILABLE)
    await a_voice_call(runner, completion("done"), continuing({"line": LINE}), completion("[silent]"))
    answer = runner.gateway.hold_request(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.app.until_window_read(PLACE_ID, runner.app.window_reads(PLACE_ID) + 1)
    answer.set()

    assert await decided(runner) == ["undecided"]
    assert runner.gateway.tool_results(3)[0].startswith("Error")
    assert runner.app.speaks_naming_holds == []


@SECOND_HOLD_DECISIONS
@ARRANGEMENTS
async def test_a_second_hold_after_the_short_line_is_decided_as_the_first(runner, told, reply, outcome):
    """TURNS-47."""
    await held_deciding(runner, told, {"line": LINE}, THREE_SENTENCES, reply, completion("[silent]"))
    assert await decided(runner) == ["short_line"]
    await runner.app.spoken_count(1 if told else 2)

    runner.app.hold(PLACE_ID, SECOND_HOLD, ("Bob", "no, wait"))
    assert await decided(runner, 2) == ["short_line", outcome]
    before_the_line = [] if told else [FIRST]
    after_the_line = [FIRST, SECOND, THIRD] if told else [SECOND, THIRD]
    heard = heard_after_the_second_hold(outcome, [*before_the_line, LINE], after_the_line)
    await play_out(runner, len(heard))
    await runner.reports.until_a_turn_completed()

    assert sentences_spoken(runner) == heard
