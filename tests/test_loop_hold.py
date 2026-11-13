"""The whole runner, the hold: a call of a turn is held once when something not his own came while he thought."""

from __future__ import annotations

import secrets

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.turn import FailureCode

from personas.config import load_config
from personas.decision import NETWORK_FAILURE_CODE, TURN_CANCELLED_CODE
from personas.reporting import Event
from tests.support.call import (
    ADDRESSED,
    ANSWER,
    PLACE_ID,
    restarted_in_the_call,
    say_in_the_call,
)
from tests.support.gateway import Gateway, play, say, says_after_a_hold, words
from tests.support.history import (
    REMARK,
    SPEAKER,
    TABLE,
    at_table,
    entry,
    said_and_taken,
    settled,
    url_of,
    waits,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, DISCOVER
from tests.support.songmaker import CALLS, CREATE_SONG, GENERATE_SONG
from tests.support.voice import in_a_voice_call_and_at_a_table

pytestmark = pytest.mark.anyio

STARTED = (Event.STARTED, None)


async def addressed_in_the_call(runner):
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)


def song(name: str):
    return tool_call(secrets.token_hex(4), name, {"title": "t"})


async def at_the_table_with_news_to_come(runner, other, *replies, holds=(1,)):
    """His table turn at the other app waits for the model's first round; the call at the first app is where news is
    said. The holds are the requests that wait for a release."""
    runner.gateway = Gateway(*replies)
    released = [runner.gateway.hold_request(number) for number in holds]
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    state = waits(other, 1)
    await runner.gateway.until_requested(1)
    return state, released


async def test_a_turn_that_meets_nothing_new_has_every_call_relayed_in_as_many_rounds(runner, other):
    texts = words(3)
    runner.gateway = Gateway(*(tool_round(play(text)) for text in texts), completion("done"))
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    waits(other, 1)

    await other.said_count(3)
    await runner.gateway.until_requested(4)

    assert [text for _, _, text in other.said] == texts
    assert runner.gateway.arrivals == 4
    assert [runner.gateway.held_results(number) for number in (2, 3, 4)] == [[], [], []]


async def test_his_own_line_read_back_and_his_own_act_hold_no_call(runner):
    first, then = words(2)
    runner.gateway = Gateway(tool_round(say(first)), tool_round(say(then)), completion("done"))
    second = runner.gateway.hold_request(2)
    await addressed_in_the_call(runner)
    await runner.app.said_count(1)
    await settled(runner.app)
    second.set()
    await runner.app.said_count(2)
    await runner.gateway.until_requested(3)

    assert [text for _, _, text in runner.app.said] == [first, then]
    assert runner.gateway.held_results(3) == []


async def test_a_call_after_a_line_of_someone_else_is_held_with_every_arrival_told_new_and_the_next_call_goes_out(
    runner,
):
    first, held_text, last = words(3)
    runner.gateway = Gateway(
        tool_round(say(first)), tool_round(say(held_text)), tool_round(say(last)), completion("done")
    )
    second = runner.gateway.hold_request(2)
    await addressed_in_the_call(runner)
    await runner.app.said_count(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    second.set()
    await runner.app.said_count(2)
    await runner.gateway.until_requested(4)

    assert [text for _, _, text in runner.app.said] == [first, last]
    (held,) = runner.gateway.held_results(3)
    arrived = held["arrived"]["entries"]
    assert held["sent"] is False
    assert held["arrived"]["older_dropped"] is False
    assert sorted(each["entry"] for each in arrived) == ["act", "line", "line"]
    assert all(each["new"] for each in arrived)
    assert arrived[-1] == entry(url_of(runner, 0), PLACE_ID, REMARK, seq=3, name=SPEAKER, new=True)
    assert [each["own"] for each in arrived if each["entry"] == "line"] == [True, False]


async def test_news_before_each_of_three_calls_holds_only_the_first(runner, other):
    texts = words(3)
    _, (first, second, third) = await at_the_table_with_news_to_come(
        runner, other, *(tool_round(play(text)) for text in texts), completion("done"), holds=(1, 2, 3)
    )
    for number, released in enumerate((first, second, third), 1):
        await runner.gateway.until_requested(number)
        await said_and_taken(runner.app, PLACE_ID, f"{REMARK} {number}")
        released.set()
    await other.said_count(2)
    await runner.gateway.until_requested(4)

    assert [text for _, _, text in other.said] == texts[1:]
    assert [len(runner.gateway.held_results(number)) for number in (2, 3, 4)] == [1, 1, 1]


async def test_two_calls_of_one_reply_after_news_hold_the_first_and_send_the_second(runner, other):
    held_text, sent_text = words(2)
    _, (running,) = await at_the_table_with_news_to_come(
        runner, other, tool_round(play(held_text), play(sent_text)), completion("done")
    )
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await other.said_count(1)
    await runner.gateway.until_requested(2)

    assert [text for _, _, text in other.said] == [sent_text]
    (held,) = runner.gateway.held_results(2)
    assert held["arrived"]["entries"] == [entry(url_of(runner, 0), PLACE_ID, REMARK, name=SPEAKER, new=True)]
    assert runner.gateway.tool_results(2)[1] == "said"


async def test_a_held_call_to_a_tools_only_app_opens_no_session_and_takes_nothing_from_its_bound(runner, songmaker):
    runner.gateway = Gateway(
        tool_round(song(CREATE_SONG)),
        tool_round(song(CREATE_SONG)),
        tool_round(song(GENERATE_SONG)),
        completion("done"),
    )
    running = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    sessions_before = songmaker.log.count(DISCOVER)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await runner.gateway.until_requested(4)

    assert [name for name, _ in songmaker.called] == [CREATE_SONG, GENERATE_SONG]
    assert songmaker.log.count(DISCOVER) == sessions_before + CALLS
    assert songmaker.log.count(CALL) == CALLS
    assert len(runner.gateway.held_results(2)) == 1
    assert not any(each.startswith("Error") for each in runner.gateway.tool_results(4))
    assert runner.reports.reported() == [STARTED]


async def test_a_held_call_takes_nothing_from_its_apps_bound_so_the_whole_bound_goes_out_after_it(runner):
    bound = runner.config.apps[0].binding.max_tool_calls
    runner.gateway = Gateway(*(tool_round(say(ANSWER)) for _ in range(bound + 1)), completion("done"))
    running = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await runner.gateway.until_requested(bound + 2)

    assert len(runner.app.said) == bound
    assert not any(each.startswith("Error") for each in runner.gateway.tool_results(bound + 2))
    assert runner.reports.reported() == [STARTED]


async def test_a_turn_that_spends_every_apps_bound_after_a_hold_ends_normally(runner, songmaker):
    bound = runner.config.apps[0].binding.max_tool_calls
    runner.gateway = Gateway(
        tool_round(say(ANSWER)),
        *(tool_round(say(ANSWER)) for _ in range(bound)),
        *(tool_round(song(CREATE_SONG)) for _ in range(CALLS)),
        completion("done"),
    )
    running = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await runner.gateway.until_requested(bound + CALLS + 2)

    assert len(runner.app.said) == bound
    assert len(songmaker.called) == CALLS
    assert runner.reports.reported() == [STARTED]


async def test_a_failure_of_the_call_that_follows_a_hold_ends_the_turn_with_one_reaction_and_one_act(runner):
    runner.gateway = Gateway(*says_after_a_hold(PLACE_ID, ANSWER), completion("again"))
    running = runner.gateway.hold_request(1)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)
    runner.app.script(CALL, _cuts)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await runner.reports.until(Event.RESUMED)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2, why="lines")

    assert runner.reports.reported() == [STARTED, (Event.BACKOFF, NETWORK_FAILURE_CODE), (Event.RESUMED, None)]
    assert runner.gateway.round_sizes() == [2, 4, 2]
    acts = [each for each in runner.gateway.turns()[1].room["history"]["entries"] if each["entry"] == "act"]
    assert [each["outcome"] for each in acts] == [{"answered": False, "code": NETWORK_FAILURE_CODE}]


async def _cuts(reply):
    await reply.open_stream()
    reply.cut()


async def test_a_place_that_leaves_while_the_round_after_a_hold_is_pending_cancels_the_turn_and_sends_nothing(runner):
    runner.gateway = Gateway(*says_after_a_hold(PLACE_ID, ANSWER))
    first = runner.gateway.hold_request(1)
    runner.gateway.hold_request(2)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    first.set()
    await runner.gateway.until_requested(2)

    runner.app.stand(SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)

    assert runner.reports.reported() == [STARTED, (Event.TURN_ENDED, TURN_CANCELLED_CODE)]
    assert runner.app.said == []
    assert runner.app.log.count(CALL) == 0


async def test_a_round_after_a_hold_that_passes_the_deadline_ends_the_turn_as_limit_exceeded_and_sends_nothing(runner):
    runner.files.persona_path.write_text(runner.files.persona_text(max_duration_seconds=1), encoding="utf-8")
    runner.config = _reloaded(runner)
    runner.gateway = Gateway(*says_after_a_hold(PLACE_ID, ANSWER))
    first = runner.gateway.hold_request(1)
    runner.gateway.hold_request(2)
    await addressed_in_the_call(runner)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    first.set()
    await runner.reports.until(Event.TURN_ENDED)

    assert runner.reports.reported() == [STARTED, (Event.TURN_ENDED, FailureCode.LIMIT_EXCEEDED.value)]
    assert runner.app.said == []
    assert runner.app.log.count(CALL) == 0


def _reloaded(runner):
    return load_config(runner.files.write())


async def test_the_table_state_his_own_card_caused_holds_his_next_call_once_and_the_call_after_it_goes_out(
    runner, other
):
    other.says_move_the_view = True
    card, remark = words(2)
    _, (first, second) = await at_the_table_with_news_to_come(
        runner,
        other,
        tool_round(play(card)),
        tool_round(play(remark)),
        tool_round(play(remark)),
        completion("done"),
        holds=(1, 2),
    )
    first.set()
    await runner.gateway.until_requested(2)
    await other.until_view_read(TABLE, 2)
    await settled(other)
    second.set()
    await other.said_count(2)
    await runner.gateway.until_requested(4)

    assert [text for _, _, text in other.said] == [card, remark]
    (held,) = runner.gateway.held_results(3)
    kinds = sorted(each["entry"] for each in held["arrived"]["entries"])
    assert kinds == ["act", "line", "view"]
    assert all(each["new"] for each in held["arrived"]["entries"])
    assert len(runner.gateway.held_results(4)) == 1


async def test_a_line_heard_at_a_voice_place_holds_the_call_of_his_table_turn(runner):
    call = words(1)[0]
    heard = "Anyone there?"
    await in_a_voice_call_and_at_a_table(
        runner, tool_round(say(call, TABLE.id)), tool_round(say(call, TABLE.id)), completion("done")
    )
    running = runner.gateway.hold_request(1)
    waits(runner.app, 1)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, heard)
    running.set()
    await runner.app.said_count(1)
    await runner.gateway.until_requested(3)

    (held,) = runner.gateway.held_results(3)
    assert [each["text"] for each in held["arrived"]["entries"] if each["entry"] == "line" and not each["own"]] == [
        heard
    ]
    assert [text for _, _, text in runner.app.said] == [call]
