"""The whole runner, a turn that holds the tools of the other apps where he holds a place: the call and the table
answer and act in one turn."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from dataclasses import replace
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.tools import ToolOutcome
from agents.turn import FailureCode, Principal, person_fingerprint
from mcp_types import ListToolsResult

from personas.decision import NETWORK_FAILURE_CODE, TURN_CANCELLED_CODE, status_code
from personas.reporting import Event
from tests.support.call import ADDRESSED, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, play, say, words
from tests.support.history import (
    REMARK,
    TABLE,
    act_entry,
    answered,
    at_table,
    said_and_taken,
    settled,
    shows,
    unanswered,
    url_of,
    waits,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import CALL, ISSUER, LIST_TOOLS, PLAY_TOOL, READ, SAY_TOOL, active
from tests.support.songmaker import CALLS, CREATE_SONG, GENERATE_SONG
from tests.support.voice import HeldSleeps, a_voice_call
from tests.support.wire import answers_with, cuts, hangs, hangs_once_entered, refuses_with_a_server_error

pytestmark = pytest.mark.anyio


def fingerprints(tokens) -> set[str]:
    """A token never enters an assertion: bearers are compared by the digest of each, the empty one left out."""
    return {hashlib.sha256(each.encode()).hexdigest()[:12] for each in set(tokens) - {""}}


def said_act(runner, text: str, outcome=None):
    return act_entry(url_of(runner, 0), SAY_TOOL, {"place": PLACE_ID, "text": text}, outcome or answered("said"))


def played_act(runner, text: str, outcome=None):
    return act_entry(url_of(runner, 1), PLAY_TOOL, {"place": TABLE.id, "text": text}, outcome or answered("said"))


async def lists_nothing(reply):
    await reply.result(ListToolsResult(tools=[]))


async def seated_in_the_call(runner, other, *replies):
    """The runner started with his table at the other app and his call at the first, both read once, none waiting."""
    runner.gateway = Gateway(*replies)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    await other.until_standing_read()
    await settled(other)


def bound_requests_to(runner, app_url: str, seconds: float) -> None:
    runner.config = replace(
        runner.config,
        apps=tuple(
            replace(each, call_seconds=seconds) if each.mcp_url == app_url else each for each in runner.config.apps
        ),
    )


async def a_call_turn_claimed_before_the_table_that_waits(runner, other, *replies):
    """A table turn runs; a line is said in the call and then the table's next revision waits; once it ends the claim
    takes the call line, with the table's view in the history. The first round of the table turn is released last."""
    runner.gateway = Gateway(completion("done"), *replies, completion("done"), completion("done"))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    waits(other, 1)
    await runner.gateway.until_requested(1)
    await shows_after_the_line(runner, other)
    running.set()


async def shows_after_the_line(runner, other):
    await said_and_taken(runner.app, PLACE_ID, ADDRESSED)
    await shows(other, 2, awaits_you=True)


async def test_a_call_turn_answers_in_the_call_and_plays_at_the_table_in_one_turn_each_call_under_its_apps_token(
    runner, other
):
    written, played = words(2)
    await a_call_turn_claimed_before_the_table_that_waits(runner, other, tool_round(say(written), play(played)))
    await runner.gateway.until_turns(3)

    assert runner.app.said == [(SUBJECT, PLACE_ID, written)]
    assert other.said == [(SUBJECT, TABLE.id, played)]
    assert len(runner.gateway.turns("lines")) == 1
    (_, following) = runner.gateway.turns("view")
    assert following.told("act") == [said_act(runner, written), played_act(runner, played)]
    scope_of_other = runner.config.apps[1].scope
    table_bearers = fingerprints(other.presented)
    issued_for_table = fingerprints(runner.realm.issued_for[scope_of_other])
    call_bearers = fingerprints(runner.app.presented)
    assert table_bearers <= issued_for_table
    assert len(table_bearers) >= 3
    assert table_bearers.isdisjoint(call_bearers)
    assert runner.reports.reported(apps=True, started=False) == []


async def test_the_table_turn_that_follows_a_call_turn_that_only_wrote_is_told_his_line(runner, other):
    written = words(1)[0]
    await a_call_turn_claimed_before_the_table_that_waits(runner, other, tool_round(say(written)))
    await runner.gateway.until_turns(3)

    (_, following) = runner.gateway.turns("view")
    assert following.told("act") == [said_act(runner, written)]
    assert other.said == []


async def test_the_move_his_call_turn_made_leaves_no_second_act_at_the_revision_he_played_at(runner, other):
    other.says_move_the_view = True
    played = words(1)[0]
    await a_call_turn_claimed_before_the_table_that_waits(runner, other, tool_round(play(played)), completion("done"))
    await other.said_count(1)
    await other.until_view_read(TABLE, 3)
    await settled(other)
    await settled(runner.app)

    assert len(runner.gateway.turns("view")) == 1
    assert len(runner.gateway.turns("lines")) == 1
    assert other.said == [(SUBJECT, TABLE.id, played)]


async def test_a_call_turn_declares_the_tools_of_the_apps_he_holds_a_place_at_in_configuration_order_and_no_other(
    runner, other, songmaker
):
    await seated_in_the_call(runner, other, completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL, PLAY_TOOL, CREATE_SONG, GENERATE_SONG]


async def test_a_call_turn_declares_no_tool_of_an_app_where_he_holds_no_place(runner, other):
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL]


async def test_a_call_turn_and_a_table_turn_that_hold_the_same_apps_declare_the_same_tools_in_the_same_order(
    runner, other, songmaker
):
    await seated_in_the_call(runner, other, completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    waits(other, 1)
    await runner.gateway.until_requested(2)

    assert runner.gateway.offered_tools(2) == runner.gateway.offered_tools(1)
    assert runner.gateway.offered_tools(2) == [SAY_TOOL, PLAY_TOOL, CREATE_SONG, GENERATE_SONG]
    assert runner.gateway.turns("view")


async def test_a_table_turn_holds_the_tools_of_his_call_and_writes_in_it(runner, other):
    written = words(1)[0]
    await seated_in_the_call(runner, other, tool_round(say(written)), completion("done"))
    waits(other, 1)
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, PLACE_ID, written)]
    call_bearers = fingerprints(runner.app.presented)
    issued_for_call = fingerprints(runner.realm.issued_for[runner.config.apps[0].scope])
    table_bearers = fingerprints(other.presented)
    assert call_bearers <= issued_for_call
    assert call_bearers
    assert table_bearers.isdisjoint(call_bearers)
    assert runner.gateway.offered_tools() == [SAY_TOOL, PLAY_TOOL]


@pytest.mark.parametrize(
    ("handler", "code", "call_seconds"),
    [
        (refuses_with_a_server_error, status_code(500), None),
        (cuts, NETWORK_FAILURE_CODE, None),
        (hangs, NETWORK_FAILURE_CODE, 0.3),
    ],
    ids=["a server error", "a cut connection", "a hang past the call time"],
)
async def test_a_failed_call_to_a_held_app_is_an_error_that_spends_its_bound_and_the_turn_goes_on(
    runner, other, handler, code, call_seconds
):
    written, played = words(2)
    if call_seconds is not None:
        bound_requests_to(runner, url_of(runner, 1), call_seconds)
    other.script(CALL, handler)
    await seated_in_the_call(
        runner, other, tool_round(say(written), play(played)), tool_round(play(played)), completion("done")
    )
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(3)

    first, second = (each for each in runner.gateway.tool_results(3) if each.startswith("Error"))
    assert first != second
    assert other.log.count(CALL) == 1
    assert runner.app.said == [(SUBJECT, PLACE_ID, written)]
    assert runner.reports.reported(apps=True, started=False) == [(Event.HELD_APP_FAILED, code, url_of(runner, 1))]


async def test_a_held_app_that_lists_none_of_its_bound_tools_holds_none_and_is_no_failure(runner, other, songmaker):
    other.script(LIST_TOOLS, lists_nothing)
    await seated_in_the_call(runner, other, completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL, CREATE_SONG, GENERATE_SONG]
    assert runner.reports.reported(apps=True, started=False) == []


async def test_a_held_app_whose_tools_cannot_be_listed_is_left_out_and_reported_once(runner, other):
    other.script(LIST_TOOLS, cuts)
    code = NETWORK_FAILURE_CODE
    written = words(1)[0]
    await seated_in_the_call(runner, other, tool_round(say(written)), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    assert runner.gateway.offered_tools() == [SAY_TOOL]
    assert runner.reports.reported(apps=True, started=False) == [(Event.HELD_APP_FAILED, code, url_of(runner, 1))]


async def test_a_failed_call_to_his_own_apps_tool_ends_the_turn_into_that_loops_reaction_and_reports_nothing_held(
    runner, other
):
    runner.app.script(CALL, cuts)
    await seated_in_the_call(runner, other, tool_round(say(words(1)[0]), play(words(1)[0])), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.RESUMED)

    assert runner.reports.reported(apps=True, started=False) == [
        (Event.BACKOFF, NETWORK_FAILURE_CODE, url_of(runner, 0)),
        (Event.RESUMED, None, url_of(runner, 0)),
    ]
    assert runner.gateway.arrivals == 1


def rounds_spending_every_bound(runner) -> list[object]:
    """One call a round until every app's bound is spent: the held app's bound plus one, the last of which its own
    bound refuses, then the place's and the tools-only app's. Their calls number the turn's bound, the sum of the
    apps' bounds plus one, so the call after them is past it."""
    held_bound = runner.config.apps[1].binding.max_tool_calls
    own_bound = runner.config.apps[0].binding.max_tool_calls
    return [
        *(tool_round(play(words(1)[0])) for _ in range(held_bound + 1)),
        *(tool_round(say(words(1)[0])) for _ in range(own_bound)),
        *(tool_round(tool_call(secrets.token_hex(4), CREATE_SONG, {"title": "t"})) for _ in range(CALLS)),
    ]


async def test_every_apps_bound_of_a_turn_is_spent_in_full_and_the_call_past_a_held_apps_bound_is_an_error(
    runner, other, songmaker
):
    held_bound = runner.config.apps[1].binding.max_tool_calls
    own_bound = runner.config.apps[0].binding.max_tool_calls
    rounds = rounds_spending_every_bound(runner)
    await seated_in_the_call(runner, other, *rounds, completion("done"))
    final_request = len(rounds) + 1
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(final_request)

    assert other.calls == [PLAY_TOOL] * held_bound
    assert len(runner.app.said) == own_bound
    assert len(songmaker.called) == CALLS
    assert runner.gateway.tool_results(final_request)[held_bound].startswith("Error")
    assert runner.reports.reported(apps=True, started=False) == []


async def test_the_call_past_the_turn_bound_of_the_three_apps_ends_the_turn_as_limit_exceeded_and_is_not_sent(
    runner, other, songmaker
):
    rounds = rounds_spending_every_bound(runner)
    await seated_in_the_call(runner, other, *rounds, tool_round(play(words(1)[0])), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.TURN_ENDED)

    assert runner.gateway.arrivals == len(rounds) + 1
    assert other.calls == [PLAY_TOOL] * runner.config.apps[1].binding.max_tool_calls
    assert len(runner.app.said) == runner.config.apps[0].binding.max_tool_calls
    assert len(songmaker.called) == CALLS
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TURN_ENDED, FailureCode.LIMIT_EXCEEDED.value, url_of(runner, 0))
    ]


async def test_a_turn_whose_place_leaves_while_a_held_apps_call_is_in_flight_ends_cancelled_and_the_act_is_told(
    runner, other
):
    entered = asyncio.Event()
    played = words(1)[0]

    other.script(CALL, hangs_once_entered(entered))
    await seated_in_the_call(runner, other, tool_round(play(played)), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await entered.wait()
    runner.app.stand(SUBJECT)
    await runner.reports.until(Event.TURN_ENDED)
    waits(other, 1)
    await runner.gateway.until_turns(2)

    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TURN_ENDED, TURN_CANCELLED_CODE, url_of(runner, 0))
    ]
    assert runner.gateway.arrivals == 2
    assert runner.gateway.turns("view")[0].told("act") == [played_act(runner, played, unanswered(TURN_CANCELLED_CODE))]


async def test_a_place_that_left_or_an_app_that_parks_lends_no_tools_until_it_is_read_again(runner, other):
    sleeps = HeldSleeps(runner)
    await seated_in_the_call(runner, other, completion("done"))
    other.stand(SUBJECT)
    await settled(other)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)

    other.script(READ, await answers_with(HTTPStatus.SERVICE_UNAVAILABLE))
    other.stand(SUBJECT, at_table())
    await sleeps.until_began(1)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(2)
    sleeps.release.set()
    await other.until_standing_read(4)
    await settled(other)
    say_in_the_call(runner, f"{ADDRESSED} Once more.")
    await runner.gateway.until_requested(3)

    assert [runner.gateway.offered_tools(number) for number in (1, 2, 3)] == [
        [SAY_TOOL],
        [SAY_TOOL],
        [SAY_TOOL, PLAY_TOOL],
    ]


async def test_a_place_that_becomes_active_after_a_turn_was_admitted_is_held_by_the_next_turn_only(runner, other):
    runner.gateway = Gateway(completion("done"))
    running = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    other.stand(SUBJECT)
    await other.until_standing_read()
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    other.stand(SUBJECT, at_table())
    await settled(other)
    running.set()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(2)

    assert [runner.gateway.offered_tools(number) for number in (1, 2)] == [[SAY_TOOL], [SAY_TOOL, PLAY_TOOL]]


async def test_each_agents_turn_holds_the_tools_of_the_apps_where_that_agent_holds_a_place(runner, other):
    second_subject = "agent-subject-2"
    runner.add_agent("agent-bach", second_subject, "Bach")
    runner.gateway = Gateway(completion("done"))
    runner.app.stand(SUBJECT, active(PLACE_ID))
    runner.app.stand(second_subject, active(PLACE_ID))
    other.stand(SUBJECT, at_table())
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await other.until_standing_read()
    await settled(other)
    await settled(runner.app, second_subject)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(2, why="lines")

    offered = {}
    for number in range(1, runner.gateway.arrivals + 1):
        offered.setdefault(runner.gateway.user_of(number), runner.gateway.offered_tools(number))
    assert offered == {
        person_fingerprint(Principal(iss=ISSUER, sub=SUBJECT)): [SAY_TOOL, PLAY_TOOL],
        person_fingerprint(Principal(iss=ISSUER, sub=second_subject)): [SAY_TOOL],
    }


async def test_a_call_of_a_table_turn_held_by_a_line_in_the_call_is_followed_by_the_jack_and_the_answer_in_the_call(
    runner, other
):
    card, jack, written = words(3)
    runner.gateway = Gateway(
        tool_round(play(card)), tool_round(play(jack), say(written)), completion("done"), completion("done")
    )
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    waits(other, 1)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await other.said_count(1)
    await runner.app.said_count(1)
    await runner.gateway.until_turns(1, why="lines")

    (held,) = runner.gateway.held_results(2)
    assert held["sent"] is False
    assert other.said == [(SUBJECT, TABLE.id, jack)]
    assert runner.app.said == [(SUBJECT, PLACE_ID, written)]
    table_bearers = fingerprints(other.presented)
    call_bearers = fingerprints(runner.app.presented)
    assert table_bearers.isdisjoint(call_bearers)


async def test_a_line_in_the_call_while_the_table_already_had_its_turn_lets_the_call_turn_play(runner, other):
    played = words(1)[0]
    await seated_in_the_call(runner, other, completion("done"), tool_round(play(played)), completion("done"))
    waits(other, 1)
    await runner.gateway.until_turns(1, why="view")
    say_in_the_call(runner, ADDRESSED)
    await other.said_count(1)

    assert other.said == [(SUBJECT, TABLE.id, played)]


async def test_a_held_app_that_refuses_after_the_turn_began_is_an_error_takes_no_more_calls_and_is_lacking_next_turn(
    runner, other
):
    sleeps = HeldSleeps(runner)
    played = words(1)[0]
    await seated_in_the_call(
        runner, other, tool_round(play(played)), tool_round(play(played)), completion("done"), completion("done")
    )
    running = runner.gateway.hold_request(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    other.refuse_with = HTTPStatus.FORBIDDEN
    other.stand(SUBJECT, at_table())
    await sleeps.until_began(1)
    running.set()
    await runner.gateway.until_requested(2)
    requests_after_the_refusal = len(other.log.entries)
    await runner.gateway.until_requested(3)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_requested(4)

    assert len(other.log.entries) == requests_after_the_refusal
    assert all(each.startswith("Error") for each in runner.gateway.tool_results(3))
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.SUSPENDED, status_code(HTTPStatus.FORBIDDEN), url_of(runner, 1)),
        (Event.HELD_APP_FAILED, status_code(HTTPStatus.FORBIDDEN), url_of(runner, 1)),
    ]
    assert runner.gateway.offered_tools(4) == [SAY_TOOL]


@pytest.mark.parametrize("answer_is_error", [False, True], ids=["an act the table allows", "a move the table refuses"])
async def test_a_call_to_the_table_while_it_is_not_his_move_is_the_tables_to_answer(runner, other, answer_is_error):
    answer = secrets.token_hex(4)
    other.say_outcomes.append(ToolOutcome(answer, is_error=answer_is_error))
    await seated_in_the_call(
        runner, other, tool_round(play(words(1)[0])), tool_round(play(words(1)[0])), completion("done")
    )
    say_in_the_call(runner, f"{ADDRESSED} steh auf")
    await runner.gateway.until_requested(3)

    first, second = runner.gateway.tool_results(3)
    assert (answer in first, first.startswith("Error")) == (True, answer_is_error)
    assert second == "said"
    assert other.calls == [PLAY_TOOL, PLAY_TOOL]
    assert runner.reports.reported(apps=True, started=False) == []


async def test_a_sentence_before_a_call_to_a_held_app_is_spoken_and_waited_for_before_the_call_goes_out(runner, other):
    sleeps = HeldSleeps(runner)
    spoken, played = "Okay, der Bube.", words(1)[0]
    other.stand(SUBJECT, at_table())
    await a_voice_call(runner, tool_round(play(played), content=spoken), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await sleeps.until_began(1)

    assert runner.app.spoken == [(SUBJECT, PLACE_ID, spoken)]
    assert other.calls == []
    sleeps.release.set()
    await other.said_count(1)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    told = runner.gateway.turns()[1].told("speech", "act")
    assert [each["entry"] for each in told[:2]] == ["speech", "act"]
    assert told[1] == played_act(runner, played)


async def test_a_table_turn_of_a_speaking_agent_is_not_spoken_and_writes_in_his_call(runner, other):
    written, unspoken = words(2)
    runner.give_voice()
    other.stand(SUBJECT, at_table())
    runner.gateway = Gateway(tool_round(say(written), content=unspoken), completion("done"))
    await restarted_in_the_call(runner)
    waits(other, 1)
    await runner.app.said_count(1)

    (turn,) = runner.gateway.turns()
    assert not turn.aloud
    assert runner.app.said == [(SUBJECT, PLACE_ID, written)]
    assert runner.app.spoken == []
    assert "speak" not in runner.gateway.offered_tools()


async def test_a_call_to_a_held_app_after_the_move_it_caused_was_read_holds_the_next_call_of_the_turn_once(
    runner, other
):
    other.says_move_the_view = True
    card, written = words(2)
    await seated_in_the_call(
        runner, other, tool_round(play(card)), tool_round(say(written)), tool_round(say(written)), completion("done")
    )
    await shows(other, 1)
    running = runner.gateway.hold_request(2)
    say_in_the_call(runner, ADDRESSED)
    await other.said_count(1)
    await other.until_view_read(TABLE, 3)
    await settled(other)
    running.set()
    await runner.app.said_count(1)
    await runner.gateway.until_requested(4)

    (held,) = runner.gateway.held_results(3)
    assert [each["entry"] for each in held["arrived"]["entries"] if each["entry"] == "view"] == ["view"]
    assert [text for _, _, text in runner.app.said] == [written]
    assert len(runner.gateway.held_results(4)) == 1
