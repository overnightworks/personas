"""The whole runner, the skills a turn lists and reads: its front's skill lines, ``read_skill`` and the round that
tells a read in the turns after it."""

from __future__ import annotations

import json
import secrets
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion, tool_round
from agents.member_outcomes import INVALID_ANSWER_CODE, status_code
from agents.skills import SkillRefusal, implicit_skill_name
from agents.turn import MAX_EVENT_BYTES
from agents.voice.reply import REPLY_RULES

from personas.decision import READ_SKILL_CALLS, READ_SKILL_TOOL
from personas.entries import ActDone, SkillRead
from personas.reporting import Event
from tests.support.call import ADDRESSED, a_turn, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, reading
from tests.support.hopin import CALL, LIST_TOOLS, READ, SKILLS_GET, SKILLS_LIST
from tests.support.logs import formatted, lines_of
from tests.support.realm import LIFETIME_SECONDS
from tests.support.skills import (
    HOPIN,
    HOPIN_DESCRIPTION,
    SONG_BODY,
    SONG_DESCRIPTION,
    SONGMAKER,
    Shelf,
    a_skill_file,
    an_entry,
    skilled,
    turns_past_a_cut_of_a_round,
    turns_telling_a_round_across_a_restart,
)
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG, OWN_DESCRIPTION, a_song, lists_only, songmaker_url
from tests.support.voice import a_voice_call
from tests.support.wire import hangs, refuses_with_a_server_error

pytestmark = pytest.mark.anyio

ERROR_MARK = "Error: "
"""How the gateway shows the model an error result."""
LARGE_BODY_BYTES = 5000


def skill_reads(runner) -> list[SkillRead]:
    return [entry for entry in runner.histories.entries() if isinstance(entry, SkillRead)]


async def test_a_turn_tells_the_skill_lines_of_the_apps_it_lists_after_the_reply_rules(runner, songmaker):
    skilled(runner, songmaker)
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await a_turn(runner, 2)

    first, second = (runner.gateway.body_of(number) for number in (1, 2))
    _, _, lines = first["messages"][0]["content"].partition(REPLY_RULES)
    assert lines == f"\n\n{HOPIN}: {HOPIN_DESCRIPTION}\n{SONGMAKER}: {SONG_DESCRIPTION}"
    assert runner.gateway.offered_tools(1, reading=True).count(READ_SKILL_TOOL) == 1
    assert second["messages"][: len(first["messages"])] == first["messages"]


async def test_read_skill_answers_a_listed_skills_body_and_the_next_turn_tells_it_as_a_round(runner, songmaker):
    skilled(runner, songmaker)
    runner.gateway = Gateway(
        tool_round(reading(SONGMAKER)), tool_round(a_song("model-song")), completion("done"), completion("again")
    )
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(3)
    await a_turn(runner, 4)
    await a_turn(runner, 5)

    (shown,) = runner.gateway.tool_results(2)
    assert SONG_BODY in shown
    assert [name for name, _ in songmaker.called] == [CREATE_SONG]
    ((assistant, result),) = runner.gateway.tool_rounds(4)
    (call,) = assistant["tool_calls"]
    assert (call["function"]["name"], json.loads(call["function"]["arguments"])) == (
        READ_SKILL_TOOL,
        {"skill": SONGMAKER},
    )
    assert (result["tool_call_id"], result["content"]) == (call["id"], shown)
    assert call["id"] != "model-read"
    assert runner.gateway.tool_rounds(5) == [[assistant, result]]


async def the_turn_ended_lines_around_a_read_of(runner, songmaker, caplog, body: str) -> list[dict[str, object]]:
    """The `turn.ended` lines of a turn that reads songmaker's skill, whose body is ``body``, and of the next turn."""
    skilled(runner, songmaker)
    songmaker.shelf = Shelf({SONGMAKER: a_skill_file(SONGMAKER, tools=f"{CREATE_SONG} {GENERATE_SONG}", body=body)})
    runner.gateway = Gateway(tool_round(reading(SONGMAKER)), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()
    await a_turn(runner, 3)
    await runner.reports.until_a_turn_completed(count=2)
    return lines_of(caplog.get_records("call"), "turn.ended")


async def test_a_read_counts_its_answer_in_the_results_bytes_of_its_turn(runner, songmaker, caplog):
    first, _ = await the_turn_ended_lines_around_a_read_of(runner, songmaker, caplog, "y" * LARGE_BODY_BYTES)

    assert first["results_bytes"] >= LARGE_BODY_BYTES


async def test_the_next_turn_counts_a_read_skills_round_in_its_told_bytes(runner, songmaker, caplog):
    first, second = await the_turn_ended_lines_around_a_read_of(runner, songmaker, caplog, "y" * LARGE_BODY_BYTES)

    assert second["told_bytes"] - first["told_bytes"] >= LARGE_BODY_BYTES


async def test_a_read_skills_round_is_byte_equal_after_a_restart(runner, songmaker):
    await turns_telling_a_round_across_a_restart(runner, songmaker, reading(SONGMAKER))

    assert len(runner.gateway.tool_rounds(3)) == 1
    assert runner.gateway.tool_rounds(4) == runner.gateway.tool_rounds(3)


async def test_a_read_skills_round_is_gone_once_a_cut_drops_its_entry_and_is_not_written_again(runner, songmaker):
    await turns_past_a_cut_of_a_round(runner, songmaker, reading(SONGMAKER), len(SONG_BODY.encode()))

    assert len(runner.gateway.tool_rounds(3)) == 1
    assert (runner.gateway.tool_rounds(5), runner.gateway.tool_rounds(6)) == ([], [])
    assert len(skill_reads(runner)) == 1


async def test_a_read_skill_beside_a_relayed_call_enters_both_in_the_order_they_ended_and_its_round_holds_only_the_read(
    runner, songmaker
):
    skilled(runner, songmaker)
    spoken = f"Let me look {secrets.token_hex(4)}."
    runner.gateway = Gateway(
        tool_round(a_song("model-song"), reading(SONGMAKER), content=spoken), completion("done"), completion("again")
    )
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    await a_turn(runner, 3)

    assert [type(entry) for entry in runner.histories.entries() if isinstance(entry, ActDone | SkillRead)] == [
        ActDone,
        SkillRead,
    ]
    ((assistant, _),) = runner.gateway.tool_rounds(3)
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == [READ_SKILL_TOOL]
    assert assistant["content"] is None
    told = runner.gateway.told_in(3)
    assert spoken not in told
    assert '"entry": "act"' in told


async def test_read_skill_for_a_name_no_line_holds_answers_an_error_and_enters_nothing(runner, songmaker):
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(reading("nobody")), completion("done"), completion("again"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)
    await a_turn(runner, 3)

    (shown,) = runner.gateway.tool_results(2)
    assert shown.startswith(ERROR_MARK)
    assert skill_reads(runner) == []
    assert runner.gateway.tool_rounds(3) == []
    assert SKILLS_GET not in songmaker.log.entries + runner.app.log.entries


async def test_a_refused_skill_answers_an_error_logs_its_code_and_enters_nothing(runner, songmaker, caplog):
    """LOG-40."""
    skilled(runner, songmaker)
    songmaker.shelf.tampered.add(SONGMAKER)
    runner.gateway = Gateway(tool_round(reading(SONGMAKER)), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)
    await runner.reports.until(Event.SKILL_REFUSED)

    (shown,) = runner.gateway.tool_results(2)
    assert shown.startswith(ERROR_MARK)
    assert SONG_BODY not in shown
    assert runner.reports.codes_of(Event.SKILL_REFUSED) == [SkillRefusal.SIZE_MISMATCH.value]
    (line,) = lines_of(caplog.get_records("call"), "skill.refused")
    assert line["reason"] == SkillRefusal.SIZE_MISMATCH.value
    assert skill_reads(runner) == []


async def test_read_skill_calls_past_their_bound_answer_an_error_and_a_later_call_still_runs(runner, songmaker):
    skilled(runner, songmaker)
    reads = [reading(SONGMAKER, f"model-read-{number}") for number in range(READ_SKILL_CALLS + 1)]
    runner.gateway = Gateway(tool_round(*reads), tool_round(a_song("model-song")), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(3)

    results = runner.gateway.tool_results(2)
    assert [each.startswith(ERROR_MARK) for each in results] == [False] * READ_SKILL_CALLS + [True]
    assert len(skill_reads(runner)) == READ_SKILL_CALLS
    assert [name for name, _ in songmaker.called] == [CREATE_SONG]


async def test_a_malformed_skill_listing_leaves_its_apps_lines_out_and_logs_the_entry_index(runner, songmaker, caplog):
    skilled(runner, songmaker)
    marker = secrets.token_hex(8)
    good = an_entry("good", a_skill_file("good"))
    broken = good | {"uri": f"skill://{marker}/not-a-skill-file"}

    async def lists_a_broken_entry(reply) -> None:
        await reply.raw_result({"skills": [good, good, broken]})

    songmaker.script(SKILLS_LIST, lists_a_broken_entry)
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)

    system = runner.gateway.system_of(1)
    assert SONG_DESCRIPTION not in system
    assert HOPIN_DESCRIPTION in system
    assert {CREATE_SONG, GENERATE_SONG} <= set(runner.gateway.offered_tools(1))
    (line,) = [each for each in lines_of(caplog.get_records("call"), "skill_list.ended") if each["outcome"] == "failed"]
    assert (line["reason"], line["entry"]) == (INVALID_ANSWER_CODE, 2)
    assert marker not in formatted(caplog.get_records("call"))


async def test_a_skills_listing_that_times_out_leaves_its_apps_lines_out_and_its_tools_held(runner, songmaker):
    skilled(runner, songmaker)
    songmaker.script(SKILLS_LIST, hangs)
    runner.bound_tools_only_calls_to(0.5)
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)

    assert SONG_DESCRIPTION not in runner.gateway.system_of(1)
    assert {CREATE_SONG, GENERATE_SONG} <= set(runner.gateway.offered_tools(1))


async def test_an_app_without_the_skills_extension_is_sent_no_request_for_its_skills(runner, songmaker, caplog):
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)

    sent = [*songmaker.log.entries, *runner.app.log.entries]
    assert SKILLS_LIST not in sent
    assert SKILLS_GET not in sent
    assert songmaker.log.count(LIST_TOOLS) == 1
    assert lines_of(caplog.get_records("call"), "skill_list.ended") == []


async def test_the_own_apps_declared_skills_are_listed_on_the_loops_session_with_its_tools(runner, caplog):
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN, description=HOPIN_DESCRIPTION)})
    await restarted_in_the_call(runner)
    listings = runner.app.log.count(SKILLS_LIST)

    await a_turn(runner, 1)
    await a_turn(runner, 2)

    assert runner.app.log.count(SKILLS_LIST) == listings + 2
    assert HOPIN_DESCRIPTION in runner.gateway.system_of(2)
    assert lines_of(caplog.get_records("call"), "session_open.ended") == []


async def test_a_refused_renewal_of_the_loops_token_does_not_hold_back_an_app_whose_skills_are_listed(
    runner, songmaker
):
    skilled(runner, songmaker)
    loops_scope = runner.config.apps[0].scope

    async def lets_the_loops_token_expire_and_be_refused(reply) -> None:
        runner.clock.time += LIFETIME_SECONDS
        runner.realm.refused_scopes.add(loops_scope)
        await reply.raw_result({"skills": []})

    runner.app.script(SKILLS_LIST, lets_the_loops_token_expire_and_be_refused)
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)

    assert {CREATE_SONG, GENERATE_SONG} <= set(runner.gateway.offered_tools(1))
    assert SONG_DESCRIPTION in runner.gateway.system_of(1)
    assert runner.reports.changes_of(Event.SUSPENDED) == []


async def test_a_skill_name_two_apps_share_goes_to_the_first(runner, songmaker):
    skilled(runner, songmaker)
    songmaker.shelf = Shelf({HOPIN: a_skill_file(HOPIN, body=SONG_BODY, description=SONG_DESCRIPTION)})
    runner.gateway = Gateway(tool_round(reading(HOPIN)), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    system = runner.gateway.system_of(1)
    assert HOPIN_DESCRIPTION in system
    assert SONG_DESCRIPTION not in system
    (shown,) = runner.gateway.tool_results(2)
    assert SONG_BODY not in shown
    assert SKILLS_GET not in songmaker.log.entries


async def test_an_apps_tool_named_read_skill_is_not_taken(runner, songmaker):
    skilled(runner, songmaker)

    songmaker.script(LIST_TOOLS, lists_only(READ_SKILL_TOOL, CREATE_SONG))
    runner.gateway = Gateway(tool_round(reading(SONGMAKER)), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    offered = [
        tool["function"] for tool in runner.gateway.body_of(1)["tools"] if tool["function"]["name"] == READ_SKILL_TOOL
    ]
    assert len(offered) == 1
    assert offered[0]["description"] != OWN_DESCRIPTION
    assert runner.reports.changes_of(Event.TOOLS_LEFT_OUT) == []
    assert SONG_BODY in runner.gateway.tool_results(2)[0]


async def test_read_skill_in_a_spoken_turn_answers_under_one_call_per_reply(runner, songmaker):
    skilled(runner, songmaker)
    await a_voice_call(runner, tool_round(reading(SONGMAKER), a_song("model-song")), completion("done"))

    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until_a_turn_completed()

    first, second = runner.gateway.tool_results(2)
    assert SONG_BODY in first
    assert second.startswith(ERROR_MARK)
    assert songmaker.called == []
    assert len(skill_reads(runner)) == 1


@pytest.mark.parametrize(
    ("failing", "first_round", "second_round", "reads_sent", "calls_sent"),
    [
        (SKILLS_GET, reading(SONGMAKER, "model-read-1"), reading(SONGMAKER, "model-read-2"), 1, 0),
        (CALL, a_song("model-song"), reading(SONGMAKER, "model-read-2"), 0, 1),
        (SKILLS_GET, reading(SONGMAKER, "model-read-1"), a_song("model-song"), 1, 0),
    ],
    ids=["a read after a failed read", "a read after a failed relayed call", "a relayed call after a failed read"],
)
async def test_an_app_that_failed_a_call_takes_no_more_this_turn_and_the_next_is_not_sent(
    runner, songmaker, failing, first_round, second_round, reads_sent, calls_sent
):
    skilled(runner, songmaker)
    songmaker.script(failing, refuses_with_a_server_error)
    runner.gateway = Gateway(tool_round(first_round), tool_round(second_round), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(3)

    assert runner.gateway.tool_results(2)[-1].startswith(ERROR_MARK)
    assert runner.gateway.tool_results(3)[-1].startswith(ERROR_MARK)
    assert songmaker.log.count(SKILLS_GET) == reads_sent
    assert songmaker.log.count(CALL) == calls_sent
    assert runner.reports.reported(apps=True, started=False) == [
        (Event.TOOLS_ONLY_APP_FAILED, status_code(HTTPStatus.INTERNAL_SERVER_ERROR), songmaker_url(runner))
    ]
    assert skill_reads(runner) == []


async def test_a_read_that_fails_on_the_loops_session_ends_the_turn_into_the_loops_reaction(runner):
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    runner.app.script(SKILLS_GET, refuses_with_a_server_error)
    runner.gateway = Gateway(tool_round(reading(HOPIN)), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.reports.until(Event.BACKOFF)

    assert runner.gateway.arrivals == 1
    assert skill_reads(runner) == []
    assert runner.reports.reported(started=False)[0] == (Event.BACKOFF, status_code(HTTPStatus.INTERNAL_SERVER_ERROR))


async def test_a_skill_body_past_the_bound_of_one_result_answers_an_error_and_enters_nothing(runner, songmaker):
    skilled(runner, songmaker)
    songmaker.shelf.files[SONGMAKER] = a_skill_file(SONGMAKER, body="x" * (MAX_EVENT_BYTES + 1))
    runner.gateway = Gateway(tool_round(reading(SONGMAKER)), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    (shown,) = runner.gateway.tool_results(2)
    assert shown.startswith(ERROR_MARK)
    assert len(shown) < MAX_EVENT_BYTES
    assert skill_reads(runner) == []


async def test_read_skill_answers_the_instructions_of_an_app_without_the_extension_and_sends_nothing(runner, songmaker):
    instructions = f"Instructions {secrets.token_hex(6)}."
    songmaker.shelf = Shelf(instructions=instructions, declares=False)
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN, description=HOPIN_DESCRIPTION)})
    implicit = implicit_skill_name(songmaker_url(runner))
    runner.gateway = Gateway(tool_round(reading(implicit)), completion("done"), completion("again"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)
    await a_turn(runner, 3)

    (shown,) = runner.gateway.tool_results(2)
    assert instructions in shown
    assert not {SKILLS_GET, READ} & set(songmaker.log.entries)
    ((assistant, _),) = runner.gateway.tool_rounds(3)
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == [READ_SKILL_TOOL]
    assert len(skill_reads(runner)) == 1
