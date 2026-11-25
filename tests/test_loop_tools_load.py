"""The whole runner, the tools of a tools-only app sent deferred, ``load_tools`` and the round that tells a load in the
turns after it."""

from __future__ import annotations

import json
import secrets

import pytest
from agents.backend.fake_gateway import completion, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL
from agents.skills import implicit_skill_name

from personas.decision import LOAD_TOOLS_CALLS, LOAD_TOOLS_TOOL, READ_SKILL_TOOL
from personas.entries import ActDone, ToolsLoaded, told_bytes
from personas.held_tools import LOADS_SPENT_TEXT, NOTHING_WAITS_TEXT
from personas.present import NO_SUCH_SKILL_TEXT, NOT_OFFERED_TEXT
from personas.reporting import Event
from tests.support.app_driving import said_and_taken
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, a_turn, restarted_in_the_call, say_in_the_call
from tests.support.gateway import ERROR_MARK, Gateway, loading, reading, say
from tests.support.hopin import LIST_TOOLS, SAY_TOOL
from tests.support.logs import lines_of
from tests.support.skills import (
    HOPIN,
    SONG_BODY,
    SONGMAKER,
    Shelf,
    a_skill_file,
    assert_a_round_is_told_byte_for_byte_after_a_restart,
    skilled,
    turns_past_a_cut_of_a_round,
)
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG, OWN_DESCRIPTION, a_song, lists_only, songmaker_url
from tests.support.voice import a_voice_call, deciding, held_in_the_first_sentence

pytestmark = pytest.mark.anyio

SONG_TOOLS = [CREATE_SONG, GENERATE_SONG]


def tools_loaded(runner) -> list[ToolsLoaded]:
    return [entry for entry in runner.histories.entries() if isinstance(entry, ToolsLoaded)]


def referenced(result) -> list[str]:
    """The names a tool result of the wire references, in order."""
    return [part["tool_name"] for part in result]


def deferred_in(runner, request_number: int) -> list[str]:
    """The names of the tools the request that arrived as this number sent deferred."""
    tools = runner.gateway.body_of(request_number)["tools"]
    return [each["function"]["name"] for each in tools if each.get("defer_loading")]


def shown_bytes(runner, request_number: int) -> int:
    """The bytes of the definitions the request that arrived as this number shows the model: those it does not defer and
    each one a tool result of it references."""
    loaded = {
        name
        for result in runner.gateway.tool_results(request_number)
        if isinstance(result, list)
        for name in referenced(result)
    }
    tools = runner.gateway.body_of(request_number)["tools"]
    return sum(
        told_bytes(each["function"])
        for each in tools
        if not each.get("defer_loading") or each["function"]["name"] in loaded
    )


def declared_skill(runner, songmaker):
    skilled(runner, songmaker)
    return SONGMAKER


def implicit_skill(runner, songmaker):
    songmaker.shelf = Shelf(declares=False)
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    return implicit_skill_name(songmaker_url(runner))


@pytest.mark.parametrize("call", [reading(SONGMAKER), loading(SONGMAKER)], ids=["read_skill", "load_tools"])
async def test_a_skill_read_and_a_tools_load_are_not_held_back_by_what_arrived_and_do_not_use_up_the_hold(
    runner, songmaker, call
):
    """TURNS-20."""
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(call), tool_round(say(ANSWER)), completion("done"))
    running = runner.gateway.hold_request(1)
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, "News that arrived meanwhile")
    running.set()
    await runner.gateway.until_requested(3)

    assert runner.gateway.held_result_texts(2) == []
    assert len(runner.gateway.held_results(3)) == 1
    assert runner.app.said == []


async def test_a_hopin_turn_sends_songmakers_tools_deferred_and_its_own_and_the_runners_loaded(runner, songmaker):
    skilled(runner, songmaker)
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)

    assert deferred_in(runner, 1) == SONG_TOOLS
    sent = runner.gateway.offered_tools(1, reading=True)
    assert {SAY_TOOL, READ_SKILL_TOOL, LOAD_TOOLS_TOOL} <= set(sent) - set(SONG_TOOLS)


@pytest.mark.parametrize(
    ("arranging", "reads_first"),
    [(declared_skill, False), (implicit_skill, False), (declared_skill, True)],
    ids=["a declared skill", "an implicit skill", "read and load in one reply"],
)
async def test_load_tools_answers_references_to_a_skills_deferred_tools_and_the_next_round_calls_one(
    runner, songmaker, arranging, reads_first
):
    name = arranging(runner, songmaker)
    first_round = tool_round(reading(name), loading(name)) if reads_first else tool_round(loading(name))
    runner.gateway = Gateway(first_round, tool_round(a_song("model-song")), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(3)

    assert referenced(runner.gateway.tool_results(2)[-1]) == SONG_TOOLS
    assert [name for name, _ in songmaker.called] == [CREATE_SONG]
    (load,) = tools_loaded(runner)
    assert (load.skill, load.tools) == (name, tuple(SONG_TOOLS))


async def test_the_next_turn_tells_a_load_as_a_load_tools_round_under_its_id_and_the_turn_after_byte_equal(
    runner, songmaker
):
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(loading(SONGMAKER)), completion("done"), completion("again"), completion("end"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)
    await a_turn(runner, 3)
    await a_turn(runner, 4)

    ((assistant, result),) = runner.gateway.tool_rounds(3)
    (call,) = assistant["tool_calls"]
    assert (call["function"]["name"], json.loads(call["function"]["arguments"])) == (
        LOAD_TOOLS_TOOL,
        {"skill": SONGMAKER},
    )
    assert call["id"] != "model-load"
    assert result["tool_call_id"] == call["id"]
    assert referenced(result["content"]) == SONG_TOOLS
    assert runner.gateway.tool_rounds(4) == [[assistant, result]]


async def test_a_loads_round_is_byte_equal_after_a_restart(runner, songmaker):
    await assert_a_round_is_told_byte_for_byte_after_a_restart(runner, songmaker, loading(SONGMAKER))


@pytest.mark.parametrize(
    ("listings", "told"),
    [
        pytest.param([[CREATE_SONG, GENERATE_SONG], [CREATE_SONG]], [CREATE_SONG], id="one of two gone"),
        pytest.param([[CREATE_SONG, GENERATE_SONG], []], NOT_OFFERED_TEXT, id="every one gone"),
        pytest.param([[CREATE_SONG, GENERATE_SONG], [], SONG_TOOLS], SONG_TOOLS, id="listed again"),
    ],
)
async def test_a_tool_the_app_stops_listing_leaves_the_loads_round_until_the_app_lists_it_again(
    runner, songmaker, listings, told
):
    skilled(runner, songmaker)
    songmaker.script(LIST_TOOLS, *(lists_only(*each) for each in listings))
    runner.gateway = Gateway(tool_round(loading(SONGMAKER)), *(completion(str(each)) for each in range(len(listings))))
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    for turn, _ in enumerate(listings[1:]):
        await a_turn(runner, 3 + turn)

    ((_, result),) = runner.gateway.tool_rounds(1 + len(listings))
    assert (referenced(result["content"]) if isinstance(told, list) else result["content"]) == told


def own_apps_skill(runner, songmaker):
    skilled(runner, songmaker)
    return HOPIN


def skill_naming_no_tool(runner, songmaker):
    skilled(runner, songmaker)
    songmaker.shelf.files["quiet"] = a_skill_file("quiet", tools=None)
    return "quiet"


def skill_naming_another_apps_tool(runner, songmaker):
    skilled(runner, songmaker)
    songmaker.shelf.files["borrowing"] = a_skill_file("borrowing", tools=SAY_TOOL)
    return "borrowing"


@pytest.mark.parametrize(
    ("arranging", "earlier"),
    [
        pytest.param(own_apps_skill, "no load", id="the own app's skill"),
        pytest.param(declared_skill, "a told round", id="loaded by a told round"),
        pytest.param(declared_skill, "the turn", id="loaded earlier in the turn"),
        pytest.param(skill_naming_no_tool, "no load", id="a declared skill naming no tool"),
        pytest.param(skill_naming_another_apps_tool, "no load", id="a skill naming only another app's tool"),
    ],
)
async def test_a_load_of_a_skill_with_no_tool_waiting_answers_so_and_enters_nothing(
    runner, songmaker, arranging, earlier
):
    name = arranging(runner, songmaker)
    first, again = tool_round(loading(name, "model-first")), tool_round(loading(name, "model-again"))
    script = {
        "no load": [again, completion("done")],
        "the turn": [first, again, completion("done")],
        "a told round": [first, completion("done"), again, completion("done")],
    }[earlier]
    runner.gateway = Gateway(*script)
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    if earlier == "a told round":
        await a_turn(runner, 3)
    await runner.reports.until_a_turn_completed(count=2 if earlier == "a told round" else 1)

    answered = runner.gateway.tool_results(len(script))[-1]
    assert answered == NOTHING_WAITS_TEXT
    assert len(tools_loaded(runner)) == (0 if earlier == "no load" else 1)


async def test_load_tools_for_a_name_no_line_holds_answers_an_error_and_enters_nothing(runner, songmaker):
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(loading("nobody")), completion("done"), completion("again"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)
    await a_turn(runner, 3)

    assert runner.gateway.tool_results(2) == [f"{ERROR_MARK}{NO_SUCH_SKILL_TEXT}"]
    assert tools_loaded(runner) == []
    assert runner.gateway.tool_rounds(3) == []


async def test_load_tools_calls_past_their_bound_answer_an_error_and_a_later_call_still_runs(runner, songmaker):
    skilled(runner, songmaker)
    loads = [loading(SONGMAKER, f"model-load-{number}") for number in range(LOAD_TOOLS_CALLS + 1)]
    runner.gateway = Gateway(tool_round(*loads), tool_round(a_song("model-song")), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(3)

    *answered, spent = runner.gateway.tool_results(2)
    assert referenced(answered[0]) == SONG_TOOLS
    assert answered[1] == NOTHING_WAITS_TEXT
    assert spent == f"{ERROR_MARK}{LOADS_SPENT_TEXT}"
    assert len(tools_loaded(runner)) == 1
    assert [name for name, _ in songmaker.called] == [CREATE_SONG]


async def test_a_load_beside_a_relayed_call_enters_both_in_the_order_they_ended_and_its_round_holds_only_the_load(
    runner, songmaker
):
    skilled(runner, songmaker)
    spoken = f"Let me look {secrets.token_hex(4)}."
    runner.gateway = Gateway(
        tool_round(a_song("model-song"), loading(SONGMAKER), content=spoken), completion("done"), completion("again")
    )
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    await a_turn(runner, 3)

    kinds = [type(entry) for entry in runner.histories.entries() if isinstance(entry, ActDone | ToolsLoaded)]
    assert kinds == [ActDone, ToolsLoaded]
    ((assistant, _),) = runner.gateway.tool_rounds(3)
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == [LOAD_TOOLS_TOOL]
    assert assistant["content"] is None
    told = runner.gateway.told_in(3)
    assert spoken not in told
    assert '"entry": "act"' in told


async def test_a_spoken_hopin_turn_reads_loads_and_calls_songmakers_tool_one_round_each(runner, songmaker):
    skilled(runner, songmaker)
    await a_voice_call(
        runner,
        tool_round(reading(SONGMAKER)),
        tool_round(loading(SONGMAKER)),
        tool_round(a_song("model-song")),
        completion("done"),
    )

    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.reports.until_a_turn_completed()

    assert SONG_BODY in runner.gateway.tool_results(2)[0]
    assert referenced(runner.gateway.tool_results(3)[-1]) == SONG_TOOLS
    assert [name for name, _ in songmaker.called] == [CREATE_SONG]
    assert runner.gateway.arrivals == 4


async def test_a_hold_round_after_a_load_in_its_turn_tells_the_load_and_is_sent(runner, songmaker):
    skilled(runner, songmaker)
    await held_in_the_first_sentence(
        runner,
        tool_round(loading(SONGMAKER)),
        completion("First sentence. Second sentence."),
        deciding(CONTINUE_SPEAKING_TOOL),
        completion(),
    )

    await runner.reports.until(Event.HOLD_DECIDED)

    (number,) = runner.gateway.rounds()
    ((_, result),) = runner.gateway.tool_rounds(number)
    assert referenced(result["content"]) == SONG_TOOLS
    assert deferred_in(runner, number) == SONG_TOOLS


async def test_a_loads_round_is_gone_once_a_cut_drops_its_entry_and_is_not_written_again(runner, songmaker):
    await turns_past_a_cut_of_a_round(runner, songmaker, loading(SONGMAKER), len(CREATE_SONG) + len(GENERATE_SONG))

    assert len(runner.gateway.tool_rounds(3)) == 1
    assert (runner.gateway.tool_rounds(5), runner.gateway.tool_rounds(6)) == ([], [])
    assert len(tools_loaded(runner)) == 1


async def test_an_apps_tool_named_load_tools_is_not_taken(runner, songmaker):
    skilled(runner, songmaker)

    songmaker.script(LIST_TOOLS, lists_only(LOAD_TOOLS_TOOL, CREATE_SONG))
    runner.gateway = Gateway(tool_round(loading(SONGMAKER)), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    own = [each for each in runner.gateway.body_of(1)["tools"] if each["function"]["name"] == LOAD_TOOLS_TOOL]
    assert len(own) == 1
    assert own[0]["function"]["description"] != OWN_DESCRIPTION
    assert referenced(runner.gateway.tool_results(2)[0]) == [CREATE_SONG]


async def test_tools_bytes_count_the_loaded_tools_and_leave_out_the_deferred_ones_none_loaded(
    runner, songmaker, caplog
):
    skilled(runner, songmaker)
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()

    (ended,) = lines_of(caplog.get_records("call"), "turn.ended")
    assert ended["tools_bytes"] == shown_bytes(runner, 1)
    assert deferred_in(runner, 1) == SONG_TOOLS


async def test_tools_bytes_count_the_loaded_tools_and_leave_out_the_deferred_ones_loaded_by_a_told_round(
    runner, songmaker, caplog
):
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(loading(SONGMAKER)), completion("done"), completion("again"))
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()
    await a_turn(runner, 3)
    await runner.reports.until_a_turn_completed(count=2)

    _, second = lines_of(caplog.get_records("call"), "turn.ended")
    assert second["tools_bytes"] == shown_bytes(runner, 3)


async def test_tools_bytes_count_the_loaded_tools_and_leave_out_the_deferred_ones_loaded_in_the_turn_once(
    runner, songmaker, caplog
):
    skilled(runner, songmaker)
    runner.gateway = Gateway(
        tool_round(loading(SONGMAKER)),
        tool_round(a_song("model-song")),
        tool_round(a_song("again")),
        completion("done"),
    )
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()

    (ended,) = lines_of(caplog.get_records("call"), "turn.ended")
    assert ended["tools_bytes"] == shown_bytes(runner, 4)


async def test_a_load_counts_its_answer_in_the_results_bytes_of_its_turn(runner, songmaker, caplog):
    skilled(runner, songmaker)
    runner.gateway = Gateway(tool_round(loading(SONGMAKER)), completion("done"))
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()

    (ended,) = lines_of(caplog.get_records("call"), "turn.ended")
    assert ended["results_bytes"] > 0


async def test_the_next_turn_counts_a_loads_round_in_its_told_bytes(runner, songmaker, caplog):
    """A turn's span of the told history grows by its round when it loaded, and not otherwise."""
    skilled(runner, songmaker)
    runner.gateway = Gateway(
        completion("done"), completion("done"), tool_round(loading(SONGMAKER)), completion("done"), completion("done")
    )
    await restarted_in_the_call(runner)
    for count, request_number in enumerate([1, 2, 3, 5], start=1):
        await a_turn(runner, request_number)
        await runner.reports.until_a_turn_completed(count=count)

    _, plain_turn_told, load_turn_told, after_load_told = [
        each["told_bytes"] for each in lines_of(caplog.get_records("call"), "turn.ended")
    ]
    assert after_load_told - load_turn_told > load_turn_told - plain_turn_told
