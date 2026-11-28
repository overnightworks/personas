"""The whole runner, the block of skills a turn appends at the end of its request: one ``read_skill`` and one
``load_tools`` round for each skill of each app he holds a place at, once, again after a cut dropped them and for a tool
a notice added."""

from __future__ import annotations

import asyncio
import json
import secrets
from http import HTTPStatus

import pytest
from agents.backend.fake_gateway import completion, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL
from agents.member_outcomes import status_code
from agents.skills import SkillRefusal, implicit_skill_name
from agents.turn import MAX_EVENT_BYTES

from personas.decision import (
    BACKOFF_BASE_SECONDS,
    LOAD_TOOLS_CALLS,
    LOAD_TOOLS_TOOL,
    NETWORK_FAILURE_CODE,
    NO_ANSWER_REASON,
    READ_SKILL_CALLS,
    READ_SKILL_TOOL,
)
from personas.entries import SkillRead, ToolsLoaded
from personas.present import EMPTY_SKILL_TEXT, NOT_OFFERED_TEXT, ROUND_CALL_PREFIX
from personas.reporting import Event
from tests.support.app_driving import said_and_taken, seated_in_the_call, settled
from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, a_turn, restarted, restarted_in_the_call, say_in_the_call
from tests.support.fakes import SweepClock
from tests.support.gateway import BUDGET_SPENT, ERROR_MARK, Gateway, loading, reading, referenced
from tests.support.history import OTHER_TABLE, at_table, bounded_history, told_line_bytes, url_of
from tests.support.hopin import PLAY_TOOL, SAY_TOOL, SKILLS_GET
from tests.support.logs import lines_of
from tests.support.skills import HOPIN, Shelf, a_skill_file
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import (
    HOLD_ID,
    SAID,
    TWO_SENTENCES,
    a_voice_call,
    deciding,
    held_in_the_first_sentence,
    holding_sentence_ends,
)
from tests.support.wire import hangs, refuses_with_a_server_error

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

OTHER = "other"
TURN_ANSWERS = 5


def turn_answers() -> list:
    return [completion(str(each)) for each in range(TURN_ANSWERS)]


def told(rounds) -> list[tuple[str, str, object]]:
    """Each round as its tool, the skill it names and what it answers: a read its text, a load the tools it references
    or its text when it loads none."""
    answered = []
    for assistant, result in rounds:
        (call,) = assistant["tool_calls"]
        content = result["content"]
        answered.append(
            (
                call["function"]["name"],
                json.loads(call["function"]["arguments"])["skill"],
                referenced(content) if isinstance(content, list) else content,
            )
        )
    return answered


def implicit_name(runner, number: int) -> str:
    return implicit_skill_name(url_of(runner, number))


def block_of(runner, number: int, *, tools: list[str]) -> list[tuple[str, str, object]]:
    """The two rounds that tell the implicit skill of the app at this place in the configuration: a read of no text and
    a load of its tools."""
    name = implicit_name(runner, number)
    return [(READ_SKILL_TOOL, name, EMPTY_SKILL_TEXT), (LOAD_TOOLS_TOOL, name, tools)]


def entries_of(runner, kind: type, number: int):
    return [
        each for each in runner.histories.entries() if isinstance(each, kind) and each.app == url_of(runner, number)
    ]


async def stood_at(other, *standings) -> None:
    """The agent's standings at the other app are these and its loop has read and taken them."""
    reads = other.standing_reads()
    other.stand(SUBJECT, *standings)
    await other.until_standing_read(reads + 1)
    await settled(other)


async def after_a_first_turn(runner, other=None, *replies) -> None:
    """The runner started in his call, whose first turn was answered; the table at the other app, when there is one, is
    joined after it."""
    runner.gateway = Gateway(*replies or turn_answers())
    await restarted_in_the_call(runner)
    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()
    if other is not None:
        await stood_at(other, at_table())


async def test_a_turn_after_a_join_ends_with_the_apps_read_and_load_rounds_and_tells_earlier_messages_byte_for_byte(
    runner, other
):
    """TURNS-15, TURNS-18."""
    await after_a_first_turn(runner, other)

    await a_turn(runner, 2)

    first, second = (runner.gateway.body_of(number)["messages"] for number in (1, 2))
    assert second[1 : len(first)] == first[1:]
    joined = [name for name in runner.gateway.deferred_tools(2) if name not in runner.gateway.deferred_tools(1)]
    assert joined == [PLAY_TOOL]
    assert told(runner.gateway.appended_rounds(2)) == block_of(runner, 1, tools=joined)
    ids = [assistant["tool_calls"][0]["id"] for assistant, _ in runner.gateway.appended_rounds(2)]
    assert all(each.startswith(ROUND_CALL_PREFIX) for each in ids)
    (read,) = entries_of(runner, SkillRead, 1)
    (load,) = entries_of(runner, ToolsLoaded, 1)
    assert tuple(ids) == (read.call_id, load.call_id)
    assert load.tools == (PLAY_TOOL,)


async def test_the_turn_after_a_block_appends_none_and_tells_it_at_its_place(runner):
    """TURNS-15."""
    await after_a_first_turn(runner)

    await a_turn(runner, 2)

    assert told(runner.gateway.appended_rounds(1)) == block_of(runner, 0, tools=[SAY_TOOL])
    assert runner.gateway.appended_rounds(2) == []
    assert runner.gateway.tool_rounds(2) == runner.gateway.tool_rounds(1)
    roles = [message["role"] for message in runner.gateway.body_of(2)["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool", "assistant", "tool", "user"]
    assert len(entries_of(runner, SkillRead, 0)) == len(entries_of(runner, ToolsLoaded, 0)) == 1


@pytest.mark.parametrize(
    ("standings", "appended"),
    [
        pytest.param(
            [[at_table(), active(OTHER_TABLE.id, OTHER_TABLE.kind, serves_view=True)]],
            [2],
            id="two joins before a turn",
        ),
        pytest.param(
            [[at_table()], [at_table(), active(OTHER_TABLE.id, OTHER_TABLE.kind, serves_view=True)]],
            [2, 0],
            id="a join after the block",
        ),
    ],
)
async def test_a_second_place_at_an_app_whose_block_his_history_holds_appends_none(runner, other, standings, appended):
    """TURNS-18."""
    await after_a_first_turn(runner)

    counts = []
    for number, each in enumerate(standings, start=2):
        await stood_at(other, *each)
        await a_turn(runner, number)
        counts.append(len(runner.gateway.appended_rounds(number)))
        await runner.reports.until_a_turn_completed(count=number)

    assert counts == appended
    assert len(entries_of(runner, SkillRead, 1)) == 1


async def test_a_rejoin_at_an_app_whose_block_his_history_holds_appends_none_and_its_load_names_the_tools_again(
    runner, other
):
    """TURNS-18."""
    await after_a_first_turn(runner, other)
    await a_turn(runner, 2)
    await runner.reports.until_a_turn_completed(count=2)
    await stood_at(other)
    await a_turn(runner, 3)
    await runner.reports.until_a_turn_completed(count=3)
    await stood_at(other, at_table())

    await a_turn(runner, 4)

    loads_of_other = [each for each in told(runner.gateway.tool_rounds(3)) if each[0] == LOAD_TOOLS_TOOL][1]
    assert loads_of_other[2] == NOT_OFFERED_TEXT
    assert [runner.gateway.appended_rounds(number) for number in (3, 4)] == [[], []]
    assert [each for each in told(runner.gateway.tool_rounds(4)) if each[0] == LOAD_TOOLS_TOOL][1][2] == [PLAY_TOOL]


async def test_a_declared_apps_block_is_a_read_and_a_load_per_skill_in_listing_order_loaded_tools_read_alone(
    runner, other
):
    """TURNS-18."""
    alpha, beta, quiet = (secrets.token_hex(4) for _ in range(3))
    other.shelf = Shelf(
        {
            "alpha": a_skill_file("alpha", tools=PLAY_TOOL, body=alpha),
            "beta": a_skill_file("beta", tools=PLAY_TOOL, body=beta),
            "quiet": a_skill_file("quiet", tools=None, body=quiet),
        }
    )
    await after_a_first_turn(runner, other)

    await a_turn(runner, 2)

    appended = told(runner.gateway.appended_rounds(2))
    assert [(tool, skill) for tool, skill, _ in appended] == [
        (READ_SKILL_TOOL, "alpha"),
        (LOAD_TOOLS_TOOL, "alpha"),
        (READ_SKILL_TOOL, "beta"),
        (READ_SKILL_TOOL, "quiet"),
    ]
    assert [alpha in appended[0][2], appended[1][2], beta in appended[2][2], quiet in appended[3][2]] == [
        True,
        [PLAY_TOOL],
        True,
        True,
    ]
    assert other.log.count(SKILLS_GET) == 3


async def test_two_present_apps_whose_skills_share_a_name_each_get_a_block_that_loads_their_own_tools(runner, other):
    """TURNS-18."""
    await seated_in_the_call(runner, other, *turn_answers())

    await a_turn(runner, 1)

    assert implicit_name(runner, 0) == implicit_name(runner, 1)
    assert told(runner.gateway.appended_rounds(1)) == [
        *block_of(runner, 0, tools=[SAY_TOOL]),
        *block_of(runner, 1, tools=[PLAY_TOOL]),
    ]


def with_a_history_of_two_lines(runner) -> None:
    """A history that holds two lines: a block entered after the first line and the second line do not both fit."""
    line_bytes = told_line_bytes(url_of(runner, 0), PLACE_ID, f"{ADDRESSED} {secrets.token_hex(3)}")
    runner.config = bounded_history(runner.config, 2 * line_bytes)


async def test_a_block_the_cut_dropped_is_appended_again_at_the_end_of_his_next_turn_while_he_holds_the_place(runner):
    """TURNS-16."""
    with_a_history_of_two_lines(runner)
    await after_a_first_turn(runner)
    for number in (2, 3):
        await a_turn(runner, number)
        await runner.reports.until_a_turn_completed(count=number)

    first, again = (runner.gateway.appended_rounds(number) for number in (1, 3))
    assert told(again) == told(first)
    assert [assistant["tool_calls"][0]["id"] for assistant, _ in again] != [
        assistant["tool_calls"][0]["id"] for assistant, _ in first
    ]
    assert runner.gateway.appended_rounds(2) == []


async def test_a_block_the_cut_dropped_after_he_left_its_app_is_not_appended_again(runner, other):
    """TURNS-16."""
    with_a_history_of_two_lines(runner)
    await after_a_first_turn(runner, other)
    await a_turn(runner, 2)
    await runner.reports.until_a_turn_completed(count=2)
    await stood_at(other)
    for number in (3, 4, 5):
        await a_turn(runner, number)
        await runner.reports.until_a_turn_completed(count=number)

    loads = {
        number: [each[2] for each in told(runner.gateway.tool_rounds(number)) if each[0] == LOAD_TOOLS_TOOL]
        for number in (2, 3, 5)
    }
    assert loads == {2: [[SAY_TOOL], [PLAY_TOOL]], 3: [NOT_OFFERED_TEXT, [SAY_TOOL]], 5: [[SAY_TOOL]]}
    assert told(runner.gateway.appended_rounds(5)) == block_of(runner, 0, tools=[SAY_TOOL])


async def test_a_restart_tells_every_block_byte_for_byte_under_the_same_ids_and_appends_none(runner, other):
    """TURNS-15."""
    await after_a_first_turn(runner, other)
    await a_turn(runner, 2)
    await runner.reports.until_a_turn_completed(count=2)

    await restarted(runner)
    await a_turn(runner, 3)

    assert runner.gateway.tool_rounds(3) == runner.gateway.tool_rounds(2)
    assert len(runner.gateway.tool_rounds(3)) == 4
    assert runner.gateway.appended_rounds(3) == []


async def test_every_tool_a_turn_lists_is_sent_deferred_and_each_present_apps_is_loaded_by_a_round_of_its_request(
    runner, other, songmaker
):
    """TURNS-18."""
    await seated_in_the_call(runner, other, *turn_answers())

    await a_turn(runner, 1)

    assert runner.gateway.deferred_tools(1) == [SAY_TOOL, PLAY_TOOL, CREATE_SONG, GENERATE_SONG]
    loaded = [name for each in told(runner.gateway.tool_rounds(1)) if each[0] == LOAD_TOOLS_TOOL for name in each[2]]
    assert loaded == [SAY_TOOL, PLAY_TOOL]


async def test_a_refused_skill_of_a_present_app_enters_no_read_logs_its_code_still_loads_its_tools_and_is_asked_again(
    runner, caplog
):
    """LOG-40."""
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)}, tampered={HOPIN})
    await after_a_first_turn(runner)

    await a_turn(runner, 2)

    assert told(runner.gateway.appended_rounds(1)) == [(LOAD_TOOLS_TOOL, HOPIN, [SAY_TOOL])]
    assert runner.gateway.appended_rounds(2) == []
    assert entries_of(runner, SkillRead, 0) == []
    assert runner.reports.codes_of(Event.SKILL_REFUSED) == [SkillRefusal.SIZE_MISMATCH.value] * 2
    assert runner.app.log.count(SKILLS_GET) == 2
    assert [each["reason"] for each in lines_of(caplog.get_records("call"), "skill.refused")] == [
        SkillRefusal.SIZE_MISMATCH.value
    ] * 2


@pytest.mark.parametrize(
    ("handler", "code", "call_seconds"),
    [
        pytest.param(refuses_with_a_server_error, NO_ANSWER_REASON, None, id="a server error"),
        pytest.param(hangs, NETWORK_FAILURE_CODE, 0.3, id="a hang past the call time"),
    ],
)
async def test_a_block_read_that_fails_or_times_out_at_the_own_app_is_left_out_and_the_turn_goes_on_with_its_load(
    runner, caplog, handler, code, call_seconds
):
    """TURNS-19, LOG-13."""
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    runner.app.script(SKILLS_GET, handler)
    if call_seconds is not None:
        runner.bound_listening_calls_to(call_seconds)
    await after_a_first_turn(runner)

    assert told(runner.gateway.appended_rounds(1)) == [(LOAD_TOOLS_TOOL, HOPIN, [SAY_TOOL])]
    (line,) = lines_of(caplog.get_records("call"), "skill_read.ended")
    assert (line["outcome"], line["reason"]) == ("failed", code)
    assert "error_type" in line
    assert runner.reports.changes_of(Event.BACKOFF) == []
    assert entries_of(runner, SkillRead, 0) == []


async def test_a_block_read_the_own_app_answers_ends_its_step_line_ok(runner, caplog):
    """LOG-13."""
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    await after_a_first_turn(runner)

    (line,) = lines_of(caplog.get_records("call"), "skill_read.ended")
    assert line["outcome"] == "ok"
    assert "reason" not in line
    assert len(entries_of(runner, SkillRead, 0)) == 1


async def test_a_block_read_the_own_app_refuses_the_agent_ends_the_turn_into_the_loops_suspension(runner):
    """TURNS-19, LOG-7."""
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    runner.app.refuse_with = HTTPStatus.FORBIDDEN
    runner.app.refused_method = SKILLS_GET
    runner.gateway = Gateway(completion("done"))
    await restarted_in_the_call(runner)

    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.SUSPENDED)

    assert runner.reports.codes_of(Event.SUSPENDED) == [status_code(HTTPStatus.FORBIDDEN)]
    assert runner.gateway.arrivals == 0


async def test_a_block_read_that_fails_at_another_app_logs_held_app_failed_and_the_turn_goes_on_with_its_load(
    runner, other
):
    """LOG-19."""
    other.shelf = Shelf({OTHER: a_skill_file(OTHER, tools=PLAY_TOOL)})
    other.script(SKILLS_GET, refuses_with_a_server_error)
    await seated_in_the_call(runner, other, *turn_answers())

    await a_turn(runner, 1)
    await runner.reports.until_a_turn_completed()
    await a_turn(runner, 2)

    assert told(runner.gateway.appended_rounds(1))[-1] == (LOAD_TOOLS_TOOL, OTHER, [PLAY_TOOL])
    assert (Event.HELD_APP_FAILED, status_code(HTTPStatus.INTERNAL_SERVER_ERROR), url_of(runner, 1)) in (
        runner.reports.reported(apps=True, started=False)
    )
    assert [each[:2] for each in told(runner.gateway.appended_rounds(2))] == [(READ_SKILL_TOOL, OTHER)]
    assert other.log.count(SKILLS_GET) == 2


async def test_a_line_that_arrives_while_the_block_reads_is_told_before_every_round_of_the_block(runner, other):
    """TURNS-15."""
    runner.app.shelf = Shelf({HOPIN: a_skill_file(HOPIN)})
    other.shelf = Shelf({OTHER: a_skill_file(OTHER, tools=PLAY_TOOL)})
    entered, released = asyncio.Event(), asyncio.Event()

    other.script(SKILLS_GET, other.shelf.entry_held_until(entered, released))
    await seated_in_the_call(runner, other, *turn_answers())
    say_in_the_call(runner, f"{ADDRESSED} {secrets.token_hex(3)}")
    await entered.wait()
    arrived = f"later {secrets.token_hex(3)}"

    await said_and_taken(runner.app, PLACE_ID, arrived)
    released.set()
    await runner.reports.until_a_turn_completed()
    await a_turn(runner, 2)

    messages = runner.gateway.body_of(2)["messages"]
    told_at = next(index for index, each in enumerate(messages) if arrived in str(each["content"]))
    first_round = next(index for index, each in enumerate(messages) if each.get("tool_calls"))
    assert told_at < first_round


@pytest.mark.parametrize("at_other", [False, True], ids=["the own app", "another app"])
async def test_a_block_read_past_one_results_bound_is_left_out_and_its_load_enters(runner, other, caplog, at_other):
    """TURNS-19, LOG-13."""
    large = a_skill_file(
        OTHER if at_other else HOPIN, tools=PLAY_TOOL if at_other else SAY_TOOL, body="x" * (MAX_EVENT_BYTES + 1)
    )
    app = other if at_other else runner.app
    app.shelf = Shelf({OTHER if at_other else HOPIN: large})
    await seated_in_the_call(runner, other, *turn_answers())

    await a_turn(runner, 1)

    tools = [(tool, skill) for tool, skill, _ in told(runner.gateway.appended_rounds(1)) if skill in {OTHER, HOPIN}]
    assert tools == [(LOAD_TOOLS_TOOL, OTHER if at_other else HOPIN)]
    (line,) = lines_of(caplog.get_records("call"), "skill_read.ended")
    assert (line["outcome"], line["reason"]) == ("failed", "too_large")
    assert entries_of(runner, SkillRead, 1 if at_other else 0) == []
    assert runner.reports.changes_of(Event.HELD_APP_FAILED) == []


async def test_a_spoken_turn_after_a_join_appends_the_block_and_speaks_under_one_call_per_reply(runner):
    """TURNS-15, TURNS-18."""
    await a_voice_call(runner, completion(ANSWER))

    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    await runner.reports.until_a_turn_completed()

    assert told(runner.gateway.appended_rounds(1)) == block_of(runner, 0, tools=[SAY_TOOL])
    assert runner.app.spoken == [(SUBJECT, PLACE_ID, ANSWER)]


async def test_a_hold_round_after_a_block_tells_its_rounds_and_is_sent(runner):
    """TURNS-30."""
    await held_in_the_first_sentence(runner, completion(TWO_SENTENCES), deciding(CONTINUE_SPEAKING_TOOL), completion())
    await runner.reports.until(Event.HOLD_DECIDED)

    (number,) = runner.gateway.rounds()
    assert told(runner.gateway.tool_rounds(number)) == told(runner.gateway.tool_rounds(1))
    assert len(runner.gateway.tool_rounds(number)) == 2
    assert runner.reports.codes_of(Event.HOLD_DECIDED) == ["continued"]


async def test_a_turn_told_a_hold_tells_the_hold_after_the_block(runner):
    """TURNS-15."""
    holding_sentence_ends(runner)
    await a_voice_call(runner, deciding(CONTINUE_SPEAKING_TOOL), completion("[silent]"))
    runner.app.hold(PLACE_ID, HOLD_ID, SAID)
    await runner.reports.until(Event.HOLD_DECIDED)

    (number,) = runner.gateway.rounds()
    messages = runner.gateway.body_of(number)["messages"]
    assert [message["role"] for message in messages[-6:]] == ["user", "assistant", "tool", "assistant", "tool", "user"]
    assert '"woke"' in messages[-6]["content"]
    assert messages[-1]["content"].startswith('{"held"')
    assert len(runner.gateway.tool_rounds(number)) == 2


async def test_a_block_takes_none_of_the_models_reads_or_loads(runner):
    """TURNS-22."""
    names = ("alpha", "beta", "gamma")
    runner.app.shelf = Shelf({each: a_skill_file(each, tools=SAY_TOOL if each == "alpha" else None) for each in names})
    calls = [reading("beta", f"model-read-{each}") for each in range(READ_SKILL_CALLS)] + [
        loading("alpha", f"model-load-{each}") for each in range(LOAD_TOOLS_CALLS)
    ]
    runner.gateway = Gateway(tool_round(*calls), completion("done"))
    await restarted_in_the_call(runner)

    await a_turn(runner, 1)
    await runner.gateway.until_requested(2)

    results = runner.gateway.tool_results(2)
    assert len(results) == len(calls)
    assert not any(str(each).startswith(ERROR_MARK) for each in results)
    assert len(told(runner.gateway.appended_rounds(1))) == len(names) + 1


async def test_a_block_of_a_turn_without_an_answer_is_told_at_its_place_in_the_next_turn(runner):
    """TURNS-15."""
    runner.clock = SweepClock(BACKOFF_BASE_SECONDS)
    runner.gateway = Gateway(BUDGET_SPENT, completion("done"))
    await restarted_in_the_call(runner)
    say_in_the_call(runner, ADDRESSED)
    await runner.clock.until_waiting(1)

    runner.clock.open_gate()
    await runner.gateway.until_requested(2)

    assert told(runner.gateway.appended_rounds(1)) == block_of(runner, 0, tools=[SAY_TOOL])
    assert runner.gateway.appended_rounds(2) == []
    assert runner.gateway.tool_rounds(2) == runner.gateway.tool_rounds(1)
    roles = [message["role"] for message in runner.gateway.body_of(2)["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool", "assistant", "tool", "user"]
