"""The whole runner, the history: what an agent took in at his call places is told to every turn of his."""

from __future__ import annotations

import json
import secrets
from dataclasses import replace

import pytest
from agents.backend.fake_gateway import completion
from agents.turn import Principal, person_fingerprint

from personas.failures import STORE_UNREACHABLE_CODE, StoreUnavailable
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call
from tests.support.fakes import memory_target
from tests.support.gateway import Gateway, says, says_after_a_hold
from tests.support.history import (
    OTHER_CALL_ID,
    REMARK,
    SPEAKER,
    STATE_BYTES,
    TABLE,
    act_entry,
    answered,
    at_table,
    entry,
    histories,
    history_of,
    said_and_taken,
    settled,
    told_line_bytes,
    url_of,
    view_entry,
    view_turns,
    waits,
    woke_at_lines,
    woke_at_table,
    written_turns,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import ISSUER, PLAY_TOOL, active, paused
from tests.support.realm import AGENT_NAME
from tests.support.voice import a_voice_call

pytestmark = pytest.mark.anyio

SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME = "agent-bach", "agent-subject-2", "Bach"


async def test_a_line_said_in_a_call_of_one_app_is_told_to_his_view_turn_of_another_app_after_a_silent_written_turn(
    runner, other
):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    other.stand(SUBJECT, at_table())
    state = waits(other, 1)
    await runner.gateway.until_turns(1, why="view")

    (written,) = written_turns(runner)
    assert written.new_seqs == [1]
    (opening,) = view_turns(runner)
    assert opening.room == {
        "you": AGENT_NAME,
        "history": history_of(
            [entry(url_of(runner, 0), PLACE_ID, REMARK), view_entry(url_of(runner, 1), state, revision=1, new=True)]
        ),
        "woke": woke_at_table(url_of(runner, 1)),
    }
    assert runner.app.said == []


async def test_a_line_read_in_the_same_wake_as_the_view_of_its_own_app_is_told_with_that_app(runner):
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    runner.app.show_view(TABLE.id, 1, {"hand": "none"}, awaits_you=False)
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_view_read(TABLE)
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, REMARK)
    state = waits(runner.app, 2)
    await runner.gateway.until_turns(1, why="view")

    (opening,) = view_turns(runner)
    assert opening.room["woke"]["app"] == url_of(runner, 0)
    assert opening.room["history"] == history_of(
        [
            view_entry(url_of(runner, 0), {"hand": "none"}, revision=1, awaits_you=False),
            entry(url_of(runner, 0), PLACE_ID, REMARK),
            view_entry(url_of(runner, 0), state, revision=2, new=True),
        ]
    )


async def test_lines_of_two_calls_of_two_apps_are_told_in_the_order_they_were_taken_each_with_its_app_and_place(
    runner, other
):
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    other.stand(SUBJECT, active(OTHER_CALL_ID))
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await other.until_window_read(OTHER_CALL_ID)
    await said_and_taken(runner.app, PLACE_ID, "first")
    await said_and_taken(other, OTHER_CALL_ID, "second")
    await said_and_taken(runner.app, PLACE_ID, "third")
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [
                entry(url_of(runner, 0), PLACE_ID, "first"),
                entry(url_of(runner, 1), OTHER_CALL_ID, "second"),
                entry(url_of(runner, 0), PLACE_ID, "third", seq=2),
                view_entry(url_of(runner, 0), state, revision=1, new=True),
            ]
        )
    ]


async def test_his_own_line_is_told_as_his_own_and_starts_no_turn(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK, author=SUBJECT, name=AGENT_NAME)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [
                entry(url_of(runner, 0), PLACE_ID, REMARK, name=AGENT_NAME, own=True),
                view_entry(url_of(runner, 0), state, revision=1, new=True),
            ]
        )
    ]
    assert len(runner.gateway.openings()) == 1


async def test_a_line_stays_told_after_the_call_it_was_said_in_left_the_standing(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    runner.app.stand(SUBJECT, at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [entry(url_of(runner, 0), PLACE_ID, REMARK), view_entry(url_of(runner, 0), state, revision=1, new=True)]
        )
    ]


async def test_what_was_said_before_the_runner_started_is_not_told_and_what_is_said_after_is_told_at_the_next_revision(
    runner,
):
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, "before the start")
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    first = waits(runner.app, 1)
    runner.start()
    await runner.gateway.until_turns(1, why="view")
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    second = waits(runner.app, 2)
    await runner.gateway.until_turns(2, why="view")

    app_url = url_of(runner, 0)
    assert histories(runner) == [
        history_of([view_entry(app_url, first, revision=1, new=True)]),
        history_of(
            [
                view_entry(app_url, first, revision=1),
                entry(app_url, PLACE_ID, REMARK, seq=2),
                view_entry(app_url, second, revision=2, new=True),
            ]
        ),
    ]


async def test_lines_past_the_bound_leave_the_oldest_out_and_the_history_says_so(runner):
    runner.config = replace(
        runner.config, history_bytes=2 * told_line_bytes(url_of(runner, 0), PLACE_ID, "line-1") + STATE_BYTES
    )
    await restarted_in_the_call(runner)
    for number in (1, 2, 3):
        await said_and_taken(runner.app, PLACE_ID, f"line-{number}")
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [
                entry(url_of(runner, 0), PLACE_ID, "line-2", seq=2),
                entry(url_of(runner, 0), PLACE_ID, "line-3", seq=3),
                view_entry(url_of(runner, 0), state, revision=1, new=True),
            ],
            older_dropped=True,
        )
    ]


async def test_a_turn_is_told_as_many_of_the_newest_lines_as_the_bytes_they_are_told_as_fit_the_bound(runner):
    app = url_of(runner, 0)
    runner.config = replace(
        runner.config,
        history_bytes=told_line_bytes(app, PLACE_ID, "second", seq=2)
        + told_line_bytes(app, PLACE_ID, ADDRESSED, seq=3),
    )
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, "first")
    await said_and_taken(runner.app, PLACE_ID, "second")
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    written = await runner.gateway.until_written_turn(PLACE_ID, 3)

    assert written.room["history"] == history_of(
        [entry(app, PLACE_ID, "second", seq=2), entry(app, PLACE_ID, ADDRESSED, seq=3, new=True)], older_dropped=True
    )


class DownStore:
    """A store that keeps nothing and cannot be read."""

    async def keep(self, read) -> None:
        raise StoreUnavailable(STORE_UNREACHABLE_CODE)

    async def kept_since(self, client_id, since):
        raise StoreUnavailable(STORE_UNREACHABLE_CODE)


@pytest.mark.parametrize(
    ("store", "failures"),
    [(None, 0), (DownStore(), 1)],
    ids=["no memory is configured", "the memory's store is down"],
)
async def test_the_history_is_told_with_no_memory_and_with_a_store_that_is_down_and_no_day_is_told(
    runner, store, failures
):
    if store is not None:
        runner.lines = store
        runner.config = replace(runner.config, memory=memory_target())
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    (opening,) = view_turns(runner)
    assert opening.room["history"] == history_of(
        [entry(url_of(runner, 0), PLACE_ID, REMARK), view_entry(url_of(runner, 0), state, revision=1, new=True)]
    )
    assert "earlier_today" not in opening.room
    assert [code for event, code in runner.reports.events() if event == "store_failed"] == [STORE_UNREACHABLE_CODE] * (
        failures
    )


async def test_a_view_turn_starts_at_once_told_what_the_history_holds_while_a_window_read_of_another_app_is_held(
    runner, other
):
    other.stand(SUBJECT, active(OTHER_CALL_ID))
    await restarted_in_the_call(runner)
    await other.until_window_read(OTHER_CALL_ID)
    release = other.hold_next_window_read()
    other.speak(OTHER_CALL_ID, PERSON, SPEAKER, REMARK)
    await other.until_window_read_held()
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    first = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")
    release.set()
    await settled(other)
    second = waits(runner.app, 2)
    await runner.gateway.until_turns(2, why="view")

    table = url_of(runner, 0)
    assert histories(runner) == [
        history_of([view_entry(table, first, revision=1, new=True)]),
        history_of(
            [
                view_entry(table, first, revision=1),
                entry(url_of(runner, 1), OTHER_CALL_ID, REMARK),
                view_entry(table, second, revision=2, new=True),
            ]
        ),
    ]


async def test_a_line_said_while_the_call_is_paused_is_never_in_the_history(runner):
    await restarted_in_the_call(runner)
    runner.app.stand(SUBJECT, paused(PLACE_ID))
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, REMARK)
    await settled(runner.app)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [history_of([view_entry(url_of(runner, 0), state, revision=1, new=True)])]


async def test_a_new_generation_of_the_call_adds_its_lines_from_the_start_and_keeps_the_earlier_ones(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, "earlier")
    runner.app.join(PLACE_ID)
    await said_and_taken(runner.app, PLACE_ID, "later")
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [
                entry(url_of(runner, 0), PLACE_ID, "earlier"),
                entry(url_of(runner, 0), PLACE_ID, "later"),
                view_entry(url_of(runner, 0), state, revision=1, new=True),
            ]
        )
    ]


async def test_a_line_heard_at_a_voice_place_is_told_to_a_view_turn_of_another_app_and_to_the_spoken_turn(
    runner, other
):
    await a_voice_call(runner, completion("Aloud."), completion("done"))
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)
    other.stand(SUBJECT, at_table())
    waits(other, 1)
    await runner.gateway.until_requested(2)

    (told,) = histories(runner)
    assert (url_of(runner, 0), PLACE_ID, ADDRESSED) in [
        (each["app"], each["place"]["id"], each["text"]) for each in told["entries"] if each["entry"] == "line"
    ]


async def test_a_window_read_again_with_the_same_lines_tells_each_line_once(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    reads = runner.app.window_reads(PLACE_ID)
    await settled(runner.app)
    await runner.app.until_window_read(PLACE_ID, reads + 1)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [entry(url_of(runner, 0), PLACE_ID, REMARK), view_entry(url_of(runner, 0), state, revision=1, new=True)]
        )
    ]


async def test_a_line_said_while_a_view_turn_runs_holds_its_call_once_and_is_told_to_the_next_turn_with_the_recall(
    runner, other
):
    runner.gateway = Gateway(*says_after_a_hold(TABLE.id, ANSWER, tool=PLAY_TOOL))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, at_table())
    await restarted_in_the_call(runner)
    first = waits(other, 1)
    await runner.gateway.until_requested(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    running.set()
    await other.said_count(1)
    await runner.gateway.until_requested(3)
    second = waits(other, 2)
    await runner.gateway.until_turns(2, why="view")

    table = url_of(runner, 1)
    (held,) = json.loads(runner.gateway.tool_results(2)[0])["arrived"]["entries"]
    assert held == entry(url_of(runner, 0), PLACE_ID, REMARK, new=True)
    assert histories(runner) == [
        history_of([view_entry(table, first, revision=1, new=True)]),
        history_of(
            [
                view_entry(table, first, revision=1),
                entry(url_of(runner, 0), PLACE_ID, REMARK),
                act_entry(table, PLAY_TOOL, {"place": TABLE.id, "text": ANSWER}, answered("said")),
                view_entry(table, second, revision=2, new=True),
            ]
        ),
    ]
    assert other.said == [(SUBJECT, TABLE.id, ANSWER)]


async def test_each_agent_is_told_the_history_of_his_own_places_only(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    runner.app.stand(SECOND_SUBJECT, at_table())
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(2, why="view")

    told = {opening.user: opening.room["history"] for opening in view_turns(runner)}
    table = view_entry(url_of(runner, 0), state, revision=1, new=True)
    assert told == {
        person_fingerprint(Principal(iss=ISSUER, sub=SUBJECT)): history_of(
            [entry(url_of(runner, 0), PLACE_ID, REMARK), table]
        ),
        person_fingerprint(Principal(iss=ISSUER, sub=SECOND_SUBJECT)): history_of([table]),
    }


async def test_a_line_that_names_him_starts_his_written_turn_told_it_as_new_and_the_next_view_turn_told_it_too(
    runner, other
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    await runner.app.said_count(1)
    other.stand(SUBJECT, at_table())
    waits(other, 1)
    await runner.gateway.until_turns(2)

    (written,) = written_turns(runner)
    assert written.room == {
        "you": AGENT_NAME,
        "history": history_of([entry(url_of(runner, 0), PLACE_ID, ADDRESSED, new=True)]),
        "woke": woke_at_lines(url_of(runner, 0), PLACE_ID),
    }
    assert written.system == runner.files.persona_prompt
    (told,) = histories(runner)
    assert told["entries"][0] == entry(url_of(runner, 0), PLACE_ID, ADDRESSED)


async def test_lines_said_earlier_in_the_call_are_told_in_the_order_taken_and_only_those_of_the_naming_read_are_new(
    runner,
):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, "first")
    await said_and_taken(runner.app, PLACE_ID, "second")
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    written = await runner.gateway.until_written_turn(PLACE_ID, 3)

    assert written.room["history"] == history_of(
        [
            entry(url_of(runner, 0), PLACE_ID, "first"),
            entry(url_of(runner, 0), PLACE_ID, "second", seq=2),
            entry(url_of(runner, 0), PLACE_ID, ADDRESSED, seq=3, new=True),
        ]
    )


async def test_a_line_that_names_him_in_one_app_is_told_after_the_lines_taken_in_a_call_of_another_app(runner, other):
    other.stand(SUBJECT, active(OTHER_CALL_ID))
    await restarted_in_the_call(runner)
    await other.until_window_read(OTHER_CALL_ID)
    await said_and_taken(other, OTHER_CALL_ID, REMARK)
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    written = await runner.gateway.until_written_turn(PLACE_ID, 1)

    assert written.room["history"] == history_of(
        [
            entry(url_of(runner, 1), OTHER_CALL_ID, REMARK),
            entry(url_of(runner, 0), PLACE_ID, ADDRESSED, new=True),
        ]
    )
    assert written.room["woke"] == woke_at_lines(url_of(runner, 0), PLACE_ID)


async def test_his_own_line_said_before_a_line_that_names_him_is_told_as_his_own_and_started_no_turn(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK, author=SUBJECT, name=AGENT_NAME)
    assert runner.gateway.openings() == []
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    await runner.gateway.until_turns(1, why="lines")

    (written,) = written_turns(runner)
    assert written.room["history"] == history_of(
        [
            entry(url_of(runner, 0), PLACE_ID, REMARK, name=AGENT_NAME, own=True),
            entry(url_of(runner, 0), PLACE_ID, ADDRESSED, seq=2, new=True),
        ]
    )


async def test_two_lines_taken_in_one_read_of_which_one_names_him_are_both_new_and_start_one_turn(runner):
    await restarted_in_the_call(runner)
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, REMARK)
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    await runner.gateway.until_turns(1)
    await settled(runner.app)

    (written,) = runner.gateway.openings()
    assert written.new_seqs == [1, 2]
    assert len(runner.gateway.requests) == 1


async def test_what_was_said_in_the_call_before_the_runner_started_is_told_to_no_turn_without_a_memory(runner):
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, "before the start")
    await restarted_in_the_call(runner)
    runner.app.speak(PLACE_ID, PERSON, SPEAKER, ADDRESSED)
    await runner.gateway.until_turns(1, why="lines")

    (written,) = written_turns(runner)
    assert written.room["history"] == history_of([entry(url_of(runner, 0), PLACE_ID, ADDRESSED, seq=2, new=True)])
    assert "earlier_today" not in written.room


async def test_a_line_taken_ninety_seconds_before_a_turn_was_admitted_is_told_with_that_age(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    runner.clock.time += 90
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [
                entry(url_of(runner, 0), PLACE_ID, REMARK, seconds_ago=90),
                view_entry(url_of(runner, 0), state, revision=1, new=True),
            ]
        )
    ]


async def test_a_window_the_app_lost_enters_no_line_again_and_the_earlier_one_stays_once(runner):
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK)
    runner.app.vanished.add(PLACE_ID)
    reads = runner.app.window_reads(PLACE_ID)
    runner.app.stand(SUBJECT, active(PLACE_ID))
    await runner.app.until_window_read(PLACE_ID, reads + 1)
    runner.app.vanished.discard(PLACE_ID)
    await settled(runner.app)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")

    assert histories(runner) == [
        history_of(
            [entry(url_of(runner, 0), PLACE_ID, REMARK), view_entry(url_of(runner, 0), state, revision=1, new=True)]
        )
    ]


async def test_no_line_state_or_name_of_the_history_reaches_a_log_line(runner, other, caplog):
    name = secrets.token_hex(8)
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, REMARK, name=name)
    other.stand(SUBJECT, at_table())
    state = waits(other, 1)
    await runner.gateway.until_turns(1)

    assert not [value for value in state.values() if value in caplog.text]
    assert name not in caplog.text
    assert REMARK not in caplog.text
