"""The whole runner, the table: every view revision an agent reads is an entry of his history, in the order read."""

from __future__ import annotations

from dataclasses import replace

import pytest
from agents.backend.fake_gateway import completion
from agents.turn import Principal, person_fingerprint

from tests.support.call import ADDRESSED, ANSWER, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says
from tests.support.history import (
    REMARK,
    SPEAKER,
    STATE_BYTES,
    TABLE,
    a_state,
    at_table,
    entry,
    history_of,
    said_and_taken,
    settled,
    shows,
    url_of,
    view_entry,
    view_turns,
    waits,
    written_turns,
)
from tests.support.hopin import AGENT_SUBJECT as SUBJECT
from tests.support.hopin import ISSUER, READ, active, paused
from tests.support.realm import AGENT_NAME

pytestmark = pytest.mark.anyio

SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME = "agent-bach", "agent-subject-2", "Bach"


async def seated_in_the_call_and_at_the_table(runner, *, state=None, revision: int = 1) -> dict[str, str]:
    """The runner started with a call and a table whose first view does not wait, read once."""
    state = state or a_state()
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    runner.app.show_view(TABLE.id, revision, state, awaits_you=False)
    runner.start()
    await runner.app.until_window_read(PLACE_ID)
    await runner.app.until_view_read(TABLE)
    await settled(runner.app)
    return state


async def test_a_table_that_does_not_wait_is_told_to_his_written_turn_with_its_age_and_known_from_the_start(
    runner,
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    state = await seated_in_the_call_and_at_the_table(runner)
    runner.clock.time += 90
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)

    (written,) = runner.gateway.openings()
    app = url_of(runner, 0)
    assert written.room["history"] == history_of(
        [
            view_entry(app, state, revision=1, awaits_you=False, seconds_ago=90),
            entry(app, PLACE_ID, ADDRESSED, new=True),
        ]
    )
    assert written.new_seqs == [1]
    assert "earlier_today" not in written.room
    assert view_turns(runner) == []


async def test_a_view_turn_is_told_the_state_once_in_its_entry_marked_new_and_woke_names_only_why_app_and_place(runner):
    runner.app.stand(SUBJECT, at_table())
    state = waits(runner.app, 1)
    runner.start()
    await runner.gateway.until_turns(1)

    (opening,) = view_turns(runner)
    assert opening.room["woke"] == {
        "why": "view",
        "app": url_of(runner, 0),
        "place": {"kind": TABLE.kind, "id": TABLE.id},
    }
    assert [each["new"] for each in opening.room["history"]["entries"]] == [True]
    assert runner.gateway.told_in(1).count(state["hand"]) == 1


async def test_a_revision_read_in_several_wakes_is_one_entry_aged_from_its_first_read(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    state = await seated_in_the_call_and_at_the_table(runner)
    runner.clock.time += 30
    await shows(runner.app, 1, state)
    runner.clock.time += 60
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)

    (written,) = runner.gateway.openings()
    assert [each["entry"] for each in written.room["history"]["entries"]] == ["view", "line"]
    assert written.room["history"]["entries"][0]["seconds_ago"] == 90


async def test_a_revision_read_again_with_another_state_keeps_the_state_of_its_first_read(runner):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    first = await seated_in_the_call_and_at_the_table(runner)
    await shows(runner.app, 1, a_state())
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)

    (written,) = runner.gateway.openings()
    assert [each["state"] for each in written.room["history"]["entries"] if each["entry"] == "view"] == [first]


async def test_revisions_and_lines_are_told_in_the_order_read_and_only_the_waiting_revision_is_new(runner):
    await restarted_in_the_call(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    first = await shows(runner.app, 1)
    await said_and_taken(runner.app, PLACE_ID, "first")
    second = await shows(runner.app, 2)
    await said_and_taken(runner.app, PLACE_ID, "second")
    third = waits(runner.app, 3)
    await runner.gateway.until_turns(1)

    app = url_of(runner, 0)
    (opening,) = view_turns(runner)
    assert opening.room["history"] == history_of(
        [
            view_entry(app, first, revision=1, awaits_you=False),
            entry(app, PLACE_ID, "first"),
            view_entry(app, second, revision=2, awaits_you=False),
            entry(app, PLACE_ID, "second", seq=2),
            view_entry(app, third, revision=3, new=True),
        ]
    )


async def test_a_revision_asked_again_by_the_app_with_the_same_state_is_a_second_turn_and_a_second_entry(runner):
    runner.gateway = Gateway(completion("done"), completion("done"))
    runner.app.stand(SUBJECT, at_table())
    state = waits(runner.app, 5)
    runner.start()
    await runner.gateway.until_turns(1)
    runner.app.show_view(TABLE.id, 6, state)
    await runner.gateway.until_turns(2)

    app = url_of(runner, 0)
    assert [each["history"] for each in (opening.room for opening in view_turns(runner))][1] == history_of(
        [view_entry(app, state, revision=5), view_entry(app, state, revision=6, new=True)]
    )


async def test_two_tables_of_two_apps_enter_in_the_order_read_each_with_its_app_and_place(runner, other):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    other.stand(SUBJECT, at_table())
    runner.app.show_view(TABLE.id, 1, first := a_state(), awaits_you=False)
    runner.start()
    await runner.app.until_view_read(TABLE)
    second = await shows(other, 1)
    third = await shows(runner.app, 2)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)

    (written,) = runner.gateway.openings()
    assert [(each["app"], each["revision"], each["state"]) for each in written.room["history"]["entries"][:3]] == [
        (url_of(runner, 0), 1, first),
        (url_of(runner, 1), 1, second),
        (url_of(runner, 0), 2, third),
    ]


@pytest.mark.parametrize("earlier_line", [True, False], ids=["a line was said before", "nothing was said before"])
async def test_a_view_turn_is_told_its_state_alone_when_the_bound_is_below_one_state_and_says_what_left(
    runner, earlier_line
):
    runner.config = replace(runner.config, history_bytes=STATE_BYTES - 1)
    await restarted_in_the_call(runner)
    if earlier_line:
        await said_and_taken(runner.app, PLACE_ID, REMARK)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    state = waits(runner.app, 1)
    await runner.gateway.until_turns(1)

    assert [opening.room["history"] for opening in view_turns(runner)] == [
        history_of([view_entry(url_of(runner, 0), state, revision=1, new=True)], older_dropped=earlier_line)
    ]


@pytest.mark.parametrize("earlier_line", [True, False], ids=["an earlier line was said", "no earlier entry"])
async def test_a_line_that_names_him_and_alone_exceeds_the_bound_is_told_alone_and_new_and_says_what_left(
    runner, earlier_line
):
    runner.config = replace(runner.config, history_bytes=len(SPEAKER + ADDRESSED) - 1)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    if earlier_line:
        await said_and_taken(runner.app, PLACE_ID, REMARK)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)

    (written,) = runner.gateway.openings()
    seq = 2 if earlier_line else 1
    assert written.room["history"] == history_of(
        [entry(url_of(runner, 0), PLACE_ID, ADDRESSED, seq=seq, new=True)], older_dropped=earlier_line
    )


async def test_a_place_that_leaves_and_returns_at_its_held_waiting_revision_gets_a_second_turn_with_one_entry(runner):
    runner.app.stand(SUBJECT, at_table())
    state = waits(runner.app, 1)
    runner.start()
    await runner.gateway.until_turns(1)
    await settled(runner.app)
    runner.app.stand(SUBJECT)
    await settled(runner.app)
    runner.app.stand(SUBJECT, at_table())
    await runner.gateway.until_turns(2)

    assert [opening.room["history"] for opening in view_turns(runner)] == [
        history_of([view_entry(url_of(runner, 0), state, revision=1, new=True)])
    ] * 2


async def test_a_table_that_serves_no_view_for_a_while_is_not_read_and_its_earlier_entries_are_told_with_their_age(
    runner,
):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER), completion("done"))
    first = await seated_in_the_call_and_at_the_table(runner)
    runner.app.stand(SUBJECT, active(PLACE_ID), active(TABLE.id, TABLE.kind))
    await settled(runner.app)
    reads = runner.app.view_reads(TABLE)
    runner.clock.time += 20
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)
    assert runner.app.view_reads(TABLE) == reads
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    third = waits(runner.app, 2)
    await runner.gateway.until_turns(2)

    app = url_of(runner, 0)
    assert runner.gateway.openings()[0].room["history"]["entries"][0] == view_entry(
        app, first, revision=1, awaits_you=False, seconds_ago=20
    )
    assert [each["entry"] for each in view_turns(runner)[0].room["history"]["entries"]] == [
        "view",
        "line",
        "act",
        "line",
        "view",
    ]
    assert view_turns(runner)[0].room["history"]["entries"][-1] == view_entry(app, third, revision=2, new=True)


async def test_a_view_he_cannot_read_enters_nothing_and_the_earlier_entries_stay_until_it_is_readable_again(runner):
    runner.app.stand(SUBJECT, at_table())
    first = await shows_at_start(runner)
    runner.app.vanished.add(TABLE.id)
    reads = runner.app.view_reads(TABLE)
    runner.app.show_view(TABLE.id, 2, a_state(), awaits_you=False)
    await runner.app.until_view_read(TABLE, reads + 1)
    runner.app.vanished.discard(TABLE.id)
    third = waits(runner.app, 3)
    await runner.gateway.until_turns(1)

    app = url_of(runner, 0)
    assert [opening.room["history"] for opening in view_turns(runner)] == [
        history_of(
            [
                view_entry(app, first, revision=1, awaits_you=False),
                view_entry(app, third, revision=3, new=True),
            ]
        )
    ]


async def shows_at_start(runner) -> dict[str, str]:
    state = a_state()
    runner.app.show_view(TABLE.id, 1, state, awaits_you=False)
    runner.start()
    await runner.app.until_view_read(TABLE)
    await settled(runner.app)
    return state


async def test_a_view_read_that_fails_enters_nothing_and_the_read_after_the_resume_enters_once(runner):
    async def cuts(reply):
        await reply.open_stream()
        reply.cut()

    runner.app.stand(SUBJECT, at_table())
    first = await shows_at_start(runner)
    reads = runner.app.view_reads(TABLE)
    runner.app.script(READ, cuts)
    second = a_state()
    runner.app.show_view(TABLE.id, 2, second, awaits_you=False)
    await runner.reports.until("resumed")
    await runner.app.until_view_read(TABLE, reads + 1)
    await settled(runner.app)
    third = waits(runner.app, 3)
    await runner.gateway.until_turns(1)

    (opening,) = view_turns(runner)
    assert [(each["revision"], each["state"], each["new"]) for each in opening.room["history"]["entries"]] == [
        (1, first, False),
        (2, second, False),
        (3, third, True),
    ]


async def test_a_table_in_a_paused_place_enters_nothing_and_the_first_read_after_the_resume_enters(runner):
    runner.app.stand(SUBJECT, paused(TABLE.id, TABLE.kind, serves_view=True))
    first = a_state()
    runner.app.show_view(TABLE.id, 1, first, awaits_you=False)
    runner.start()
    await runner.app.until_standing_read(1)
    await settled(runner.app)
    assert runner.app.view_reads(TABLE) == 0
    runner.app.stand(SUBJECT, at_table())
    second = waits(runner.app, 2)
    await runner.gateway.until_turns(1)

    assert view_turns(runner)[0].room["history"]["entries"][-1] == view_entry(
        url_of(runner, 0), second, revision=2, new=True
    )
    assert runner.app.view_reads(TABLE) > 0


async def test_each_agent_is_told_only_the_tables_of_his_own_places_in_his_written_turn(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    runner.gateway = Gateway(completion("done"), completion("done"))
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    state = a_state()
    runner.app.show_view(TABLE.id, 1, state, awaits_you=False)
    runner.start()
    await runner.app.until_window_read(PLACE_ID, 2)
    await runner.app.until_view_read(TABLE)
    await settled(runner.app)
    runner.app.speak(PLACE_ID, "visitor-subject", "Ada", f"{AGENT_NAME} and {SECOND_NAME}, are you there?")
    await runner.gateway.until_turns(2)

    told = {opening.user: opening.room["history"]["entries"] for opening in written_turns(runner)}
    app = url_of(runner, 0)
    assert [each["entry"] for each in told[person_fingerprint(Principal(iss=ISSUER, sub=SUBJECT))]] == ["view", "line"]
    assert told[person_fingerprint(Principal(iss=ISSUER, sub=SECOND_SUBJECT))] == [
        entry(app, PLACE_ID, f"{AGENT_NAME} and {SECOND_NAME}, are you there?", new=True)
    ]


async def test_without_a_memory_the_table_displaces_the_early_call_lines_and_they_reach_no_part_of_the_message(runner):
    early = "early-call-line"
    runner.config = replace(runner.config, history_bytes=len(SPEAKER + ADDRESSED) + 2 * STATE_BYTES)
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER))
    await restarted_in_the_call(runner)
    await said_and_taken(runner.app, PLACE_ID, early)
    runner.app.stand(SUBJECT, active(PLACE_ID), at_table())
    await shows(runner.app, 1)
    second = await shows(runner.app, 2)
    third = await shows(runner.app, 3)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(1)

    (written,) = runner.gateway.openings()
    app = url_of(runner, 0)
    assert written.room["history"] == history_of(
        [
            view_entry(app, second, revision=2, awaits_you=False),
            view_entry(app, third, revision=3, awaits_you=False),
            entry(app, PLACE_ID, ADDRESSED, seq=2, new=True),
        ],
        older_dropped=True,
    )
    assert early not in runner.gateway.told_in(1)
    assert "earlier_today" not in written.room


async def test_a_table_that_moves_during_his_view_turn_enters_at_the_first_read_after_that_turn(runner):
    runner.gateway = Gateway(completion("done"), completion("done"))
    ending = runner.gateway.hold_request(1)
    runner.app.stand(SUBJECT, at_table())
    first = waits(runner.app, 1)
    runner.start()
    await runner.gateway.until_requested(1)
    second = waits(runner.app, 2)
    ending.set()
    await runner.gateway.until_turns(2)

    app = url_of(runner, 0)
    assert [opening.room["history"] for opening in view_turns(runner)] == [
        history_of([view_entry(app, first, revision=1, new=True)]),
        history_of([view_entry(app, first, revision=1), view_entry(app, second, revision=2, new=True)]),
    ]


async def test_a_table_that_moves_during_a_call_turn_is_not_in_that_turns_input_and_is_in_the_next(runner, other):
    runner.gateway = Gateway(*says(PLACE_ID, ANSWER, ANSWER))
    running = runner.gateway.hold_request(1)
    other.stand(SUBJECT, at_table())
    other.show_view(TABLE.id, 1, first := a_state(), awaits_you=False)
    await restarted_in_the_call(runner)
    await other.until_view_read(TABLE)
    await settled(other)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_requested(1)
    second = await shows(other, 2)
    running.set()
    await runner.app.said_count(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(2)

    table = url_of(runner, 1)
    during, after = (opening.room["history"]["entries"] for opening in written_turns(runner))
    assert [each["state"] for each in during if each["entry"] == "view"] == [first]
    assert [each["state"] for each in after if each["entry"] == "view"] == [first, second]
    assert after[-1] == entry(url_of(runner, 0), PLACE_ID, ADDRESSED, seq=3, new=True)
    assert all(each["app"] == table for each in after if each["entry"] == "view")
