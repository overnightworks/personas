"""The whole runner, the spoken turn: which turns are spoken, what they are told and how their text is spoken."""

from __future__ import annotations

import json

import pytest
from agents.backend.fake_gateway import completion, tool_call, tool_round
from agents.member import CONTINUE_SPEAKING_TOOL, STOP_SPEAKING_TOOL, SpeakRefusal, SpeakRefused, VoiceState
from agents.tools import ToolOutcome
from agents.turn import Principal, person_fingerprint
from agents.voice.reply import REPLY_RULES

from personas.reporting import Event
from tests.support.call import ADDRESSED, ANSWER, PERSON, PLACE_ID, restarted_in_the_call, say_in_the_call
from tests.support.gateway import Gateway, says, speaks_and_says
from tests.support.history import REMARK, SPEAKER, bounded_history, said_and_taken, settled, waits
from tests.support.hopin import PLAY_TOOL, SAY_TOOL, QuietApp
from tests.support.hopin_places import ISSUER, Line
from tests.support.logs import lines_of
from tests.support.realm import LIFETIME_SECONDS
from tests.support.songmaker import CREATE_SONG, GENERATE_SONG
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active
from tests.support.voice import (
    TWO_SENTENCES,
    a_voice_call,
    holding_sentence_ends,
    in_a_voice_call_and_at_a_table,
    sentences_spoken,
)

pytestmark = pytest.mark.anyio

QUIET_PLACE_ID = "room-8"
SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME = "agent-bach", "agent-subject-2", "Bach"
SPEAK = "speak"


@pytest.fixture
async def quiet(runner):
    """A second app the agent listens to, in which it is in a call, that declares no ``speak``."""
    app = QuietApp(line_tool=PLAY_TOOL)
    async with app.serving() as url:
        runner.add_listening_app(app, url, "mcp-quiet")
        app.stand(SUBJECT, active(QUIET_PLACE_ID))
        yield app


def refusing(refusal: SpeakRefusal) -> ToolOutcome:
    return ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True)


async def turns_over(runner, count: int) -> None:
    """Lines are said, one after another, until the turn after the first ``count`` has begun: those have ended."""
    for number in range(count + 1):
        say_in_the_call(runner, f"{ADDRESSED} {number}")
        await runner.gateway.until_turns(number + 1)


@pytest.mark.parametrize(
    ("text", "spoken"),
    [("Anyone there?", True), (ADDRESSED, False)],
    ids=["a spoken line that does not name him", "a written line"],
)
async def test_a_line_at_a_voice_place_makes_a_turn_told_aloud_that_holds_every_apps_tools_and_never_speak(
    runner, songmaker, text, spoken
):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, text, spoken=spoken)
    await runner.app.spoken_count(1)

    (turn,) = runner.gateway.turns()
    assert turn.aloud
    assert turn.woke_seqs == [1]
    assert set(runner.gateway.offered_tools()) == {
        SAY_TOOL,
        CREATE_SONG,
        GENERATE_SONG,
        CONTINUE_SPEAKING_TOOL,
        STOP_SPEAKING_TOOL,
    }


async def test_sentences_released_while_a_reply_is_written_are_sent_in_order_the_next_once_the_one_before_plays(runner):
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion("done"))
    reply = runner.gateway.pause_request(1, "First sentence. Sec", "ond sentence.")
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.spoken_count(1)

    assert not reply.finished
    assert sentences_spoken(runner) == ["First sentence."]
    reply.release()
    await runner.app.until_called(2)
    assert sentences_spoken(runner) == ["First sentence."]
    runner.app.end_sentence()
    await runner.app.spoken_count(2)

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]


@pytest.mark.parametrize("cut", [False, True], ids=["whole", "cut"])
async def test_each_sentence_answered_as_spoken_lets_the_next_out_at_once_never_two_ahead_of_the_one_playing(
    runner, cut
):
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion("First sentence. Second sentence. Third sentence."))
    say_in_the_call(runner, ADDRESSED, spoken=True)

    await runner.app.until_called(2)
    assert sentences_spoken(runner) == ["First sentence."]
    runner.app.end_sentence(cut=cut)
    await runner.app.until_called(3)
    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]
    runner.app.end_sentence(cut=cut)
    await runner.app.spoken_count(3)
    runner.app.end_sentence(cut=cut)
    await runner.reports.until_a_turn_completed()

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence.", "Third sentence."]
    assert runner.app.calls == [SPEAK, SPEAK, SPEAK]
    assert runner.app.most_in_flight == 1
    assert [(each.text, each.spoken, each.cut) for each in runner.app.lines[PLACE_ID] if each.author == SUBJECT] == [
        (text, True, cut) for text in ("First sentence.", "Second sentence.", "Third sentence.")
    ]


async def test_the_next_turn_is_told_each_sentence_and_his_line_for_it_in_order_whole_or_cut_with_the_part_heard(
    runner,
):
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion("First sentence. Second sentence."), completion("done"))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.end_sentence()
    await runner.app.spoken_count(2)
    runner.app.end_sentence(cut=True)
    await runner.reports.until_a_turn_completed()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    second = runner.gateway.turns()[1]
    assert [each["text"] for each in second.told("speech")] == ["First sentence.", "Second sentence."]
    assert [(each["text"], each["spoken"], each.get("cut")) for each in second.told("line") if each["own"]] == [
        ("First sentence.", True, None),
        ("Second sentence.", True, True),
    ]


async def test_a_turn_that_calls_say_and_writes_no_text_writes_the_line_and_speaks_nothing(runner):
    await a_voice_call(runner, *says(PLACE_ID, "Written."))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    assert runner.app.said == [(SUBJECT, PLACE_ID, "Written.")]
    assert runner.app.spoken == []


async def test_a_turn_that_speaks_a_sentence_and_calls_say_does_both_in_the_order_it_wrote_them(runner):
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, "Spoken first.", "Written second."))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    assert runner.app.calls == [SPEAK, SAY_TOOL]
    assert sentences_spoken(runner) == ["Spoken first."]
    assert runner.app.said == [(SUBJECT, PLACE_ID, "Written second.")]


async def test_a_sentence_a_call_and_a_sentence_go_out_in_that_order_and_the_next_turn_is_told_each_as_it_happened(
    runner,
):
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, "Before.", "Written.", after="After."), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(2)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert runner.app.calls == [SPEAK, SAY_TOOL, SPEAK]
    assert runner.gateway.tool_results(2) == ["said"]
    told = runner.gateway.turns()[1].told("speech", "act")
    assert runner.gateway.turns()[1].told_own("speech", "act") == told
    assert [(each["entry"], each.get("text") or each["tool"]) for each in told] == [
        ("speech", "Before."),
        ("act", SAY_TOOL),
        ("speech", "After."),
    ]
    assert [each["outcome"]["answered"] for each in told] == [True, True, True]


@pytest.mark.parametrize(
    "reply", [completion("[silent]"), completion("[quiet]")], ids=["silent", "a marker the reader does not know"]
)
async def test_a_turn_that_stays_silent_says_nothing_relays_nothing_and_logs_nothing(runner, reply):
    await a_voice_call(runner, reply)
    await turns_over(runner, 1)

    assert runner.app.spoken == []
    assert runner.app.said == []
    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_an_aloud_turns_system_text_ends_with_the_reply_rules_and_a_written_turns_lacks_them(runner, quiet):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(1)
    await quiet.until_window_read(QUIET_PLACE_ID)
    quiet.speak(QUIET_PLACE_ID, PERSON, "Ada", ADDRESSED)
    await runner.gateway.until_turns(2)

    aloud, written = runner.gateway.turns()
    assert aloud.aloud
    assert not written.aloud
    assert aloud.system.endswith(REPLY_RULES)
    assert aloud.system.startswith(runner.files.persona_prompt)
    assert REPLY_RULES not in written.system


async def test_a_line_at_an_app_that_declares_no_speak_makes_a_turn_not_told_aloud_and_its_text_is_not_spoken(
    runner, quiet
):
    await a_voice_call(runner, completion("Not spoken."))
    await quiet.until_window_read(QUIET_PLACE_ID)
    quiet.speak(QUIET_PLACE_ID, PERSON, "Ada", ADDRESSED)
    await runner.gateway.until_turns(1)
    quiet.speak(QUIET_PLACE_ID, PERSON, "Ada", f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert not any(turn.aloud for turn in runner.gateway.turns())
    assert runner.app.spoken == []
    assert quiet.said == []


async def test_a_line_for_an_agent_without_a_voice_table_makes_a_turn_not_told_aloud_and_its_text_is_not_spoken(
    runner,
):
    runner.gateway = Gateway(completion("Not spoken."))
    await restarted_in_the_call(runner)
    await turns_over(runner, 1)

    assert not any(turn.aloud for turn in runner.gateway.turns())
    assert runner.app.spoken == []


async def test_a_table_turn_of_a_voice_call_is_not_told_aloud_and_its_text_is_not_spoken(runner):
    await in_a_voice_call_and_at_a_table(runner, completion("Not spoken."))
    waits(runner.app, 1)
    await runner.gateway.until_turns(1, why="view")
    waits(runner.app, 2)
    await runner.gateway.until_turns(2, why="view")

    assert not any(turn.aloud for turn in runner.gateway.turns("view"))
    assert runner.app.spoken == []


@pytest.mark.parametrize(
    "refusal", [refusal for refusal in SpeakRefusal if refusal is not SpeakRefusal.STOPPED], ids=lambda each: each.value
)
async def test_a_refused_sentence_ends_the_turns_speech_but_not_the_turn_and_the_next_turn_is_told_the_refusal(
    runner, refusal
):
    replies = speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written.", after="Muted.")
    await a_voice_call(runner, *replies, completion("Later."))
    runner.app.speak_outcomes.append(refusing(refusal))
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.SPEAK_REFUSED)
    await runner.app.said_count(1)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.app.spoken_count(1)

    assert sentences_spoken(runner) == ["Later."]
    assert runner.reports.reported() == [(Event.STARTED, None), (Event.SPEAK_REFUSED, refusal.value)]
    assert runner.app.said == [(SUBJECT, PLACE_ID, "Written.")]
    (refused,) = runner.gateway.turns()[1].told("speech")
    assert (refused["text"], refused["outcome"]) == ("First sentence.", {"answered": True, "refused": refusal.value})


async def test_a_turn_at_a_place_whose_window_says_his_voice_is_unavailable_is_written_and_a_say_call_goes_through(
    runner,
):
    runner.app.state_voice(PLACE_ID, VoiceState.UNAVAILABLE)
    await a_voice_call(runner, *says(PLACE_ID, "Written."))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.said_count(1)

    (turn,) = runner.gateway.turns()
    assert not turn.aloud
    assert runner.app.said == [(SUBJECT, PLACE_ID, "Written.")]
    assert runner.app.spoken == []


async def test_a_voice_that_turns_unavailable_during_a_spoken_turn_leaves_that_turn_spoken_and_makes_the_next_written(
    runner,
):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(ANSWER), completion("Second answer."))
    say_in_the_call(runner, ADDRESSED)
    await sleeps.until_began(1)
    runner.app.voice_states[PLACE_ID] = VoiceState.UNAVAILABLE
    await said_and_taken(runner.app, PLACE_ID, REMARK, name=SPEAKER)
    runner.app.end_sentence()
    await runner.gateway.until_turns(2)

    first, second = runner.gateway.turns()
    assert (first.aloud, second.aloud) == (True, False)
    assert sentences_spoken(runner) == [ANSWER]


async def test_a_voice_down_at_one_call_writes_only_the_turn_at_that_place(runner):
    other = "room-9"
    runner.app.state_voice(other, VoiceState.UNAVAILABLE)
    await a_voice_call(runner, completion("Hello."), completion("Written."))
    runner.app.stand(SUBJECT, active(PLACE_ID), active(other))
    await runner.app.until_window_read(other)
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(1)
    runner.app.speak(other, PERSON, "Ada", ADDRESSED)
    await runner.gateway.until_turns(2)

    spoken, written = runner.gateway.turns()
    assert (spoken.aloud, written.aloud) == (True, False)
    assert sentences_spoken(runner) == ["Hello."]


async def test_a_sentence_the_app_wrote_to_the_place_instead_is_told_as_written_and_his_speech_goes_on(runner):
    await a_voice_call(runner, completion(TWO_SENTENCES), completion("done"))
    runner.app.written_speaks = 1
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(2)
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert sentences_spoken(runner) == ["First sentence.", "Second sentence."]
    assert runner.reports.reported() == [(Event.STARTED, None)]
    second = runner.gateway.turns()[1]
    assert [(each["text"], each["outcome"]) for each in second.told("speech")] == [
        ("First sentence.", {"answered": True, "written": True}),
        ("Second sentence.", {"answered": True}),
    ]
    assert [(each["text"], each["own"], each["spoken"]) for each in second.told("line") if each["own"]] == [
        ("First sentence.", True, False),
        ("Second sentence.", True, True),
    ]


async def test_a_line_said_while_he_speaks_is_not_in_the_running_turn_and_is_told_to_the_next_one(runner):
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(ANSWER), completion("Second answer."))
    say_in_the_call(runner, ADDRESSED)
    await sleeps.until_began(1)
    await said_and_taken(runner.app, PLACE_ID, REMARK, name=SPEAKER)

    assert runner.gateway.arrivals == 1
    runner.app.end_sentence()
    await runner.gateway.until_turns(2)

    first, second = runner.gateway.turns()
    assert first.woke_seqs == [1]
    assert REMARK in [each["text"] for each in second.own["entries"]]


async def test_a_call_reaches_the_app_only_after_the_sentence_before_it_was_answered_and_his_line_for_it_entered(
    runner,
):
    holding_sentence_ends(runner)
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, ANSWER, "Written."))
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(1)
    answer.set()
    await runner.app.spoken_count(1)

    assert runner.app.calls == [SPEAK]
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert runner.app.calls == [SPEAK, SAY_TOOL]
    assert runner.app.most_in_flight == 1


async def test_his_line_for_the_playing_sentence_that_the_bound_cuts_while_the_next_speak_is_open_is_not_awaited(
    runner,
):
    runner.config = bounded_history(runner.config, 1)
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    first_answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(1)
    first_answer.set()
    await runner.app.spoken_count(1)
    second_answer = runner.app.hold_next_speak()
    await runner.app.until_called(2)
    runner.app.end_sentence()
    await settled(runner.app)
    second_answer.set()
    await runner.app.spoken_count(2)
    runner.app.end_sentence()

    await runner.reports.until_a_turn_completed()

    assert runner.reports.reported() == [(Event.STARTED, None)]


async def test_a_sentence_dropped_by_a_stop_yields_no_line_and_the_turn_does_not_wait_for_one(runner):
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES))
    runner.app.speak_outcomes.append(refusing(SpeakRefusal.STOPPED))
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.SPEAK_REFUSED)
    await runner.reports.until_a_turn_completed()

    assert runner.reports.reported()[:2] == [(Event.STARTED, None), (Event.SPEAK_REFUSED, SpeakRefusal.STOPPED.value)]


async def test_a_sentence_refused_while_the_one_before_plays_holds_the_call_after_it_until_that_ones_line_entered(
    runner,
):
    holding_sentence_ends(runner)
    await a_voice_call(runner, *speaks_and_says(PLACE_ID, TWO_SENTENCES, "Written."))
    runner.app.speak_outcomes.extend([None, refusing(SpeakRefusal.NOT_ACTIVE)])
    say_in_the_call(runner, ADDRESSED)
    await runner.reports.until(Event.SPEAK_REFUSED)

    assert sentences_spoken(runner) == ["First sentence."]
    runner.app.end_sentence()
    await runner.app.said_count(1)

    assert runner.app.calls == [SPEAK, SPEAK, SAY_TOOL]
    assert [(each.text, each.spoken) for each in runner.app.lines[PLACE_ID] if each.author == SUBJECT] == [
        ("First sentence.", True),
        ("Written.", False),
    ]


async def test_a_stop_while_a_sentence_plays_leaves_its_cut_line_awaited_and_the_next_stopped_and_none_sent(
    runner,
):
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion("First sentence. Second sentence. Third sentence."), completion("done"))
    say_in_the_call(runner, ADDRESSED, spoken=True)
    await runner.app.until_called(2)
    runner.app.stop_speaking()
    await runner.reports.until_a_turn_completed()
    say_in_the_call(runner, f"{ADDRESSED} Again.")
    await runner.gateway.until_turns(2)

    assert runner.app.calls == [SPEAK, SPEAK]
    assert runner.reports.reported()[:2] == [(Event.STARTED, None), (Event.SPEAK_REFUSED, SpeakRefusal.STOPPED.value)]
    second = runner.gateway.turns()[1]
    assert [(each["text"], each["outcome"]) for each in second.told("speech")] == [
        ("First sentence.", {"answered": True}),
        ("Second sentence.", {"answered": True, "refused": SpeakRefusal.STOPPED.value}),
    ]
    assert [(each["text"], each.get("cut")) for each in second.told("line") if each["own"]] == [
        ("First sentence.", True)
    ]


async def test_a_turn_that_outlives_its_token_speaks_its_next_sentence_with_a_renewed_one(runner):
    holding_sentence_ends(runner)
    await a_voice_call(runner, completion(TWO_SENTENCES), max_duration_seconds=2 * LIFETIME_SECONDS)
    answer = runner.app.hold_next_speak()
    say_in_the_call(runner, ADDRESSED)
    await runner.app.until_called(1)
    runner.clock.time += LIFETIME_SECONDS
    answer.set()
    await runner.app.until_called(2)
    runner.app.end_sentence()
    await runner.app.spoken_count(2)

    first, second = runner.app.speak_bearers
    assert first != second
    assert second == runner.realm.issued[-1]


async def test_two_agents_in_one_place_make_no_turn_of_their_own_spoken_lines_and_one_of_the_others(runner):
    runner.add_agent(SECOND_AGENT, SECOND_SUBJECT, SECOND_NAME)
    runner.app.stand(SECOND_SUBJECT, active(PLACE_ID))
    await a_voice_call(runner, completion("[silent]"))
    mozart = person_fingerprint(runner.app_principal)
    bach = person_fingerprint(Principal(iss=ISSUER, sub=SECOND_SUBJECT))
    runner.gateway.script_for(mozart, completion("Mozart answers."), completion("[silent]"))
    runner.gateway.script_for(bach, completion("Bach answers."), completion("[silent]"))
    say_in_the_call(runner, "Both of you, hello.")
    await runner.gateway.until_requested(4)
    await said_and_taken(runner.app, PLACE_ID, "Goodbye.")
    await runner.gateway.until_requested(6)

    mozarts, bachs = ([turn for turn in runner.gateway.turns() if turn.user == user] for user in (mozart, bach))
    assert (len(mozarts), len(bachs)) == (3, 3)
    assert "Mozart answers." in [each["text"] for each in bachs[1].own["entries"]]
    assert "Bach answers." in [each["text"] for each in mozarts[1].own["entries"]]


async def test_a_window_that_lost_older_lines_makes_a_turn_told_it_missed_lines(runner, monkeypatch):
    monkeypatch.setattr("tests.support.hopin_places.WINDOW_LINES", 1)
    sleeps = holding_sentence_ends(runner)
    await a_voice_call(runner, completion(ANSWER), completion("Second answer."))
    say_in_the_call(runner, ADDRESSED)
    await sleeps.until_began(1)
    runner.app.lines[PLACE_ID].extend(Line(seq, PERSON, "Ada", f"Line {seq}") for seq in (3, 4))
    await said_and_taken(runner.app, PLACE_ID, f"{ADDRESSED} Again.", author=PERSON, name="Ada")
    runner.app.end_sentence()
    await runner.gateway.until_turns(2)

    first, second = runner.gateway.turns()
    assert first.missed is False
    assert second.missed is True


async def test_a_line_heard_while_he_speaks_holds_the_call_after_the_sentence_once_and_the_model_is_told_the_line(
    runner,
):
    sleeps = holding_sentence_ends(runner)
    written = {"place": PLACE_ID, "text": "Written."}
    first, again = tool_call("call-0", SAY_TOOL, written), tool_call("call-1", SAY_TOOL, written)
    await a_voice_call(runner, tool_round(first, content=ANSWER), tool_round(again), completion("done"))
    say_in_the_call(runner, ADDRESSED)
    await sleeps.until_began(1)
    await said_and_taken(runner.app, PLACE_ID, "Anyone there?", spoken=True)
    runner.app.end_sentence()
    await runner.app.said_count(1)
    await runner.gateway.until_answered(3)

    (held,) = (json.loads(each) for each in runner.gateway.tool_results(3) if each.startswith('{"sent": false'))
    arrived = held["arrived"]["entries"]
    assert [(each["text"], each["spoken"]) for each in arrived if each["entry"] == "line" and not each["own"]] == [
        ("Anyone there?", True)
    ]
    assert [each["text"] for each in arrived if each["entry"] == "speech"] == [ANSWER]
    assert runner.app.said == [(SUBJECT, PLACE_ID, "Written.")]
    assert sentences_spoken(runner) == [ANSWER]


async def test_the_next_turn_is_told_which_lines_were_said_aloud_and_that_his_own_sentence_was_one(runner):
    await a_voice_call(runner, completion(ANSWER), completion("done"))
    say_in_the_call(runner, "Anyone there?", spoken=True)
    await runner.app.spoken_count(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(2)

    second = runner.gateway.turns()[1]
    assert [(each["text"], each["own"], each["spoken"]) for each in second.told("line")] == [
        ("Anyone there?", False, True),
        (ANSWER, True, True),
        (ADDRESSED, False, False),
    ]
    assert [each["text"] for each in second.told("speech")] == [ANSWER]


async def test_the_next_turn_is_told_a_line_the_app_marked_as_cut_as_cut_with_the_part_heard(runner):
    await a_voice_call(runner, completion("[silent]"), completion("done"))
    say_in_the_call(runner, "Anyone th", spoken=True, cut=True)
    await runner.gateway.until_turns(1)
    say_in_the_call(runner, ADDRESSED)
    await runner.gateway.until_turns(2)

    lines = [each for each in runner.gateway.turns()[1].told("line") if not each["own"]]
    assert [(each["text"], each["spoken"], each.get("cut")) for each in lines] == [
        ("Anyone th", True, True),
        (ADDRESSED, False, None),
    ]


async def test_a_spoken_turn_logs_its_start_and_no_sentence(runner, caplog):
    await a_voice_call(runner, completion(ANSWER))
    say_in_the_call(runner, ADDRESSED)
    await runner.app.spoken_count(1)

    events = [line["event"] for line in lines_of(caplog.get_records("call"))]
    assert [event for event in events if not event.endswith(".ended")] == ["loop.started"]
