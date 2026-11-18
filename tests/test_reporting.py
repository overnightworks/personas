"""What a state change writes as a log line."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agents.turn import FailureCode, RejectionCode, TurnUsage
from logline import Outcome
from logline.testing import configured

from personas.decision import TurnEnding
from personas.entries import InputSizes
from personas.failures import StatusFailure, Unreachable
from personas.reporting import Event, StateChange, log_state_change, start_turn_tally

if TYPE_CHECKING:
    from collections.abc import Callable

AGENT = "mozart"
APP_URL = "https://hopin.example:8443/mcp?token=hidden"


@pytest.fixture
def written(capsys: pytest.CaptureFixture[str]) -> Callable[[StateChange], dict[str, object]]:
    """Reports a change under the entry point's logging and returns the one JSON line it wrote."""

    def report(change: StateChange) -> dict[str, object]:
        with configured("personas"):
            log_state_change(change)
        return json.loads(capsys.readouterr().out)

    return report


def of(event: Event, code: str | None = None, **fields: object) -> StateChange:
    return StateChange(event, AGENT, APP_URL, code, **fields)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("change", "event", "fields"),
    [
        pytest.param(of(Event.STARTED, model="text/sonnet"), "loop.started", {"model": "text/sonnet"}, id="started"),
        pytest.param(of(Event.RESUMED), "loop.resumed", {}, id="resumed"),
        pytest.param(of(Event.JOINED), "invitation.accepted", {}, id="joined"),
        pytest.param(of(Event.REFUSED, "stale"), "invitation.refused", {"reason": "stale"}, id="refused"),
        pytest.param(
            of(Event.INVITATION_LEFT_OUT, "no_turns", place_kind="table"),
            "invitation.left_out",
            {"reason": "no_turns", "place_kind": "table"},
            id="left out",
        ),
        pytest.param(
            of(Event.PAUSED, "budget_spent", seconds=1.5),
            "turns.paused",
            {"reason": "budget_spent", "pause_ms": 1500},
            id="paused",
        ),
        pytest.param(of(Event.SPEAK_REFUSED, "busy"), "speak.refused", {"reason": "busy"}, id="speak refused"),
        pytest.param(of(Event.STORE_FAILED, "store_timeout"), "store.failed", {"reason": "store_timeout"}, id="store"),
        pytest.param(of(Event.STORE_RESUMED), "store.resumed", {}, id="store resumed"),
        pytest.param(
            of(Event.BACKOFF, failure=Unreachable("down")),
            "loop.backed_off",
            {"reason": "network"},
            id="backed off by the network",
        ),
        pytest.param(
            of(Event.SUSPENDED, failure=StatusFailure(403, "tools/list", "application/json")),
            "loop.suspended",
            {"status": 403, "mcp_method": "tools/list", "reason": "json"},
            id="suspended by a status",
        ),
        pytest.param(
            of(Event.HELD_APP_FAILED, failure=StatusFailure(500, "tools/call")),
            "held_app.failed",
            {"status": 500, "mcp_method": "tools/call", "reason": "no_answer"},
            id="held app failed",
        ),
        pytest.param(
            of(Event.TOOLS_ONLY_APP_FAILED, failure=StatusFailure(400, "tools/list", "text/x-private-front")),
            "tools_only_app.failed",
            {"status": 400, "mcp_method": "tools/list", "reason": "other_media_type"},
            id="an unknown media type is one fixed word",
        ),
    ],
)
def test_a_change_is_one_literal_event_with_ids_the_host_of_the_app_and_closed_codes(written, change, event, fields):
    """LOG-2, LOG-6, LOG-7, LOG-8, LOG-9, LOG-10, LOG-11, LOG-15, LOG-16, LOG-19, LOG-20."""
    line = written(change)

    assert line["event"] == event
    assert {key: line[key] for key in fields} == fields
    assert (line["agent_id"], line["app_id"]) == (AGENT, "hopin.example")
    assert line["service"] == "personas"
    assert "hidden" not in json.dumps(line)


def test_the_sweep_belongs_to_no_agent_and_no_app_and_says_its_line_count(written):
    """LOG-21."""
    line = written(StateChange(Event.SWEPT, None, None, lines=7))

    assert (line["event"], line["line_count"]) == ("lines.swept", 7)
    assert not {"agent_id", "app_id"} & line.keys()


@pytest.fixture
def turn_line(capsys: pytest.CaptureFixture[str]) -> Callable[..., dict[str, object]]:
    """Ends a turn's tally under the entry point's logging and returns the one JSON line it wrote."""

    def end(ending: TurnEnding, **counted: object) -> dict[str, object]:
        with configured("personas"):
            tally = start_turn_tally(AGENT, APP_URL)
            for name, value in counted.items():
                setattr(tally, name, value)
            tally.end(ending)
        return json.loads(capsys.readouterr().out)

    return end


def test_a_turn_says_its_counts_tokens_and_input_bytes_beside_its_duration(turn_line):
    usage = TurnUsage(input_tokens=250, cached_input_tokens=140, cache_write_tokens=10, output_tokens=8, requests=2)

    line = turn_line(
        TurnEnding(Outcome.OK),
        usage=usage,
        sizes=InputSizes(1, 2, 3, 4),
        sentence_count=3,
        silent_count=1,
        tool_call_count=2,
    )

    assert (line["event"], line["outcome"], line["agent_id"], line["app_id"]) == (
        "turn.ended",
        "ok",
        AGENT,
        "hopin.example",
    )
    assert isinstance(line["duration_ms"], float)
    assert (line["sentence_count"], line["silent_count"], line["tool_call_count"]) == (3, 1, 2)
    assert {key: line[key] for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens")} == {
        "input_tokens": 250,
        "cached_input_tokens": 140,
        "cache_write_tokens": 10,
    }
    assert (line["output_tokens"], line["model_request_count"]) == (8, 2)
    assert [line[key] for key in ("tools_bytes", "told_bytes", "results_bytes", "images_bytes")] == [1, 2, 3, 4]


def test_a_turn_that_sent_nothing_says_zero_counts_and_no_first_output_tokens_or_sizes(turn_line):
    line = turn_line(TurnEnding(Outcome.OK))

    assert (line["sentence_count"], line["silent_count"], line["tool_call_count"]) == (0, 0, 0)
    assert not {"first_output_ms", "input_tokens", "tools_bytes"} & line.keys()


def test_the_first_sentence_sent_marks_the_first_output(capsys):
    with configured("personas"):
        tally = start_turn_tally(AGENT, APP_URL)
        tally.sentence_sent()
        tally.sentence_sent()
        tally.end(TurnEnding(Outcome.OK))
    line = json.loads(capsys.readouterr().out)

    assert (line["sentence_count"], isinstance(line["first_output_ms"], float)) == (2, True)


@pytest.mark.parametrize(
    ("ending", "outcome", "level", "reason"),
    [
        (TurnEnding(Outcome.OK), "ok", "info", None),
        (TurnEnding(Outcome.CANCELLED), "cancelled", "info", None),
        (TurnEnding(Outcome.REFUSED, RejectionCode.BUDGET_SPENT.value), "refused", "info", "budget_spent"),
        (TurnEnding(Outcome.FAILED, FailureCode.PROVIDER_ERROR.value), "failed", "error", "provider_error"),
    ],
    ids=["answered", "cancelled", "refused by the gateway", "failed"],
)
def test_a_turn_ends_with_its_outcome_its_reason_when_it_has_one_and_an_error_only_when_it_failed(
    turn_line, ending, outcome, level, reason
):
    """LOG-12, LOG-28."""
    line = turn_line(ending)

    assert (line["outcome"], line["level"], line.get("reason")) == (outcome, level, reason)
