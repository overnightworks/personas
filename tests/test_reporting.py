"""What a state change writes as a log line."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agents.turn import FailureCode, RejectionCode, TurnUsage
from logline.testing import configured

from personas.decision import TURN_CANCELLED_CODE
from personas.entries import InputSizes
from personas.failures import StatusFailure, Unreachable
from personas.reporting import Event, StateChange, log_state_change

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
    line = written(change)

    assert line["event"] == event
    assert {key: line[key] for key in fields} == fields
    assert (line["agent_id"], line["app_id"]) == (AGENT, "hopin.example")
    assert line["service"] == "personas"
    assert "hidden" not in json.dumps(line)


def test_the_sweep_belongs_to_no_agent_and_no_app_and_says_its_line_count(written):
    line = written(StateChange(Event.SWEPT, None, None, lines=7))

    assert (line["event"], line["line_count"]) == ("lines.swept", 7)
    assert not {"agent_id", "app_id"} & line.keys()


def test_a_turn_that_used_tokens_and_sent_a_request_says_its_counts_and_bytes(written):
    usage = TurnUsage(input_tokens=250, cached_input_tokens=140, cache_write_tokens=10, output_tokens=8, requests=2)

    line = written(of(Event.TURN_ENDED, usage=usage, sizes=InputSizes(1, 2, 3, 4)))

    assert line["outcome"] == "ok"
    assert {key: line[key] for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens")} == {
        "input_tokens": 250,
        "cached_input_tokens": 140,
        "cache_write_tokens": 10,
    }
    assert (line["output_tokens"], line["model_request_count"]) == (8, 2)
    assert [line[key] for key in ("tools_bytes", "told_bytes", "results_bytes", "images_bytes")] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("code", "outcome", "level"),
    [
        (None, "ok", "info"),
        (TURN_CANCELLED_CODE, "cancelled", "info"),
        (RejectionCode.BUDGET_SPENT.value, "refused", "info"),
        (FailureCode.PROVIDER_ERROR.value, "failed", "error"),
    ],
    ids=["answered", "cancelled", "refused by the gateway", "failed"],
)
def test_a_turn_ends_with_the_outcome_its_code_means_and_an_error_only_when_it_failed(written, code, outcome, level):
    line = written(of(Event.TURN_ENDED, code))

    assert (line["outcome"], line["level"]) == (outcome, level)
    assert line.get("reason") == code
