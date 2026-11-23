"""The runner's log lines as a test reads them: from the records a capture holds or the text the entry point wrote."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from personas.main import configure_logging

if TYPE_CHECKING:
    import io

    import pytest


_RECORD_ATTRIBUTES = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def fields_of(record: logging.LogRecord) -> dict[str, object]:
    """The fields a record carries beside its message, as the JSON line holds them."""
    return {key: value for key, value in vars(record).items() if key not in _RECORD_ATTRIBUTES}


class _FormatterWithFields(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{super().format(record)} {json.dumps(fields_of(record), default=str, sort_keys=True)}"


def configure_logging_into(caplog: pytest.LogCaptureFixture) -> None:
    """The entry point's logging, with the lines of every logger also in ``caplog`` and its fields in their text."""
    configure_logging()
    caplog.handler.setFormatter(_FormatterWithFields())
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)


def formatted(records: list[logging.LogRecord]) -> str:
    """The records as a handler writes them, with the fields and the traceback of each."""
    formatter = _FormatterWithFields()
    return "\n".join(formatter.format(record) for record in records)


def lines_of(records: list[logging.LogRecord], event: str | None = None) -> list[dict[str, object]]:
    """The runner's lines among ``records`` as their JSON holds them, only those of ``event`` when it is given."""
    lines = [{"event": record.getMessage(), **fields_of(record)} for record in records if record.name == "personas"]
    return lines if event is None else [line for line in lines if line["event"] == event]


def written_by_the_runner(written: io.StringIO) -> list[dict[str, object]]:
    """The runner's lines among what the entry point's handler wrote into ``written``, as their JSON holds them."""
    lines = map(json.loads, written.getvalue().splitlines())
    return [line for line in lines if line["logger"] == "personas"]


def trace_of(traceparent: str) -> str:
    """The trace id of a W3C traceparent; a request made inside a unit has a span of its own."""
    return traceparent.split("-")[1]
