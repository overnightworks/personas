"""The entry point's logging: only the runner's own state changes reach a log line."""

from __future__ import annotations

import logging

import pytest

from agents.main import QUIET_LIBRARIES, configure_logging


@pytest.mark.parametrize("library", QUIET_LIBRARIES)
def test_an_error_record_of_a_library_the_runner_uses_reaches_no_log_line(
    library, library_loggers, caplog: pytest.LogCaptureFixture
):
    configure_logging()
    caplog.set_level(logging.DEBUG)

    logging.getLogger(f"{library}.client.streamable_http").error("a response body")

    assert caplog.records == []
