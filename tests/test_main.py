"""The entry point's logging: only the runner's own state changes reach a log line."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from agents.config import CONFIG_ENVIRONMENT_VARIABLE
from agents.main import QUIET_LIBRARIES, READINESS_ERROR_EXIT_CODE, configure_logging, main

if TYPE_CHECKING:
    from tests.support.configfiles import Files


@pytest.mark.parametrize("library", QUIET_LIBRARIES)
def test_an_error_record_of_a_library_the_runner_uses_reaches_no_log_line(
    library, library_loggers, caplog: pytest.LogCaptureFixture
):
    configure_logging()
    caplog.set_level(logging.DEBUG)

    logging.getLogger(f"{library}.client.streamable_http").error("a response body")

    assert caplog.records == []


def _start_with_loops(monkeypatch: pytest.MonkeyPatch, files: Files, behaviour) -> None:
    """Starts the real entry point whose loops are replaced by `behaviour(boundaries)`."""

    async def loops(config, boundaries):
        behaviour(boundaries)

    monkeypatch.setattr("agents.main.run_all", loops)
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(files.config_path))
    files.write()


def test_a_marker_left_from_an_earlier_run_is_gone_before_any_loop_starts(files, monkeypatch):
    files.ready_path.touch()
    seen: list[bool] = []
    _start_with_loops(monkeypatch, files, lambda boundaries: seen.append(files.ready_path.exists()))

    main()

    assert seen == [False]


def test_the_marker_is_created_once_when_a_loop_announces_it_and_stays(files, monkeypatch):
    _start_with_loops(monkeypatch, files, lambda boundaries: (boundaries.ready.announce(), boundaries.ready.announce()))

    main()

    assert files.ready_path.exists()


def test_a_marker_that_cannot_be_removed_stops_the_start_and_names_its_path(files, monkeypatch, capsys):
    files.ready_path.mkdir()
    (files.ready_path / "held").touch()
    _start_with_loops(monkeypatch, files, lambda boundaries: None)

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == READINESS_ERROR_EXIT_CODE
    assert str(files.ready_path) in capsys.readouterr().err


def test_a_marker_that_cannot_be_created_stops_the_runner_and_names_its_path(files, monkeypatch, capsys):
    files.text = files.text.replace(str(files.ready_path), str(files.ready_path / "missing-directory" / "ready"))
    _start_with_loops(monkeypatch, files, lambda boundaries: boundaries.ready.announce())

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == READINESS_ERROR_EXIT_CODE
    assert "missing-directory" in capsys.readouterr().err
