"""The entry point: the console script ``agents``, which composes the real boundaries."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Final

import httpx2

from agents.app_session import open_mcp_session
from agents.clock import SystemClock
from agents.config import Config, ConfigError, load_config
from agents.loop import Boundaries, run_all
from agents.readiness import FileReadyMarker, ReadinessError, ReadyMarker
from agents.reporting import log_state_change
from agents.tokens import HttpTokenSource

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
READINESS_ERROR_EXIT_CODE: Final = 3
QUIET_LIBRARIES: Final = ("httpx2", "httpcore2", "mcp", "client")
"""Their records carry request details and response content; only the runner's own state changes are logged."""
SILENT_LEVEL: Final = logging.CRITICAL + 1


async def _run(config: Config, ready: ReadyMarker) -> None:
    async with httpx2.AsyncClient() as token_client:
        boundaries = Boundaries(
            tokens=HttpTokenSource(config.token_endpoint, token_client),
            open_session=open_mcp_session,
            clock=SystemClock(),
            report=log_state_change,
            ready=ready,
        )
        await run_all(config, boundaries)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(SILENT_LEVEL)


def main() -> None:
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"agents: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    marker = FileReadyMarker(config.ready_file)
    failure: Exception | None = None
    try:
        # A marker from an earlier run never counts: it goes before any loop asks for a token.
        marker.clear()
        asyncio.run(_run(config, marker))
    except* ReadinessError as group:
        failure = group.exceptions[0]
    if failure is not None:
        sys.stderr.write(f"agents: {failure}\n")
        raise SystemExit(READINESS_ERROR_EXIT_CODE)
