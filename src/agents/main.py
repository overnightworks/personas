"""The entry point: the console script ``agents``, which composes the real boundaries."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Final

import httpx2

from agents.app_session import open_mcp_session
from agents.clock import SystemClock
from agents.config import Config, ConfigError, load_config
from agents.loop import Boundaries, run_all
from agents.readiness import FileReadyMarker, ReadinessError, ReadyMarker
from agents.reporting import log_state_change
from agents.tokens import HttpTokenSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

type Compose = Callable[[Config, ReadyMarker], AbstractAsyncContextManager[Boundaries]]

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
READINESS_ERROR_EXIT_CODE: Final = 3
QUIET_LIBRARIES: Final = ("httpx2", "httpcore2", "mcp", "client")
"""Their records carry request details and response content; only the runner's own state changes are logged."""
SILENT_LEVEL: Final = logging.CRITICAL + 1


@asynccontextmanager
async def real_boundaries(config: Config, ready: ReadyMarker) -> AsyncGenerator[Boundaries]:
    async with httpx2.AsyncClient() as token_client:
        yield Boundaries(
            tokens=HttpTokenSource(config.token_endpoint, token_client),
            open_session=open_mcp_session,
            clock=SystemClock(),
            report=log_state_change,
            ready=ready,
        )


async def _run(config: Config, ready: ReadyMarker, compose: Compose) -> None:
    async with compose(config, ready) as boundaries:
        await run_all(config, boundaries)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(SILENT_LEVEL)


def main(compose: Compose = real_boundaries) -> None:
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"agents: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    marker = FileReadyMarker(config.ready_file)
    try:
        # A marker from an earlier run never counts: it goes before any loop asks for a token.
        marker.clear()
        asyncio.run(_run(config, marker, compose))
    except* ReadinessError as group:
        sys.stderr.write(f"agents: {group.exceptions[0]}\n")
        raise SystemExit(READINESS_ERROR_EXIT_CODE) from None
