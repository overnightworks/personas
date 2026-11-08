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
from agents.reporting import log_state_change
from agents.tokens import HttpTokenSource

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
# Their records carry request details; only the runner's own state changes are logged.
_QUIET_LIBRARIES: Final = ("httpx2", "httpcore2", "mcp")


async def _run(config: Config) -> None:
    async with httpx2.AsyncClient() as token_client:
        boundaries = Boundaries(
            tokens=HttpTokenSource(config.token_endpoint, token_client),
            open_session=open_mcp_session,
            clock=SystemClock(),
            report=log_state_change,
        )
        await run_all(config, boundaries)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"agents: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    asyncio.run(_run(config))
