"""The entry point: the console script ``personas``, which composes the real boundaries."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Final

import httpx2
from agent_runtime_client.backend.gateway import GatewayBackend

from personas.app_session import open_mcp_session
from personas.clock import SystemClock
from personas.config import Config, ConfigError, load_config
from personas.failures import leaves_of
from personas.loop import Boundaries, run_all
from personas.readiness import FileReadyMarker, ReadinessError, ReadyMarker
from personas.reporting import log_state_change
from personas.tokens import HttpTokenSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable

    from agent_runtime_client.backend import AgentBackend
    from agent_runtime_client.config import AgentConfig
    from agent_runtime_client.tools import HandTool

    from personas.loop import BackendFactory

type Compose = Callable[[Config, ReadyMarker], AbstractAsyncContextManager[Boundaries]]

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
READINESS_ERROR_EXIT_CODE: Final = 3
QUIET_LIBRARIES: Final = ("httpx2", "httpcore2", "mcp", "client", "agent_runtime_client")
"""Their records carry request details and response content; only the runner's own state changes are logged."""
SILENT_LEVEL: Final = logging.CRITICAL + 1


def _gateway_backends(gateway: AgentConfig | None) -> BackendFactory | None:
    if gateway is None:
        return None

    def backend(tools: Iterable[HandTool]) -> AgentBackend:
        return GatewayBackend(gateway, tools=tools)

    return backend


@asynccontextmanager
async def real_boundaries(config: Config, ready: ReadyMarker) -> AsyncGenerator[Boundaries]:
    async with httpx2.AsyncClient() as token_client:
        yield Boundaries(
            tokens=HttpTokenSource(config.token_endpoint, token_client),
            open_session=open_mcp_session,
            clock=SystemClock(),
            report=log_state_change,
            ready=ready,
            backend=_gateway_backends(config.gateway),
        )


async def _compose_and_run(config: Config, ready: ReadyMarker, compose: Compose) -> None:
    async with compose(config, ready) as boundaries:
        await run_all(config, boundaries)


async def _run(config: Config, ready: ReadyMarker, compose: Compose) -> None:
    """Runs the runner as its own task, which a SIGTERM cancels; that cancellation is the clean stop.

    Waiting on the task, not awaiting it, leaves the cancellation unraised, so any error the task ends with is a defect.
    """
    runner = asyncio.create_task(_compose_and_run(config, ready, compose))
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, runner.cancel)
    await asyncio.wait({runner})
    if not runner.cancelled():
        runner.result()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(SILENT_LEVEL)


def main(compose: Compose = real_boundaries) -> None:
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    marker = FileReadyMarker(config.ready_file)
    try:
        # A marker from an earlier run never counts: it goes before any loop asks for a token.
        marker.clear()
        asyncio.run(_run(config, marker, compose))
    except* ReadinessError as group:
        sys.stderr.write(f"personas: {next(leaves_of(group))}\n")
        raise SystemExit(READINESS_ERROR_EXIT_CODE) from None
