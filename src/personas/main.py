"""The entry point: the console script ``personas``, which composes the real boundaries or migrates the memory."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Final

import httpx2
from agents.backend.gateway import GatewayBackend

from personas.app_session import open_mcp_session
from personas.clock import SystemClock
from personas.config import Config, ConfigError, load_config
from personas.failures import leaves_of
from personas.loop import Boundaries, run_all
from personas.readiness import FileReadyMarker, ReadinessError, ReadyMarker
from personas.reporting import log_state_change
from personas.tokens import HttpTokenSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable, Sequence

    from agents.backend import AgentBackend
    from agents.config import AgentConfig
    from agents.tools import HandTool

    from personas.config import MemoryTarget
    from personas.loop import BackendFactory, LineStore

type Compose = Callable[[Config, ReadyMarker], AbstractAsyncContextManager[Boundaries]]

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
READINESS_ERROR_EXIT_CODE: Final = 3
QUIET_LIBRARIES: Final = ("httpx2", "httpcore2", "mcp", "client", "agents", "sqlalchemy", "psycopg", "alembic")
"""Their records carry request details and response content; only the runner's own state changes are logged."""
SILENT_LEVEL: Final = logging.CRITICAL + 1
MIGRATE_COMMAND: Final = "migrate"


def _gateway_backends(gateway: AgentConfig | None) -> BackendFactory | None:
    if gateway is None:
        return None

    def backend(tools: Iterable[HandTool]) -> AgentBackend:
        return GatewayBackend(gateway, tools=tools)

    return backend


@asynccontextmanager
async def _line_store(target: MemoryTarget | None) -> AsyncGenerator[LineStore | None]:
    """The store the configuration names, opened on its first write; without the table there is none.

    The database library is loaded only here, so a runner without a memory never imports it.
    """
    if target is None:
        yield None
        return
    from personas.memory import PostgresLineStore

    store = PostgresLineStore(target)
    try:
        yield store
    finally:
        await store.close()


@asynccontextmanager
async def real_boundaries(config: Config, ready: ReadyMarker) -> AsyncGenerator[Boundaries]:
    async with httpx2.AsyncClient() as token_client, _line_store(config.memory) as lines:
        yield Boundaries(
            tokens=HttpTokenSource(config.token_endpoint, token_client),
            open_session=open_mcp_session,
            clock=SystemClock(),
            report=log_state_change,
            ready=ready,
            backend=_gateway_backends(config.gateway),
            lines=lines,
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


def _migrate(config: Config) -> None:
    if config.memory is None:
        sys.stderr.write("personas: memory: the table is required to migrate\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE)
    from personas.memory import migrate

    migrate(config.memory)


def _run_runner(config: Config, compose: Compose) -> None:
    marker = FileReadyMarker(config.ready_file)
    try:
        # A marker from an earlier run never counts: it goes before any loop asks for a token.
        marker.clear()
        asyncio.run(_run(config, marker, compose))
    except* ReadinessError as group:
        sys.stderr.write(f"personas: {next(leaves_of(group))}\n")
        raise SystemExit(READINESS_ERROR_EXIT_CODE) from None


def main(compose: Compose = real_boundaries, arguments: Sequence[str] = ()) -> None:
    """Runs the runner, or with the argument ``migrate`` brings the memory's schema to its head."""
    if list(arguments) not in ([], [MIGRATE_COMMAND]):
        sys.stderr.write(f"personas: unknown command; the only one is {MIGRATE_COMMAND}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE)
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    if arguments:
        _migrate(config)
    else:
        _run_runner(config, compose)


def command_line() -> None:
    """The console script's entry point."""
    main(arguments=sys.argv[1:])
