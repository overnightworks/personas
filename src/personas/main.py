"""The entry point: the console script ``personas``, which composes the real boundaries, or migrates, lists or erases
the memory."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import signal
import sys
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Final, NoReturn

import httpx2
from agents.backend.gateway import GatewayBackend

from personas.app_session import open_mcp_session
from personas.clock import SystemClock
from personas.config import Config, ConfigError, load_config
from personas.failures import StoreDefect, StoreUnavailable, leaves_of
from personas.loop import Boundaries, run_all
from personas.readiness import FileReadyMarker, ReadinessError, ReadyMarker
from personas.reporting import log_state_change
from personas.sweep import SweptStore, sweep_lines
from personas.tokens import HttpTokenSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator, Iterable, Sequence

    from agents.backend import AgentBackend
    from agents.config import AgentConfig
    from agents.tools import HandTool

    from personas.config import MemoryTarget
    from personas.loop import BackendFactory, LineStore
    from personas.memory import KeptVisit

type Compose = Callable[[Config, ReadyMarker], AbstractAsyncContextManager[Boundaries]]

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
READINESS_ERROR_EXIT_CODE: Final = 3
STORE_ERROR_EXIT_CODE: Final = 4
QUIET_LIBRARIES: Final = ("httpx2", "httpcore2", "mcp", "client", "agents", "sqlalchemy", "psycopg", "alembic")
"""Their records carry request details and response content; only the runner's own state changes are logged."""
SILENT_LEVEL: Final = logging.CRITICAL + 1
MIGRATE_COMMAND: Final = "migrate"
MEMORY_COMMAND: Final = "memory"
LIST_COMMAND: Final = "list"
ERASE_COMMAND: Final = "erase"
VISIT_COLUMNS: Final = ("agent", "app", "place_kind", "place_id", "generation", "lines", "first_read", "last_read")


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


async def run_with_sweep(config: Config, boundaries: Boundaries) -> NoReturn:
    """The loops, and beside them the sweep when the configuration has a memory; a defect of either ends both."""
    async with asyncio.TaskGroup() as group:
        if config.memory is not None and isinstance(boundaries.lines, SweptStore):
            group.create_task(sweep_lines(boundaries.lines, config.memory, boundaries.clock, boundaries.report))
        await run_all(config, boundaries)


async def _compose_and_run(config: Config, ready: ReadyMarker, compose: Compose) -> None:
    async with compose(config, ready) as boundaries:
        await run_with_sweep(config, boundaries)


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


class _MemoryArgumentParser(argparse.ArgumentParser):
    """Refuses a wrong use with its usage and the exit code of every refused command, before anything is read."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: {message}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE)


class _StoreOnce(argparse.Action):
    """Refuses an option given twice, which argparse would otherwise settle silently by its last value."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest) is not None:
            parser.error(f"argument {option_string}: given more than once")
        setattr(namespace, self.dest, values)


def _memory_request(arguments: Sequence[str]) -> argparse.Namespace:
    # Prefix matching is off, or a mistyped option such as ``--al`` would select every agent's lines.
    parser = _MemoryArgumentParser(prog=f"personas {MEMORY_COMMAND}", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        LIST_COMMAND, help="one row per visit, without any line's content", allow_abbrev=False
    )
    listing.add_argument("--agent", action=_StoreOnce, metavar="CLIENT_ID", help="only this agent's visits")
    erasing = commands.add_parser(ERASE_COMMAND, help="delete kept lines and print how many", allow_abbrev=False)
    selector = erasing.add_mutually_exclusive_group(required=True)
    selector.add_argument("--agent", action=_StoreOnce, metavar="CLIENT_ID", help="every line this agent keeps")
    selector.add_argument("--all", action="store_true", dest="every_agent", help="every line of every agent")
    return parser.parse_args(arguments)


def _memory_target(config: Config, command: str) -> MemoryTarget:
    if config.memory is None:
        sys.stderr.write(f"personas: memory: the table is required to {command}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE)
    return config.memory


@contextmanager
def _stopping_on_store_failure() -> Generator[None]:
    """A failure of the store ends the command with its code or its revisions, and no traceback."""
    from personas.memory import SchemaNotAtHeadError

    try:
        yield
    except (StoreUnavailable, StoreDefect) as error:
        sys.stderr.write(f"personas: memory: {error}\n")
        raise SystemExit(STORE_ERROR_EXIT_CODE) from None
    except SchemaNotAtHeadError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(STORE_ERROR_EXIT_CODE) from None


def _migrate(config: Config) -> None:
    target = _memory_target(config, MIGRATE_COMMAND)
    from personas.memory import migrate

    with _stopping_on_store_failure():
        migrate(target)


def _print_visits(visits: Iterable[KeptVisit]) -> None:
    """Tab-separated; a field holding a tab or a line break, as a place id from an app may, is quoted."""
    rows = csv.writer(sys.stdout, dialect="excel-tab", lineterminator="\n")
    rows.writerow(VISIT_COLUMNS)
    for each in visits:
        place, times = each.visit.place, (each.first_read_at.isoformat(), each.last_read_at.isoformat())
        rows.writerow((each.client_id, each.visit.app, place.kind, place.id, each.visit.generation, each.lines, *times))


async def _perform_memory(target: MemoryTarget, request: argparse.Namespace) -> None:
    from personas.memory import PostgresLineStore

    store = PostgresLineStore(target)
    try:
        if request.command == LIST_COMMAND:
            _print_visits(await store.kept_visits(request.agent))
            return
        erased = await (store.erase_all() if request.every_agent else store.erase_agent(request.agent))
        sys.stdout.write(f"{erased} lines erased\n")
    finally:
        await store.close()


def _memory(config: Config, request: argparse.Namespace) -> None:
    target = _memory_target(config, request.command)
    with _stopping_on_store_failure():
        asyncio.run(_perform_memory(target, request))


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
    """Runs the runner; ``migrate`` brings the memory's schema to its head, ``memory`` lists or erases what it keeps."""
    command = list(arguments)
    memory_request = _memory_request(command[1:]) if command[:1] == [MEMORY_COMMAND] else None
    if memory_request is None and command not in ([], [MIGRATE_COMMAND]):
        sys.stderr.write(f"personas: unknown command; the commands are {MIGRATE_COMMAND} and {MEMORY_COMMAND}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE)
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    if memory_request is not None:
        _memory(config, memory_request)
    elif command:
        _migrate(config)
    else:
        _run_runner(config, compose)


def command_line() -> None:
    """The console script's entry point."""
    main(arguments=sys.argv[1:])
