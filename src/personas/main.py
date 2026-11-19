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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NoReturn, Protocol

import httpx2
from agents.backend.gateway import GatewayBackend
from logline import configure

from personas.app_session import open_mcp_session
from personas.clock import SystemClock
from personas.config import Config, ConfigError, load_config
from personas.failures import SchemaNotAtHeadError, StoreDefect, StoreUnavailable, leaves_of
from personas.locks import PostgresAgentLocks
from personas.loop import Boundaries, run_all
from personas.memory import PostgresHistoryStore, PostgresLineStore, PostgresMemory, migrate
from personas.ownership import LockedOwnership
from personas.readiness import FileReadyMarker, ReadinessError, ReadyMarker
from personas.reporting import (
    MCP_CLIENT_LOGGER,
    Event,
    StateChange,
    StoreOutage,
    carry_trace,
    demote_a_closed_stream,
    log_state_change,
)
from personas.sweep import sweep_lines
from personas.tokens import HttpTokenSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator, Iterable, Sequence

    from agents.config import AgentConfig
    from agents.tools import HandTool

    from personas.config import DatabaseTarget
    from personas.memory import KeptAgentHistory, KeptVisit
    from personas.ownership import Ownership
    from personas.sweep import SweptStore
    from personas.turning import BackendFactory

type Compose = Callable[[Config, ReadyMarker], AbstractAsyncContextManager[SweepingBoundaries]]

CONFIGURATION_ERROR_EXIT_CODE: Final = 2
READINESS_ERROR_EXIT_CODE: Final = 3
STORE_ERROR_EXIT_CODE: Final = 4
STOP_CUT_EXIT_CODE: Final = 5
"""A stop its deadline cut. Kubernetes neither restarts a pod it deleted nor holds a rollout by its exit code, so the
code costs nothing there and leaves in the container's last state that the stop did not end cleanly."""
MIGRATE_COMMAND: Final = "migrate"
MEMORY_COMMAND: Final = "memory"
LIST_COMMAND: Final = "list"
ERASE_COMMAND: Final = "erase"
VISIT_COLUMNS: Final = ("agent", "app", "place_kind", "place_id", "generation", "lines", "first_read", "last_read")
HISTORY_COLUMNS: Final = ("agent", "history_entries", "answered_turns", "first_at", "last_at")


def _gateway_backends(gateway: AgentConfig | None) -> BackendFactory | None:
    if gateway is None:
        return None

    def backend(tools: Iterable[HandTool]) -> GatewayBackend:
        return GatewayBackend(gateway, tools=tools)

    return backend


class SchemaChecked(Protocol):
    async def require_head(self) -> None:
        """Raises ``SchemaNotAtHeadError`` when the schema is behind; ``StoreUnavailable`` when it cannot tell now."""
        ...


@dataclass(frozen=True, kw_only=True)
class SweepingBoundaries(Boundaries):
    """The loops' boundaries, the ownership they run under, the store the sweep erases from, none without a memory,
    and the store whose schema the runner checks before it starts."""

    ownership: Ownership
    swept: SweptStore | None = None
    schema: SchemaChecked | None = None


@asynccontextmanager
async def _line_store(config: Config) -> AsyncGenerator[PostgresLineStore | None]:
    """The store the configuration names, opened on its first write; without a memory table there is none."""
    if config.memory is None:
        yield None
        return
    store = PostgresLineStore(config.database)
    try:
        yield store
    finally:
        await store.close()


@asynccontextmanager
async def real_boundaries(config: Config, ready: ReadyMarker) -> AsyncGenerator[SweepingBoundaries]:
    clock = SystemClock()
    locks = PostgresAgentLocks(config.database, config.ownership)
    histories = PostgresHistoryStore(config.database)
    try:
        async with (
            httpx2.AsyncClient(event_hooks={"request": [carry_trace]}) as token_client,
            _line_store(config) as lines,
        ):
            yield SweepingBoundaries(
                tokens=HttpTokenSource(config.token_endpoint, token_client),
                open_session=open_mcp_session,
                clock=clock,
                report=log_state_change,
                ready=ready,
                backend=_gateway_backends(config.gateway),
                lines=lines,
                ownership=LockedOwnership(locks, histories, clock, config.ownership, log_state_change),
                swept=lines,
                schema=histories,
            )
    finally:
        await histories.close()
        # Closing the connection frees the pod's locks: the loops have ended by now, so the next owner may start.
        await locks.close()


async def _schema_at_head(schema: SchemaChecked, config: Config, boundaries: SweepingBoundaries) -> None:
    """Waits until the database answers, an outage reported once, and raises ``SchemaNotAtHeadError`` when its schema
    is behind; the loops and the ready marker wait for it."""
    outage = StoreOutage(lambda event, code: boundaries.report(StateChange(event, None, None, code)))
    while True:
        try:
            await schema.require_head()
        except StoreUnavailable as unavailable:
            outage.failed(unavailable.code)
        else:
            outage.answered()
            return
        await boundaries.clock.sleep(config.ownership.contend_seconds)


async def run_with_sweep(config: Config, boundaries: SweepingBoundaries) -> NoReturn:
    """Once the database's schema is at its head, the loops, and beside them the sweep when the configuration has a
    memory; a defect of either ends both."""
    if boundaries.schema is not None:
        await _schema_at_head(boundaries.schema, config, boundaries)
    async with asyncio.TaskGroup() as group:
        group.create_task(run_all(config, boundaries, boundaries.ownership))
        if config.memory is not None and boundaries.swept is not None:
            group.create_task(sweep_lines(boundaries.swept, config.memory, boundaries.clock, boundaries.report))
    raise AssertionError("a loop ended")  # pragma: no cover


async def _compose_and_run(config: Config, ready: ReadyMarker, compose: Compose) -> None:
    async with compose(config, ready) as boundaries:
        await run_with_sweep(config, boundaries)


async def _run(config: Config, ready: ReadyMarker, compose: Compose) -> None:
    """Runs the runner as its own task, which a SIGTERM cancels; that cancellation is the clean stop, which is cut if it
    has not ended ``stop_seconds`` after the SIGTERM.

    Waiting on the task, not awaiting it, leaves the cancellation unraised, so any error the task ends with is a defect.
    """
    runner = asyncio.create_task(_compose_and_run(config, ready, compose))
    loop = asyncio.get_running_loop()

    def stop() -> None:
        runner.cancel()
        loop.call_later(config.ownership.stop_seconds, _cut_the_stop)

    loop.add_signal_handler(signal.SIGTERM, stop)
    await asyncio.wait({runner})
    if not runner.cancelled():
        runner.result()


def _cut_the_stop() -> None:
    """Ends the process at once: its stop outlasted the deadline, since the loop that runs this has not closed.

    No timeout inside the loop could bound the stop: a close may await again after its cancellation, and the loop's own
    shutdown waits for every task and thread. Each agent's write within his margin has ended by now, and the kernel
    closes the lock connection, which frees his locks.
    """
    log_state_change(StateChange(Event.STOP_CUT, None, None))
    logging.shutdown()
    os._exit(STOP_CUT_EXIT_CODE)


# alembic writes its migration context at INFO on every schema check, which is no state change of the runner.
ALEMBIC_LOGGER: Final = "alembic"


def configure_logging() -> None:
    """Every line one JSON object on stdout; alembic's lines below WARNING stay out, and so does the mcp client's ERROR
    for a closed stream. A ``LOG_LEVEL`` logline refuses is refused with one plain line on stderr."""
    try:
        configure("personas")
    except ValueError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    logging.getLogger(ALEMBIC_LOGGER).setLevel(logging.WARNING)
    logging.getLogger(MCP_CLIENT_LOGGER).addFilter(demote_a_closed_stream)


class _RefusingArgumentParser(argparse.ArgumentParser):
    """Refuses a wrong use with its usage and the exit code of every refused command, before anything is read."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: {message}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE)

    # Restating the base's three overloads, generic in the namespace, serves no caller: none here passes a namespace.
    def parse_known_args(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, args: Iterable[str] | None = None, namespace: None = None
    ) -> tuple[argparse.Namespace, list[str]]:
        """Refuses an argument it does not know itself, so the usage shown is that of the command it was given to;
        argparse would hand it up and refuse it with the outermost usage."""
        parsed, unknown = super().parse_known_args(args, namespace)
        if unknown:
            self.error(f"unrecognized arguments: {' '.join(unknown)}")
        return parsed, unknown


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


def _request(arguments: Sequence[str]) -> argparse.Namespace:
    """The command asked for; none runs the runner."""
    # Prefix matching is off, or a mistyped option such as ``--al`` would select every agent's lines.
    parser = _RefusingArgumentParser(
        prog="personas", description="Without a command, runs the runner.", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser(MIGRATE_COMMAND, help="bring the memory's schema to its head", allow_abbrev=False)
    memory = commands.add_parser(
        MEMORY_COMMAND, help="list or erase what the agents keep, without any content", allow_abbrev=False
    )
    _add_memory_commands(memory)
    return parser.parse_args(arguments)


def _add_memory_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="memory_command", metavar="COMMAND", required=True)
    listing = commands.add_parser(
        LIST_COMMAND, help="one row per visit and per agent's history, without any content", allow_abbrev=False
    )
    listing.add_argument("--agent", action=_StoreOnce, metavar="CLIENT_ID", help="only this agent's visits and history")
    erasing = commands.add_parser(
        ERASE_COMMAND,
        help="delete kept lines, history entries and answered turns and print how many",
        allow_abbrev=False,
    )
    selector = erasing.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--agent",
        action=_StoreOnce,
        metavar="CLIENT_ID",
        help="every line, history entry and answered turn this agent keeps",
    )
    selector.add_argument(
        "--all", action="store_true", dest="every_agent", help="every agent's lines, history entries and answered turns"
    )


@contextmanager
def _stopping_on_store_failure() -> Generator[None]:
    """A failure of the store ends the command with its code or its revisions, and no traceback."""
    try:
        yield
    except (StoreUnavailable, StoreDefect) as error:
        sys.stderr.write(f"personas: memory: {error}\n")
        raise SystemExit(STORE_ERROR_EXIT_CODE) from None
    except SchemaNotAtHeadError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(STORE_ERROR_EXIT_CODE) from None


def _migrate(config: Config) -> None:
    with _stopping_on_store_failure():
        migrate(config.database)


def _print_visits(visits: Iterable[KeptVisit]) -> None:
    """Tab-separated; a field holding a tab or a line break, as a place id from an app may, is quoted."""
    rows = csv.writer(sys.stdout, dialect="excel-tab", lineterminator="\n")
    rows.writerow(VISIT_COLUMNS)
    for each in visits:
        place, times = each.visit.place, (each.first_read_at.isoformat(), each.last_read_at.isoformat())
        rows.writerow((each.client_id, each.visit.app, place.kind, place.id, each.visit.generation, each.lines, *times))


def _print_histories(histories: Iterable[KeptAgentHistory]) -> None:
    """A blank line, then tab-separated below the visits."""
    sys.stdout.write("\n")
    rows = csv.writer(sys.stdout, dialect="excel-tab", lineterminator="\n")
    rows.writerow(HISTORY_COLUMNS)
    for each in histories:
        times = (each.first_at.isoformat(), each.last_at.isoformat())
        rows.writerow((each.client_id, each.entries, each.answered_turns, *times))


async def _list_memory(target: DatabaseTarget, client_id: str | None) -> None:
    lines, histories = PostgresLineStore(target), PostgresHistoryStore(target)
    try:
        _print_visits(await lines.kept_visits(client_id))
        _print_histories(await histories.kept_histories(client_id))
    finally:
        await lines.close()
        await histories.close()


async def _erase_memory(target: DatabaseTarget, request: argparse.Namespace) -> None:
    memory = PostgresMemory(target)
    try:
        erased = await (memory.erase_all() if request.every_agent else memory.erase_agent(request.agent))
    finally:
        await memory.close()
    lines = _counted(erased.lines, "line", "lines")
    entries = _counted(erased.entries, "history entry", "history entries")
    turns = _counted(erased.answered_turns, "answered turn", "answered turns")
    sys.stdout.write(f"{lines}, {entries} and {turns} erased\n")


def _counted(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


async def _perform_memory(target: DatabaseTarget, request: argparse.Namespace) -> None:
    if request.memory_command == LIST_COMMAND:
        await _list_memory(target, request.agent)
    else:
        await _erase_memory(target, request)


def _memory(config: Config, request: argparse.Namespace) -> None:
    with _stopping_on_store_failure():
        asyncio.run(_perform_memory(config.database, request))


def _run_runner(config: Config, compose: Compose) -> None:
    marker = FileReadyMarker(config.ready_file)
    try:
        # A marker from an earlier run never counts: it goes before any loop asks for a token.
        marker.clear()
        asyncio.run(_run(config, marker, compose))
    except* ReadinessError as group:
        sys.stderr.write(f"personas: {next(leaves_of(group))}\n")
        raise SystemExit(READINESS_ERROR_EXIT_CODE) from None
    except* SchemaNotAtHeadError as group:
        sys.stderr.write(f"personas: {next(leaves_of(group))}\n")
        raise SystemExit(STORE_ERROR_EXIT_CODE) from None


def main(compose: Compose = real_boundaries, arguments: Sequence[str] = ()) -> None:
    """Runs the runner; ``migrate`` brings the memory's schema to its head, ``memory`` lists or erases what it keeps."""
    request = _request(arguments)
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        sys.stderr.write(f"personas: {error}\n")
        raise SystemExit(CONFIGURATION_ERROR_EXIT_CODE) from None
    configure_logging()
    if request.command == MEMORY_COMMAND:
        _memory(config, request)
    elif request.command == MIGRATE_COMMAND:
        _migrate(config)
    else:
        _run_runner(config, compose)


def command_line() -> None:
    """The console script's entry point."""
    main(arguments=sys.argv[1:])
