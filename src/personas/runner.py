"""The runner: one loop for each agent and app that listens, while this pod owns the agent."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, NoReturn

from personas.due import resumed_turns
from personas.loop import AgentLoop
from personas.mind import Mind
from personas.present import present_of
from personas.recall import LookupBounds
from personas.task_record import ending_loud_when_cancelled, unrequested_unless_cancelling
from personas.write_through import WriteThrough

if TYPE_CHECKING:
    from personas.config import Config
    from personas.loop import Boundaries
    from personas.ownership import Owned, Ownership


async def _run_loop(loop: AgentLoop, agent_task: asyncio.Task[object]) -> None:
    """Runs a loop; one that ends cancelled while its agent's task is not being cancelled ends the runner loud.

    The agent task's own count says so: it is above zero when the ownership or the runner cancels it and while its
    group aborts after a loop's loud end. A loop task's count is not read, a cancelled lifecycle session lowers it."""
    return await ending_loud_when_cancelled(loop.run(), partial(unrequested_unless_cancelling, agent_task))


async def run_all(config: Config, boundaries: Boundaries, ownership: Ownership) -> NoReturn:
    """One loop for every app that listens, for each agent while this pod owns it; a loop never ends on its own."""
    memory = config.memory
    lookup = None if memory is None else LookupBounds(memory.lookup_calls, memory.lookup_bytes)
    agents = {agent.client_id: agent for agent in config.agents}

    async def serve_agent(owned: Owned) -> NoReturn:
        """Every loop of the agent, over the history he was taken over with, which is written through as it grows."""
        agent_task = asyncio.current_task()
        assert agent_task is not None  # noqa: S101 - an agent is served inside its own task
        clock = boundaries.clock
        resumed = resumed_turns(owned.history.turns, clock.now(), clock.wall())
        mind = Mind(
            config.history_bytes,
            config.history_keep_bytes,
            config.budget_pause_cap_seconds,
            present_of(owned.history),
            resumed,
            keeping_record=memory is not None,
        )
        agent = agents[owned.client_id]
        write_through = WriteThrough(mind, owned, boundaries)
        loops = [AgentLoop(agent, app, boundaries, mind, config.apps, lookup) for app in config.apps if app.listens]
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(write_through.run())
                for loop in loops:
                    group.create_task(_run_loop(loop, agent_task))
                try:
                    await asyncio.Event().wait()
                finally:
                    # The agent's cancel lands here before the group cancels his loops and turns, so a turn it cuts
                    # finds his mind stopping.
                    mind.stopping()
                    write_through.stopping(config.ownership.cancel_margin_seconds)
        finally:
            if agent_task.cancelling():
                # The statements share his stop's margin with the write, beside it, so neither waits on the other; the
                # write runs outside their group, so a statement's defect does not cancel his last write.
                flush = asyncio.create_task(write_through.flush_on_stop())
                try:
                    async with asyncio.TaskGroup() as stop:
                        for loop in loops:
                            stop.create_task(loop.state_on_stop(write_through.seconds_left_to_stop()))
                finally:
                    await flush
        raise AssertionError("a loop ended")  # pragma: no cover

    await ownership.serve(list(agents), serve_agent, boundaries.ready.announce)
    raise AssertionError("the ownership ended")  # pragma: no cover
