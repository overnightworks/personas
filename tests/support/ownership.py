"""The ownership of a runner that no other pod competes with: every agent is served at once."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import NoReturn

    from personas.ownership import ServeAgent


class AllAgentsOwned:
    """Serves every agent without a lock and without a grace."""

    async def serve(
        self, client_ids: Sequence[str], serve_agent: ServeAgent, contending: Callable[[], None]
    ) -> NoReturn:
        contending()
        async with asyncio.TaskGroup() as group:
            for each in client_ids:
                group.create_task(serve_agent(each))
        raise AssertionError("an agent ended")  # pragma: no cover
