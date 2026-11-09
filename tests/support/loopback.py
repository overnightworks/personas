"""An ASGI app on a real loopback socket: an in-process transport would buffer a stream and hold its acknowledgement."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from uvicorn._types import ASGI3Application

SHUTDOWN_SECONDS = 0.1


@asynccontextmanager
async def serving(app: ASGI3Application) -> AsyncIterator[str]:
    """The origin of ``app`` served on an ephemeral loopback port, stopped at the end."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        lifespan="off",
        interface="asgi3",
        log_config=None,
        log_level="critical",
        access_log=False,
    )
    config.timeout_graceful_shutdown = SHUTDOWN_SECONDS
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup()
    main_loop = asyncio.create_task(server.main_loop())
    try:
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await main_loop
        await server.shutdown()
