"""What a misbehaving MCP server writes: one request read, one answer, a stream or a cut connection."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY
from mcp_types import (
    ErrorData,
    JSONRPCError,
    ResourceUpdatedNotification,
    ResourceUpdatedNotificationParams,
    SubscriptionFilter,
    SubscriptionsAcknowledgedNotification,
    SubscriptionsAcknowledgedNotificationParams,
    SubscriptionsListenResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

JSON = "application/json"
EVENT_STREAM = "text/event-stream"


class ConnectionCut(Exception):  # noqa: N818 - raised in the app to make the server drop the connection
    """Ends the answer without its end, so the client sees a lost connection."""


@dataclass(frozen=True)
class Rpc:
    """The JSON-RPC request a handler answers."""

    id: int | str
    method: str
    params: dict[str, Any]


@dataclass
class Reply:
    """The answer to one request, written through the ASGI ``send``."""

    rpc: Rpc
    send: Callable[[dict[str, Any]], Awaitable[None]]
    receive: Callable[[], Awaitable[dict[str, Any]]]
    keep_alives: int = 0

    async def result(self, result: BaseModel) -> None:
        await self.raw_result(result.model_dump(by_alias=True, mode="json", exclude_none=True))

    async def raw_result(self, result: dict[str, Any]) -> None:
        await self._whole(HTTPStatus.OK, {"jsonrpc": "2.0", "id": self.rpc.id, "result": result})

    async def end(self) -> None:
        await self.send({"type": "http.response.body", "body": b"", "more_body": False})

    async def error(self, code: int, *, status: HTTPStatus = HTTPStatus.BAD_REQUEST, data: object = None) -> None:
        error = JSONRPCError(jsonrpc="2.0", id=self.rpc.id, error=ErrorData(code=code, message="refused", data=data))
        await self._whole(status, error.model_dump(by_alias=True, mode="json", exclude_none=True))

    async def status(self, status: HTTPStatus) -> None:
        await self._start(status, None)
        await self.send({"type": "http.response.body", "body": b""})

    async def open_stream(self) -> None:
        await self._start(HTTPStatus.OK, EVENT_STREAM)

    async def acknowledge(self, honored: list[str]) -> None:
        notification = SubscriptionsAcknowledgedNotification(
            params=SubscriptionsAcknowledgedNotificationParams(
                notifications=SubscriptionFilter(resource_subscriptions=honored or None), _meta=self._meta
            )
        )
        await self.event(notification.model_dump(by_alias=True, mode="json", exclude_none=True) | {"jsonrpc": "2.0"})

    async def pulse(self, uri: str) -> None:
        notification = ResourceUpdatedNotification(params=ResourceUpdatedNotificationParams(uri=uri, _meta=self._meta))
        await self.event(notification.model_dump(by_alias=True, mode="json", exclude_none=True) | {"jsonrpc": "2.0"})

    async def keep_alive(self) -> None:
        await self.send({"type": "http.response.body", "body": b": traffic\n\n", "more_body": True})
        self.keep_alives += 1

    async def keep_alives_until_closed(self) -> None:
        """Writes keep-alives, paced by the connection's own backpressure, until the client closes it."""
        closed = asyncio.create_task(self.receive())
        try:
            while not closed.done():
                await self.keep_alive()
                await asyncio.sleep(0)
        finally:
            closed.cancel()

    async def complete(self) -> None:
        result = SubscriptionsListenResult(_meta=self._meta)
        await self.event(
            {
                "jsonrpc": "2.0",
                "id": self.rpc.id,
                "result": result.model_dump(by_alias=True, mode="json", exclude_none=True),
            }
        )
        await self.end()

    def cut(self) -> None:
        raise ConnectionCut

    @property
    def _meta(self) -> dict[str, Any]:
        return {SUBSCRIPTION_ID_META_KEY: self.rpc.id}

    async def event(self, document: dict[str, Any]) -> None:
        body = f"event: message\ndata: {json.dumps(document)}\n\n".encode()
        await self.send({"type": "http.response.body", "body": body, "more_body": True})

    async def _whole(self, status: HTTPStatus, document: dict[str, Any]) -> None:
        await self._start(status, JSON)
        await self.send({"type": "http.response.body", "body": json.dumps(document).encode()})

    async def _start(self, status: HTTPStatus, content_type: str | None) -> None:
        headers = [(b"content-type", content_type.encode())] if content_type else []
        await self.send({"type": "http.response.start", "status": status.value, "headers": headers})


type Handler = Callable[[Reply], Awaitable[None]]
