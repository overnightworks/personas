"""The loop of one agent in one app, and the runner that keeps one per agent and app."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from agents.decision import (
    SUSPENDED_RETRY_SECONDS,
    WAIT_ERROR_CODE,
    CallSource,
    Memory,
    Reaction,
    Token,
    accepts_for,
    after_refusal,
    backoff_delay,
    forget_gone,
    reaction_to_status,
    renewal_due,
    status_code,
)
from agents.failures import CallFailure, StatusFailure, TokenStatusFailure, WaitToolError
from agents.reporting import Event, Reporter, StateChange

if TYPE_CHECKING:
    from agent_runtime_client.wait import WaitAnswer

    from agents.app_session import AppSession, SessionOpener
    from agents.clock import Clock
    from agents.config import AgentCredentials, AppTarget, Config
    from agents.tokens import TokenSource


@dataclass(frozen=True)
class Boundaries:
    """Everything a loop touches outside itself."""

    tokens: TokenSource
    open_session: SessionOpener
    clock: Clock
    report: Reporter


def _raise_if_cancelled() -> None:
    """Stops a loop whose cancellation an SDK task group swallowed while a session was opening.

    The task still counts the request; without this the loop would open the next session instead of ending.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


class AgentLoop:
    """Keeps one ``wait`` in flight for one agent in one app and accepts what it is invited to.

    A refusal by the realm or the app suspends the loop: it makes no further call until its next
    retry, and resumes once ``wait`` answers again. A network failure or a 5xx backs off.
    """

    def __init__(self, agent: AgentCredentials, app: AppTarget, boundaries: Boundaries) -> None:
        self._agent = agent
        self._app = app
        self._boundaries = boundaries
        self._token: Token | None = None
        self._cursor: str | None = None
        self._memory = Memory()
        self._failures = 0
        self._renewed_after_401 = False
        self._suspension: str | None = None

    async def run(self) -> NoReturn:
        self._report(Event.STARTED)
        while True:
            _raise_if_cancelled()
            try:
                await self._serve()
            except CallFailure as failure:
                # The next wait starts without a cursor, so its answer is the whole standing again.
                self._cursor = None
                await self._react(failure)

    async def _serve(self) -> NoReturn:
        token = await self._current_token()
        async with self._boundaries.open_session(self._app, token.value) as session:
            while True:
                session.use_token((await self._current_token()).value)
                answer = await session.wait(self._cursor)
                self._answered(answer)
                await self._accept_invitations(session, answer)

    async def _current_token(self) -> Token:
        if self._token is None or renewal_due(self._token, self._boundaries.clock.now()):
            granted = await self._boundaries.tokens.fetch(self._agent, self._app.scope)
            expires_at = self._boundaries.clock.now() + granted.lifetime_seconds
            self._token = Token(granted.value, expires_at)
        return self._token

    def _answered(self, answer: WaitAnswer) -> None:
        self._cursor = answer.cursor
        self._failures = 0
        self._renewed_after_401 = False
        if self._suspension is not None:
            self._suspension = None
            self._report(Event.RESUMED)
        self._memory = forget_gone(self._memory, answer)

    async def _accept_invitations(self, session: AppSession, answer: WaitAnswer) -> None:
        for accept in accepts_for(answer, self._memory):
            refusal = await session.accept(accept)
            if refusal is None:
                self._report(Event.JOINED)
                continue
            self._memory, is_news = after_refusal(self._memory, accept, refusal)
            if is_news:
                self._report(Event.REFUSED, refusal.value)

    async def _react(self, failure: CallFailure) -> None:
        match failure:
            case WaitToolError():
                await self._suspend(WAIT_ERROR_CODE)
            case StatusFailure(status=status):
                source = CallSource.TOKEN_ENDPOINT if isinstance(failure, TokenStatusFailure) else CallSource.APP
                match reaction_to_status(status, source, already_renewed=self._renewed_after_401):
                    case Reaction.RENEW:
                        self._renewed_after_401 = True
                        self._token = None
                    case Reaction.SUSPEND:
                        await self._suspend(status_code(status))
                    case Reaction.BACKOFF:
                        await self._back_off()
            case _:
                await self._back_off()

    async def _suspend(self, code: str) -> None:
        if self._suspension != code:
            self._suspension = code
            self._report(Event.SUSPENDED, code)
        self._token = None
        await self._boundaries.clock.sleep(SUSPENDED_RETRY_SECONDS)

    async def _back_off(self) -> None:
        self._failures += 1
        await self._boundaries.clock.sleep(backoff_delay(self._failures))

    def _report(self, event: Event, code: str | None = None) -> None:
        self._boundaries.report(StateChange(event, self._agent.client_id, self._app.mcp_url, code))


async def run_all(config: Config, boundaries: Boundaries) -> NoReturn:
    """One loop for every agent and every app; a loop never ends on its own."""
    async with asyncio.TaskGroup() as group:
        for agent in config.agents:
            for app in config.apps:
                group.create_task(AgentLoop(agent, app, boundaries).run())
    raise AssertionError("a loop ended")  # pragma: no cover
