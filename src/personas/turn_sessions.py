"""How a turn reaches another app: in a session of its own for each request, under the token for that app's scope and
within its call time."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from personas.decision import decisive_failure
from personas.failures import leaves_of, raise_if_cancelled, within_call_time
from personas.reporting import step_timer, stepping
from personas.task_record import ending_loud_when_cancelled, unreachable_unless_cancelling

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents.mcp_client import MemberSession
    from agents.member_outcomes import CallFailure

    from personas.app_session import SessionOpener
    from personas.config import AgentCredentials, AppTarget
    from personas.secret import Secret
    from personas.task_record import TaskRecord
    from personas.tokens import TokenSource


class TurnSessions:
    """Asks another app for one agent's turns in a session of its own."""

    def __init__(self, agent: AgentCredentials, tokens: TokenSource, open_session: SessionOpener) -> None:
        self._agent = agent
        self._tokens = tokens
        self._open_session = open_session

    async def ask_elsewhere[Answer](
        self,
        app: AppTarget,
        ask: Callable[[MemberSession], Awaitable[Answer]],
        record: TaskRecord,
        *,
        opening: bool = False,
        bounds_own_steps: bool = False,
    ) -> Answer | CallFailure:
        """Asks another app in a session of its own, under the token for its own scope; ``opening`` says the turn
        waits for the session's opening, which is then a step of its own. An ``ask`` that ``bounds_own_steps`` by the
        app's call time each is bounded by none else, and the token and the opening have the call time to themselves.

        The session lives only for this request. A failure of the call, a session that died of its own child and
        an expiry of the app's call time come back as the failure; any other error is raised, and a turn whose
        place left stops. A defect recorded in the turn ends the request, raised from the failure.
        """
        try:
            return await ending_loud_when_cancelled(
                self._within_call_time(app, ask, record, opening=opening, bounds_own_steps=bounds_own_steps),
                unreachable_unless_cancelling,
            )
        except Exception as error:
            raise_if_cancelled()
            failure = decisive_failure(tuple(leaves_of(error)), already_renewed=False)
            if failure is None:
                raise
            defect = record.defect
            if defect is not None:
                raise defect from failure
            return failure

    async def opened(
        self, opened: AsyncExitStack, app: AppTarget, bearer: Callable[[], Secret], *, measured: bool
    ) -> MemberSession:
        """The session at another ``app``, closed with ``opened``; its opening is a step when ``measured``."""
        if not measured:
            return await opened.enter_async_context(self._open_session(app, bearer, self._agent.client_id))
        with stepping(step_timer("session_open.ended", self._agent.client_id, app.mcp_url), turn_goes_on=True):
            return await opened.enter_async_context(self._open_session(app, bearer, self._agent.client_id))

    async def _within_call_time[Answer](
        self,
        app: AppTarget,
        ask: Callable[[MemberSession], Awaitable[Answer]],
        record: TaskRecord,
        *,
        opening: bool,
        bounds_own_steps: bool,
    ) -> Answer:
        """The token, the session and the request under the call time of the app; when the ``ask`` bounds its own
        steps, it runs outside that call time, which bounds the token, the session's opening and its close."""
        async with within_call_time(app.call_seconds) as call_time, AsyncExitStack() as opened:
            session = await self._session_at(opened, app, opening=opening)
            if bounds_own_steps:
                with call_time.paused():
                    return await self._noted(ask, session, record)
            return await self._noted(ask, session, record)

    async def _session_at(self, opened: AsyncExitStack, app: AppTarget, *, opening: bool) -> MemberSession:
        granted = await self._tokens.fetch(self._agent, app.scope)
        return await self.opened(opened, app, lambda: granted.value, measured=opening)

    @staticmethod
    async def _noted[Answer](
        ask: Callable[[MemberSession], Awaitable[Answer]], session: MemberSession, record: TaskRecord
    ) -> Answer:
        try:
            return await ask(session)
        except Exception as error:
            record.note(error)
            raise
