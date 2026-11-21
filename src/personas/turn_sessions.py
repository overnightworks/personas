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

    from personas.app_session import AppSession, SessionOpener
    from personas.config import AgentCredentials, AppTarget
    from personas.failures import CallFailure
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
        ask: Callable[[AppSession], Awaitable[Answer]],
        record: TaskRecord,
        *,
        opening: bool = False,
    ) -> Answer | CallFailure:
        """Asks another app in a session of its own, under the token for its own scope; ``opening`` says the turn
        waits for the session's opening, which is then a step of its own.

        The session lives only for this request. A failure of the call, a session that died of its own child and
        an expiry of the app's call time come back as the failure; any other error is raised, and a turn whose
        place left stops. A defect recorded in the turn ends the request, raised from the failure.
        """
        try:
            return await ending_loud_when_cancelled(
                self._within_call_time(app, ask, record, opening=opening), unreachable_unless_cancelling
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

    async def opened(self, opened: AsyncExitStack, app: AppTarget, token: Secret, *, measured: bool) -> AppSession:
        """The session at another ``app``, closed with ``opened``; its opening is a step when ``measured``."""
        if not measured:
            return await opened.enter_async_context(self._open_session(app, token, self._agent.client_id))
        with stepping(step_timer("session_open.ended", self._agent.client_id, app.mcp_url), turn_goes_on=True):
            return await opened.enter_async_context(self._open_session(app, token, self._agent.client_id))

    async def _within_call_time[Answer](
        self, app: AppTarget, ask: Callable[[AppSession], Awaitable[Answer]], record: TaskRecord, *, opening: bool
    ) -> Answer:
        """The token, the session and the request, all under the call time of the app."""
        async with within_call_time(app.call_seconds):
            granted = await self._tokens.fetch(self._agent, app.scope)
            async with AsyncExitStack() as opened:
                session = await self.opened(opened, app, granted.value, measured=opening)
                try:
                    return await ask(session)
                except Exception as error:
                    record.note(error)
                    raise
