"""What a turn lists of the apps it may hold tools and skills from: the place's own app within its call time and each
other app the admission names or the realm grants, with the reports of a listing that failed, suspended or resumed an
app."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.member_outcomes import CallFailure, UnusableAnswer
from logline import StepTimer

from personas.app_session import read_skills
from personas.decision import failure_code, is_not_granted
from personas.failures import CallTimedOut, within_call_time
from personas.reporting import LOG, Event, agent_app_fields, failure_event, stepping

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from agents.mcp_client import MemberSession
    from agents.tools import ToolDeclaration

    from personas.admission import Admitted
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget
    from personas.mind import Mind
    from personas.present import AppSkill
    from personas.task_record import TaskRecord
    from personas.turn_sessions import TurnSessions

type Declared = tuple[ToolDeclaration, ...] | None


@dataclass(frozen=True)
class Listed:
    """What an app lists for a turn: its tools, or ``None`` when the turn holds none of its, and its skills, which an
    app the turn holds no tools of has none of."""

    tools: Declared
    skills: tuple[AppSkill, ...] = ()


NOT_LISTED: Final = Listed(None)


class ToolLists:
    """Lists the tools and skills of one agent's apps for a turn, in the loop's session for the place's own app and in a
    session of its own for each other app."""

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        apps: tuple[AppTarget, ...],
        sessions: TurnSessions,
        mind: Mind,
        clock: Clock,
        renew_token: Callable[[], Awaitable[None]],
        report: Callable[[Event, str | None, AppTarget | None, CallFailure | None], None],
    ) -> None:
        self._agent = agent
        self._app = app
        self._apps = apps
        self._sessions = sessions
        self._mind = mind
        self._clock = clock
        self._renew_token = renew_token
        self._report = report

    async def listed(self, session: MemberSession, admitted: Admitted, record: TaskRecord) -> tuple[Listed, ...]:
        """What each configured app lists for the turn, in the configuration's order: its declarations, or ``None``
        when the turn holds none of its, and its skills. The apps are listed side by side, so their reports come in any
        order; the place's own app is listed whether the configuration names it or not."""
        async with asyncio.TaskGroup() as group:
            own = group.create_task(self._list_own(session, record))
            elsewhere = {
                app.mcp_url: group.create_task(self._listed_elsewhere(app, admitted, record))
                for app in self._apps
                if app.mcp_url != self._app.mcp_url
            }
        return tuple(
            own.result() if app.mcp_url == self._app.mcp_url else elsewhere[app.mcp_url].result() for app in self._apps
        )

    async def _listed_elsewhere(self, app: AppTarget, admitted: Admitted, record: TaskRecord) -> Listed:
        """What another app lists for the turn: nothing for a listening app the admission does not name, a tools-only
        app the realm holds back, or an app whose listing failed."""
        if app.listens and app.mcp_url not in admitted.apps:
            return NOT_LISTED
        if not app.listens and self._mind.holds_back(app.mcp_url, self._clock.now()):
            return NOT_LISTED
        return await self._list_elsewhere(app, record)

    async def _list_elsewhere(self, app: AppTarget, record: TaskRecord) -> Listed:
        """What another app lists, or nothing after the report of its failed listing."""
        declared = await self._sessions.ask_elsewhere(app, partial(self._list, app), record, listing=True)
        if isinstance(declared, CallFailure):
            if not app.listens and is_not_granted(declared):
                if self._mind.refused(app.mcp_url, self._clock.now()):
                    self._report(Event.SUSPENDED, failure_code(declared), app, declared)
            else:
                self._report(failure_event(app), failure_code(declared), app, declared)
            return NOT_LISTED
        if not app.listens and self._mind.granted(app.mcp_url):
            self._report(Event.RESUMED, None, app, None)
        return declared

    async def _list_own(self, session: MemberSession, record: TaskRecord) -> Listed:
        """What the place's own app lists, on the loop's session."""
        tools = await self._list_own_tools(session, record)
        if tools is None:
            return NOT_LISTED
        return Listed(tools, await self._skills_of(self._app, session, tools))

    async def _list_own_tools(self, session: MemberSession, record: TaskRecord) -> Declared:
        """The tools the place's app lists, or ``None`` when it did not answer within its call time or its listing
        breaks a rule; the step then ends failed with its reason and the turn goes on. The list the session kept is
        used as it is: no request is sent and no step is written. The record keeps how many tools notices the session
        had received when the turn read its list."""
        record.tools_notices_read = session.tools_notices()
        if (kept := session.kept_tools()) is not None:
            return kept
        timer = StepTimer("tool_list.ended", LOG, **agent_app_fields(self._agent.client_id, self._app.mcp_url))
        with stepping(timer):
            try:
                async with within_call_time(self._app.call_seconds):
                    await self._renew_token()
                    try:
                        return await session.declared_tools()
                    except UnusableAnswer as failure:
                        return self._ended_failed(timer, failure)
            except CallTimedOut as failure:
                return self._ended_failed(timer, failure)

    @staticmethod
    def _ended_failed(timer: StepTimer, failure: CallTimedOut | UnusableAnswer) -> None:
        timer.handled()
        timer.fail(failure_code(failure))
        timer.end()

    async def _list_tools(self, app: AppTarget, session: MemberSession) -> tuple[ToolDeclaration, ...]:
        with stepping(
            StepTimer("tool_list.ended", LOG, **agent_app_fields(self._agent.client_id, app.mcp_url)),
            turn_goes_on=True,
        ):
            async with within_call_time(app.call_seconds):
                return await session.declared_tools()

    async def _list(self, app: AppTarget, session: MemberSession) -> Listed:
        tools = await self._list_tools(app, session)
        return Listed(tools, await self._skills_of(app, session, tools))

    async def _skills_of(
        self, app: AppTarget, session: MemberSession, tools: tuple[ToolDeclaration, ...]
    ) -> tuple[AppSkill, ...]:
        """The skills of an app that listed these tools, in the session that listed them. An app whose skills listing
        fails has none this turn, and the turn goes on; a refusal is raised as its tools listing's is."""
        try:
            return await read_skills(
                session, app.mcp_url, [each.name for each in tools], listing=partial(self._listing_skills, app)
            )
        except CallFailure as failure:
            if is_not_granted(failure):
                raise
            return ()

    @asynccontextmanager
    async def _listing_skills(self, app: AppTarget) -> AsyncGenerator[None]:
        """A ``skills/list`` that is sent: a step of its own, within the app's call time. The loop's token is renewed
        for its own app alone; another app's listing runs under that app's own token."""
        with stepping(
            StepTimer("skill_list.ended", LOG, **agent_app_fields(self._agent.client_id, app.mcp_url)),
            turn_goes_on=True,
        ):
            async with within_call_time(app.call_seconds):
                if app.mcp_url == self._app.mcp_url:
                    await self._renew_token()
                yield
