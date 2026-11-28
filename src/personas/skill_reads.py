"""How a skill is read from its app and entered into his history: for the model's ``read_skill`` and for the block
a turn appends at the end of its request."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.member_outcomes import CallFailure
from agents.skills import Skill, SkillEntry, SkillRefused
from agents.turn import is_within_event_bound
from logline import StepTimer

from personas.decision import failure_code, is_not_granted
from personas.failures import within_call_time
from personas.present import blocks_due, load_round, skill_round, skill_text
from personas.reporting import LOG, Event, agent_app_fields, failure_event, stepping

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from agents.mcp_client import MemberSession
    from agents.turn import ToolRound

    from personas.admission import Admitted
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget
    from personas.entries import SkillRead
    from personas.held_tools import TurnOffer
    from personas.mind import Mind
    from personas.present import AppSkill, BlockDue
    from personas.reporting import LoopReport
    from personas.task_record import TaskRecord
    from personas.turn_sessions import TurnSessions

TOO_LARGE_REASON: Final = "too_large"


async def got_skill(session: MemberSession, uri: str) -> Skill | SkillRefused:
    """The skill at ``uri``, or the refusal of a file that is not what its entry promised, which is an answer, not a
    defect."""
    try:
        return await session.get_skill(uri)
    except SkillRefused as refused:
        return refused


class SkillReads:
    """Reads the skills of one agent's apps and enters them into his history: on the loop's session for the place's
    own app and in a session of its own for each other app."""

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        app_at: Mapping[str, AppTarget],
        sessions: TurnSessions,
        mind: Mind,
        clock: Clock,
        renew_token: Callable[[], Awaitable[None]],
        report: LoopReport,
    ) -> None:
        self._agent = agent
        self._app = app
        self._app_at = app_at
        self._sessions = sessions
        self._mind = mind
        self._clock = clock
        self._renew_token = renew_token
        self._report = report

    async def own(self, session: MemberSession, uri: str) -> Skill | SkillRefused:
        """The skill at ``uri`` of the place's own app, on the loop's session under a renewed token; a failure is
        raised."""
        await self._renew_token()
        return await got_skill(session, uri)

    async def elsewhere(self, app: AppTarget, uri: str, record: TaskRecord) -> Skill | SkillRefused | CallFailure:
        """The skill at ``uri`` of another app, in a session of its own, or the failure of the call."""
        return await self._sessions.ask_elsewhere(app, partial(got_skill, uri=uri), record)

    def skill_of(self, app: AppTarget, answer: Skill | SkillRefused) -> Skill | None:
        """The skill an app answered, or ``None`` after the report of its refusal."""
        if isinstance(answer, SkillRefused):
            self._report(Event.SKILL_REFUSED, answer.refusal.value, app)
            return None
        return answer

    def entered(self, app: str, skill: Skill) -> SkillRead | None:
        """The entry a skill read joins his history as, or ``None`` when its text is past one result's bound."""
        if not is_within_event_bound(skill_text(skill.body), None):
            return None
        return self._mind.skill_read(app, skill.name, skill.identity.uri, skill.body, self._clock.wall())

    async def block(
        self, session: MemberSession, admitted: Admitted, offered: TurnOffer, record: TaskRecord
    ) -> tuple[ToolRound, ...]:
        """The rounds that end a turn's request, entered into his history as they are made: for each skill of the apps
        he holds a place at that is due (``blocks_due``), its read when its app answered it, then its load. A read
        that fails, is refused or is too large is left out, and its load enters all the same; the own app's refusal of
        the agent is raised."""
        present = admitted.apps | {self._app.mcp_url}
        rounds: list[ToolRound] = []
        for due in blocks_due(offered.offers, present, offered.deferred, admitted.history):
            if due.read and (read := await self._read(session, due, record)) is not None:
                rounds.append(skill_round(read))
            if due.tools:
                load = self._mind.tools_loaded(due.app, due.skill.name, due.tools, self._clock.wall())
                rounds.append(load_round(load, offered.deferred))
        return tuple(rounds)

    async def _read(self, session: MemberSession, due: BlockDue, record: TaskRecord) -> SkillRead | None:
        """The entry of a block's read, or ``None`` when it is left out. A read sent on the loop's session is a step
        of the turn, ``skill_read.ended``, as is a text past one result's bound, at any app."""
        app = self._app_at[due.app]
        sent = isinstance(due.skill, SkillEntry) and app.mcp_url == self._app.mcp_url
        timer = StepTimer("skill_read.ended", LOG, **agent_app_fields(self._agent.client_id, app.mcp_url))
        timer.handled()
        try:
            with stepping(timer) if sent else nullcontext(timer):
                skill = await self._fetched(session, app, due.skill, record)
                if skill is None:
                    return None
                if (entry := self.entered(app.mcp_url, skill)) is None:
                    timer.fail(TOO_LARGE_REASON)
                    timer.end()
                return entry
        except CallFailure as failure:
            if is_not_granted(failure):
                raise
            return None

    async def _fetched(
        self, session: MemberSession, app: AppTarget, skill: AppSkill, record: TaskRecord
    ) -> Skill | None:
        """The skill of a block, read from its app unless it is the app's implicit skill, whose body its listing kept;
        or ``None`` after the report of a read that failed or was refused."""
        if isinstance(skill, Skill):
            return skill
        if app.mcp_url == self._app.mcp_url:
            async with within_call_time(app.call_seconds):
                answer = await self.own(session, skill.uri)
        else:
            answer = await self.elsewhere(app, skill.uri, record)
            if isinstance(answer, CallFailure):
                self._report(failure_event(app), failure_code(answer), app, answer)
                return None
        return self.skill_of(app, answer)
