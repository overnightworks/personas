"""The app boundary: a loop's session is agents' member client, opened here against an app's ``/mcp``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from functools import partial
from typing import TYPE_CHECKING

from agents.mcp_client import READ_SECONDS, MemberSession, open_member_session
from agents.skills import Skill, implicit_skill, implicit_skill_name
from pydantic import SecretStr

from personas.reporting import report_argument_left_out

if TYPE_CHECKING:
    from personas.config import AppTarget
    from personas.present import AppSkill
    from personas.secret import Secret

_IMPLICIT_DESCRIPTION = "What the app {} says of itself, and the tools it offers."

type SessionOpener = Callable[[AppTarget, Callable[[], Secret], str], AbstractAsyncContextManager[MemberSession]]
"""Opens a session at an app; the bearer callable is read at each call, so a renewed token reaches the next one."""


def open_app_session(
    app: AppTarget, bearer: Callable[[], Secret], agent_id: str, *, stream_seconds: float = READ_SECONDS
) -> AbstractAsyncContextManager[MemberSession]:
    """A member session at the app, under the bearer; its calls fail as ``MemberSession`` names. An argument the app's
    tool does not declare, which a call leaves out, is reported once per session for ``agent_id``."""
    return open_member_session(
        app.mcp_url,
        bearer=lambda: SecretStr(bearer().reveal()),
        on_argument_left_out=partial(report_argument_left_out, agent_id, app.mcp_url),
        stream_seconds=stream_seconds,
    )


async def read_skills(
    session: MemberSession,
    app_url: str,
    tools: Sequence[str],
    *,
    listing: Callable[[], AbstractAsyncContextManager[object]],
) -> tuple[AppSkill, ...]:
    """The app's skills: its one implicit skill, of the ``tools`` its turn listed, when it declares no skills
    extension, else every entry of its listing in its order, however many pages. An empty listing is no skills. No
    skill file is read. The listing requests, which only an app that declares the extension is sent, run inside
    ``listing()``.

    A listing that breaks a rule fails whole as an ``UnusableAnswer`` that carries the entry's index; raises
    ``StatusFailure``, ``Unreachable`` or ``UnusableAnswer`` otherwise.
    """
    discovery = await session.read_discovery()
    if not discovery.declares_skills:
        return (_implicit_skill(app_url, discovery.instructions, tools),)
    skills: list[AppSkill] = []
    cursor: str | None = None
    async with listing():
        while True:
            page = await session.list_skills(cursor, listed=len(skills))
            skills.extend(page.skills)
            cursor = page.next_cursor
            if cursor is None:
                return tuple(skills)


def _implicit_skill(app_url: str, instructions: str | None, tools: Sequence[str]) -> Skill:
    description = _IMPLICIT_DESCRIPTION.format(implicit_skill_name(app_url))
    return implicit_skill(app_url, app_url, instructions, tools, description)
