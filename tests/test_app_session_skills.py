"""The runner reads an app's skills from its discover answer and its listing, paging through the listing."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

import pytest
from agents.member_outcomes import INVALID_ANSWER_CODE, UnusableAnswer
from agents.skills import SKILLS_EXTENSION, Skill, SkillEntry
from mcp_types import ListToolsResult, Tool

from personas.app_session import read_skills
from tests.support.app_session import opened
from tests.support.hopin import DISCOVER, LIST_TOOLS, READ, SKILLS_LIST, discovering
from tests.support.skills import a_skill_file, an_entry

if TYPE_CHECKING:
    from tests.support.wire import Handler

pytestmark = pytest.mark.anyio

WITH_THE_EXTENSION = discovering(extensions={SKILLS_EXTENSION: {}})


def a_listing(entries: list[dict], next_cursor: str | None = None) -> Handler:
    async def lists(reply):
        await reply.raw_result({"skills": entries} | ({"nextCursor": next_cursor} if next_cursor else {}))

    return lists


async def skills_of(session, app) -> tuple:
    """The skills of the app, read with the tools its listing gave."""
    tools = [tool.name for tool in await session.declared_tools()]
    return await read_skills(session, app.url, tools)


def named(skills) -> list[str]:
    return [skill.name for skill in skills]


async def test_an_app_that_declares_no_skills_extension_has_its_implicit_skill_named_from_its_host(app, bearer):
    instructions = secrets.token_hex(8)
    app.script(DISCOVER, discovering(instructions=instructions))

    async with opened(app, bearer) as session:
        (skill,) = await skills_of(session, app)
        declared = await session.declared_tools()

    assert isinstance(skill, Skill)
    assert skill.name == "127-0-0-1"
    assert skill.body == instructions
    assert skill.tools == tuple(tool.name for tool in declared)
    assert skill.name in skill.description
    assert skill.identity.server == app.url


async def test_an_implicit_skill_holds_the_tools_it_is_given_and_sends_no_request_for_them(app, bearer):
    app.script(DISCOVER, discovering())
    tools = [secrets.token_hex(4), secrets.token_hex(4)]

    async with opened(app, bearer) as session:
        (skill,) = await read_skills(session, app.url, tools)

    assert skill.tools == tuple(tools)
    assert app.log.count(LIST_TOOLS) == 0


async def test_an_app_with_the_extension_has_every_entry_of_its_listing_over_its_pages_in_order(app, bearer):
    files = {name: a_skill_file(name) for name in ("one", "two", "three")}
    entries = {name: an_entry(name, content) for name, content in files.items()}
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, a_listing([entries["one"], entries["two"]], "next"), a_listing([entries["three"]]))

    async with opened(app, bearer) as session:
        skills = await skills_of(session, app)

    assert all(isinstance(skill, SkillEntry) for skill in skills)
    assert named(skills) == ["one", "two", "three"]


async def test_an_empty_listing_is_no_skills_and_never_the_implicit_skill(app, bearer):
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, a_listing([]))

    async with opened(app, bearer) as session:
        assert await skills_of(session, app) == ()


@pytest.mark.parametrize(
    ("pages", "index"),
    [
        pytest.param([["ok", "broken"]], 1, id="the second entry of the one page"),
        pytest.param([["ok"], ["broken"]], 1, id="the first entry of the second page counts the page before"),
        pytest.param([[], ["broken"]], 0, id="the first entry after an empty page"),
    ],
)
async def test_a_listing_that_breaks_a_rule_fails_whole_naming_the_entry_and_no_text_of_the_app(
    app, bearer, pages, index
):
    marker = secrets.token_hex(8)
    good = an_entry("good", a_skill_file("good"))
    broken = an_entry("good", a_skill_file("good")) | {"uri": f"skill://{marker}/not-a-skill-file"}
    served = [a_listing([good if each == "ok" else broken for each in page], "next") for page in pages[:-1]]
    served.append(a_listing([good if each == "ok" else broken for each in pages[-1]]))
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, *served)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await skills_of(session, app)

    assert (failure.value.code, failure.value.entry) == (INVALID_ANSWER_CODE, index)
    assert marker not in f"{failure.value} {failure.value!r} {failure.value.answer}"


async def test_reading_the_skills_fetches_no_skill_file(app, bearer):
    content = a_skill_file("greeting")
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, a_listing([an_entry("greeting", content)]))

    async with opened(app, bearer) as session:
        await skills_of(session, app)

    assert app.log.count(READ) == 0


async def test_an_app_that_lists_a_tool_twice_fails_its_implicit_skill_as_an_unusable_answer(app, bearer):
    marker = f"tool_{secrets.token_hex(8)}"

    async def lists_a_tool_twice(reply):
        tool = Tool(name=marker, description="Does a thing.", input_schema={"type": "object"})
        await reply.result(ListToolsResult(tools=[tool, tool]))

    app.script(DISCOVER, discovering())
    app.script(LIST_TOOLS, lists_a_tool_twice)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await skills_of(session, app)

    assert failure.value.code == INVALID_ANSWER_CODE
    assert marker not in f"{failure.value} {failure.value!r} {failure.value.answer}"
