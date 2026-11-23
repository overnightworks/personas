"""An app session reads an app's skills from its discover answer and its listing, and one skill's file on need."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

import pytest
from agents.skills import SKILLS_EXTENSION, Skill, SkillEntry, SkillRefusal, SkillRefused, skill_digest, skill_uri
from mcp_types import ReadResourceResult, TextResourceContents

from personas.failures import INVALID_ANSWER_CODE, UnusableAnswer
from tests.support.app_session import opened
from tests.support.hopin import DISCOVER, READ, SKILLS_GET, SKILLS_LIST, discovering

if TYPE_CHECKING:
    from tests.support.wire import Handler

pytestmark = pytest.mark.anyio

WITH_THE_EXTENSION = discovering(extensions={SKILLS_EXTENSION: {}})


def a_skill_file(name: str, *, tools: str = "say play", body: str = "Greet people by name.") -> bytes:
    return f"---\nname: {name}\ndescription: Greets.\nmetadata:\n  tools: {tools}\n---\n{body}\n".encode()


def an_entry(name: str, content: bytes) -> dict:
    uri = skill_uri(name)
    return {
        "uri": uri,
        "frontmatter": {"name": name, "description": "Greets.", "metadata": {"tools": "say play"}},
        "resources": [{"uri": uri, "digest": skill_digest(content), "size": len(content)}],
    }


def a_listing(entries: list[dict], next_cursor: str | None = None) -> Handler:
    async def lists(reply):
        await reply.raw_result({"skills": entries} | ({"nextCursor": next_cursor} if next_cursor else {}))

    return lists


def a_skill_answer(entry: dict) -> Handler:
    async def answers(reply):
        await reply.raw_result({"skill": entry})

    return answers


def serving_file(content: bytes, uri: str) -> Handler:
    async def shows(reply):
        contents = TextResourceContents(uri=uri, text=content.decode(), mime_type="text/markdown")
        await reply.result(ReadResourceResult(contents=[contents], ttl_ms=0, cache_scope="private"))

    return shows


def named(skills) -> list[str]:
    return [skill.name for skill in skills]


async def test_an_app_that_declares_no_skills_extension_has_its_implicit_skill_named_from_its_host(app, bearer):
    instructions = secrets.token_hex(8)
    app.script(DISCOVER, discovering(instructions=instructions))

    async with opened(app, bearer) as session:
        (skill,) = await session.read_skills()
        declared = await session.declared_tools()

    assert isinstance(skill, Skill)
    assert skill.name == "127-0-0-1"
    assert skill.body == instructions
    assert skill.tools == tuple(tool.name for tool in declared)
    assert skill.name in skill.description
    assert skill.identity.server == app.url


async def test_an_app_with_the_extension_has_every_entry_of_its_listing_over_its_pages_in_order(app, bearer):
    files = {name: a_skill_file(name) for name in ("one", "two", "three")}
    entries = {name: an_entry(name, content) for name, content in files.items()}
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, a_listing([entries["one"], entries["two"]], "next"), a_listing([entries["three"]]))

    async with opened(app, bearer) as session:
        skills = await session.read_skills()

    assert all(isinstance(skill, SkillEntry) for skill in skills)
    assert named(skills) == ["one", "two", "three"]


async def test_an_empty_listing_is_no_skills_and_never_the_implicit_skill(app, bearer):
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, a_listing([]))

    async with opened(app, bearer) as session:
        assert await session.read_skills() == ()


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
            await session.read_skills()

    assert (failure.value.code, failure.value.entry) == (INVALID_ANSWER_CODE, index)
    assert marker not in f"{failure.value} {failure.value!r} {failure.value.answer}"


async def test_a_listing_page_that_is_no_list_of_entries_fails_with_no_index(app, bearer):
    async def lists_nothing(reply):
        await reply.raw_result({"skills": "none"})

    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, lists_nothing)

    async with opened(app, bearer) as session:
        with pytest.raises(UnusableAnswer) as failure:
            await session.read_skills()

    assert (failure.value.code, failure.value.entry) == (INVALID_ANSWER_CODE, None)


async def test_reading_the_skills_fetches_no_skill_file(app, bearer):
    content = a_skill_file("greeting")
    app.script(DISCOVER, WITH_THE_EXTENSION)
    app.script(SKILLS_LIST, a_listing([an_entry("greeting", content)]))

    async with opened(app, bearer) as session:
        await session.read_skills()

    assert app.log.count(READ) == 0


async def test_a_skill_whose_file_matches_its_entry_is_read_with_its_body_and_tools_under_the_apps_label(app, bearer):
    content = a_skill_file("greeting", body="Greet people by name.")
    entry = an_entry("greeting", content)
    app.script(SKILLS_GET, a_skill_answer(entry))
    app.script(READ, serving_file(content, entry["uri"]))

    async with opened(app, bearer) as session:
        skill = await session.read_skill(entry["uri"])

    assert (skill.name, skill.body, skill.tools) == ("greeting", "Greet people by name.", ("say", "play"))
    assert skill.identity.server == app.url
    assert app.log.count(READ) == 1


async def test_a_skill_whose_file_differs_from_its_entry_is_refused_with_its_code(app, bearer):
    content = a_skill_file("greeting", body="Greet people by name.")
    entry = an_entry("greeting", content)
    app.script(SKILLS_GET, a_skill_answer(entry))
    app.script(READ, serving_file(a_skill_file("greeting", body="Greet people by none."), entry["uri"]))

    async with opened(app, bearer) as session:
        with pytest.raises(SkillRefused) as refused:
            await session.read_skill(entry["uri"])

    assert refused.value.refusal == SkillRefusal.DIGEST_MISMATCH
