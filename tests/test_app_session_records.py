"""A resource record is read with the keys this release does not know ignored, and refused on a wrong known field."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from agents.member import MemberStanding, PlaceRef, PlaceView
from mcp_types import ReadResourceResult, TextResourceContents

from personas.failures import INVALID_ANSWER_CODE, UnusableAnswer
from tests.support.app_session import opened
from tests.support.hopin import READ
from tests.support.standings import AGENT_SUBJECT as SUBJECT
from tests.support.standings import active, line, window_of

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.anyio


@pytest.fixture
def table():
    return PlaceRef(kind="table", id="table-1")


@dataclass(frozen=True)
class ResourceCase:
    """One resource record: the document an app shows, where a known field and a nested record sit, and its read."""

    document: dict
    known_field: tuple[str, ...]
    nested: tuple[str, ...]
    read: Callable


def _read_standing(session, _place):
    return session.read_standing()


@pytest.fixture(params=["standing", "window", "view"])
def resource(request, app, table):
    """The resource of one kind, served by the app as a document the test edits before it answers."""
    app.stand(SUBJECT, active(table.id, table.kind, serves_view=True))
    if request.param == "standing":
        document = MemberStanding(standings=(active(table.id, table.kind),)).model_dump(mode="json")
        return ResourceCase(document, ("standings", "0", "state"), ("standings", "0"), _read_standing)
    if request.param == "window":
        document = window_of(line(1, "hello")).model_dump(mode="json")
        return ResourceCase(document, ("entries", "0", "seq"), ("entries", "0"), lambda s, place: s.read_place(place))
    document = PlaceView(revision=3, awaits_you=True, state={"hand": [1]}).model_dump(mode="json")
    return ResourceCase(document, ("revision",), (), lambda s, place: s.read_view(place))


def _at(document, path):
    for step in path:
        document = document[int(step) if step.isdigit() else step]
    return document


async def reads_the_document(app, bearer, resource, table, document):
    async def shows(reply):
        contents = TextResourceContents(uri="member://x", text=json.dumps(document), mime_type="application/json")
        await reply.result(ReadResourceResult(contents=[contents], ttl_ms=0, cache_scope="private"))

    app.script(READ, shows)
    async with opened(app, bearer) as session:
        return await resource.read(session, table)


async def test_a_record_with_keys_this_release_does_not_know_at_any_depth_is_read_as_before(
    app, bearer, resource, table
):
    expected = await reads_the_document(app, bearer, resource, table, resource.document)
    served = copy.deepcopy(resource.document)
    _at(served, resource.nested)["added_by_the_app"] = {"deeper": [1]}
    served["added_by_the_app"] = "top"

    assert await reads_the_document(app, bearer, resource, table, served) == expected


async def test_a_record_with_a_wrong_known_field_is_an_unusable_answer(app, bearer, resource, table):
    served = copy.deepcopy(resource.document)
    _at(served, resource.known_field[:-1])[resource.known_field[-1]] = ["not", "valid"]

    with pytest.raises(UnusableAnswer) as failure:
        await reads_the_document(app, bearer, resource, table, served)

    assert failure.value.code == INVALID_ANSWER_CODE
