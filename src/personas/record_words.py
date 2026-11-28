"""The words of a record row as Postgres full text: the part of each kind of entry that a lookup matches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sqlalchemy import Text, case, func, literal, literal_column
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement

SIMPLE_CONFIGURATION: Final = literal_column("'simple'::regconfig", Text)
"""Postgres full text without stemming or stop words: a word is what the parser makes of the text, lower case."""

LINE_KIND: Final = "line"
_TEXT_KINDS: Final = (LINE_KIND, "speech")
_VIEW_KIND: Final = "view"
_CALL_KIND: Final = "act"
_JSON_VALUES: Final = literal_column("""'["string", "numeric", "boolean"]'::jsonb""", JSONB)
"""The kinds of JSON value a word is taken from; ``jsonb_to_tsvector`` takes no key with them."""
_NO_OBJECT: Final = literal_column("'{}'::jsonb", JSONB)


def _words_of(text: ColumnElement[object]) -> ColumnElement[object]:
    return func.to_tsvector(SIMPLE_CONFIGURATION, func.coalesce(text, ""), type_=TSVECTOR)


def _values_of(document: ColumnElement[object]) -> ColumnElement[object]:
    return func.jsonb_to_tsvector(
        SIMPLE_CONFIGURATION, func.coalesce(document, _NO_OBJECT), _JSON_VALUES, type_=TSVECTOR
    )


def entry_words(kind: ColumnElement[str], entry: ColumnElement[object]) -> ColumnElement[object]:
    """The words of a record row, each in its kind's own text: a line's and a sentence's text, a view's state values,
    and a call's tool name, its arguments' values and its result's text. Never a key, a kind, an app or a place.

    ``entry`` is the row's JSON column, whose subscripts are the JSON paths of the stored entries.
    """
    call = (
        _words_of(entry["tool"].astext)
        .op("||")(_values_of(entry["arguments"]))
        .op("||")(_words_of(entry["outcome"]["text"].astext))
    )
    return case(
        (kind.in_(_TEXT_KINDS), _words_of(entry["text"].astext)),
        (kind == _VIEW_KIND, _values_of(entry["view"]["state"])),
        (kind == _CALL_KIND, call),
        else_=_words_of(literal("")),
    )
