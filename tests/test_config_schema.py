"""The configuration file's JSON schema, the one source the release notes read: each rule across fields is named."""

import re
from typing import Self

import pytest
from pydantic import ValidationInfo, field_validator, model_validator

from personas.config import ConfigError, _Entry, configuration_schema, load_config

TOOLS_ONLY_APP = '[[apps]]\nmcp_url = "https://songmaker.test/mcp"\nscope = "mcp-songmaker"\nlistens = false\n'


def schema_refusals(node: object) -> set[str]:
    """Every sentence under ``refusals`` anywhere in a JSON schema."""
    if isinstance(node, list):
        return {sentence for each in node for sentence in schema_refusals(each)}
    if not isinstance(node, dict):
        return set()
    found = {sentence for value in node.values() for sentence in schema_refusals(value)}
    return found | set(node.get("refusals", []))


@pytest.mark.parametrize(
    "break_it",
    [
        lambda text: text + TOOLS_ONLY_APP,
        lambda text: text + "[ownership]\ncancel_margin_seconds = 1.0\nstop_seconds = 1.0\n",
        lambda text: text + "[ownership]\ngrace_seconds = 1.0\n",
        lambda text: f"history_bytes = 1024\nhistory_keep_bytes = 1025\n{text}",
        lambda text: re.sub(r"\[gateway\].*?(?=\[\[apps\]\])", "", text, flags=re.DOTALL),
        lambda text: re.sub(r"^persona_file = .*\n", "", text, flags=re.MULTILINE),
        lambda text: re.sub(r"^persona_file = .*\n", 'model = "text/sonnet-low"\n', text, flags=re.MULTILINE),
        lambda text: re.sub(r"^persona_file = .*\n", 'hold_model = "text/sonnet-low"\n', text, flags=re.MULTILINE),
    ],
    ids=["app", "stop", "grace", "top level", "gateway", "persona file", "model", "hold model"],
)
def test_a_rule_across_fields_that_refuses_a_file_is_named_in_the_configuration_schema(files, break_it):
    files.text = break_it(files.text)
    path = files.write()

    with pytest.raises(ConfigError) as refused:
        load_config(path)

    assert [sentence for sentence in schema_refusals(configuration_schema()) if sentence in str(refused.value)]


class CheckedByTheModel(_Entry):
    low: int = 1
    high: int = 2

    @model_validator(mode="after")
    def _low_is_below_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("high: must be above low")
        return self


class CheckedByAField(_Entry):
    low: int = 1
    high: int = 2

    @field_validator("high")
    @classmethod
    def _high_is_above_low(cls, high: int, info: ValidationInfo) -> int:
        if high <= info.data["low"]:
            raise ValueError("high: must be above low")
        return high


@pytest.mark.parametrize(
    ("entry", "check"),
    [(CheckedByTheModel, "_low_is_below_high"), (CheckedByAField, "_high_is_above_low")],
    ids=["model validator", "field validator reading the fields before it"],
)
def test_a_check_across_fields_that_is_no_refusal_makes_the_schema_refuse_to_be_written(entry, check):
    with pytest.raises(TypeError, match=check):
        entry.model_json_schema()
