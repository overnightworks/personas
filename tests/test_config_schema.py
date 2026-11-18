"""The configuration file's JSON schema, the one source the release notes read: each rule across fields is named."""

import pytest

from personas.config import ConfigError, configuration_schema, load_config

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
    ],
    ids=["app", "stop", "grace", "top level"],
)
def test_a_rule_across_fields_that_refuses_a_file_is_named_in_the_configuration_schema(files, break_it):
    files.text = break_it(files.text)

    with pytest.raises(ConfigError) as refused:
        load_config(files.write())

    assert [sentence for sentence in schema_refusals(configuration_schema()) if sentence in str(refused.value)]
