"""The configuration file, read through the entry point: a bad value stops the start and names itself."""

from __future__ import annotations

import pytest

from agents.config import CONFIG_ENVIRONMENT_VARIABLE, ToolBinding, load_config
from agents.main import CONFIGURATION_ERROR_EXIT_CODE, main
from tests.support.configfiles import MAX_TOOL_CALLS, PERSONA_NAME
from tests.support.realm import API_GROUP

APP_TABLE = (
    '[[apps]]\nmcp_url = "https://hopin.test/mcp"\nscope = "mcp-hopin"\n'
    f'tools = ["say"]\nmax_tool_calls = {MAX_TOOL_CALLS}\n'
)


def _with_ready_file(text: str, value: str | None) -> str:
    """The valid text with its ready_file line removed (None) or set to a value."""
    others = [line for line in text.splitlines(keepends=True) if not line.startswith("ready_file")]
    return "".join(others if value is None else [f'ready_file = "{value}"\n', *others])


def fails_at_start(environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == CONFIGURATION_ERROR_EXIT_CODE
    return capsys.readouterr().err


def test_a_valid_file_gives_every_agent_its_secret_and_every_app_its_scope(files):
    config = load_config(files.write())

    assert config.token_endpoint.startswith("https://realm.test/")
    assert config.ready_file == files.ready_path
    assert [(app.mcp_url, app.scope) for app in config.apps] == [("https://hopin.test/mcp", "mcp-hopin")]
    assert [(agent.client_id, agent.client_secret.reveal()) for agent in config.agents] == [
        ("agent-mozart", files.secret)
    ]


def test_an_environment_without_the_variable_stops_the_start(monkeypatch, capsys):
    error = fails_at_start({}, monkeypatch, capsys)

    assert CONFIG_ENVIRONMENT_VARIABLE in error


def test_a_file_that_cannot_be_read_stops_the_start_and_names_its_path(files, monkeypatch, capsys):
    missing = files.config_path.parent / "nowhere.toml"

    error = fails_at_start({CONFIG_ENVIRONMENT_VARIABLE: str(missing)}, monkeypatch, capsys)

    assert str(missing) in error


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (lambda text: text.replace("token_endpoint", "token_url"), "token_endpoint"),
        (lambda text: text.replace("https://realm.test/protocol/openid-connect/token", "realm"), "token_endpoint"),
        (lambda text: text.replace('mcp_url = "https://hopin.test/mcp"', 'mcp_url = "ftp://x"'), "apps.0.mcp_url"),
        (lambda text: text.replace('scope = "mcp-hopin"', 'scope = ""'), "apps.0.scope"),
        (lambda text: text.replace('scope = "mcp-hopin"\n', ""), "apps.0.scope"),
        (lambda text: text.replace('client_id = "agent-mozart"', 'client_id = ""'), "agents.0.client_id"),
        (lambda text: text.replace('client_id = "agent-mozart"\n', ""), "agents.0.client_id"),
        (lambda text: text.replace("secret_file", "secret"), "agents.0.secret_file"),
        (lambda text: text.replace(APP_TABLE, ""), "apps"),
        (lambda text: text.split("[[agents]]")[0], "agents"),
        (lambda text: "apps = []\n" + text.replace(APP_TABLE, ""), "apps"),
        (lambda text: "agents = []\n" + text.split("[[agents]]")[0], "agents"),
        (lambda text: text + "unknown = 1\n", "unknown"),
        (lambda text: text + APP_TABLE, "apps"),
        (lambda text: text + "[[agents]]" + text.split("[[agents]]")[1], "agents"),
        (lambda text: text.replace("[[agents]]", "[[agents"), "TOML"),
        (lambda text: _with_ready_file(text, None), "ready_file"),
        (lambda text: _with_ready_file(text, ""), "ready_file"),
        (lambda text: _with_ready_file(text, "ready"), "ready_file"),
    ],
    ids=[
        "token endpoint missing",
        "token endpoint not a URL",
        "app URL not http",
        "app scope empty",
        "app scope missing",
        "client id empty",
        "client id missing",
        "secret file missing",
        "no apps",
        "no agents",
        "empty list of apps",
        "empty list of agents",
        "unknown field",
        "app URL listed twice",
        "client id listed twice",
        "not TOML",
        "ready file missing",
        "ready file empty",
        "ready file relative",
    ],
)
def test_a_missing_or_malformed_field_stops_the_start_and_names_it(files, monkeypatch, capsys, break_it, named):
    files.text = break_it(files.text)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert files.secret not in error


@pytest.mark.parametrize(
    "arrange",
    [
        lambda files: files.secret_path.unlink(),
        lambda files: files.secret_path.write_text("  \n", encoding="utf-8"),
        lambda files: files.secret_path.write_bytes(b"\xff\xfe"),
        lambda files: setattr(files, "text", files.text.replace(str(files.secret_path), "mozart.secret")),
    ],
    ids=["missing", "empty", "not text", "relative path"],
)
def test_a_secret_file_that_is_unusable_stops_the_start_and_names_the_field(files, monkeypatch, capsys, arrange):
    arrange(files)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "agents.0.secret_file" in error


def test_a_secret_written_into_the_file_is_refused_without_being_echoed(files, monkeypatch, capsys):
    files.text += f'client_secret = "{files.secret}"\n'

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "client_secret" in error
    assert files.secret not in error


def test_a_valid_file_gives_the_agent_its_persona_the_app_its_tools_and_the_runner_its_gateway(files):
    config = load_config(files.write())

    [agent], [app] = config.agents, config.apps
    assert agent.persona is not None
    assert (agent.persona.name, agent.persona.prompt) == (PERSONA_NAME, files.persona_prompt)
    assert app.binding == ToolBinding(("say",), MAX_TOOL_CALLS)
    assert config.gateway is not None
    assert (config.gateway.api_group, config.gateway.gateway.key_file) == (API_GROUP, files.key_path)


def test_a_configuration_without_tools_in_any_app_needs_no_gateway_and_no_persona(files):
    files.text = files.valid_text(binds_tools=False)

    config = load_config(files.write())

    assert config.gateway is None
    assert [app.binding for app in config.apps] == [None]
    assert [agent.persona for agent in config.agents] == [None]


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (lambda text: text.replace(f"max_tool_calls = {MAX_TOOL_CALLS}\n", ""), "apps.0"),
        (lambda text: text.replace('tools = ["say"]\n', ""), "apps.0"),
        (
            lambda text: text.replace(f"max_tool_calls = {MAX_TOOL_CALLS}", "max_tool_calls = 0"),
            "apps.0.max_tool_calls",
        ),
        (lambda text: text.replace('tools = ["say"]', 'tools = ["say", "say"]'), "apps.0"),
        (lambda text: text.replace('tools = ["say"]', 'tools = [""]'), "apps.0"),
        (lambda text: text.replace('[gateway]\nbackend = "litellm"\n', '[gatewa]\nbackend = "litellm"\n'), "gatewa"),
        (lambda text: text.split("[gateway]")[0] + "[[apps]]" + text.split("[[apps]]")[1], "gateway"),
        (lambda text: text.replace('backend = "litellm"', 'backend = "other"'), "gateway.backend"),
        (lambda text: text.replace("origin = ", "origins = "), "gateway.gateway"),
        (lambda text: text.replace("persona_file", "persona"), "persona"),
        (lambda text: "\n".join(line for line in text.splitlines() if "persona_file" not in line), "persona_file"),
    ],
    ids=[
        "tools without a call bound",
        "a call bound without tools",
        "no call allowed",
        "a tool listed twice",
        "an empty tool name",
        "unknown gateway key",
        "tools without a gateway",
        "unknown backend",
        "gateway origin missing",
        "persona file misnamed",
        "tools without a persona file",
    ],
)
def test_tools_need_their_call_bound_and_the_gateway_and_a_persona_to_run_with(
    files, monkeypatch, capsys, break_it, named
):
    files.text = break_it(files.text)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert files.secret not in error
    assert files.gateway_key not in error


@pytest.mark.parametrize(
    "arrange",
    [
        lambda files: files.persona_path.unlink(),
        lambda files: files.persona_path.write_text("no front matter\n", encoding="utf-8"),
        lambda files: files.persona_path.write_text(files.persona_text().replace("name: mozart", "name: bach")),
        lambda files: files.persona_path.write_text(
            files.persona_text().replace("---\n" + files.persona_prompt, "---\n")
        ),
        lambda files: files.persona_path.write_text(files.persona_text().replace("[admin, user]", "[user]")),
        lambda files: files.persona_path.write_text(files.persona_text().replace("[admin, user]", "[admin]")),
        lambda files: files.persona_path.write_text(
            files.persona_text()
            .replace("model: text/", "model: image/")
            .replace("---\n" + files.persona_prompt, "picture:\n  size: 64x64\n---\n" + files.persona_prompt)
        ),
        lambda files: files.persona_path.write_text(
            files.persona_text().replace("---\n" + files.persona_prompt, "tools: [say]\n---\n")
        ),
        lambda files: files.persona_path.write_text(
            files.persona_text().replace("maxDurationSeconds: 30", "maxDurationSeconds: 30\nmaxToolCalls: 3")
        ),
        lambda files: setattr(files, "text", files.text.replace(str(files.persona_path), "mozart.md")),
    ],
    ids=[
        "missing",
        "no front matter",
        "name not the file's",
        "no prompt",
        "roles invalid",
        "no user role",
        "picture agent",
        "undeclared tool",
        "call bound without tools",
        "relative path",
    ],
)
def test_a_persona_file_that_is_unusable_stops_the_start_and_names_the_field(files, monkeypatch, capsys, arrange):
    arrange(files)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "agents.0.persona_file" in error
    assert files.persona_prompt not in error
