"""The configuration file, read through the entry point: a bad value stops the start and names itself."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

import pytest

from agents.config import CONFIG_ENVIRONMENT_VARIABLE, load_config
from agents.main import CONFIGURATION_ERROR_EXIT_CODE, main

if TYPE_CHECKING:
    from pathlib import Path

APP_TABLE = '[[apps]]\nmcp_url = "https://hopin.test/mcp"\nscope = "mcp-hopin"\n'


class Files:
    """A configuration directory: a secret file and a TOML that points at it, each changeable by a test."""

    def __init__(self, directory: Path) -> None:
        self.secret = secrets.token_hex(16)
        self.secret_path = directory / "mozart.secret"
        self.secret_path.write_text(f"{self.secret}\n", encoding="utf-8")
        self.config_path = directory / "agents.toml"
        self.text = self.valid_text()

    def valid_text(self) -> str:
        return (
            'token_endpoint = "https://realm.test/protocol/openid-connect/token"\n'
            '[[apps]]\nmcp_url = "https://hopin.test/mcp"\nscope = "mcp-hopin"\n'
            f'[[agents]]\nclient_id = "agent-mozart"\nsecret_file = "{self.secret_path}"\n'
        )

    def write(self) -> dict[str, str]:
        self.config_path.write_text(self.text, encoding="utf-8")
        return {CONFIG_ENVIRONMENT_VARIABLE: str(self.config_path)}


@pytest.fixture
def files(tmp_path: Path) -> Files:
    return Files(tmp_path)


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
        (lambda text: text.replace("[[agents]]", "[[agents"), "TOML"),
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
        "not TOML",
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
