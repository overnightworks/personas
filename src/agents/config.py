"""The operator's configuration: one TOML file named by ``AGENTS_CONFIG``.

The file holds the realm's token endpoint, the apps (an MCP URL and its scope) and the agents
(a client id and the path of the file holding its client secret). Which apps an agent may reach
is the realm's group membership, never this file. Every value is validated at start and an
error names the field or path, never a secret.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError

from agents.secret import Secret

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_ENVIRONMENT_VARIABLE: Final = "AGENTS_CONFIG"


class ConfigError(Exception):
    """The configuration is missing or malformed; the message names the field or path."""


@dataclass(frozen=True)
class AppTarget:
    mcp_url: str
    scope: str


@dataclass(frozen=True)
class AgentCredentials:
    client_id: str
    client_secret: Secret


@dataclass(frozen=True)
class Config:
    token_endpoint: str
    apps: tuple[AppTarget, ...]
    agents: tuple[AgentCredentials, ...]


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _AppEntry(_Entry):
    mcp_url: AnyHttpUrl
    scope: str = Field(min_length=1)


class _AgentEntry(_Entry):
    client_id: str = Field(min_length=1)
    secret_file: Path


class _FileModel(_Entry):
    token_endpoint: AnyHttpUrl
    apps: tuple[_AppEntry, ...] = Field(min_length=1)
    agents: tuple[_AgentEntry, ...] = Field(min_length=1)

    @classmethod
    def parse(cls, text: str) -> Self:
        try:
            return cls.model_validate(tomllib.loads(text))
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"the file is not valid TOML: {error}") from None
        except ValidationError as error:
            # The error's input values stay out: only where the problem is and what kind it is.
            problems = (f"{'.'.join(str(part) for part in each['loc'])}: {each['msg']}" for each in error.errors())
            raise ConfigError("; ".join(problems)) from None


def _read_secret(index: int, path: Path) -> Secret:
    field_name = f"agents.{index}.secret_file"
    if not path.is_absolute():
        raise ConfigError(f"{field_name}: the path must be absolute")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        raise ConfigError(f"{field_name}: cannot read {path}") from None
    secret = text.strip()
    if not secret:
        raise ConfigError(f"{field_name}: {path} is empty")
    return Secret(secret)


def load_config(environment: Mapping[str, str]) -> Config:
    location = environment.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not location:
        raise ConfigError(f"{CONFIG_ENVIRONMENT_VARIABLE} names no configuration file")
    try:
        text = Path(location).read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        raise ConfigError(f"{CONFIG_ENVIRONMENT_VARIABLE}: cannot read {location}") from None
    parsed = _FileModel.parse(text)
    return Config(
        token_endpoint=str(parsed.token_endpoint),
        apps=tuple(AppTarget(str(app.mcp_url), app.scope) for app in parsed.apps),
        agents=tuple(
            AgentCredentials(agent.client_id, _read_secret(index, agent.secret_file))
            for index, agent in enumerate(parsed.agents)
        ),
    )
