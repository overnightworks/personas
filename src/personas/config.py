"""The operator's configuration: one TOML file named by ``PERSONAS_CONFIG``.

The file holds the realm's token endpoint, the ready file's path, the apps (an MCP URL and its scope) and the agents
(a client id and the path of the file holding its client secret). Which apps an agent may reach
is the realm's group membership, never this file. An app that binds tools to its agents' turns
names them with their call bound, and then the model gateway and each agent's persona file are required.
An app with ``listens = false`` serves tools only: it starts no turn, its bound tools reach the turns of the apps that
listen, and ``call_seconds`` bounds each request to it. No tool name is shared with a tools-only app.
An agent has a turn cap only when it sets ``max_turns`` in ``turn_period_seconds``, both or neither;
with neither its turns are bounded by the budget and the off switches alone.
Every value is validated at start and an error names the field or path, never a secret.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from agent_runtime_client.agents import AgentDefinition, AgentFileError, load_agent
from agent_runtime_client.config import AgentConfig  # noqa: TC002 - pydantic resolves the field's type at run time
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

from personas.decision import TURN_ROLE, TurnCap
from personas.secret import Secret

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_ENVIRONMENT_VARIABLE: Final = "PERSONAS_CONFIG"
DEFAULT_CALL_SECONDS: Final = 30.0


class ConfigError(Exception):
    """The configuration is missing or malformed; the message names the field or path."""


@dataclass(frozen=True)
class ToolBinding:
    """The tools of an app that its agents' turns may call, and how many calls one turn may make."""

    tools: tuple[str, ...]
    max_tool_calls: int


@dataclass(frozen=True)
class AppTarget:
    mcp_url: str
    scope: str
    binding: ToolBinding | None = None
    listens: bool = True
    call_seconds: float = DEFAULT_CALL_SECONDS


@dataclass(frozen=True)
class AgentCredentials:
    client_id: str
    client_secret: Secret
    persona: AgentDefinition | None = None
    turn_cap: TurnCap | None = None


@dataclass(frozen=True)
class Config:
    token_endpoint: str
    ready_file: Path
    apps: tuple[AppTarget, ...]
    agents: tuple[AgentCredentials, ...]
    gateway: AgentConfig | None = None


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _AppEntry(_Entry):
    mcp_url: AnyHttpUrl
    scope: str = Field(min_length=1)
    tools: tuple[str, ...] = ()
    max_tool_calls: PositiveInt | None = None
    listens: bool = True
    call_seconds: PositiveFloat | None = None

    @model_validator(mode="after")
    def _a_tools_only_app_has_tools_and_only_it_a_call_time(self) -> Self:
        if not self.listens and not self.tools:
            raise ValueError("listens = false needs tools")
        if self.listens and self.call_seconds is not None:
            raise ValueError("call_seconds is for an app that does not listen")
        return self

    @model_validator(mode="after")
    def _tools_come_with_their_call_bound(self) -> Self:
        if bool(self.tools) != (self.max_tool_calls is not None):
            raise ValueError("tools and max_tool_calls are set together")
        if len(set(self.tools)) != len(self.tools) or not all(self.tools):
            raise ValueError("tools holds each name once and no empty name")
        return self

    @property
    def binding(self) -> ToolBinding | None:
        if self.max_tool_calls is None:
            return None
        return ToolBinding(self.tools, self.max_tool_calls)


class _AgentEntry(_Entry):
    client_id: str = Field(min_length=1)
    secret_file: Path
    persona_file: Path | None = None
    max_turns: PositiveInt | None = Field(default=None, strict=True)
    turn_period_seconds: PositiveFloat | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def _turn_cap_amounts_come_together(self) -> Self:
        for present, missing in (("max_turns", "turn_period_seconds"), ("turn_period_seconds", "max_turns")):
            if getattr(self, present) is not None and getattr(self, missing) is None:
                raise ValueError(f"{missing}: required when {present} is set")
        return self

    @property
    def turn_cap(self) -> TurnCap | None:
        if self.max_turns is None or self.turn_period_seconds is None:
            return None
        return TurnCap(self.max_turns, self.turn_period_seconds)


class _FileModel(_Entry):
    token_endpoint: AnyHttpUrl
    ready_file: Path
    apps: tuple[_AppEntry, ...] = Field(min_length=1)
    agents: tuple[_AgentEntry, ...] = Field(min_length=1)
    gateway: AgentConfig | None = None

    @model_validator(mode="after")
    def _tools_need_a_gateway_and_personas(self) -> Self:
        if any(app.binding for app in self.apps):
            if self.gateway is None:
                raise ValueError("gateway: required when an app binds tools")
            if any(agent.persona_file is None for agent in self.agents):
                raise ValueError("persona_file: required for every agent when an app binds tools")
        return self

    @field_validator("ready_file")
    @classmethod
    def _ready_file_is_absolute(cls, ready_file: Path) -> Path:
        if not ready_file.is_absolute():
            raise ValueError("the path must be absolute")
        return ready_file

    @field_validator("apps")
    @classmethod
    def _one_entry_per_app(cls, apps: tuple[_AppEntry, ...]) -> tuple[_AppEntry, ...]:
        if len({app.mcp_url for app in apps}) != len(apps):
            raise ValueError("an mcp_url is listed twice")
        return apps

    @field_validator("apps")
    @classmethod
    def _a_turn_holds_no_tool_name_twice(cls, apps: tuple[_AppEntry, ...]) -> tuple[_AppEntry, ...]:
        for index, app in enumerate(apps):
            held_elsewhere = {name for other in apps[:index] + apps[index + 1 :] for name in other.tools}
            if not app.listens and held_elsewhere & set(app.tools):
                raise ValueError("a tools-only app shares a tool name with another app")
        return apps

    @field_validator("apps")
    @classmethod
    def _some_app_listens(cls, apps: tuple[_AppEntry, ...]) -> tuple[_AppEntry, ...]:
        if not any(app.listens for app in apps):
            raise ValueError("no app listens")
        return apps

    @field_validator("agents")
    @classmethod
    def _one_entry_per_agent(cls, agents: tuple[_AgentEntry, ...]) -> tuple[_AgentEntry, ...]:
        if len({agent.client_id for agent in agents}) != len(agents):
            raise ValueError("a client_id is listed twice")
        return agents

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


def _read_persona(index: int, path: Path | None) -> AgentDefinition | None:
    if path is None:
        return None
    field_name = f"agents.{index}.persona_file"
    if not path.is_absolute():
        raise ConfigError(f"{field_name}: the path must be absolute")
    try:
        persona = load_agent(path, [])
    except AgentFileError as error:
        raise ConfigError(f"{field_name}: {error}") from None
    except OSError:
        raise ConfigError(f"{field_name}: cannot read {path}") from None
    if not persona.offered_to(TURN_ROLE):
        raise ConfigError(f"{field_name}: appRoles must include {TURN_ROLE}, the role a turn acts under")
    if persona.picture_size is not None:
        raise ConfigError(f"{field_name}: a picture agent cannot take a turn")
    return persona


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
        ready_file=parsed.ready_file,
        apps=tuple(
            AppTarget(str(app.mcp_url), app.scope, app.binding, app.listens, app.call_seconds or DEFAULT_CALL_SECONDS)
            for app in parsed.apps
        ),
        agents=tuple(
            AgentCredentials(
                agent.client_id,
                _read_secret(index, agent.secret_file),
                _read_persona(index, agent.persona_file),
                agent.turn_cap,
            )
            for index, agent in enumerate(parsed.agents)
        ),
        gateway=parsed.gateway,
    )
