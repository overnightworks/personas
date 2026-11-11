"""The operator's configuration: one TOML file named by ``PERSONAS_CONFIG``.

The file holds the realm's token endpoint, the ready file's path, the apps (an MCP URL and its scope) and the agents
(a client id and the path of the file holding its client secret). Which apps an agent may reach
is the realm's group membership, never this file. An app that binds tools to its agents' turns
names them with their call bound, and then the model gateway and each agent's persona file are required.
An app with ``listens = false`` serves tools only: it starts no turn, its bound tools reach the turns of the apps that
listen, and ``call_seconds`` bounds each request to it. No tool name is shared with a tools-only app, and no app names
``speak``, the voice's own tool, which is never offered to a turn.
An agent has a turn cap only when it sets ``max_turns`` in ``turn_period_seconds``, both or neither;
with neither its turns are bounded by the budget and the off switches alone.
The optional ``[memory]`` table names the Postgres database that keeps every line the agents read: its host, port,
database, user, the file holding the password, ``write_seconds``, the bound on one write or read, and how far back
(``today_hours``) and how much (``today_bytes``) a turn is told of the lines of the agent's other visits, and how long
a line is kept (``lines_keep_days``) and how often older ones are swept (``sweep_minutes``); ``inf`` keeps every line
or sweeps once, at the start.
Every value is validated at start and an error names the field or path, never a secret.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from agents.agent_file import AgentDefinition, AgentFileError, load_agent
from agents.config import AgentConfig  # noqa: TC002 - pydantic resolves the field's type at run time
from agents.member import SPEAK_TOOL
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

from personas.decision import TURN_ROLE, TodayBounds, TurnCap
from personas.secret import Secret

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_ENVIRONMENT_VARIABLE: Final = "PERSONAS_CONFIG"
DEFAULT_CALL_SECONDS: Final = 30.0
DEFAULT_WRITE_SECONDS: Final = 2.0
DEFAULT_TODAY: Final = TodayBounds(hours=24.0, size_bytes=32768)
DEFAULT_LINES_KEEP_DAYS: Final = 30.0
DEFAULT_SWEEP_MINUTES: Final = 60.0


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
class MemoryTarget:
    """The database that keeps the lines, how long one write or read of it may take, what a turn is told of the day
    and how long a line is kept, with how often older ones are swept."""

    host: str
    port: int
    dbname: str
    user: str
    password: Secret
    write_seconds: float = DEFAULT_WRITE_SECONDS
    today: TodayBounds = DEFAULT_TODAY
    lines_keep_days: float = DEFAULT_LINES_KEEP_DAYS
    sweep_minutes: float = DEFAULT_SWEEP_MINUTES


@dataclass(frozen=True)
class Config:
    token_endpoint: str
    ready_file: Path
    apps: tuple[AppTarget, ...]
    agents: tuple[AgentCredentials, ...]
    gateway: AgentConfig | None = None
    memory: MemoryTarget | None = None


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

    @field_validator("tools")
    @classmethod
    def _speak_is_never_offered_to_the_model(cls, tools: tuple[str, ...]) -> tuple[str, ...]:
        if SPEAK_TOOL in tools:
            raise ValueError(f"{SPEAK_TOOL} is the voice's own tool and is never offered to a turn")
        return tools

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


class _MemoryEntry(_Entry):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535, strict=True)
    dbname: str = Field(min_length=1)
    user: str = Field(min_length=1)
    password_file: Path
    write_seconds: PositiveFloat = Field(default=DEFAULT_WRITE_SECONDS, strict=True)
    today_hours: PositiveFloat = Field(default=DEFAULT_TODAY.hours, strict=True)
    today_bytes: PositiveInt = Field(default=DEFAULT_TODAY.size_bytes, strict=True)
    lines_keep_days: PositiveFloat = Field(default=DEFAULT_LINES_KEEP_DAYS, strict=True)
    sweep_minutes: PositiveFloat = Field(default=DEFAULT_SWEEP_MINUTES, strict=True)


class _FileModel(_Entry):
    token_endpoint: AnyHttpUrl
    ready_file: Path
    apps: tuple[_AppEntry, ...] = Field(min_length=1)
    agents: tuple[_AgentEntry, ...] = Field(min_length=1)
    gateway: AgentConfig | None = None
    memory: _MemoryEntry | None = None

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


def _read_secret(field_name: str, path: Path) -> Secret:
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


def _memory_target(entry: _MemoryEntry | None) -> MemoryTarget | None:
    if entry is None:
        return None
    password = _read_secret("memory.password_file", entry.password_file)
    today = TodayBounds(entry.today_hours, entry.today_bytes)
    return MemoryTarget(
        entry.host,
        entry.port,
        entry.dbname,
        entry.user,
        password,
        entry.write_seconds,
        today,
        entry.lines_keep_days,
        entry.sweep_minutes,
    )


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
                _read_secret(f"agents.{index}.secret_file", agent.secret_file),
                _read_persona(index, agent.persona_file),
                agent.turn_cap,
            )
            for index, agent in enumerate(parsed.agents)
        ),
        gateway=parsed.gateway,
        memory=_memory_target(parsed.memory),
    )
