"""The operator's configuration: one TOML file named by ``PERSONAS_CONFIG``.

The file holds the realm's token endpoint, the ready file's path, the apps (an MCP URL and its scope) and the agents
(a client id and the path of the file holding its client secret, and optionally the ``figure`` table, a ``url`` and
a ``sha256``, that every accept of the agent carries, and the ``voice`` table, which turns the agent's spoken turns on
and may name the speech service's voice that every accept of the agent carries). Which apps an agent may reach
is the realm's group membership, never this file. An app that binds tools to its agents' turns
names them with their call bound, and then the model gateway and each agent's persona file are required.
An app with ``listens = false`` serves tools only: it starts no turn, and its bound tools reach the turns of the apps
that listen. An app that listens lends its bound tools to a turn of another app too, while the agent holds a place
there. ``call_seconds`` bounds each request a turn of another app makes to an app. No tool name is bound by two apps,
since one turn holds the tools of several, and no app names ``speak``, the voice's own tool, which is never offered to
a turn.
The top-level ``history_bytes`` bounds the history each agent keeps in memory of the lines he took in, the views he
read and the calls relayed for him at any of his apps and tells every turn.
The optional ``[memory]`` table names the Postgres database that keeps every line the agents read: its host, port,
database, user, the file holding the password, ``write_seconds``, the bound on one write or read, and how far back
(``today_hours``) and how much (``today_bytes``) a turn is told of the lines he kept that his history does not hold,
and how long a line is kept (``lines_keep_days``) and how often older ones are swept (``sweep_minutes``); ``inf``
keeps every line or sweeps once, at the start.
Every value is validated at start and an error names the field or path, never a secret.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from agents.agent_file import AgentDefinition, AgentFileError, load_agent
from agents.config import AgentConfig  # noqa: TC002 - pydantic resolves the field's type at run time
from agents.member import SPEAK_TOOL, FigureRef, VoiceName
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

from personas.decision import TURN_ROLE, TodayBounds
from personas.secret import Secret

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_ENVIRONMENT_VARIABLE: Final = "PERSONAS_CONFIG"
DEFAULT_CALL_SECONDS: Final = 30.0
DEFAULT_WRITE_SECONDS: Final = 2.0
DEFAULT_TODAY: Final = TodayBounds(hours=24.0, size_bytes=32768)
DEFAULT_HISTORY_BYTES: Final = 32768
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
class Voice:
    """An agent whose turns at the call places of an app that declares ``speak`` are spoken, in the voice ``name`` the
    speech service knows, or in the app's default voice."""

    name: VoiceName | None


@dataclass(frozen=True)
class AgentCredentials:
    client_id: str
    client_secret: Secret
    persona: AgentDefinition | None = None
    figure: FigureRef | None = field(default=None, repr=False)
    voice: Voice | None = None


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
    history_bytes: int = DEFAULT_HISTORY_BYTES


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
    def _a_tools_only_app_has_tools(self) -> Self:
        if not self.listens and not self.tools:
            raise ValueError("listens = false needs tools")
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


class _VoiceEntry(_Entry):
    name: VoiceName | None = None


class _AgentEntry(_Entry):
    client_id: str = Field(min_length=1)
    secret_file: Path
    persona_file: Path | None = None
    figure: FigureRef | None = None
    voice: _VoiceEntry | None = None


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
    history_bytes: PositiveInt = Field(default=DEFAULT_HISTORY_BYTES, strict=True)

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
        names = [name for app in apps for name in app.tools]
        if len(set(names)) != len(names):
            raise ValueError("a tool name is bound by two apps")
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
    if persona.voice_window_bytes is not None:
        raise ConfigError(
            f"{field_name}: voice: the voice is set in the agent's [agents.voice] table, not in the persona file"
        )
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
                agent.figure,
                None if agent.voice is None else Voice(agent.voice.name),
            )
            for index, agent in enumerate(parsed.agents)
        ),
        gateway=parsed.gateway,
        memory=_memory_target(parsed.memory),
        history_bytes=parsed.history_bytes,
    )
