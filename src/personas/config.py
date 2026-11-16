"""The operator's configuration: one TOML file named by ``PERSONAS_CONFIG``.

The file holds the realm's token endpoint, the ready file's path, the apps (an MCP URL and its scope) and the agents
(a client id and the path of the file holding its client secret, and optionally the ``figure`` table, a ``url`` and
a ``sha256``, that every accept of the agent carries, and the ``voice`` table, which turns the agent's spoken turns on
and may name, in ``names``, an ordered list of the speech service's voices that every accept of the agent carries
as it stands, first choice first). Which apps an agent may reach
is the realm's group membership, never this file, and what an agent may do in an app is the app's, through the tools
it offers and the role in the token. An app that sets ``max_tool_calls`` binds every tool it offers to its agents'
turns, that many calls to it in one turn, and then the model gateway and each agent's persona file are required.
An app with ``listens = false`` serves tools only: it starts no turn, and its tools reach the turns of the apps
that listen. An app that listens lends its tools to a turn of another app too, while the agent holds a place
there. ``call_seconds`` bounds each request a turn of another app makes to an app. A tool name two apps offer goes to
the first app in this file, and the runner's own names, ``speak``, the voice's tool, and ``recall``, the lookup, which
every turn holds while ``[memory]`` is set, are never taken from an app.
The top-level ``history_bytes`` bounds the history each agent keeps in memory of the lines he took in, the views he
read and the calls relayed for him at any of his apps and tells every turn; past it the oldest messages leave in one
step until what stays fits ``history_keep_bytes``, which is never above ``history_bytes``. The top-level
``budget_pause_cap_seconds`` is the longest wait between his turns while the gateway refuses them for a spent budget.
The required ``[database]`` table names the Postgres database the runner shares with its other pods: its host, port,
database, user, the file holding the password and ``write_seconds``, the bound on one write or read. A pod serves an
agent only while it holds that agent's advisory lock there. The optional ``[ownership]`` table sets how the pods
share the agents: ``contend_seconds`` is how often a pod checks its locks and takes the free ones,
``heartbeat_timeout_seconds`` how long one such check may take, ``cancel_margin_seconds`` the time a pod that lost
its hold is given to stop its loops and ``grace_seconds`` how long a pod that took an agent waits before it
starts; the grace is above the other three together, so that the pod that lost the agent has stopped by then.
The optional ``[memory]`` table keeps every line the agents read in that database: how long a line is
kept (``lines_keep_days``) and how often older ones are swept (``sweep_minutes``); ``inf`` keeps every line or sweeps
once, at the start. A turn may look up what was said in its place before its history (``lookup_calls``, the lookups one
turn makes, and ``lookup_bytes``, the bytes of one result and the longest text a lookup accepts); both are integers,
the calls above 0 and the bytes at least what a result without a line takes, and neither turns the lookup off.
Every value is validated at start and an error names the field or path, never a secret.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from agents.agent_file import AgentDefinition, AgentFileError, load_agent
from agents.config import AgentConfig  # noqa: TC002 - pydantic resolves the field's type at run time
from agents.member import FigureRef, VoiceName, Voices  # noqa: TC002 - pydantic resolves the fields' types at run time
from agents.turn import GatewayModelName, ModelKind, model_kind_of
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from personas.decision import BACKOFF_BASE_SECONDS, DEFAULT_BUDGET_PAUSE_CAP_SECONDS, TURN_ROLE
from personas.recall import SMALLEST_RESULT_BYTES
from personas.secret import Secret

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic_core import ErrorDetails

CONFIG_ENVIRONMENT_VARIABLE: Final = "PERSONAS_CONFIG"
DEFAULT_CALL_SECONDS: Final = 30.0
DEFAULT_WRITE_SECONDS: Final = 2.0
DEFAULT_HISTORY_BYTES: Final = 49152
DEFAULT_HISTORY_KEEP_BYTES: Final = 32768
DEFAULT_LINES_KEEP_DAYS: Final = 30.0
DEFAULT_SWEEP_MINUTES: Final = 60.0
DEFAULT_LOOKUP_CALLS: Final = 2
DEFAULT_LOOKUP_BYTES: Final = 8192
DEFAULT_CONTEND_SECONDS: Final = 5.0
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS: Final = 3.0
DEFAULT_CANCEL_MARGIN_SECONDS: Final = 2.0
DEFAULT_GRACE_SECONDS: Final = 12.0


class ConfigError(Exception):
    """The configuration is missing or malformed; the message names the field or path."""


@dataclass(frozen=True)
class AppTarget:
    mcp_url: str
    scope: str
    max_tool_calls: int | None = None
    listens: bool = True
    call_seconds: float = DEFAULT_CALL_SECONDS


@dataclass(frozen=True)
class Voice:
    """An agent whose turns at the call places of an app that declares ``speak`` are spoken, in the first of the voices
    ``names`` that the app offers, or in the app's default voice when there are none."""

    names: tuple[VoiceName, ...]


@dataclass(frozen=True)
class AgentCredentials:
    client_id: str
    client_secret: Secret
    persona: AgentDefinition | None = None
    figure: FigureRef | None = field(default=None, repr=False)
    voice: Voice | None = field(default=None, repr=False)


@dataclass(frozen=True)
class DatabaseTarget:
    """The Postgres database every pod shares, and how long one write or read of it may take."""

    host: str
    port: int
    dbname: str
    user: str
    password: Secret
    write_seconds: float = DEFAULT_WRITE_SECONDS


@dataclass(frozen=True)
class OwnershipTimings:
    """How the pods share the agents, in seconds; the grace is above the other three together."""

    contend_seconds: float = DEFAULT_CONTEND_SECONDS
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    cancel_margin_seconds: float = DEFAULT_CANCEL_MARGIN_SECONDS
    grace_seconds: float = DEFAULT_GRACE_SECONDS


@dataclass(frozen=True)
class MemoryTarget:
    """How long a line is kept, how often older ones are swept, and what a turn's lookups may take."""

    lines_keep_days: float = DEFAULT_LINES_KEEP_DAYS
    sweep_minutes: float = DEFAULT_SWEEP_MINUTES
    lookup_calls: int = DEFAULT_LOOKUP_CALLS
    lookup_bytes: int = DEFAULT_LOOKUP_BYTES


@dataclass(frozen=True)
class Config:
    token_endpoint: str
    ready_file: Path
    apps: tuple[AppTarget, ...]
    agents: tuple[AgentCredentials, ...]
    database: DatabaseTarget
    ownership: OwnershipTimings = OwnershipTimings()
    gateway: AgentConfig | None = None
    memory: MemoryTarget | None = None
    history_bytes: int = DEFAULT_HISTORY_BYTES
    history_keep_bytes: int = DEFAULT_HISTORY_KEEP_BYTES
    budget_pause_cap_seconds: float = DEFAULT_BUDGET_PAUSE_CAP_SECONDS


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _AppEntry(_Entry):
    mcp_url: AnyHttpUrl
    scope: str = Field(min_length=1)
    max_tool_calls: PositiveInt | None = None
    listens: bool = True
    call_seconds: PositiveFloat | None = None

    @model_validator(mode="after")
    def _a_tools_only_app_has_a_call_bound(self) -> Self:
        if not self.listens and self.max_tool_calls is None:
            raise ValueError("listens = false needs max_tool_calls")
        return self


class _VoiceEntry(_Entry):
    names: Voices | None = None


class _AgentEntry(_Entry):
    client_id: str = Field(min_length=1)
    secret_file: Path
    persona_file: Path | None = None
    model: GatewayModelName | None = None
    figure: FigureRef | None = None
    voice: _VoiceEntry | None = None

    @field_validator("model")
    @classmethod
    def _model_is_a_text_model_of_an_agent_file(cls, model: str, info: ValidationInfo) -> str:
        if model_kind_of(model) is not ModelKind.TEXT:
            raise ValueError(f"must start with {ModelKind.TEXT}/")
        if info.data.get("persona_file") is None:
            raise ValueError("needs a persona_file, whose agent it names the model of")
        return model


class _DatabaseEntry(_Entry):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535, strict=True)
    dbname: str = Field(min_length=1)
    user: str = Field(min_length=1)
    password_file: Path
    write_seconds: PositiveFloat = Field(default=DEFAULT_WRITE_SECONDS, strict=True)


class _OwnershipEntry(_Entry):
    contend_seconds: PositiveFloat = Field(default=DEFAULT_CONTEND_SECONDS, strict=True, allow_inf_nan=False)
    heartbeat_timeout_seconds: PositiveFloat = Field(
        default=DEFAULT_HEARTBEAT_TIMEOUT_SECONDS, strict=True, allow_inf_nan=False
    )
    cancel_margin_seconds: PositiveFloat = Field(
        default=DEFAULT_CANCEL_MARGIN_SECONDS, strict=True, allow_inf_nan=False
    )
    grace_seconds: PositiveFloat = Field(default=DEFAULT_GRACE_SECONDS, strict=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def _the_grace_outlasts_the_hold_it_takes_over_from(self) -> Self:
        """A pod that lost its lock notices within a contend interval and a heartbeat timeout and then needs its margin
        to stop; a taker that starts sooner than that could speak beside it."""
        stopped_by = self.contend_seconds + self.heartbeat_timeout_seconds + self.cancel_margin_seconds
        if self.grace_seconds <= stopped_by:
            raise ValueError(
                "grace_seconds: must be above contend_seconds + heartbeat_timeout_seconds + cancel_margin_seconds"
            )
        return self


class _MemoryEntry(_Entry):
    lines_keep_days: PositiveFloat = Field(default=DEFAULT_LINES_KEEP_DAYS, strict=True)
    sweep_minutes: PositiveFloat = Field(default=DEFAULT_SWEEP_MINUTES, strict=True)
    lookup_calls: PositiveInt = Field(default=DEFAULT_LOOKUP_CALLS, strict=True)
    lookup_bytes: int = Field(default=DEFAULT_LOOKUP_BYTES, strict=True, ge=SMALLEST_RESULT_BYTES)


class _FileModel(_Entry):
    token_endpoint: AnyHttpUrl
    ready_file: Path
    apps: tuple[_AppEntry, ...] = Field(min_length=1)
    agents: tuple[_AgentEntry, ...] = Field(min_length=1)
    database: _DatabaseEntry
    ownership: _OwnershipEntry = _OwnershipEntry()
    gateway: AgentConfig | None = None
    memory: _MemoryEntry | None = None
    history_bytes: PositiveInt = Field(default=DEFAULT_HISTORY_BYTES, strict=True)
    history_keep_bytes: PositiveInt = Field(default=DEFAULT_HISTORY_KEEP_BYTES, strict=True)
    budget_pause_cap_seconds: float = Field(
        default=DEFAULT_BUDGET_PAUSE_CAP_SECONDS, strict=True, ge=BACKOFF_BASE_SECONDS, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _tools_need_a_gateway_and_personas(self) -> Self:
        if any(app.max_tool_calls for app in self.apps):
            if self.gateway is None:
                raise ValueError("gateway: required when an app binds tools")
            if any(agent.persona_file is None for agent in self.agents):
                raise ValueError("persona_file: required for every agent when an app binds tools")
        return self

    @model_validator(mode="after")
    def _a_step_keeps_no_more_than_the_history_may_hold(self) -> Self:
        if self.history_keep_bytes > self.history_bytes:
            raise ValueError("history_keep_bytes: must not be above history_bytes")
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
            problems = (
                f"{'.'.join(str(part) for part in each['loc'])}: {each['msg']}"
                for each in _without_the_length_of_entries_that_failed(error.errors())
            )
            raise ConfigError("; ".join(problems)) from None


def _without_the_length_of_entries_that_failed(problems: list[ErrorDetails]) -> list[ErrorDetails]:
    """A tuple whose entry failed is also counted short of the entries that held: that follows the entry's own error."""

    def an_entry_failed(short: ErrorDetails) -> bool:
        inside = len(short["loc"])
        return any(len(each["loc"]) > inside and each["loc"][:inside] == short["loc"] for each in problems)

    return [each for each in problems if each["type"] != "too_short" or not an_entry_failed(each)]


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


def _read_persona(index: int, path: Path | None, model: GatewayModelName | None) -> AgentDefinition | None:
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
    return persona if model is None else persona.model_copy(update={"model": model})


def _database_target(entry: _DatabaseEntry) -> DatabaseTarget:
    password = _read_secret("database.password_file", entry.password_file)
    return DatabaseTarget(entry.host, entry.port, entry.dbname, entry.user, password, entry.write_seconds)


def _memory_target(entry: _MemoryEntry | None) -> MemoryTarget | None:
    if entry is None:
        return None
    return MemoryTarget(
        entry.lines_keep_days,
        entry.sweep_minutes,
        entry.lookup_calls,
        entry.lookup_bytes,
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
            AppTarget(
                str(app.mcp_url), app.scope, app.max_tool_calls, app.listens, app.call_seconds or DEFAULT_CALL_SECONDS
            )
            for app in parsed.apps
        ),
        agents=tuple(
            AgentCredentials(
                agent.client_id,
                _read_secret(f"agents.{index}.secret_file", agent.secret_file),
                _read_persona(index, agent.persona_file, agent.model),
                agent.figure,
                None if agent.voice is None else Voice(agent.voice.names or ()),
            )
            for index, agent in enumerate(parsed.agents)
        ),
        database=_database_target(parsed.database),
        ownership=OwnershipTimings(**parsed.ownership.model_dump()),
        gateway=parsed.gateway,
        memory=_memory_target(parsed.memory),
        history_bytes=parsed.history_bytes,
        history_keep_bytes=parsed.history_keep_bytes,
        budget_pause_cap_seconds=parsed.budget_pause_cap_seconds,
    )
