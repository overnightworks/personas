"""The schemas the configuration release notes tests compare, each a file model before or after a change."""

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Backend(Enum):
    LITELLM = "litellm"


class OwnershipBefore(Entry):
    grace_seconds: float = 12.0


class OwnershipAfter(Entry):
    grace_seconds: float = 12.0
    stop_seconds: float = 10.0


class AppBefore(Entry):
    mcp_url: str
    tools: list[str] = []


class AppAfter(Entry):
    mcp_url: str


class Gateway(Entry):
    origin: str
    ca_file: str | None = None


class Database(Entry):
    host: str
    port: int
    backend: Backend
    gateway: Gateway
    write_seconds: float = 2.0
    listens: bool = True
    keep_days: float = math.inf
    names: list[str] = ["first"]
    choice: int | str


class Ownership(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppBefore]


class WithStop(BaseModel):
    ownership: OwnershipAfter = OwnershipAfter()
    apps: list[AppBefore]


class WithoutTools(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppAfter]


class WithDatabase(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppBefore]
    database: Database


class TopLevelBefore(BaseModel):
    ready_file: str | None = None


class TopLevelAfter(BaseModel):
    ready_file: str
    history_bytes: int | None = None


class HistoryBefore(BaseModel):
    history_bytes: int = 49152


class HistoryAfter(BaseModel):
    history_bytes: int = 1000


class HistoryCountedInKibibytes(BaseModel):
    history_bytes: float = 48.0


class HistoryRequired(BaseModel):
    history_bytes: int


class HistoryRequiredAsText(BaseModel):
    history_bytes: str


class IntPort(Entry):
    port: int


class StrPort(Entry):
    port: str


class WithIntPort(BaseModel):
    database: IntPort


class WithStrPort(BaseModel):
    database: StrPort


class LowPort(Entry):
    port: int = Field(ge=1)


class HighPort(Entry):
    port: int = Field(ge=1024)


class WithLowPort(BaseModel):
    database: LowPort


class WithHighPort(BaseModel):
    database: HighPort


class UnboundLookups(BaseModel):
    lookup_calls: int = Field(default=2, gt=0)


class BoundLookups(BaseModel):
    lookup_calls: int = Field(default=2, gt=0, le=4)


class Clip(Entry):
    url: str
    note: str = ""


class TwoGestures(Entry):
    gestures: dict[Literal["wave", "nod"], Clip] = {}


class OneGesture(Entry):
    gestures: dict[Literal["wave"], Clip] = {}


class WithTwoGestures(BaseModel):
    figure: TwoGestures


class WithOneGesture(BaseModel):
    figure: OneGesture


class NotedClip(Entry):
    url: str
    note: str


class WithClipOrName(BaseModel):
    intro: Clip | str = ""


class WithNotedClipOrName(BaseModel):
    intro: NotedClip | str = ""


class StringVoices(Entry):
    names: tuple[str, ...] = ()


class IntegerVoices(Entry):
    names: tuple[int, ...] = ()


class WithStringVoices(BaseModel):
    voice: StringVoices = StringVoices()


class WithIntegerVoices(BaseModel):
    voice: IntegerVoices = IntegerVoices()


class AppWithScope(Entry):
    mcp_url: str
    tools: list[str] = []
    scope: str


class WithScope(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()
    apps: list[AppWithScope]


class Memory(Entry):
    keep_days: float = 30.0


class MemoryWithLookup(Entry):
    keep_days: float = 30.0
    lookup_bytes: int


class WithMemory(BaseModel):
    memory: Memory | None = None


class WithLookup(BaseModel):
    memory: MemoryWithLookup | None = None


class AgentWithMemory(Entry):
    client_id: str
    memory: Memory | None = None


class AgentWithLookup(Entry):
    client_id: str
    memory: MemoryWithLookup | None = None


class MemoryWithMode(Entry):
    keep_days: float = 30.0
    mode: str = "x"


class WithModedMemory(BaseModel):
    memory: MemoryWithMode | None = None


class WithAgentMemory(BaseModel):
    agents: list[AgentWithMemory]


class WithAgentLookup(BaseModel):
    agents: list[AgentWithLookup]


class IntPortBehindGateway(Entry):
    port: int
    gateway: Gateway


class WithGateway(BaseModel):
    database: IntPortBehindGateway


class AgentWithScope(Entry):
    client_id: str
    memory: Memory | None = None
    scope: str


class WithOptionalAgents(BaseModel):
    agents: list[AgentWithMemory] = []


class WithOptionalScopedAgents(BaseModel):
    agents: list[AgentWithScope] = []


class AgentWithVoice(Entry):
    client_id: str
    memory: Memory | None = None
    voice: Gateway | None = None


class WithAgentVoice(BaseModel):
    agents: list[AgentWithVoice]


class VoiceBehindGateway(Entry):
    origin: str
    ca_file: str | None = None
    gateway: Gateway


class AgentWithVoiceBehindGateway(Entry):
    client_id: str
    memory: Memory | None = None
    voice: VoiceBehindGateway | None = None


class WithAgentVoiceBehindGateway(BaseModel):
    agents: list[AgentWithVoiceBehindGateway]


class WithNoClips(BaseModel):
    clips: list[Clip] = []


class WithOneClip(BaseModel):
    clips: list[Clip] = [Clip(url="https://clip.test")]


class OwnershipOnly(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore()


class TunedOwnershipOnly(BaseModel):
    ownership: OwnershipBefore = OwnershipBefore(grace_seconds=20.0)


class MemoryOnByDefault(BaseModel):
    memory: Memory = Memory()


class RuledHistory(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "refusals": [
                "history_keep_bytes: must be a multiple of 1024",
                "history_keep_bytes: must not be above history_bytes",
            ]
        }
    )

    history_bytes: int = 49152
    history_keep_bytes: int = 32768


class RuledShorterHistory(RuledHistory):
    history_bytes: int = 16384


class DeclaresNoRules(BaseModel):
    model_config = ConfigDict(json_schema_extra={"refusals": []})

    ownership: OwnershipAfter = OwnershipAfter()


class RuledOwnership(Entry):
    model_config = ConfigDict(json_schema_extra={"refusals": ["stop_seconds: must be above grace_seconds"]})

    grace_seconds: float = 12.0
    stop_seconds: float = 10.0


class DeclaresRules(BaseModel):
    model_config = ConfigDict(json_schema_extra={"refusals": ["history_keep_bytes: must not be above history_bytes"]})

    ownership: RuledOwnership = RuledOwnership()
