"""What an agent took in, did and said, as the history keeps it, and how a turn is told of it: pure values with no
process, network or clock."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, TypeIs, cast

from agents.member import (
    HoldPlace,
    PartialSpeech,
    PlaceEntry,
    PlaceRef,
    PlaceView,
    PlaceWindow,
    SpeakAnswer,
    SpeakRefusal,
    Spoken,
    VoiceState,
    Written,
)
from agents.turn import JsonObject  # noqa: TC002 - pydantic resolves the fields' types at run time
from agents.voice.conversation import told_hold as agents_told_hold
from agents.voice.conversation import told_went_on as agents_told_went_on
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.tools import ToolImage


@dataclass(frozen=True)
class WindowRead:
    """A window as one agent read it from one app's place, and the wall time of that read."""

    client_id: str
    app: str
    place: PlaceRef
    window: PlaceWindow
    read_at: datetime


@dataclass(frozen=True)
class Visit:
    """One stay of an agent in a place: the app, the place and the window's generation, which a new join renews."""

    app: str
    place: PlaceRef
    generation: str


@dataclass(frozen=True)
class Room:
    """Where a call was made: the app and place of the turn that made it and that place's visit, which is none where
    the entry entered while no window of the place was read."""

    app: str
    place: PlaceRef
    generation: str | None


@dataclass(frozen=True)
class KeptLine:
    """A line the memory kept as a visit's window gave it, with the wall time it was first read."""

    visit: Visit
    seq: int
    first_read_at: datetime
    name: str = field(repr=False)
    kind: str
    text: str = field(repr=False)
    own: bool
    spoken: bool
    cut: bool
    act: bool


@dataclass(frozen=True)
class ViewSeen:
    """A revision of a place's view as an agent read it, with the wall time of that read."""

    app: str
    place: PlaceRef
    view: PlaceView = field(repr=False)
    read_at: datetime
    generation: str | None = None


@dataclass(frozen=True)
class Answered:
    """The app answered a call; an error result is an answer too."""

    text: str = field(repr=False)
    is_error: bool
    image_shown: bool = False
    """Whether the answer carried a picture, which is shown to the model in its turn and never kept."""


@dataclass(frozen=True)
class Unanswered:
    """The app gave no answer, so whether the call took effect is unknown: ``code`` is the failure's, or the code of a
    call cut in flight."""

    code: str


type ActOutcome = Answered | Unanswered


@dataclass(frozen=True)
class ActDone:
    """A call relayed for an agent to the app at ``app``, with the wall time its relay ended and the room it was made
    in."""

    app: str
    tool: str
    arguments: JsonObject = field(repr=False)
    outcome: ActOutcome
    at: datetime
    room: Room | None = None


type SpeechOutcome = SpeakAnswer | SpeakRefusal | Unanswered


@dataclass(frozen=True)
class SaidAloud:
    """A sentence an agent's turn sent to the ``speak`` of the app at ``app``, with the wall time its request ended."""

    app: str
    place: PlaceRef
    text: str = field(repr=False)
    outcome: SpeechOutcome
    at: datetime
    generation: str | None = None


@dataclass(frozen=True)
class SkillRead:
    """A skill of the app at ``app`` an agent loaded: its name and uri, its body and the wall time. The round that
    read it is told under ``call_id``, which the runner minted when it wrote the entry, so a replay tells the round
    under the same id."""

    app: str
    name: str
    uri: str
    body: str = field(repr=False)
    at: datetime
    call_id: str


@dataclass(frozen=True)
class ToolsLoaded:
    """The tools of the skill ``skill`` of the app at ``app`` an agent loaded, and the wall time. The round that
    loaded them is told under ``call_id``, which the runner minted when it wrote the entry."""

    app: str
    skill: str
    tools: tuple[str, ...]
    at: datetime
    call_id: str


type ToldEntry = KeptLine | ViewSeen | ActDone | SaidAloud
type Entry = ToldEntry | SkillRead | ToolsLoaded


def is_told(entry: Entry) -> TypeIs[ToldEntry]:
    """Whether a turn's message tells the entry; a skill and a load are told by their round, never by a message."""
    return not isinstance(entry, SkillRead | ToolsLoaded)


def is_his_own(entry: ToldEntry) -> bool:
    """Whether he said or did the entry: his own line, a call he made or a sentence he spoke; a view and another's line
    are told to him."""
    return entry.own if isinstance(entry, KeptLine) else isinstance(entry, ActDone | SaidAloud)


@dataclass(frozen=True)
class Lines:
    """Why a written turn runs: the lines of one visit's window taken since his last turn there, which the history
    holds, and whether older ones were never seen. ``traceparent`` is the trace the newest line that made the turn due
    entered in, which the turn continues; none opens a trace of its own."""

    generation: str
    seqs: frozenset[int]
    missed: bool
    traceparent: str | None = None


@dataclass(frozen=True)
class Waiting:
    """Why a view turn runs: the place's view waits for the agent at this revision, which the history holds.
    ``traceparent`` is the trace the view moved to it in, which the turn continues; none opens a trace of its own."""

    revision: int
    traceparent: str | None = None


type Occasion = Lines | Waiting


@dataclass(frozen=True)
class TurnTold:
    """An answered turn and what its message told: the turn's app, place, occasion and admission time, and ``through``,
    the ordinal after the last entry of the history it told. It tells the entries of the history from the ordinal
    where the message before it ended, or from the oldest the history holds."""

    at: datetime
    app: str
    place: PlaceRef
    occasion: Occasion
    aloud: bool
    through: int


@dataclass(frozen=True)
class HistoryDelta:
    """What one agent's history gained and lost since its last write: the entries appended, the first of them at the
    ordinal ``first``, the turns answered in their order, the first of them at the ordinal ``first_told``, and
    ``left_out``, the ordinal of the oldest entry it still holds; every older entry has left it, and so has every
    answered turn that told none of the rest. An answered turn's ordinal, not its wall time, is its identity and order.

    ``kept`` are the entries the record takes, the first of them at the ordinal ``kept_first``: every entry taken and
    not yet kept, whether the history still holds it or not.
    """

    first: int
    entries: tuple[Entry, ...] = field(repr=False)
    first_told: int
    told: tuple[TurnTold, ...]
    left_out: int
    kept_first: int = 0
    kept: tuple[Entry, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class Reading:
    """What the last read of a window stated: its generation, the last seq taken and, when the app stated one, the
    state of his voice at the place."""

    generation: str
    last_seq: int
    voice_state: VoiceState | None = None


@dataclass(frozen=True)
class KeptCheckpoint:
    """Where reading one of his window places stands, as the store keeps it: its last read, or none for a place he
    joined and has not read since. A place with no kept checkpoint resumes as one whose window was said before."""

    app: str
    place: PlaceRef
    reading: Reading | None


@dataclass(frozen=True)
class KeptDue:
    """A place of his that has a turn due, and what for, as the store keeps it."""

    app: str
    place: PlaceRef
    occasion: Occasion


@dataclass(frozen=True)
class KeptTurns:
    """His turns as the store keeps them: where reading each place stands, the places that have a turn due in the
    order they became due, how many of his turns in a row ended without an answer, and the wall time before which no
    turn of his may start, if a pause was set."""

    checkpoints: tuple[KeptCheckpoint, ...] = ()
    due: tuple[KeptDue, ...] = ()
    unanswered: int = 0
    paused_until: datetime | None = None


NO_TURNS_KEPT: Final = KeptTurns()


@dataclass(frozen=True)
class KeptHistory:
    """One agent's history as the store keeps it: its entries in order, the first of them at the ordinal ``first``,
    and his answered turns that tell one of them in the order they were answered, the first of them at the ordinal
    ``first_told``, and his turns as the last write left them."""

    first: int
    entries: tuple[Entry, ...] = field(repr=False)
    first_told: int
    told: tuple[TurnTold, ...]
    turns: KeptTurns = NO_TURNS_KEPT


STORED_KIND_KEY: Final = "entry"
"""The key of a stored entry that names its kind."""
_ENTRY_KINDS: Final = (
    ("line", KeptLine),
    ("view", ViewSeen),
    ("act", ActDone),
    ("speech", SaidAloud),
    ("skill", SkillRead),
    ("load", ToolsLoaded),
)
_STORED_ENTRIES: Final = {kind: TypeAdapter(of_kind) for kind, of_kind in _ENTRY_KINDS}
KIND_OF_ENTRY: Final = {of_kind: kind for kind, of_kind in _ENTRY_KINDS}
"""The stored kind of each type of entry."""
_ANY_ENTRY: Final = TypeAdapter[Entry](Entry)
_STORED_TURN: Final = TypeAdapter(TurnTold)
_STORED_TURNS: Final = TypeAdapter(KeptTurns)


def stored_entry(entry: Entry) -> JsonObject:
    """The entry as a JSON object, its kind under ``STORED_KIND_KEY``, which ``entry_from_stored`` turns back into an
    equal entry."""
    return {STORED_KIND_KEY: KIND_OF_ENTRY[type(entry)]} | _ANY_ENTRY.dump_python(entry, mode="json")


def entry_from_stored(stored: object) -> Entry:
    """The entry ``stored_entry`` made ``stored`` of; a value it did not make raises a ``ValueError``."""
    kind = cast("dict[str, object]", stored).get(STORED_KIND_KEY) if isinstance(stored, dict) else None
    if not isinstance(kind, str) or kind not in _STORED_ENTRIES:
        raise ValueError("the stored value names no kind of entry")
    return _STORED_ENTRIES[kind].validate_python(stored)


def stored_turn(turn: TurnTold) -> JsonObject:
    """The answered turn as a JSON object, which ``turn_from_stored`` turns back into an equal turn."""
    return _STORED_TURN.dump_python(turn, mode="json")


def turn_from_stored(stored: JsonObject) -> TurnTold:
    """The answered turn ``stored_turn`` made ``stored`` of; a value it did not make raises a ``ValueError``."""
    return _STORED_TURN.validate_python(stored)


def stored_turns(turns: KeptTurns) -> JsonObject:
    """His kept turns as a JSON object, which ``turns_from_stored`` turns back into equal turns."""
    return _STORED_TURNS.dump_python(turns, mode="json")


def turns_from_stored(stored: JsonObject) -> KeptTurns:
    """The kept turns ``stored_turns`` made ``stored`` of; a value it did not make raises a ``ValueError``."""
    return _STORED_TURNS.validate_python(stored)


def entry_time(entry: Entry) -> datetime:
    """The wall time an entry was first read, seen or done."""
    if isinstance(entry, KeptLine):
        return entry.first_read_at
    if isinstance(entry, ViewSeen):
        return entry.read_at
    return entry.at


def entry_room(entry: Entry) -> Room | None:
    """The room an entry was taken in: a line's visit, a view's and a sentence's app, place and visit, a call's room;
    none for a call a release before the room took, a skill and a load."""
    if isinstance(entry, KeptLine):
        return Room(entry.visit.app, entry.visit.place, entry.visit.generation)
    if isinstance(entry, ViewSeen | SaidAloud):
        return Room(entry.app, entry.place, entry.generation)
    return entry.room if isinstance(entry, ActDone) else None


@dataclass(frozen=True)
class RecordRow:
    """One entry as the record keeps it: its ordinal in his history, its time and kind, the app, place and visit it was
    taken in, whether he said or did it, the display name of another's line and the entry's JSON."""

    ordinal: int
    at: datetime
    kind: str
    app: str
    place: PlaceRef | None
    generation: str | None
    own: bool
    speaker: str | None = field(repr=False)
    entry: JsonObject = field(repr=False)


def _record_row(ordinal: int, entry: ToldEntry) -> RecordRow:
    stored = stored_entry(entry)
    room = entry_room(entry)
    # Only a call a release before the room took has none: it keeps its target app, and no place.
    app, place, generation = (
        (cast("ActDone", entry).app, None, None) if room is None else (room.app, room.place, room.generation)
    )
    speaker = entry.name if isinstance(entry, KeptLine) and not entry.own else None
    kind = str(stored[STORED_KIND_KEY])
    return RecordRow(ordinal, entry_time(entry), kind, app, place, generation, is_his_own(entry), speaker, stored)


def record_rows(first: int, entries: Iterable[Entry]) -> tuple[RecordRow, ...]:
    """The rows the record takes of entries the history holds from the ordinal ``first``: what a turn is told, each at
    its ordinal; a skill and a load are told by their round and take an ordinal but no row."""
    return tuple(_record_row(ordinal, entry) for ordinal, entry in enumerate(entries, start=first) if is_told(entry))


def told_time(at: datetime) -> str:
    """A wall time as a turn is told it: UTC, whole seconds."""
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def told_bytes(told: dict[str, object]) -> int:
    return len(json.dumps(told, ensure_ascii=False).encode())


@dataclass(frozen=True)
class InputSizes:
    """The bytes a turn's input is made of: the tool declarations each model request carries, the told message and
    the tool results relayed back to the model, their pictures counted apart from their text."""

    tools_bytes: int
    told_bytes: int
    results_bytes: int
    images_bytes: int


def message_bytes(role: str, text: str) -> int:
    """The bytes of one chat message of the given role, counted as ``told_bytes`` counts."""
    return told_bytes({"role": role, "content": text})


def picture_bytes(image: ToolImage | None) -> int:
    """The decoded bytes of a tool result's picture, as the library's bound counts them; 0 without one."""
    return 0 if image is None else len(image.content)


def _names_bytes(names: Iterable[str]) -> int:
    return sum(len(name.encode()) for name in names)


def entry_size(entry: Entry) -> int:
    """A view counts the bytes of its state's JSON; a line, an act and a spoken sentence count the bytes of the JSON
    they are told as; a skill the bytes of its body, a load those of its tool names."""
    if isinstance(entry, SkillRead):
        return len(entry.body.encode())
    if isinstance(entry, ToolsLoaded):
        return _names_bytes(entry.tools)
    if isinstance(entry, ViewSeen):
        return len(json.dumps(entry.view.state, ensure_ascii=False).encode())
    return told_bytes(told_entry(entry))


def newest_that_fit(sizes_newest_first: Iterable[int], size_bytes: int) -> int:
    """How many entries, whose sizes are given newest first, fit ``size_bytes``."""
    return sum(1 for total in itertools.accumulate(sizes_newest_first) if total <= size_bytes)


@dataclass(frozen=True)
class History:
    """Every line one agent took in, every view revision he read, every call relayed for him and every sentence sent to
    a ``speak`` for him since the runner started, in the order they came, within the bound of the history.

    ``left_out`` counts the entries that ever left it by the bound and ``heard`` the entries not his own that were ever
    appended. ``entries`` are the entries ever appended with the ordinals ``left_out`` up to ``taken - 1``, without a
    gap: ``taken_into`` is the only writer, appends only at the newest end and removes only a prefix. A change that
    removes entries from elsewhere gives each entry its ordinal first.
    """

    entries: tuple[Entry, ...] = field(repr=False)
    left_out: int
    heard: int

    @property
    def older_dropped(self) -> bool:
        return self.left_out > 0

    @property
    def taken(self) -> int:
        """How many entries were ever appended."""
        return self.left_out + len(self.entries)


def taken_lines(visit: Visit, new: Iterable[PlaceEntry], read_at: datetime) -> tuple[KeptLine, ...]:
    """The lines a read of the visit's window took, as the history keeps them."""
    return tuple(
        KeptLine(visit, line.seq, read_at, line.name, line.kind, line.text, line.own, line.spoken, line.cut, line.act)
        for line in new
    )


def told_line(line: KeptLine) -> dict[str, object]:
    told: dict[str, object] = {
        "name": line.name,
        "kind": line.kind,
        "text": line.text,
        "own": line.own,
        "spoken": line.spoken,
    }
    if line.cut:
        told["cut"] = True
    if line.act:
        told["act"] = True
    return told


def told_place(place: PlaceRef) -> dict[str, str]:
    return {"kind": place.kind, "id": place.id}


def told_outcome(outcome: ActOutcome) -> dict[str, object]:
    if isinstance(outcome, Answered):
        answer: dict[str, object] = {"answered": True, "is_error": outcome.is_error, "text": outcome.text}
        return answer | {"image": True} if outcome.image_shown else answer
    return {"answered": False, "code": outcome.code}


def told_speech_outcome(outcome: SpeechOutcome) -> dict[str, object]:
    if isinstance(outcome, Written):
        return {"answered": True, "written": True}
    if isinstance(outcome, Spoken):
        return {"answered": True}
    if isinstance(outcome, SpeakRefusal):
        return {"answered": True, "refused": outcome.value}
    return {"answered": False, "code": outcome.code}


def told_entry(entry: ToldEntry) -> dict[str, object]:
    if isinstance(entry, SaidAloud):
        return {
            "entry": "speech",
            "app": entry.app,
            "place": told_place(entry.place),
            "at": told_time(entry.at),
            "text": entry.text,
            "outcome": told_speech_outcome(entry.outcome),
        }
    if isinstance(entry, ActDone):
        return {
            "entry": "act",
            "app": entry.app,
            "at": told_time(entry.at),
            "tool": entry.tool,
            "arguments": entry.arguments,
            "outcome": told_outcome(entry.outcome),
        }
    if isinstance(entry, ViewSeen):
        return {
            "entry": "view",
            "app": entry.app,
            "place": told_place(entry.place),
            "at": told_time(entry.read_at),
            "revision": entry.view.revision,
            "awaits_you": entry.view.awaits_you,
            "state": entry.view.state,
        }
    return {
        "entry": "line",
        "app": entry.visit.app,
        "place": told_place(entry.visit.place),
        "at": told_time(entry.first_read_at),
        "seq": entry.seq,
    } | told_line(entry)


def told_hold(
    hold_id: str,
    speech: Iterable[PartialSpeech],
    place: HoldPlace,
    unsaid: Iterable[str],
    waiting: tuple[str, JsonObject] | None,
) -> str:
    """The message that tells him he is held: the hold's id, what people said over him, where it stopped him and the
    sentences he had not yet said as agents tells them, and the call of his that waits, as its tool and arguments."""
    held: dict[str, object] = {
        "id": hold_id,
        "talked_over": [{"name": each.name, "text": each.text} for each in speech],
        **agents_told_hold(place, list(unsaid)),
    }
    if waiting is not None:
        held["call_waiting"] = {"tool": waiting[0], "arguments": waiting[1]}
    return json.dumps({"held": held}, ensure_ascii=False)


def told_went_on() -> str:
    """The message that tells him a sound without words held him and he went on by himself."""
    return json.dumps(agents_told_went_on())


def told_words(words: Iterable[PartialSpeech]) -> str:
    """The result of a call that was not sent because people kept talking over him after he went on: that it was not,
    and what each has said that he was not told."""
    return json.dumps(
        {"sent": False, "talked_over": [{"name": each.name, "text": each.text} for each in words]}, ensure_ascii=False
    )


def told_stop(unsaid: Iterable[str], waiting_tool: str | None) -> str:
    """The result of his stop: the sentences that were not said and the tool of the call that was never sent."""
    stopped: dict[str, object] = {"stopped": True, "unsaid": list(unsaid)}
    if waiting_tool is not None:
        stopped["call_dropped"] = waiting_tool
    return json.dumps(stopped, ensure_ascii=False)
