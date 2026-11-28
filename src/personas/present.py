"""What his next request holds: the history and the answered turns kept present, what entered since a mark, and the
messages they are told in; with no process, network or clock.

Within the pure chain it imports ``entries`` and ``decision`` and is imported by ``due``: ``entries`` <- ``recall`` <-
``decision`` <- ``present`` <- ``due``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain
from typing import TYPE_CHECKING, Final

from agents.skills import Skill, SkillEntry
from agents.tools import ToolDeclaration, ToolEffect
from agents.turn import MAX_EVENT_BYTES, RoundCall, ToolRound, TurnMessage, TurnRole

from personas.decision import LOAD_TOOLS_TOOL, READ_SKILL_TOOL
from personas.entries import (
    ActDone,
    Entry,
    History,
    HistoryDelta,
    KeptHistory,
    KeptLine,
    Lines,
    SaidAloud,
    SkillRead,
    ToolsLoaded,
    TurnTold,
    ViewSeen,
    Visit,
    entry_size,
    is_his_own,
    is_told,
    newest_that_fit,
    told_bytes,
    told_entry,
    told_place,
    told_time,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from agents.member import PlaceRef
    from agents.turn import JsonObject


def _revision_key(view: ViewSeen) -> tuple[str, PlaceRef, int]:
    return view.app, view.place, view.view.revision


def _holds_revision(entries: Iterable[Entry], view: ViewSeen) -> bool:
    return any(isinstance(entry, ViewSeen) and _revision_key(entry) == _revision_key(view) for entry in entries)


def _not_his_own(entry: Entry) -> bool:
    return is_told(entry) and not is_his_own(entry)


@dataclass(frozen=True)
class Present:
    """What one agent's next request holds besides its header and its own turn: the history and the answered turns whose
    messages are still in it, oldest first. ``told_left_out`` counts the answered turns that ever left it, so the turn
    ``told[i]`` was the ``told_left_out + i``-th answered."""

    history: History
    told: tuple[TurnTold, ...] = ()
    told_left_out: int = 0

    @property
    def answered(self) -> int:
        """How many answered turns ever entered it."""
        return self.told_left_out + len(self.told)


EMPTY_PRESENT: Final = Present(History((), left_out=0, heard=0))


def _step_cut(history: History, boundaries: Iterable[int], keep_bytes: int) -> int:
    """The ordinal the entries before which leave in a step: the oldest end of a message from which on what stays fits
    ``keep_bytes``, or, when the newest message alone does not fit, the oldest of its entries that fits with the
    newer ones. The newest entry stays."""
    fitting = max(1, newest_that_fit(map(entry_size, reversed(history.entries)), keep_bytes))
    oldest_that_fits = history.taken - fitting
    return min((end for end in boundaries if oldest_that_fits <= end < history.taken), default=oldest_that_fits)


def _stays_told(turn: TurnTold, left_out: int) -> bool:
    """Whether the message of ``turn`` has an entry left to tell once the history has dropped the first ``left_out``."""
    return turn.through > left_out


def appended_by(held: Collection[Entry], new: Iterable[Entry]) -> tuple[Entry, ...]:
    """The entries of ``new`` that taking them appends to the entries ``held``: a view revision they hold, or an earlier
    entry of ``new`` holds, adds nothing and an act is always added."""
    appended: list[Entry] = []
    for entry in new:
        if not (isinstance(entry, ViewSeen) and _holds_revision(chain(held, appended), entry)):
            appended.append(entry)
    return tuple(appended)


def taken_into(present: Present, new: Iterable[Entry], history_bytes: int, keep_bytes: int) -> Present:
    """The present with ``new`` appended, a view revision it already holds adding nothing and an act always added. Past
    ``history_bytes`` the oldest whole messages leave in one step until what stays fits ``keep_bytes``; only the newest
    message may be cut, and its newest entry always stays. A step's cut points are the ends of the answered messages, so
    the cut can fall inside the span of a turn still in flight: when it is answered, its message renders partly cut,
    and nothing is told twice or lost."""
    appended = appended_by(present.history.entries, new)
    entries = (*present.history.entries, *appended)
    heard = present.history.heard + sum(map(_not_his_own, appended))
    grown = History(entries, present.history.left_out, heard)
    if sum(map(entry_size, entries)) <= history_bytes:
        return Present(grown, present.told, present.told_left_out)
    cut = _step_cut(grown, (each.through for each in present.told), keep_bytes)
    history = History(entries[cut - grown.left_out :], cut, heard)
    told = tuple(each for each in present.told if _stays_told(each, cut))
    return Present(history, told, present.told_left_out + len(present.told) - len(told))


def answered_into(present: Present, turn: TurnTold) -> Present:
    """The present once ``turn`` was answered: its message is part of every request after it, while it stays. A turn
    whose entries all left in a step while it ran has no message left to tell."""
    if not _stays_told(turn, present.history.left_out):
        return present
    return Present(present.history, (*present.told, turn), present.told_left_out)


def history_delta(
    written: Present, current: Present, kept_first: int = 0, kept: tuple[Entry, ...] = ()
) -> HistoryDelta | None:
    """What ``current`` gained and lost since ``written``, an earlier value of the same present that the store holds,
    and the entries to keep in the record from the ordinal ``kept_first``; ``None`` when nothing. What entered and left
    again between the two is never written to the history, and is kept."""
    first = max(written.history.taken, current.history.left_out)
    first_told = max(written.answered, current.told_left_out)
    entries = current.history.entries[first - current.history.left_out :]
    told = current.told[first_told - current.told_left_out :]
    if not entries and not told and not kept and current.history.left_out == written.history.left_out:
        return None
    return HistoryDelta(first, entries, first_told, told, current.history.left_out, kept_first, kept)


def present_of(kept: KeptHistory) -> Present:
    """The present a history the store keeps resumes as: what he heard before the store's oldest entry is not
    counted, since only a change of the count matters to what arrives next."""
    heard = sum(map(_not_his_own, kept.entries))
    return Present(History(kept.entries, kept.first, heard), kept.told, kept.first_told)


@dataclass(frozen=True)
class Arrived:
    """The entries that entered a history since an earlier value of it, and whether any of them left again by the
    bound."""

    entries: tuple[Entry, ...] = field(repr=False)
    older_dropped: bool


def _entered_since(current: History, taken: int) -> tuple[Entry, ...]:
    """The entries of ``current`` that were appended after the first ``taken`` ones that were ever appended."""
    return current.entries[max(0, taken - current.left_out) :]


def arrived_since(current: History, admitted: History) -> Arrived | None:
    """What entered ``current`` since ``admitted``, an earlier value of the same history, when any of it is not his own;
    otherwise ``None``. His own lines, his acts, his spoken sentences and his skills alone arrive as nothing."""
    if current.heard == admitted.heard:
        return None
    return Arrived(_entered_since(current, admitted.taken), current.left_out > admitted.taken)


def spoke_since(history: History, mark: int) -> bool:
    """Whether a call was relayed or a sentence sent for him since ``history`` had taken ``mark`` entries. A call of a
    tool the app hints is read-only counts too, since the hint is only the app's word. Entries the bound dropped since
    then may have been either, so a history that dropped any of them says he did."""
    return history.left_out > mark or any(
        isinstance(entry, ActDone | SaidAloud) for entry in _entered_since(history, mark)
    )


class WakeReason(StrEnum):
    LINES = "lines"
    VIEW = "view"


def held_told(arrived: Arrived) -> str:
    """The result of a call that was not sent: that it was not, and the entries that arrived. When their text would
    pass ``MAX_EVENT_BYTES`` the oldest are left out, and so is an entry that alone does not fit."""
    told = [told_entry(entry) for entry in arrived.entries if is_told(entry)]
    empty = _held_json([], older_dropped=False)
    separator = len(", ")
    sizes = reversed([told_bytes(each) + separator for each in told])
    fitting = newest_that_fit(sizes, MAX_EVENT_BYTES - len(empty.encode()) + separator)
    kept = told[len(told) - fitting :]
    return _held_json(kept, older_dropped=arrived.older_dropped or fitting < len(told))


def _held_json(entries: list[dict[str, object]], *, older_dropped: bool) -> str:
    return json.dumps(
        {"sent": False, "arrived": {"older_dropped": older_dropped, "entries": entries}}, ensure_ascii=False
    )


def _told_wake(turn: TurnTold) -> dict[str, object]:
    occasion = turn.occasion
    if isinstance(occasion, Lines):
        woke: dict[str, object] = {
            "why": WakeReason.LINES,
            "app": turn.app,
            "place": told_place(turn.place),
            "missed": occasion.missed,
            "seqs": sorted(occasion.seqs),
        }
        return woke | {"aloud": True} if turn.aloud else woke
    return {
        "why": WakeReason.VIEW,
        "app": turn.app,
        "place": told_place(turn.place),
        "revision": occasion.revision,
    }


def told_header(you: str, history: History) -> str:
    """The header message: his name and whether older entries have left the history. It holds no instruction; those
    live in the persona."""
    return json.dumps({"you": you, "older_dropped": history.left_out > 0}, ensure_ascii=False)


def told_turn(turn: TurnTold, entries: Iterable[Entry]) -> str:
    """The message of a turn: when it was admitted, what woke him and every entry taken since the message before it,
    the facts in the order things happened. A turn for lines is told when its answer is heard aloud."""
    told = {
        "at": told_time(turn.at),
        "woke": _told_wake(turn),
        "entries": [told_entry(entry) for entry in entries if is_told(entry)],
    }
    return json.dumps(told, ensure_ascii=False)


def _naming_told_lines(turn: TurnTold, history: History) -> TurnTold:
    """``turn`` with the lines its occasion names cut down to those of its visit the history still holds, since a
    message partly cut in a step no longer tells the rest; a line of another place with the same seq is no such line."""
    occasion = turn.occasion
    if not isinstance(occasion, Lines):
        return turn
    visit = Visit(turn.app, turn.place, occasion.generation)
    held = {entry.seq for entry in history.entries if isinstance(entry, KeptLine) and entry.visit == visit}
    still_told = Lines(occasion.generation, occasion.seqs & held, occasion.missed)
    return TurnTold(turn.at, turn.app, turn.place, still_told, turn.aloud, turn.through)


def user_message(text: str) -> TurnMessage:
    return TurnMessage(role=TurnRole.USER, text=text)


def _told_entries(entries: Iterable[Entry]) -> TurnMessage:
    """The message of the entries that came before a round in a turn's span: the entries alone."""
    return user_message(
        json.dumps({"entries": [told_entry(entry) for entry in entries if is_told(entry)]}, ensure_ascii=False)
    )


def _told_in_turn(turn: TurnTold, entries: Iterable[Entry], deferred: Deferred) -> list[TurnMessage | ToolRound]:
    """What a turn tells of its span of the history: a message of the entries before each skill the agent read or
    tools it loaded, that round, and the turn's own message with what came after the last round."""
    told: list[TurnMessage | ToolRound] = []
    before: list[Entry] = []
    for entry in entries:
        if isinstance(entry, SkillRead | ToolsLoaded):
            if before:
                told.append(_told_entries(before))
                before = []
            told.append(skill_round(entry) if isinstance(entry, SkillRead) else load_round(entry, deferred))
        else:
            before.append(entry)
    told.append(user_message(told_turn(turn, before)))
    return told


def turn_messages(
    you: str, present: Present, current: TurnTold, deferred: Deferred
) -> tuple[TurnMessage | ToolRound, ...]:
    """The messages of a turn's request after the persona: the header, then for each answered turn the present holds
    its span of the history as it stands, and the turn's own, which tells what the history holds from where the last of
    those ended up to ``current.through``. A span is told split around each skill the agent read in it, which
    comes as the round of a ``read_skill`` call, and each load of tools, a ``load_tools`` round that names the tools
    ``deferred`` still holds. A message's ``woke`` names only the lines the history holds."""
    history = present.history
    messages: list[TurnMessage | ToolRound] = [user_message(told_header(you, history))]
    start = history.left_out
    for turn in (*present.told, current):
        entries = history.entries[start - history.left_out : turn.through - history.left_out]
        messages += _told_in_turn(_naming_told_lines(turn, history), entries, deferred)
        start = max(start, turn.through)
    return tuple(messages)


type AppSkill = Skill | SkillEntry
"""A skill an app serves as a listing entry, which proves nothing of its file, or its one implicit skill."""

type Deferred = Mapping[str, tuple[str, ...]]
"""The tools a turn sends deferred, in the listing's order of the app at each URL."""

EMPTY_SKILL_TEXT: Final = "this skill has no text"
NO_SUCH_SKILL_TEXT: Final = "no skill of that name is listed in your instructions"
NOT_OFFERED_TEXT: Final = "these tools are not offered now"
_SKILL_ARGUMENT: Final = "skill"
_SKILL_PARAMETERS: Final = {
    "type": "object",
    "properties": {_SKILL_ARGUMENT: {"type": "string", "description": "The skill's name as listed."}},
    "required": [_SKILL_ARGUMENT],
    "additionalProperties": False,
}
READ_SKILL_DECLARATION: Final = ToolDeclaration(
    name=READ_SKILL_TOOL,
    description=(
        "Reads a skill listed in your instructions and answers with its text. `skill` is the skill's name as listed."
    ),
    parameters=_SKILL_PARAMETERS,
    effect=ToolEffect.READ_ONLY,
)
LOAD_TOOLS_DECLARATION: Final = ToolDeclaration(
    name=LOAD_TOOLS_TOOL,
    description=(
        "Loads the tools of a skill listed in your instructions, so you can call them from your next step on. `skill` "
        "is the skill's name as listed."
    ),
    parameters=_SKILL_PARAMETERS,
    effect=ToolEffect.READ_ONLY,
)
ROUND_CALL_PREFIX: Final = "personas-skill-"


@dataclass(frozen=True)
class ListedSkill:
    """A skill a turn lists, and the app at that URL that serves it."""

    app: str
    skill: AppSkill


def listed_skills(by_app: Iterable[tuple[str, Iterable[AppSkill]]]) -> Mapping[str, ListedSkill]:
    """The skills a turn lists by name, in the order of the apps given; a name two skills share goes to the first."""
    listed: dict[str, ListedSkill] = {}
    for app, skills in by_app:
        for skill in skills:
            listed.setdefault(skill.name, ListedSkill(app, skill))
    return listed


def skill_lines(listed: Iterable[ListedSkill]) -> str:
    """One line for each skill, its name and its description as the app declared them with whitespace made single
    spaces, so that no description makes a line of its own."""
    return "\n".join(f"{each.skill.name}: {' '.join(each.skill.description.split())}" for each in listed)


def round_call_id(ordinal: int) -> str:
    """The call id of the round whose entry has this ordinal, a skill's read or a load of tools. The prefix keeps it
    from a model's call id."""
    return f"{ROUND_CALL_PREFIX}{ordinal}"


def listed_skill_named(skills: Mapping[str, ListedSkill], arguments: JsonObject) -> ListedSkill | None:
    """The skill a ``read_skill`` or ``load_tools`` call names among the skills a turn lists, or ``None`` for arguments
    that name none the turn lists."""
    name = arguments.get(_SKILL_ARGUMENT)
    return skills.get(name) if isinstance(name, str) else None


def skill_text(body: str) -> str:
    """What a read of a skill with this body answers."""
    return body or EMPTY_SKILL_TEXT


def skill_round(entry: SkillRead) -> ToolRound:
    """The round that tells a skill the agent read: one ``read_skill`` call under the entry's call id, answered with
    what the model was shown."""
    arguments = json.dumps({_SKILL_ARGUMENT: entry.name}, ensure_ascii=False)
    call = RoundCall(call_id=entry.call_id, tool=READ_SKILL_TOOL, arguments=arguments, result=skill_text(entry.body))
    return ToolRound(calls=(call,))


def load_round(entry: ToolsLoaded, deferred: Deferred) -> ToolRound:
    """The round that tells tools the agent loaded: one ``load_tools`` call under the entry's call id, answered with
    references to the tools its app still sends deferred in this turn, or, when it sends none, a text that loads
    nothing."""
    loads = tuple(name for name in entry.tools if name in deferred.get(entry.app, ()))
    arguments = json.dumps({_SKILL_ARGUMENT: entry.skill}, ensure_ascii=False)
    result = "" if loads else NOT_OFFERED_TEXT
    call = RoundCall(call_id=entry.call_id, tool=LOAD_TOOLS_TOOL, arguments=arguments, result=result, loads=loads)
    return ToolRound(calls=(call,))


def loaded_by(messages: Iterable[TurnMessage | ToolRound]) -> frozenset[str]:
    """The tools the rounds among ``messages`` loaded."""
    return frozenset(
        name for each in messages if isinstance(each, ToolRound) for call in each.calls for name in call.loads
    )


def tools_to_load(skill: AppSkill, deferred: Sequence[str], loaded: Collection[str]) -> tuple[str, ...]:
    """The tools of ``skill``, in its order, that its app sends ``deferred`` and that ``loaded`` does not hold."""
    return tuple(name for name in skill.tools if name in deferred and name not in loaded)


@dataclass(frozen=True)
class AppOffer:
    """What an app offers a turn: its URL, the declarations of its listing the turn is offered, in its order, and the
    skills of that listing."""

    app: str
    declarations: tuple[ToolDeclaration, ...]
    skills: tuple[AppSkill, ...]


def deferred_by_app(offers: Iterable[AppOffer]) -> Deferred:
    """The tools a turn sends deferred by the URL of their app, in its listing's order: every tool of every app the
    turn lists."""
    return {offer.app: tuple(each.name for each in offer.declarations) for offer in offers if offer.declarations}


@dataclass(frozen=True)
class BlockDue:
    """The rounds a turn appends for one skill of one app: its read when ``read``, then the load of ``tools``."""

    app: str
    skill: AppSkill
    read: bool
    tools: tuple[str, ...]


def blocks_due(
    offers: Iterable[AppOffer], present: Collection[str], deferred: Deferred, history: History
) -> tuple[BlockDue, ...]:
    """The block a turn appends at the end of its request: for each skill, in the listing's order, of each app of
    ``offers`` in ``present``, its read when ``history`` holds none of that skill of that app, and the load of the
    tools it names that the app sends ``deferred`` and no load of that app in ``history`` names. A skill with neither
    due is left out."""
    due: list[BlockDue] = []
    for offer in offers:
        if offer.app not in present:
            continue
        read = {entry.name for entry in history.entries if isinstance(entry, SkillRead) and entry.app == offer.app}
        loaded = {
            name
            for entry in history.entries
            if isinstance(entry, ToolsLoaded) and entry.app == offer.app
            for name in entry.tools
        }
        for skill in offer.skills:
            tools = tools_to_load(skill, deferred.get(offer.app, ()), loaded)
            loaded.update(tools)
            if skill.name not in read or tools:
                due.append(BlockDue(offer.app, skill, read=skill.name not in read, tools=tools))
    return tuple(due)
