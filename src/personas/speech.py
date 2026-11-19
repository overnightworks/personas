"""What he still owes of his speech at one place and the hold that stands there; with no process, network or clock."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.member import PartialSpeech, PlaceWindow

SENTENCES_AHEAD: Final = 1
"""How many sentences he sends ahead of the one playing: the wire answers a sentence at its turn, in the order sent."""


def may_send_next(owed: int) -> bool:
    """Whether his next sentence may go to ``speak`` while ``owed`` own spoken lines, one per sentence answered as
    spoken, have not entered: the wire holds a sentence back until the one before it ends, so only the sentence playing
    may still owe its line."""
    return owed <= SENTENCES_AHEAD


@dataclass(frozen=True)
class SpeechRecord:
    """What he still owes of his speech at one place: the requests in flight, the lines still owed, one per sentence
    answered as spoken, the sentence that plays and the time by which the turn that sent them is over. A line credited
    before the answer of its request leaves the count below zero until that answer. It also holds the hold that stands
    at the place and the id of the last hold decided there. All of it outlives the turn that sent."""

    in_flight: int = 0
    credit: int = 0
    saying: str | None = None
    hold: Held | None = None
    decided: str | None = None
    bound: float | None = None

    @property
    def owed(self) -> int:
        """The own spoken lines that sentences answered as spoken have not yet brought into the history."""
        return max(self.credit, 0)


NO_SPEECH: Final = SpeechRecord()


@dataclass(frozen=True)
class OwedLines:
    """What a waiter finds of the lines he still owes where it looks: how many, the hold that stands over one of them,
    the earliest time at which the turn that sent one is over, and how many lines it dropped because that time had
    passed."""

    count: int = 0
    held: Held | None = None
    until: float | None = None
    dropped: int = 0


def _settled(record: SpeechRecord) -> SpeechRecord:
    """With no request in flight the count is not below zero, and with nothing owed there is no time bound."""
    if record.in_flight > 0:
        return record
    credit = max(record.credit, 0)
    return replace(record, credit=credit, bound=record.bound if credit else None)


def speech_sent(record: SpeechRecord, bound: float) -> SpeechRecord:
    """A request to ``speak`` counts from its send; the turn that sends it is over by ``bound``."""
    return replace(record, in_flight=record.in_flight + 1, bound=bound)


def speech_answered(record: SpeechRecord, sentence: str, *, spoken: bool) -> SpeechRecord:
    """A request ended: it stops counting on any answer other than ``Spoken``, which owes a line and is the sentence
    that plays."""
    answered = replace(record, in_flight=record.in_flight - 1)
    if spoken:
        answered = replace(answered, credit=record.credit + 1, saying=sentence)
    return _settled(answered)


def speech_line_entered(record: SpeechRecord) -> SpeechRecord:
    """His own spoken line entered at the place: it is credited while a request is in flight or a line is still owed,
    and ignored otherwise."""
    if record.in_flight > 0 or record.credit > 0:
        return _settled(replace(record, credit=record.credit - 1))
    return record


def speech_ended(record: SpeechRecord) -> SpeechRecord:
    """The turn that sent ended: its requests in flight are gone, and the lines it still owes stay for whoever waits on
    them, with the sentence that plays; with nothing owed and no hold the sentence is forgotten."""
    ended = _settled(replace(record, in_flight=0))
    return ended if ended.hold is not None or ended.owed else replace(ended, saying=None)


def speech_lapsed(record: SpeechRecord, now: float, by: float) -> bool:
    """Whether the lines owed at the place will not come, for a waiter whose own turn is over by ``by``: the turn that
    sent them was over before that and at ``now``, and no hold stands, since a held sentence's line enters only once it
    is decided. The lines of a turn over at the waiter's own bound are left to that turn's end."""
    return (
        record.owed > 0
        and record.hold is None
        and record.bound is not None
        and record.bound <= now
        and record.bound < by
    )


def speech_dropped(record: SpeechRecord) -> SpeechRecord:
    """The lines owed are given up."""
    return _settled(replace(record, credit=min(record.credit, 0)))


def lines_owed(records: Iterable[SpeechRecord], dropped: int = 0) -> OwedLines:
    """What the records owe together: the lines, the first hold standing over a sentence that owes one, and the earliest
    bound of the sentences that owe one without a hold."""
    owing = [each for each in records if each.owed]
    bounds = [each.bound for each in owing if each.hold is None and each.bound is not None]
    held = next((each.hold for each in owing if each.hold is not None), None)
    return OwedLines(sum(each.owed for each in owing), held, min(bounds, default=None), dropped)


class Telling(StrEnum):
    """Whether a turn was told a hold: not yet, a turn runs that was, or it ended without deciding."""

    NOT_TOLD = "not_told"
    TELLING = "telling"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Held:
    """A hold as the window states it: its id and what people said over him, which stays what the last read that had
    a record said once the record is gone, and whether a turn was told it."""

    id: str
    speech: tuple[PartialSpeech, ...]
    record_gone: bool
    telling: Telling = Telling.NOT_TOLD


class HoldOutcome(StrEnum):
    """How a hold ended on the runner's side, as the one line per hold id reports it."""

    CONTINUED = "continued"
    STOPPED = "stopped"
    UNDECIDED = "undecided"


def hold_stated(window: PlaceWindow, seen: Held | None) -> Held | None:
    """The hold the window states, or ``None`` when he is not held; a hold whose record has gone keeps the words an
    earlier read of the same hold had, and what a turn was told of it."""
    hold = window.interrupted
    if hold is None:
        return None
    talked_over = window.talked_over
    same = seen is not None and seen.id == hold.id
    kept = seen.speech if seen is not None and same else ()
    return Held(
        hold.id,
        kept if talked_over is None else talked_over.speech,
        record_gone=talked_over is None,
        telling=seen.telling if seen is not None and same else Telling.NOT_TOLD,
    )


def hold_read(record: SpeechRecord, window: PlaceWindow) -> SpeechRecord:
    """The record once the window is read: the hold it states, none when that is the one decided last."""
    if window.interrupted is not None and window.interrupted.id == record.decided:
        return replace(record, hold=None)
    return replace(record, hold=hold_stated(window, record.hold))


def hold_untold(record: SpeechRecord) -> Held | None:
    """The hold that stands, that no turn was told and that may be told."""
    held = record.hold
    return held if held is not None and held.telling is Telling.NOT_TOLD and hold_tellable(held) else None


def hold_marked_told(record: SpeechRecord) -> SpeechRecord:
    """A turn is told the hold that stands."""
    assert record.hold is not None  # noqa: S101 - a turn is told only a hold that stands
    return replace(record, hold=replace(record.hold, telling=Telling.TELLING))


def hold_turn_ended(record: SpeechRecord) -> SpeechRecord:
    """The turn that was told the hold ended: a hold it did not decide stays undecided for the next read."""
    held = record.hold
    if held is None or held.telling is not Telling.TELLING:
        return record
    return replace(record, hold=replace(held, telling=Telling.UNDECIDED))


def hold_decided(record: SpeechRecord, hold_id: str) -> SpeechRecord:
    """The hold ended with a decision: it stands no more and is never decided again."""
    return replace(record, hold=None, decided=hold_id)


def hold_tellable(held: Held) -> bool:
    """Whether a hold may be told: a person has said words over him, or the record is gone and no more will come."""
    return held.record_gone or any(each.text for each in held.speech)


def untold_words(said: Iterable[PartialSpeech], told: Iterable[PartialSpeech]) -> tuple[PartialSpeech, ...]:
    """What people have said over him that he was not told: each partial with words that differs from the one of the
    same name that was told."""
    known = {(each.name, each.text) for each in told}
    return tuple(each for each in said if each.text and (each.name, each.text) not in known)
