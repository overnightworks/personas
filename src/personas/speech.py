"""What he still owes of his speech at one place and the hold that stands there; with no process, network or clock."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agents.member import ContinueSpeakingRefusal, StopSpeakingRefusal

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
    answered as spoken, the sentence that plays and, per line, the time by which the turn that sent its sentence is
    over. A line credited
    before the answer of its request leaves the count below zero until that answer. It also holds the hold that stands
    at the place and the id of the last hold decided there. All of it outlives the turn that sent."""

    in_flight: int = 0
    credit: int = 0
    saying: str | None = None
    hold: Held | None = None
    decided: str | None = None
    sending: tuple[float, ...] = ()
    bounds: tuple[float, ...] = ()

    @property
    def bound(self) -> float | None:
        """The time by which the turn that sent the oldest owed line is over."""
        return self.bounds[0] if self.bounds else None

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
    """With no request in flight the count is not below zero, and a time bound is kept only for a line still owed."""
    if record.in_flight > 0:
        return record
    credit = max(record.credit, 0)
    return replace(record, credit=credit, sending=(), bounds=record.bounds[:credit])


def speech_sent(record: SpeechRecord, bound: float) -> SpeechRecord:
    """A request to ``speak`` counts from its send; the turn that sends it is over by ``bound``."""
    return replace(record, in_flight=record.in_flight + 1, sending=(*record.sending, bound))


def speech_answered(record: SpeechRecord, sentence: str, *, spoken: bool) -> SpeechRecord:
    """A request ended: it stops counting on any answer other than ``Spoken``, which owes a line and is the sentence
    that plays."""
    sent, *sending = record.sending
    answered = replace(record, in_flight=record.in_flight - 1, sending=tuple(sending))
    if spoken:
        bounds = (*record.bounds, sent) if record.credit >= 0 else record.bounds
        answered = replace(answered, credit=record.credit + 1, saying=sentence, bounds=bounds)
    return _settled(answered)


def speech_line_entered(record: SpeechRecord) -> SpeechRecord:
    """His own spoken line entered at the place: it is credited while a request is in flight or a line is still owed,
    and ignored otherwise."""
    if record.in_flight > 0 or record.credit > 0:
        return _settled(replace(record, credit=record.credit - 1, bounds=record.bounds[1:]))
    return record


def speech_ended(record: SpeechRecord) -> SpeechRecord:
    """The turn that sent ended: its requests in flight are gone, and the lines it still owes stay for whoever waits on
    them, with the sentence that plays; with nothing owed and no hold the sentence is forgotten."""
    ended = _settled(replace(record, in_flight=0))
    return ended if ended.hold is not None or ended.owed else replace(ended, saying=None)


def _lapsed_lines(record: SpeechRecord, now: float, by: float) -> int:
    """How many of the oldest lines owed have a sending turn that was over before ``by`` and is at ``now``."""
    if record.hold is not None:
        return 0
    lapsed = 0
    for bound in record.bounds:
        if bound > now or bound >= by:
            break
        lapsed += 1
    return lapsed


def speech_lapsed(record: SpeechRecord, now: float, by: float) -> bool:
    """Whether a line owed at the place will not come, for a waiter whose own turn is over by ``by``: the turn that
    sent its sentence was over before that and at ``now``, and no hold stands, since a held sentence's line enters only
    once it is decided. The lines of a turn over at the waiter's own bound are left to that turn's end."""
    return _lapsed_lines(record, now, by) > 0


def speech_dropped(record: SpeechRecord, now: float, by: float) -> SpeechRecord:
    """The lines whose sending turn is over are given up."""
    lapsed = _lapsed_lines(record, now, by)
    return _settled(replace(record, credit=record.credit - lapsed, bounds=record.bounds[lapsed:]))


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
    GONE = "gone"
    """The hold no longer stood at the app, so the decision changed nothing there."""


_HOLD_NOT_STANDING: Final = frozenset({ContinueSpeakingRefusal.NOT_INTERRUPTED, StopSpeakingRefusal.NOT_INTERRUPTED})


def hold_ended_as(decided: HoldOutcome, refusal: ContinueSpeakingRefusal | StopSpeakingRefusal | None) -> HoldOutcome:
    """How the hold ended once the app answered the decision: as decided, or gone when the app refused it because no
    such hold stood."""
    return HoldOutcome.GONE if refusal in _HOLD_NOT_STANDING else decided


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


def hold_decided(record: SpeechRecord, hold_id: str, bound: float) -> SpeechRecord:
    """The hold ended with a decision: it stands no more and is never decided again, and its sentence's line, which
    enters only now, is owed over by ``bound``."""
    bounds = (bound, *record.bounds[1:]) if record.bounds else record.bounds
    return replace(record, hold=None, decided=hold_id, bounds=bounds)


def hold_tellable(held: Held) -> bool:
    """Whether a hold may be told: a person has said words over him, or the record is gone and no more will come."""
    return held.record_gone or any(each.text for each in held.speech)


def untold_words(said: Iterable[PartialSpeech], told: Iterable[PartialSpeech]) -> tuple[PartialSpeech, ...]:
    """What people have said over him that he was not told: each partial with words that differs from the one of the
    same name that was told."""
    known = {(each.name, each.text) for each in told}
    return tuple(each for each in said if each.text and (each.name, each.text) not in known)
