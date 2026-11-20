"""What he holds of his speech at one place and the hold that stands there; with no process, network or clock."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from agents.member import ContinueSpeakingRefusal, StopSpeakingRefusal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.member import PartialSpeech, PlaceWindow


@dataclass(frozen=True)
class SpeechRecord:
    """What he holds of his speech at one place: the requests in flight, the sentence that plays, the hold that stands
    there and the id of the last hold decided there. All of it outlives the turn that sent."""

    in_flight: int = 0
    saying: str | None = None
    hold: Held | None = None
    decided: str | None = None


NO_SPEECH: Final = SpeechRecord()


def speech_sent(record: SpeechRecord) -> SpeechRecord:
    """A request to ``speak`` counts from its send."""
    return SpeechRecord(record.in_flight + 1, record.saying, record.hold, record.decided)


def speech_answered(record: SpeechRecord, sentence: str, *, spoken: bool) -> SpeechRecord:
    """A request ended: it stops counting, and a sentence answered as ``Spoken`` is the one that plays."""
    return SpeechRecord(record.in_flight - 1, sentence if spoken else record.saying, record.hold, record.decided)


def speech_line_entered(record: SpeechRecord) -> SpeechRecord:
    """His own spoken line entered at the place: the sentence that played has ended."""
    return SpeechRecord(record.in_flight, None, record.hold, record.decided)


def speech_ended(record: SpeechRecord) -> SpeechRecord:
    """The turn that sent ended: its requests in flight are gone; the sentence that plays and the hold stay."""
    return SpeechRecord(0, record.saying, record.hold, record.decided)


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
        return SpeechRecord(record.in_flight, record.saying, None, record.decided)
    return SpeechRecord(record.in_flight, record.saying, hold_stated(window, record.hold), record.decided)


def hold_untold(record: SpeechRecord) -> Held | None:
    """The hold that stands, that no turn was told and that may be told."""
    held = record.hold
    return held if held is not None and held.telling is Telling.NOT_TOLD and hold_tellable(held) else None


def hold_marked_told(record: SpeechRecord) -> SpeechRecord:
    """A turn is told the hold that stands."""
    assert record.hold is not None  # noqa: S101 - a turn is told only a hold that stands
    return SpeechRecord(record.in_flight, record.saying, replace(record.hold, telling=Telling.TELLING), record.decided)


def hold_turn_ended(record: SpeechRecord) -> SpeechRecord:
    """The turn that was told the hold ended: a hold it did not decide stays undecided for the next read."""
    held = record.hold
    if held is None or held.telling is not Telling.TELLING:
        return record
    return SpeechRecord(record.in_flight, record.saying, replace(held, telling=Telling.UNDECIDED), record.decided)


def hold_decided(record: SpeechRecord, hold_id: str) -> SpeechRecord:
    """The hold ended with a decision: it stands no more and is never decided again."""
    return SpeechRecord(record.in_flight, record.saying, None, hold_id)


def hold_tellable(held: Held) -> bool:
    """Whether a hold may be told: a person has said words over him, or the record is gone and no more will come."""
    return held.record_gone or any(each.text for each in held.speech)


def untold_words(said: Iterable[PartialSpeech], told: Iterable[PartialSpeech]) -> tuple[PartialSpeech, ...]:
    """What people have said over him that he was not told: each partial with words that differs from the one of the
    same name that was told."""
    known = {(each.name, each.text) for each in told}
    return tuple(each for each in said if each.text and (each.name, each.text) not in known)
