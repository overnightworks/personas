"""What he holds of his speech at one place and the hold that stands there; with no process, network or clock."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final, NamedTuple, cast

from agents.member import ContinueSpeakingRefusal, HoldPlace, SpeakRefusal, Spoken, StopSpeakingRefusal, Written

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.member import PartialSpeech, PlaceWindow


@dataclass(frozen=True)
class Sentence:
    """A sentence released into the record, with the time its sending turn's longest duration ends at, on the clock's
    monotonic scale."""

    text: str
    bound: float


@dataclass(frozen=True)
class Request:
    """The request to ``speak`` that is open for a sentence, whether his own stop reached it, whether his own line
    for it entered before its answer did and, once a short line of his cut the hold, the bound the sentence keeps if the
    app drops it."""

    sentence: Sentence
    cut: bool = False
    line_entered: bool = False
    line_bound: float | None = None

    def cut_by_stop(self, refusal: SpeakRefusal) -> bool:
        """Whether the refusal is the app's answer to his own stop or to his short line, which is no news."""
        return (self.cut or self.line_bound is not None) and refusal is SpeakRefusal.STOPPED


class Answered(StrEnum):
    """How the request to ``speak`` for a sentence ended."""

    SPOKEN = "spoken"
    WRITTEN = "written"
    HELD = "held"
    STOPPED = "stopped"
    """The app dropped a queued sentence for his own stop or for the answer that replaced it."""
    REFUSED = "refused"
    """Any refusal but held and stopped."""
    UNANSWERED = "unanswered"
    """The request failed, was cut or was given up."""


class DropReason(StrEnum):
    """Why unsent sentences were dropped, as the one line per drop reports it."""

    REPLACED = "replaced"
    LIMIT_EXCEEDED = "limit_exceeded"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class SpeechRecord:
    """What he holds of his speech at one place: the request in flight, the sentences released and not yet sent, the
    sentence that plays, the hold that stands there and the key of the last hold decided there. All of it outlives the
    turn that released the sentences. ``held_unread`` says the app answered a sentence as held and no read of the window
    has stated the hold since."""

    in_flight: Request | None = None
    unsent: tuple[Sentence, ...] = ()
    saying: str | None = None
    hold: Held | None = None
    decided: HoldKey | None = None
    held_unread: bool = False


NO_SPEECH: Final = SpeechRecord()


def _revised(record: SpeechRecord, **changes: object) -> SpeechRecord:
    """The record with the named fields changed; the cast states the type a static scanner reads as a bare dataclass."""
    return cast("SpeechRecord", replace(record, **changes))  # pyright: ignore[reportUnnecessaryCast]


_REFUSALS_ANSWERED: Final = {SpeakRefusal.HELD: Answered.HELD, SpeakRefusal.STOPPED: Answered.STOPPED}


def answered_as(ended: object) -> Answered:
    """How a request ended, from its answer or from the error or cancellation it ended with."""
    if isinstance(ended, Spoken):
        return Answered.SPOKEN
    if isinstance(ended, Written):
        return Answered.WRITTEN
    if isinstance(ended, SpeakRefusal):
        return _REFUSALS_ANSWERED.get(ended, Answered.REFUSED)
    return Answered.UNANSWERED


def bound_reached(now: float, bound: float) -> bool:
    """Whether a bound is reached at ``now``, a tie going to the bound."""
    return now >= bound


def sentences_released(
    record: SpeechRecord, texts: Iterable[str], bound: float, *, replacing: bool
) -> tuple[SpeechRecord, int]:
    """The record once a turn released sentences with its bound, and how many unsent ones it dropped: the first
    sentence of a spoken turn replaces the rest of the turns before it, the sentence in flight and the one that plays
    staying."""
    kept = () if replacing else record.unsent
    released = tuple(Sentence(text, bound) for text in texts)
    return replace(record, unsent=(*kept, *released)), len(record.unsent) - len(kept)


def sentence_taken(record: SpeechRecord, now: float) -> tuple[SpeechRecord, Sentence | None, int]:
    """The record once the next sentence goes out, that sentence and how many unsent ones past their bound it dropped.
    Nothing is dropped or sent while a hold stands; nothing is sent while a request is open or after one answered as
    held until a read of the window."""
    if record.hold is not None:
        return record, None, 0
    alive = tuple(each for each in record.unsent if not bound_reached(now, each.bound))
    expired = len(record.unsent) - len(alive)
    if record.in_flight is not None or record.held_unread or not alive:
        return replace(record, unsent=alive), None, expired
    first, *rest = alive
    return replace(record, in_flight=Request(first), unsent=tuple(rest)), first, expired


def speech_answered(record: SpeechRecord, answered: Answered) -> SpeechRecord:
    """The open request ended. A sentence answered as spoken is the one that plays, unless his own line for it entered
    already; one answered as held goes back first among the unsent; any other refusal drops the unsent ones."""
    request = record.in_flight
    assert request is not None  # noqa: S101 - an answer ends the request that was sent
    ended = _revised(record, in_flight=None)
    match answered:
        case Answered.SPOKEN:
            return ended if request.line_entered else _revised(ended, saying=request.sentence.text)
        case Answered.HELD:
            return _revised(ended, unsent=(request.sentence, *record.unsent), held_unread=True)
        case Answered.STOPPED if request.cut:
            return ended
        case Answered.STOPPED if request.line_bound is not None:
            return _revised(ended, unsent=(Sentence(request.sentence.text, request.line_bound), *record.unsent))
        case Answered.REFUSED | Answered.STOPPED:
            return _revised(ended, unsent=())
        case _:
            return ended


def speech_line_entered(record: SpeechRecord) -> SpeechRecord:
    """His own spoken line entered at the place: the sentence that played has ended, or, when none plays, the request
    open for a sentence has already played to its end."""
    if record.saying is not None:
        return _revised(record, saying=None)
    if record.in_flight is not None:
        return _revised(record, in_flight=replace(record.in_flight, line_entered=True))
    return record


def speech_unsaid(record: SpeechRecord) -> tuple[str, ...]:
    """The sentences not said yet: the one in flight, then the unsent ones in order."""
    open_ = () if record.in_flight is None else (record.in_flight.sentence,)
    return tuple(each.text for each in (*open_, *record.unsent))


def speech_stopped(record: SpeechRecord) -> tuple[SpeechRecord, int]:
    """His own stop: the unsent sentences drop, which is how many; a request in flight is marked cut and runs to its
    answer."""
    cut = None if record.in_flight is None else replace(record.in_flight, cut=True)
    return replace(record, in_flight=cut, unsent=()), len(record.unsent)


def speech_line_said(record: SpeechRecord, bound: float) -> SpeechRecord:
    """His short line before he goes on: the unsent sentences have the deciding turn's bound, and the sentence in flight
    goes back first among them when the app drops it for the line."""
    continued = speech_continued(record, bound)
    if continued.in_flight is None:
        return continued
    return _revised(continued, in_flight=replace(continued.in_flight, line_bound=bound))


def speech_started(record: SpeechRecord, text: str) -> SpeechRecord:
    """The sentence that answered his hold plays now."""
    return _revised(record, saying=text)


def speech_continued(record: SpeechRecord, bound: float) -> SpeechRecord:
    """His continue: the unsent sentences have the deciding turn's bound."""
    return _revised(record, unsent=tuple(Sentence(each.text, bound) for each in record.unsent))


def speech_ended(record: SpeechRecord) -> tuple[SpeechRecord, int]:
    """The sender ended: its request in flight is gone and the unsent sentences drop, which is how many; the sentence
    that plays and the hold stay."""
    return replace(record, in_flight=None, unsent=(), held_unread=False), len(record.unsent)


def unsent_dropped(record: SpeechRecord) -> tuple[SpeechRecord, int]:
    """The unsent sentences drop, which is how many."""
    return replace(record, unsent=()), len(record.unsent)


class Telling(StrEnum):
    """Whether a turn was told a hold: not yet, a turn runs that was, or it ended without deciding."""

    NOT_TOLD = "not_told"
    TELLING = "telling"
    UNDECIDED = "undecided"


class HoldKey(NamedTuple):
    """A hold is told apart by its id and the window generation that stated it: the app numbers holds per window
    generation, so an id comes again in a new one."""

    generation: str
    id: str


@dataclass(frozen=True)
class Held:
    """A hold as the window states it: its id and window generation, the place the app says it stopped him at, if it
    does, and what people said over him, which stays what the last read that had a record said once the record is
    gone, and whether a turn was told it."""

    id: str
    generation: str
    speech: tuple[PartialSpeech, ...]
    record_gone: bool
    telling: Telling = Telling.NOT_TOLD
    place: HoldPlace | None = None

    @property
    def key(self) -> HoldKey:
        return HoldKey(self.generation, self.id)


class HoldOutcome(StrEnum):
    """How a hold ended on the runner's side, as the one line per hold id reports it."""

    CONTINUED = "continued"
    SHORT_LINE = "short_line"
    STOPPED = "stopped"
    ANSWERED = "answered"
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
    same = seen is not None and seen.key == HoldKey(window.generation, hold.id)
    kept = seen.speech if seen is not None and same else ()
    return Held(
        hold.id,
        window.generation,
        kept if talked_over is None else talked_over.speech,
        record_gone=talked_over is None,
        telling=seen.telling if seen is not None and same else Telling.NOT_TOLD,
        place=hold.place,
    )


def hold_read(record: SpeechRecord, window: PlaceWindow) -> SpeechRecord:
    """The record once the window is read: the hold it states, none when that is the one decided last."""
    if window.interrupted is not None and HoldKey(window.generation, window.interrupted.id) == record.decided:
        return _revised(record, hold=None, held_unread=False)
    return _revised(record, hold=hold_stated(window, record.hold), held_unread=False)


def hold_untold(record: SpeechRecord) -> Held | None:
    """The hold that stands, that no turn was told and that may be told."""
    held = record.hold
    return held if held is not None and held.telling is Telling.NOT_TOLD and hold_tellable(held) else None


def hold_marked_told(record: SpeechRecord) -> SpeechRecord:
    """A turn is told the hold that stands."""
    assert record.hold is not None  # noqa: S101 - a turn is told only a hold that stands
    return _revised(record, hold=replace(record.hold, telling=Telling.TELLING))


def hold_turn_ended(record: SpeechRecord) -> SpeechRecord:
    """The turn that was told the hold ended: a hold it did not decide stays undecided for the next read."""
    held = record.hold
    if held is None or held.telling is not Telling.TELLING:
        return record
    return _revised(record, hold=replace(held, telling=Telling.UNDECIDED))


def hold_decided(record: SpeechRecord, key: HoldKey) -> SpeechRecord:
    """The hold ended with a decision: it stands no more and is never decided again in its window generation."""
    return _revised(record, hold=None, decided=key, held_unread=False)


def hold_forgotten(record: SpeechRecord) -> SpeechRecord:
    """The place left the standing: the hold decided there is forgotten, and the rest of the record stays."""
    return _revised(record, decided=None)


def hold_tellable(held: Held) -> bool:
    """Whether a hold may be told: a person has said words over him, or the record is gone and no more will come."""
    return held.record_gone or any(each.text for each in held.speech)


def untold_words(said: Iterable[PartialSpeech], told: Iterable[PartialSpeech]) -> tuple[PartialSpeech, ...]:
    """What people have said over him that he was not told: each partial with words that differs from the one of the
    same name that was told."""
    known = {(each.name, each.text) for each in told}
    return tuple(each for each in said if each.text and (each.name, each.text) not in known)
