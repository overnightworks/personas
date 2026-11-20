"""Which of his places has a turn due, when his next turn may start, and what a pod that takes him over keeps of his
turns; with no process, network or clock.

It imports ``entries``, ``decision`` and ``present`` and is imported by none of them: ``entries`` <- ``recall`` <-
``decision`` <- ``present`` <- ``due``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from personas.decision import AppPlace, Checkpoint, Joined, active_window_places
from personas.entries import KeptCheckpoint, KeptDue, KeptTurns, Lines, Reading
from personas.present import spoke_since

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from agents.member import Standing

    from personas.entries import History, Occasion


class WaitReason(StrEnum):
    """What first stood directly in front of a due place once its wait began: an older due place, else his running
    turn at the place's app or at another, else a pause of his turns; nothing when its turn could start at once."""

    NONE = "none"
    OLDER_DUE = "older_due"
    TURN_HERE = "turn_here"
    TURN_ELSEWHERE = "turn_elsewhere"
    PAUSED = "paused"


@dataclass(frozen=True)
class Due:
    """What a due place's turn is for, ``since``, the earliest moment the place was due that the turn answers, on
    the clock's monotonic scale, and what first kept it waiting; a turn due again after one with no answer is due from
    the end of its pause."""

    occasion: Occasion
    since: float
    waited_on: WaitReason = WaitReason.NONE


@dataclass(frozen=True)
class Turns:
    """The place of the one turn of an agent that runs, if one does, which places have a turn due, in the order they
    became due, with what the turn is for, and until when, on the clock's monotonic scale, no turn of his may start."""

    running: AppPlace | None = None
    due: Mapping[AppPlace, Due] = field(default_factory=dict[AppPlace, Due])
    not_before: float = 0.0


def _joined(held: Occasion | None, occasion: Occasion) -> Occasion:
    """Lines of one generation join in the trace of the newer ones; any other occasion replaces what was held."""
    if isinstance(held, Lines) and isinstance(occasion, Lines) and held.generation == occasion.generation:
        newer = max(held, occasion, key=lambda lines: max(lines.seqs, default=0))
        return Lines(held.generation, held.seqs | occasion.seqs, held.missed or occasion.missed, newer.traceparent)
    return occasion


def with_due(turns: Turns, at: AppPlace, occasion: Occasion, since: float) -> Turns:
    """A turn due at ``at`` for ``occasion`` from ``since``: a place already due keeps its position and its earliest
    ``since``, lines of one generation join, and lines of another generation or a newer view revision replace what
    was due."""
    held = turns.due.get(at)
    due = (
        Due(occasion, since)
        if held is None
        else Due(_joined(held.occasion, occasion), min(held.since, since), held.waited_on)
    )
    return Turns(turns.running, {**turns.due, at: due}, turns.not_before)


def lines_of_a_hold(generation: str) -> Lines:
    """What a hold makes a place due for: lines of the window's generation with none."""
    return Lines(generation, frozenset(), missed=False)


def due_for_hold(turns: Turns, at: AppPlace, generation: str, since: float) -> Turns:
    """A hold that may be told stands at ``at``: the place is due from ``since`` for lines of the window's generation
    with none, unless it is due already, when the turn it has is the one that is told the hold."""
    if at in turns.due:
        return turns
    return with_due(turns, at, lines_of_a_hold(generation), since)


def due_again(turns: Turns, at: AppPlace, occasion: Occasion, since: float) -> Turns:
    """A turn due again at ``at`` for ``occasion`` from ``since``, the end of the pause the turn left, which a turn
    ended without an answer: a place due meanwhile keeps what is newer, except that lines of the same generation join;
    any other place goes to the back of the order. A hold that made it due without a line brings no turn again: the
    read after the turn decides it."""
    held = turns.due.get(at)
    if held is None and isinstance(occasion, Lines) and not occasion.seqs:
        return turns
    if held is None or (
        isinstance(held.occasion, Lines)
        and isinstance(occasion, Lines)
        and held.occasion.generation == occasion.generation
    ):
        return with_due(turns, at, occasion, since)
    return turns


def without_due(turns: Turns, leaving: Callable[[AppPlace], bool]) -> Turns:
    """The turns without the due places for which ``leaving`` holds."""
    due = {at: held for at, held in turns.due.items() if not leaving(at)}
    return Turns(turns.running, due, turns.not_before)


def _in_front(turns: Turns, at: AppPlace, now: float) -> WaitReason:
    """What stands directly in front of the due place ``at`` at ``now``."""
    if next(iter(turns.due)) != at:
        return WaitReason.OLDER_DUE
    if turns.running is not None:
        return WaitReason.TURN_HERE if turns.running.app == at.app else WaitReason.TURN_ELSEWHERE
    if now < turns.not_before:
        return WaitReason.PAUSED
    return WaitReason.NONE


def waited(turns: Turns, now: float) -> Turns:
    """The turns once each due place whose wait has begun by ``now`` and that nothing kept waiting yet names what stands
    in front of it now; a place keeps the first reason it named."""
    due = {
        at: held
        if held.waited_on is not WaitReason.NONE or held.since > now
        else replace(held, waited_on=_in_front(turns, at, now))
        for at, held in turns.due.items()
    }
    return Turns(turns.running, due, turns.not_before)


def _oldest_in(turns: Turns, app: str) -> tuple[AppPlace, Due] | None:
    """The place that has waited longest and what its turn is for, when that place is in ``app``."""
    if not turns.due:
        return None
    at, due = next(iter(turns.due.items()))
    return (at, due) if at.app == app else None


def claimed(turns: Turns, app: str, now: float) -> tuple[Turns, AppPlace, Due] | None:
    """The turns after a turn starts at the place that has waited longest, with that place and what the turn is for
    and since when; none while a turn runs, while a pause holds his turns at ``now``, or when that place is not in
    ``app``."""
    oldest = _oldest_in(turns, app)
    if oldest is None or _in_front(turns, oldest[0], now) is not WaitReason.NONE:
        return None
    at, due = oldest
    return replace(without_due(turns, lambda each: each == at), running=at), at, due


def paused_for(turns: Turns, now: float, seconds: float) -> Turns:
    """The turns while no turn of his may start for ``seconds`` from ``now``."""
    return Turns(turns.running, turns.due, now + seconds)


def pause_left(turns: Turns, app: str, now: float) -> float:
    """The seconds a pause still holds the turn of the place that has waited longest, when it is in ``app``; zero when
    that turn waits for something else or for nothing."""
    oldest = _oldest_in(turns, app)
    waits_for_pause = oldest is not None and _in_front(turns, oldest[0], now) is WaitReason.PAUSED
    return turns.not_before - now if waits_for_pause else 0.0


def released(turns: Turns) -> Turns:
    """The turns once the running turn ended; what is due stays."""
    return Turns(None, turns.due, turns.not_before)


@dataclass(frozen=True)
class Claim:
    """The running turn's place, what it was due for, and how many entries his history had taken when it was claimed."""

    at: AppPlace
    due: Due
    mark: int


def cut_by_stop(turns: Turns, claim: Claim, history: History) -> Turns:
    """The turns once his stop cut the claimed turn. One that spoke leaves nothing due, since a line said twice is worse
    than one left unanswered. One that did not leaves its place due again first, for what it was due for joined with
    what made it due meanwhile, so the pod that takes him over answers it once."""
    if spoke_since(history, claim.mark):
        return turns
    held = turns.due.get(claim.at)
    occasion = claim.due.occasion if held is None else _joined(claim.due.occasion, held.occasion)
    rest = {at: due for at, due in turns.due.items() if at != claim.at}
    return Turns(turns.running, {claim.at: replace(claim.due, occasion=occasion), **rest}, turns.not_before)


def left_standing(app: str, standings: tuple[Standing, ...]) -> Callable[[AppPlace], bool]:
    """Whether a due place is one of ``app`` that is no active window place any more; a view place is one."""
    active = {AppPlace(app, place) for place in active_window_places(standings)}
    return lambda at: at.app == app and at not in active


@dataclass(frozen=True)
class TurnState:
    """What a pod that takes him over needs of his turns: the turns, where reading each of his window places stands,
    and how many of his turns in a row ended without an answer."""

    turns: Turns = field(default_factory=Turns)
    checkpoints: Mapping[AppPlace, Checkpoint] = field(default_factory=dict[AppPlace, Checkpoint])
    unanswered: int = 0


NO_TURNS: Final = TurnState()
"""The turns of an agent no pod served before: none due, no checkpoint, no pause."""


def pause_deadline(not_before: float, now: float, wall: datetime) -> datetime | None:
    """The wall time at which a pause that holds his turns until ``not_before`` ends, read at ``now`` on the same
    monotonic scale and at ``wall``; none once it is over."""
    return wall + timedelta(seconds=not_before - now) if not_before > now else None


def kept_turns(state: TurnState, paused_until: datetime | None) -> KeptTurns:
    """His turns as the store keeps them. A running turn is not kept, so a pod that takes him over never answers what
    it was answering again; a place whose window was said before or while its sponsor was away keeps no checkpoint,
    since it resumes as such."""
    checkpoints = tuple(
        KeptCheckpoint(at.app, at.place, None if isinstance(checkpoint, Joined) else checkpoint)
        for at, checkpoint in state.checkpoints.items()
        if isinstance(checkpoint, Joined | Reading)
    )
    due = tuple(KeptDue(at.app, at.place, held.occasion) for at, held in state.turns.due.items())
    return KeptTurns(checkpoints, due, state.unanswered, paused_until)


def resumed_turns(kept: KeptTurns, now: float, wall: datetime) -> TurnState:
    """The turns a pod resumes him with at ``now`` and ``wall``: each kept due place due from ``now``, in the kept
    order, and the kept pause holding his turns for what is left of it."""
    left = 0.0 if kept.paused_until is None else max(0.0, (kept.paused_until - wall).total_seconds())
    due = {AppPlace(held.app, held.place): Due(held.occasion, now) for held in kept.due}
    checkpoints: dict[AppPlace, Checkpoint] = {
        AppPlace(held.app, held.place): Joined() if held.reading is None else held.reading for held in kept.checkpoints
    }
    return TurnState(Turns(due=due, not_before=now + left), checkpoints, kept.unanswered)
