"""What a turn is told of the day beyond its history: the kept lines of the day, bounded, and the period after which a
line is swept. Pure values with no process, network or clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final

from personas.entries import KeptLine, Visit, newest_that_fit, told_bytes, told_line

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from personas.entries import History

SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_HOUR: Final = 3600
SECONDS_PER_DAY: Final = 24 * SECONDS_PER_HOUR


@dataclass(frozen=True)
class TodayBounds:
    """What a turn is told of the day: the lines first read within ``hours``, at most ``size_bytes`` of them as told."""

    hours: float
    size_bytes: int


@dataclass(frozen=True)
class EarlierVisit:
    """An earlier visit as a turn is told it: how many minutes ago its first line was read, and its lines. It is a
    visit whose lines the history does not hold."""

    minutes_ago: int
    lines: tuple[KeptLine, ...]


@dataclass(frozen=True)
class EarlierToday:
    """The kept lines of the day that the history does not hold, as earlier visits in the order they happened, and
    whether older lines were left out."""

    visits: tuple[EarlierVisit, ...]
    older_dropped: bool


def _before(now: datetime, seconds: float) -> datetime | None:
    """The time ``seconds`` before ``now``, or nothing when that lies before the calendar's first moment."""
    earliest = datetime.min.replace(tzinfo=now.tzinfo)
    if seconds >= (now - earliest).total_seconds():
        return None
    return now - timedelta(seconds=seconds)


def today_since(now: datetime, bounds: TodayBounds) -> datetime:
    """The earliest first read a turn at ``now`` is told of; a day longer than the calendar reaches any line."""
    return _before(now, bounds.hours * SECONDS_PER_HOUR) or datetime.min.replace(tzinfo=now.tzinfo)


def swept_before(now: datetime, lines_keep_days: float) -> datetime | None:
    """The time before which a line first read is swept at ``now``, or nothing when the period reaches any line."""
    return _before(now, lines_keep_days * SECONDS_PER_DAY)


def _remaining(kept: Iterable[KeptLine], leaving_out: History, now: datetime) -> Iterable[KeptLine]:
    """The kept lines a turn is still to be told: those the history holds are left out by visit and seq, and those first
    read after ``now``, which no history of that turn holds yet."""
    held = {(entry.visit, entry.seq) for entry in leaving_out.entries if isinstance(entry, KeptLine)}
    return (line for line in kept if (line.visit, line.seq) not in held and line.first_read_at <= now)


def earlier_today(kept: Iterable[KeptLine], now: datetime, bounds: TodayBounds, leaving_out: History) -> EarlierToday:
    """The lines of the day a turn is told besides its history: grouped by visit, visits in the order of their first
    line and lines in seq order; past ``size_bytes`` of the lines as told the oldest lines are left out, and what is
    left out beforehand (see ``_remaining``) counts nothing toward that bound."""
    by_visit: dict[Visit, list[KeptLine]] = {}
    for line in _remaining(kept, leaving_out, now):
        by_visit.setdefault(line.visit, []).append(line)
    groups = sorted((sorted(lines, key=lambda line: line.seq) for lines in by_visit.values()), key=_first_read)
    newest_first = (line for lines in reversed(groups) for line in reversed(lines))
    fitting = newest_that_fit((told_bytes(told_line(line)) for line in newest_first), bounds.size_bytes)
    left_out = sum(len(lines) for lines in groups) - fitting
    older_dropped = left_out > 0
    visits: list[EarlierVisit] = []
    for lines in groups:
        told, left_out = lines[left_out:], max(0, left_out - len(lines))
        if told:
            visits.append(EarlierVisit(int((now - _first_read(lines)).total_seconds() // 60), tuple(told)))
    return EarlierToday(tuple(visits), older_dropped)


def _first_read(lines: Sequence[KeptLine]) -> datetime:
    return min(line.first_read_at for line in lines)


def told_earlier_today(earlier: EarlierToday) -> dict[str, object]:
    visits = [
        {"minutes_ago": visit.minutes_ago, "lines": [told_line(each) for each in visit.lines]}
        for visit in earlier.visits
    ]
    return {"older_dropped": earlier.older_dropped, "visits": visits}
