"""The period after which a kept line is swept. Pure values with no process, network or clock."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_DAY: Final = 24 * 3600


def swept_before(now: datetime, lines_keep_days: float) -> datetime | None:
    """The time before which a line first read is swept at ``now``, or nothing when the period reaches any line."""
    earliest = datetime.min.replace(tzinfo=now.tzinfo)
    seconds = lines_keep_days * SECONDS_PER_DAY
    if seconds >= (now - earliest).total_seconds():
        return None
    return now - timedelta(seconds=seconds)
