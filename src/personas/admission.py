"""What a turn was decided on: the admission its agent's turn runs from."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from datetime import datetime

    from agents.member import PlaceRef

    from personas.entries import History, Occasion
    from personas.present import Present
    from personas.speech import Held
    from personas.tokens import TokenFacts


@dataclass(frozen=True)
class Admitted:
    """A turn decided on, with the facts it was decided from: the token's, what woke it, what its agent had taken in,
    the wall time it was admitted at, whether its text is spoken and the other apps whose tools it holds."""

    facts: TokenFacts
    place: PlaceRef
    occasion: Occasion
    present: Present = field(repr=False)
    at: datetime
    aloud: bool
    apps: frozenset[str]
    held: Held | None = None
    """The hold the turn is told first, which the mind marked as told when the turn was claimed."""
    went_on: bool = False
    """Whether the turn is told first that a hold went on by itself, which the mind marked as told when the turn was
    claimed."""

    @property
    def history(self) -> History:
        return self.present.history

    def groups(self) -> Coroutine[None, None, frozenset[str]]:
        """The library's GroupSource wants an awaitable; the groups are already known."""
        return asyncio.sleep(0, self.facts.groups)
