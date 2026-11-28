"""One agent's mind: the present his requests hold, which is the history of the lines he took in, the views he read and
the calls relayed for him at any of his apps with the turns that were told of it, the apps where he holds an active
place, where reading each of his window places stands, and the one turn he takes at a time, with the places that have
one due. His loops and turns feed it and every
turn reads it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from personas.decision import (
    AppPlace,
    backoff_delay,
    budget_pause,
    held_apps,
    refusal_retry_at,
    speaks_aloud,
    sync_checkpoints,
)
from personas.due import (
    NO_TURNS,
    Claim,
    TurnState,
    claimed,
    cut_by_stop,
    cut_for_spoken,
    due_again,
    due_for_hold,
    pause_left,
    paused_for,
    released,
    waited,
    with_due,
    without_due,
    yields_to_spoken,
)
from personas.entries import KeptLine, Reading, SkillRead, ToolsLoaded
from personas.present import EMPTY_PRESENT, answered_into, round_call_id, taken_into
from personas.speech import (
    NO_SPEECH,
    Answered,
    DropReason,
    Held,
    HoldKey,
    Request,
    Sentence,
    SpeechRecord,
    hold_decided,
    hold_forgotten,
    hold_marked_told,
    hold_read,
    hold_turn_ended,
    hold_untold,
    sentence_taken,
    sentences_released,
    speech_answered,
    speech_continued,
    speech_ended,
    speech_line_entered,
    speech_line_said,
    speech_started,
    speech_stopped,
    speech_unsaid,
    unsent_dropped,
    went_on_marked_told,
    went_on_untold,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime

    from agents.member import PlaceRef, PlaceWindow, Standing

    from personas.decision import Checkpoint
    from personas.due import Due
    from personas.entries import Entry, History, Occasion, TurnTold
    from personas.present import Present


class Mind:
    """Holds one agent's present and turns in memory while this pod serves him, from the present and turns his kept
    history resumed as; the rules that bound the history and move the turns are ``decision``'s. Each method but
    ``until_written`` is synchronous, so a loop changes it between its awaits."""

    def __init__(
        self,
        history_bytes: int,
        history_keep_bytes: int,
        budget_pause_cap_seconds: float,
        present: Present = EMPTY_PRESENT,
        resumed: TurnState = NO_TURNS,
    ) -> None:
        self._history_bytes = history_bytes
        self._history_keep_bytes = history_keep_bytes
        self._budget_pause_cap_seconds = budget_pause_cap_seconds
        self._unanswered_turns = resumed.unanswered
        self._present = present
        self._changed = asyncio.Event()
        self._changes = 0
        self._written_changes = 0
        self._next_write = asyncio.Event()
        self._turns = resumed.turns
        self._claim: Claim | None = None
        self._stopping = False
        self._cut_wanted = asyncio.Event()
        self._voice_apps = frozenset[str]()
        self._checkpoints = resumed.checkpoints
        self._placed = frozenset[str]()
        self._active: dict[str, frozenset[PlaceRef]] = {}
        self._claimed: dict[str, frozenset[PlaceRef]] = {}
        self._not_granted: dict[str, float] = {}
        self._wake_events: list[asyncio.Event] = []
        self._speech: dict[AppPlace, SpeechRecord] = {}
        self._speech_wakes: dict[AppPlace, asyncio.Event] = {}

    @property
    def present(self) -> Present:
        return self._present

    @property
    def history(self) -> History:
        return self._present.history

    def take(self, entries: Iterable[Entry]) -> None:
        """Appends the lines, spoken sentences, views and acts taken, in order; taking nothing leaves the history as it
        is."""
        taken = tuple(entries)
        if not taken:
            return
        for entry in taken:
            if isinstance(entry, KeptLine) and entry.own and entry.spoken:
                at = AppPlace(entry.visit.app, entry.visit.place)
                self._set_speech(at, speech_line_entered(self._speech.get(at, NO_SPEECH)))
        self._present = taken_into(self._present, taken, self._history_bytes, self._history_keep_bytes)
        self._change()

    def sentences_released(
        self, at: AppPlace, texts: Iterable[str], bound: float, allowed: float, *, replacing: bool
    ) -> int:
        """A spoken turn released sentences with its bound, which it gave for ``allowed`` seconds; how many unsent ones
        of the turns before it dropped."""
        record, dropped = sentences_released(
            self._speech.get(at, NO_SPEECH), texts, bound, allowed, replacing=replacing
        )
        self._set_speech(at, record)
        return dropped

    def sentence_to_send(self, at: AppPlace, now: float) -> tuple[Sentence | None, int]:
        """The next sentence to send at the place, which then counts as in flight, none while a hold stands or a request
        is open; and how many unsent ones past their bound dropped."""
        record, sentence, expired = sentence_taken(self._speech.get(at, NO_SPEECH), now)
        self._set_speech(at, record)
        return sentence, expired

    def speech_answered(self, at: AppPlace, answered: Answered) -> Request:
        """The request in flight at the place ended; the request it was."""
        record = self._speech.get(at, NO_SPEECH)
        assert record.in_flight is not None  # noqa: S101 - an answer ends the request that was sent
        self._set_speech(at, speech_answered(record, answered))
        return record.in_flight

    def speech_saying(self, at: AppPlace) -> str | None:
        """The sentence that plays at the place: the one last answered as spoken, until his own line enters."""
        return self._speech.get(at, NO_SPEECH).saying

    def speech_unsaid(self, at: AppPlace) -> tuple[str, ...]:
        """The sentences not said yet at the place: the one in flight, then the unsent ones."""
        return speech_unsaid(self._speech.get(at, NO_SPEECH))

    def speech_stopped(self, at: AppPlace) -> tuple[tuple[str, ...], int]:
        """His stop at the place: the sentences not said yet, and how many unsent ones dropped. A request in flight runs
        to its answer."""
        record = self._speech.get(at, NO_SPEECH)
        stopped, dropped = speech_stopped(record)
        self._set_speech(at, stopped)
        return speech_unsaid(record), dropped

    def unsent_dropped(self, at: AppPlace) -> int:
        """The unsent sentences at the place drop: how many."""
        record, dropped = unsent_dropped(self._speech.get(at, NO_SPEECH))
        self._set_speech(at, record)
        return dropped

    def speech_ended(self, at: AppPlace) -> int:
        """The sender at the place ended: its request in flight is gone and its unsent sentences drop, how many. The
        sentence that plays and the hold that stands stay."""
        record, dropped = speech_ended(self._speech.get(at, NO_SPEECH))
        self._set_speech(at, record)
        return dropped

    def drop_reason_at_end(self) -> DropReason:
        """Why the unsent sentences of a turn or a sender that ended unanswered drop: this pod stops serving him, or
        the turn or session failed."""
        return DropReason.STOPPED if self._stopping else DropReason.FAILED

    def speech_wake(self, at: AppPlace) -> asyncio.Event:
        """The event the place's sender waits on, set by every change of the place's record and by every read of its
        window; the sender clears it."""
        return self._speech_wakes.setdefault(at, asyncio.Event())

    def hold_read(self, at: AppPlace, window: PlaceWindow, now: float) -> Held | None:
        """The place's window was read at ``now``: the hold it states stands in the record, with the words people said
        over him kept after their record ends and whether a turn was told it; what waited has a renewed bound when a
        hold went on by itself. The hold that stands, none when it was decided."""
        record = hold_read(self._speech.get(at, NO_SPEECH), window, now)
        self._set_speech(at, record)
        self._speech_wake(at)
        return record.hold

    def hold_to_tell(self, at: AppPlace) -> Held | None:
        """The hold a turn that starts at the place is told, marked as told: the one that stands, that no turn was
        told and that may be told."""
        record = self._speech.get(at, NO_SPEECH)
        held = hold_untold(record)
        if held is not None:
            self._set_speech(at, hold_marked_told(record))
        return held

    def went_on_to_tell(self, at: AppPlace) -> bool:
        """Whether a round started now is told that a hold went on by itself, once per hold, which it then counts as."""
        record = self._speech.get(at, NO_SPEECH)
        if not went_on_untold(record):
            return False
        self._set_speech(at, went_on_marked_told(record))
        return True

    def sentence_bound(self, at: AppPlace, sentence: Sentence) -> float:
        """The bound of the request in flight for ``sentence``, which a hold going on by itself renews."""
        request = self._speech.get(at, NO_SPEECH).in_flight
        return sentence.bound if request is None else request.sentence.bound

    def hold_makes_due(self, at: AppPlace) -> bool:
        """Whether the hold that stands at the place may be told to a turn, which a read of it then makes due."""
        return hold_untold(self._speech.get(at, NO_SPEECH)) is not None

    def hold_turn_ended(self, at: AppPlace) -> None:
        """The turn told the hold at the place ended: a hold it did not decide stays undecided."""
        self._set_speech(at, hold_turn_ended(self._speech.get(at, NO_SPEECH)))

    def hold_decided(self, at: AppPlace, key: HoldKey) -> None:
        """The hold ``key`` at the place was decided: it stands no more and is not decided again."""
        self._set_speech(at, hold_decided(self._speech.get(at, NO_SPEECH), key))

    def speech_continued(self, at: AppPlace, bound: float) -> None:
        """His continue at the place: the unsent sentences have the deciding turn's ``bound``."""
        self._set_speech(at, speech_continued(self._speech.get(at, NO_SPEECH), bound))

    def speech_line_said(self, at: AppPlace, bound: float) -> None:
        """His short line before he goes on is on its way at the place: the unsent sentences have the deciding turn's
        ``bound``, and the request in flight, if the app drops it for the line, goes back first among them."""
        self._set_speech(at, speech_line_said(self._speech.get(at, NO_SPEECH), bound))

    def speech_started(self, at: AppPlace, text: str) -> None:
        """The sentence that answered his hold at the place plays now."""
        self._set_speech(at, speech_started(self._speech.get(at, NO_SPEECH), text))

    def hold_decided_key(self, at: AppPlace) -> HoldKey | None:
        """The key of the hold decided last at the place."""
        return self._speech.get(at, NO_SPEECH).decided

    def hold_standing(self, at: AppPlace) -> Held | None:
        """The hold that stands at the place."""
        return self._speech.get(at, NO_SPEECH).hold

    def skill_read(self, app: str, name: str, uri: str, body: str, at: datetime) -> None:
        """Appends the entry of a skill the agent read, under the call id minted from its ordinal: the count of entries
        ever appended, which no cut or resume lowers while an entry is kept, so no kept entry holds the id."""
        self.take((SkillRead(app, name, uri, body, at, round_call_id(self.history.taken)),))

    def tools_loaded(self, app: str, skill: str, tools: tuple[str, ...], at: datetime) -> None:
        """Appends the entry of tools the agent loaded, under a call id minted as a read's is."""
        self.take((ToolsLoaded(app, skill, tools, at, round_call_id(self.history.taken)),))

    def _set_speech(self, at: AppPlace, record: SpeechRecord) -> None:
        if record == self._speech.get(at, NO_SPEECH):
            return
        if record == NO_SPEECH:
            del self._speech[at]
        else:
            self._speech[at] = record
        self._speech_wake(at)

    def _speech_wake(self, at: AppPlace) -> None:
        if (wake := self._speech_wakes.get(at)) is not None:
            wake.set()

    def changed(self) -> asyncio.Event:
        """The event every change of his present or of what ``turn_state`` holds sets, which stays set until its
        waiter clears it."""
        return self._changed

    def changes(self) -> int:
        """How many changes ``changed`` was set for so far: a writer reads it in the same synchronous step as what it
        writes, and a claim right after its change."""
        return self._changes

    def written(self, changes: int) -> None:
        """Everything that changed up to the ``changes``-th change is kept in the store now."""
        self._written_changes = changes
        previous, self._next_write = self._next_write, asyncio.Event()
        previous.set()

    async def until_written(self, changes: int) -> None:
        """Returns once everything that changed up to the ``changes``-th change is kept in the store."""
        while self._written_changes < changes:
            await self._next_write.wait()

    def _change(self) -> None:
        self._changes += 1
        self._changed.set()

    def turn_state(self) -> TurnState:
        """His turns, his checkpoints and his unanswered turns in a row, as a pod that takes him over resumes them."""
        return TurnState(self._turns, self._checkpoints, self._unanswered_turns)

    def checkpoint(self, at: AppPlace) -> Checkpoint | None:
        """Where reading the place stands, if it is one of his window places."""
        return self._checkpoints.get(at)

    def generation_at(self, at: AppPlace) -> str | None:
        """The visit an entry taken now at the place belongs to: the generation of its last window read, none for a
        place joined and not read since, restarted or paused, and for one without a checkpoint."""
        checkpoint = self._checkpoints.get(at)
        return checkpoint.generation if isinstance(checkpoint, Reading) else None

    def move_checkpoint(self, at: AppPlace, checkpoint: Checkpoint | None) -> None:
        """Reading the place stands at ``checkpoint`` now; none drops the place's checkpoint."""
        moved = dict(self._checkpoints)
        if checkpoint is None:
            moved.pop(at, None)
        else:
            moved[at] = checkpoint
        self._set_checkpoints(moved)

    def hold_places(self, app: str, places: frozenset[PlaceRef]) -> None:
        """He holds ``places`` active at ``app``, as its loop last read his standing; what it claimed meanwhile is
        settled by that read."""
        self._active = {**self._active, app: places}
        self._claimed = {**self._claimed, app: frozenset()}

    def claim_places(self, app: str, places: frozenset[PlaceRef]) -> None:
        """He is about to take ``places`` at ``app``: each counts as held at once, before the accept is awaited, until
        the next standing read of ``app`` or its ``release``."""
        self._claimed = {**self._claimed, app: self._claimed.get(app, frozenset()) | places}

    def release_places(self, app: str, places: frozenset[PlaceRef]) -> None:
        """An accept of ``places`` at ``app`` did not take them: they count as held no more."""
        self._claimed = {**self._claimed, app: self._claimed.get(app, frozenset()) - places}

    def holds_active_place_elsewhere(self, app: str) -> bool:
        """Whether he holds, or has claimed, an active place of any kind in an app other than ``app``; a parked app's
        places are recorded as none."""
        return any(places for each, places in (*self._active.items(), *self._claimed.items()) if each != app)

    def sync_checkpoints(self, app: str, standings: tuple[Standing, ...], before: tuple[Standing, ...]) -> None:
        """Keeps the checkpoints of ``app``'s places in line with a standing read there, given the one read before."""
        own = {at.place: checkpoint for at, checkpoint in self._checkpoints.items() if at.app == app}
        others = {at: checkpoint for at, checkpoint in self._checkpoints.items() if at.app != app}
        synced = sync_checkpoints(own, standings, before)
        self._set_checkpoints(others | {AppPlace(app, place): checkpoint for place, checkpoint in synced.items()})

    def _set_checkpoints(self, checkpoints: dict[AppPlace, Checkpoint]) -> None:
        if checkpoints != self._checkpoints:
            self._checkpoints = checkpoints
            self._change()

    def unanswered(self, *, budget_spent: bool, now: float) -> float:
        """A turn of his ended without an answer, refused for a spent budget or not: no turn of his starts for the
        seconds this returns, counted from ``now``, whatever becomes of the turn's own task or place meanwhile."""
        self._unanswered_turns += 1
        seconds = (
            budget_pause(self._unanswered_turns, self._budget_pause_cap_seconds)
            if budget_spent
            else backoff_delay(self._unanswered_turns)
        )
        self._turns = paused_for(self._turns, now, seconds)
        self._change()
        return seconds

    def answered(self, turn: TurnTold) -> None:
        """A turn of his ended with an answer: the wait starts over and its message joins the present."""
        self._unanswered_turns = 0
        self._present = answered_into(self._present, turn)
        self._change()

    def stand_at(self, app: str, *, holds: bool) -> None:
        """Records whether he holds an active place at ``app``, as its loop last read his standing."""
        self._placed = self._placed | {app} if holds else self._placed - {app}

    def held_apps(self, own: str) -> frozenset[str]:
        """The apps, other than ``own``, where he holds an active place."""
        return held_apps(own, self._placed)

    def holds_back(self, app: str, now: float) -> bool:
        """Whether the realm refused him ``app`` and its retry is not yet due at ``now``."""
        retry_at = self._not_granted.get(app)
        return retry_at is not None and now < retry_at

    def refused(self, app: str, now: float) -> bool:
        """The realm or ``app`` did not grant him ``app`` at ``now``: it is asked again when the retry is due. Whether
        this is the first refusal since the last grant."""
        first = app not in self._not_granted
        self._not_granted[app] = refusal_retry_at(now)
        return first

    def granted(self, app: str) -> bool:
        """``app`` answered him. Whether it was refused before."""
        return self._not_granted.pop(app, None) is not None

    def wake_event(self) -> asyncio.Event:
        """A new event a loop waits on; the end of a turn or of a pause sets it, and so does a place dropped while no
        turn runs."""
        event = asyncio.Event()
        self._wake_events.append(event)
        return event

    def due(self, at: AppPlace, occasion: Occasion, since: float) -> bool:
        """The place is due for ``occasion`` from ``since``, on the clock's monotonic scale. Whether it was not due
        before."""
        became_due = at not in self._turns.due
        self._turns = with_due(self._turns, at, occasion, since)
        self._change()
        self._ask_for_a_cut()
        return became_due

    def due_for_hold(self, at: AppPlace, generation: str, since: float) -> bool:
        """A hold that may be told stands at the place: it is due from ``since`` as lines of the window's generation
        with none, unless it is due already, when the turn that comes is told the hold. Whether it became due."""
        turns = due_for_hold(self._turns, at, generation, since)
        if turns is self._turns:
            return False
        self._turns = turns
        self._change()
        self._ask_for_a_cut()
        return True

    def due_again(self, at: AppPlace, occasion: Occasion, since: float) -> None:
        """The turn at ``at`` ended without an answer: the place is due for ``occasion`` again from ``since``, the end
        of the pause the turn left, unless something newer made it due meanwhile. It waits for the running turn to
        end."""
        self._turns = due_again(self._turns, at, occasion, since)
        self._change()

    def left(self, leaving: Callable[[AppPlace], bool]) -> None:
        """The places for which ``leaving`` holds left the standing: the hold decided at each is forgotten, and the
        place is no longer due."""
        for at in [each for each in self._speech if leaving(each)]:
            self._set_speech(at, hold_forgotten(self._speech[at]))
        self.undue(leaving)

    def undue(self, leaving: Callable[[AppPlace], bool]) -> None:
        """Drops the due places for which ``leaving`` holds; with no turn running a younger place may then go first."""
        remaining = without_due(self._turns, leaving)
        dropped = len(remaining.due) < len(self._turns.due)
        self._turns = remaining
        if dropped:
            self._change()
        if dropped and not remaining.running:
            self.wake()

    def claim(self, app: str, now: float) -> tuple[AppPlace, Due] | None:
        """Takes the one turn for the place claimed first when it is at ``app`` and no pause holds it. Before,
        each due place that nothing kept waiting yet names what stands in front of it at ``now``."""
        self._turns = waited(self._turns, now, self.spoken)
        taken = claimed(self._turns, app, now, self.spoken)
        if taken is None:
            return None
        self._turns, at, due = taken
        self._claim = Claim(at, due, self._present.history.taken, self.spoken(at, due))
        self._cut_wanted.clear()
        self._change()
        return at, due

    def pause_left(self, app: str, now: float) -> float:
        """The seconds a pause still holds the turn of the place claimed first, when it is at ``app``."""
        return pause_left(self._turns, app, now, self.spoken)

    def voice_at(self, app: str, *, voice: bool) -> None:
        """Records whether the turns at the call places of ``app`` are spoken, as its loop decided for its session."""
        self._voice_apps = self._voice_apps | {app} if voice else self._voice_apps - {app}

    def speaks(self, at: AppPlace, occasion: Occasion) -> bool:
        """Whether a turn at the place for ``occasion`` would be spoken."""
        return speaks_aloud(at.app in self._voice_apps, occasion, self._checkpoints.get(at))

    def spoken(self, at: AppPlace, due: Due) -> bool:
        """Whether the turn of the due place would be spoken."""
        return self.speaks(at, due.occasion)

    def _ask_for_a_cut(self) -> None:
        """A running written turn that has not given way yet is to end at its next safe point once a spoken place is
        due."""
        if self._claim is not None and yields_to_spoken(self._turns, self._claim, self.spoken):
            self._cut_wanted.set()

    def cut_wanted(self) -> asyncio.Event:
        """The event set while the running written turn is to end at its next safe point for a spoken place that is
        due; the turn's claim and its release clear it."""
        return self._cut_wanted

    def stopping(self) -> None:
        """This pod stops serving him; a SIGKILL never gets here."""
        self._stopping = True

    def cut(self) -> None:
        """The running turn was cancelled before its answer, and not for its place. While he stops, a turn that relayed
        no call and sent no sentence leaves its place due again, so the pod that takes him over answers it once."""
        if not self._stopping:
            return
        assert self._claim is not None  # noqa: S101 - a turn runs only from its claim until its release
        turns = cut_by_stop(self._turns, self._claim, self._present.history)
        if turns != self._turns:
            self._turns = turns
            self._change()

    def cut_for_spoken(self) -> None:
        """The running written turn ended at a safe point for a spoken place: its place is due again for what it was
        for, after the spoken places."""
        assert self._claim is not None  # noqa: S101 - a turn runs only from its claim until its release
        self._turns = cut_for_spoken(self._turns, self._claim)
        self._change()

    def release(self) -> None:
        """The running turn ended: every loop of his may claim now."""
        self._turns = released(self._turns)
        self._claim = None
        self._cut_wanted.clear()
        self.wake()

    def wake(self) -> None:
        """Wakes every loop of his: the end of a pause is as much a reason to claim as the end of a turn."""
        for event in self._wake_events:
            event.set()
