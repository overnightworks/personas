"""What a fake app's voice does for its members: ``speak`` and the sentences it plays, and the hold of a member with
what people say over him."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.member import (
    CONTINUE_SPEAKING_TOOL,
    SPEAK_TOOL,
    STOP_SPEAKING_TOOL,
    ContinueSpeakingArguments,
    ContinueSpeakingRefusal,
    ContinueSpeakingRefused,
    DroppedSpeaking,
    HoldPlace,
    PartialSpeech,
    SpeakArguments,
    SpeakRefusal,
    SpeakRefused,
    Spoken,
    StoppedSpeaking,
    StopSpeakingArguments,
    StopSpeakingRefusal,
    StopSpeakingRefused,
    VoiceState,
    Written,
    has_words,
)
from agents.member_speaking import words_of
from agents.tools import ToolDeclaration, ToolEffect, ToolOutcome

from tests.support.hopin_places import ACCEPTED_NAME, AppPlaces

if TYPE_CHECKING:
    from agents.mcp_server import PersonRequest
    from agents.turn import JsonObject

SPEAK_DECLARATION = ToolDeclaration(
    name=SPEAK_TOOL,
    description="Speaks a sentence in a call.",
    parameters=SpeakArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)

CONTINUE_DECLARATION = ToolDeclaration(
    name=CONTINUE_SPEAKING_TOOL,
    description="Lets a held sentence go on.",
    parameters=ContinueSpeakingArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)

STOP_DECLARATION = ToolDeclaration(
    name=STOP_SPEAKING_TOOL,
    description="Cuts the member's sentence and queue, or drops a held one.",
    parameters=StopSpeakingArguments.model_json_schema(),
    effect=ToolEffect.SIDE_EFFECTING,
)


@dataclass
class AppVoice(AppPlaces):
    """The sentences the app takes through ``speak`` and plays, and each hold of a member with what is said over it."""

    speak_outcomes: list[ToolOutcome | None] = field(default_factory=list[ToolOutcome | None])
    """The answer to each next ``speak`` as it arrives, in order; ``None`` lets the app play that sentence."""
    spoken: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    spoken_added: asyncio.Event = field(default_factory=asyncio.Event)
    sentence_ends_held: bool = False
    """Whether a voiced sentence plays until ``end_sentence`` or ``stop_speaking`` is called: its own line enters then,
    and a ``speak`` that arrives meanwhile is answered only then, as the app answers it at its sentence's turn."""
    waiting_speaks: list[asyncio.Future[bool]] = field(default_factory=list[asyncio.Future[bool]])
    """The ``speak`` calls waiting for the playing sentence to end, in the order sent; a result says if it plays."""
    playing: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    """The place, the author and the text of each voiced sentence whose own line is held, in the order sent."""
    written_speaks: int = 0
    """How many of the next ``speak`` calls the app answers as written to the place instead of voiced."""
    voice_states: dict[str, VoiceState] = field(default_factory=dict[str, VoiceState])
    held_speak: asyncio.Event | None = None
    declares_hold_tools: bool = True
    """Whether the app declares ``continue_speaking`` and ``stop_speaking``, as one that can hold the member does."""
    holds: dict[str, tuple[str, tuple[PartialSpeech, ...] | None]] = field(
        default_factory=dict[str, tuple[str, tuple[PartialSpeech, ...] | None]]
    )
    """Each place whose member is held: the hold's id and what is said over it, none once the record is gone."""
    wordless: set[str] = field(default_factory=set[str])
    """Each place whose hold stands with no word of its talk-over heard yet."""
    went_on: dict[str, str] = field(default_factory=dict[str, str])
    """Each place whose latest hold went on by itself: that hold's id."""
    hold_places: dict[str, HoldPlace] = field(default_factory=dict[str, HoldPlace])
    """Each place whose app states, with every hold, where it stopped the member; the others state none."""
    talked_over: dict[str, tuple[PartialSpeech, ...]] = field(default_factory=dict[str, tuple[PartialSpeech, ...]])
    """Each place where people still talk over a member whose hold a continue ended: what each has said so far."""
    hold_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """The tool and the hold id of each ``continue_speaking`` and ``stop_speaking`` that arrived."""
    speaks_naming_holds: list[tuple[str, str | None]] = field(default_factory=list[tuple[str, str | None]])
    """The text and the hold id (none when it names none) of each ``speak`` that arrived."""
    speak_styles: list[str | None] = field(default_factory=list[str | None])
    """The style of each ``speak``, in the order of ``speaks_naming_holds``."""
    hold_outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    """The answer to each next ``continue_speaking`` or ``stop_speaking`` as it arrives, in order."""
    worded: dict[str, int] = field(default_factory=dict[str, int])
    """Each place whose hold states how many times the words of its talk-over changed; a decision told fewer is refused.
    The app counts only once a test turns it on with ``count_worded_changes``."""
    counts_worded_changes: bool = False
    hold_told: list[int | None] = field(default_factory=list[int | None])
    """The ``told_changes`` of each decision that arrived (a continue, a stop or a ``speak`` naming a hold)."""
    words_before_decision: list[tuple[tuple[str, str], ...]] = field(default_factory=list[tuple[tuple[str, str], ...]])
    """What people say over the held member before each next decision is judged, in order, as he decides."""
    new_holds_before_decision: list[tuple[str, tuple[str, str]]] = field(
        default_factory=list[tuple[str, tuple[str, str]]]
    )
    """A new hold (its id and what a person says over it) that stands before each next decision is judged, in order."""
    choices_wait: bool = False
    """Whether a decision that plays waits while people still talk over him, which ``talk_over_ends`` ends."""
    choices_waited: int = 0
    """How many decisions have waited for the talk-over to end."""
    choice_waited: asyncio.Event = field(default_factory=asyncio.Event)
    waiting_choices: dict[str, asyncio.Future[bool]] = field(default_factory=dict[str, asyncio.Future[bool]])
    """The decision that waits at each place; its result says whether it plays or was voided."""

    @staticmethod
    def _partial_speeches(said: tuple[tuple[str, str], ...]) -> tuple[PartialSpeech, ...]:
        return tuple(PartialSpeech(name=name, text=text) for name, text in said)

    def hold(self, place_id: str, hold_id: str, *said: tuple[str, str]) -> None:
        """The member is held at the place: the window states the hold and, over it, what each person has said so far
        (a name and its words), and signals the change."""
        self._count_changes(place_id, hold_id, said)
        self.holds[place_id] = (hold_id, self._partial_speeches(said))
        if any(has_words(text) for _, text in said):
            self.wordless.discard(place_id)
        self._signal_window(place_id)

    def count_worded_changes(self) -> None:
        """The app states ``worded_changes`` on its holds and refuses a decision told fewer than it counted."""
        self.counts_worded_changes = True

    def _count_changes(self, place_id: str, hold_id: str, said: tuple[tuple[str, str], ...]) -> None:
        """A hold counts each person whose words differ from the ones the same hold had; a new hold counts afresh."""
        if not self.counts_worded_changes:
            return
        stands = self.holds.get(place_id, ("", None))
        before = {each.name: words_of(each.text) for each in (stands[1] or ()) if stands[0] == hold_id}
        changed = sum(1 for name, text in said if words_of(text) and words_of(text) != before.get(name))
        kept = self.worded.get(place_id, 0) if stands[0] == hold_id else 0
        if kept + changed:
            self.worded[place_id] = kept + changed
        else:
            self.worded.pop(place_id, None)
        if stands[0] != hold_id:
            self._settle(place_id, plays=False)

    def hold_wordless(self, place_id: str, hold_id: str, *said: tuple[str, str]) -> None:
        """The member is held and no word has been heard of the talk-over: the app ends the hold by itself."""
        self.wordless.add(place_id)
        self.hold(place_id, hold_id, *said)

    def go_on(self, place_id: str) -> None:
        """The app ends a wordless hold by itself: it states the hold's id as the one that went on, and a sentence
        waiting for its turn plays after the playing one."""
        hold_id, _ = self.holds.pop(place_id)
        self.wordless.discard(place_id)
        self.went_on[place_id] = hold_id
        self._signal_window(place_id)

    def talk_over(self, place_id: str, *said: tuple[str, str]) -> None:
        """What is said over the held member changes, under the same hold."""
        if place_id in self.talked_over:
            self.talked_over[place_id] = self._partial_speeches(said)
            self._signal_window(place_id)
            return
        hold_id, _ = self.holds[place_id]
        self.hold(place_id, hold_id, *said)

    def talk_over_ends(self, place_id: str) -> None:
        """Nobody of the talk-over speaks any more: the decision that waits at the place plays."""
        self._settle(place_id, plays=True)

    def talk_over_unheld(self, place_id: str, *said: tuple[str, str]) -> None:
        """People talk over the member and the window states what they say, with no hold yet."""
        self.talked_over[place_id] = self._partial_speeches(said)
        self._signal_window(place_id)

    def end_record(self, place_id: str) -> None:
        """The record of what was said over the held member is gone while the hold stands."""
        hold_id, _ = self.holds[place_id]
        self.holds[place_id] = (hold_id, None)
        self._signal_window(place_id)

    def state_voice(self, place_id: str, state: VoiceState | None) -> None:
        """Sets what every window of the place states of his voice, none when ``state`` is ``None``, and signals it."""
        if state is None:
            self.voice_states.pop(place_id, None)
        else:
            self.voice_states[place_id] = state
        self._signal_window(place_id)

    def hold_next_speak(self) -> asyncio.Event:
        """The next ``speak`` that arrives is answered once the event is set."""
        self.held_speak = asyncio.Event()
        return self.held_speak

    async def _speak(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Takes a sentence, appends it as the member's own line and answers that it starts or that it was written; a
        held member's sentence that names no hold is refused ``held`` and changes nothing. A sentence that names the
        hold ends it, cuts the held line and answers the queued sentences stopped, and one that names a hold that does
        not stand is refused ``not_interrupted``."""
        taken = SpeakArguments.model_validate(arguments)
        self.speaks_naming_holds.append((taken.text, taken.interrupted))
        self.speak_styles.append(taken.style)
        if (refusal := self._refusal_before_playing(taken)) is not None:
            return ToolOutcome(SpeakRefused(refused=refusal).model_dump_json(), is_error=True)
        self.content.add(taken.text)
        if (held := self.held_speak) is not None:
            self.held_speak = None
            await held.wait()
        if self.speak_outcomes and (scripted := self.speak_outcomes.pop(0)) is not None:
            return scripted
        if taken.interrupted is not None:
            self._ends_hold(taken.place.id, taken.interrupted, keep_record=True)
            if self.playing:
                self.stop_speaking()
        if self.sentence_ends_held and self.playing:
            turn = asyncio.get_running_loop().create_future()
            self.waiting_speaks.append(turn)
            if not await turn:
                return ToolOutcome(SpeakRefused(refused=SpeakRefusal.STOPPED).model_dump_json(), is_error=True)
        self.spoken.append((request.principal.sub, taken.place.id, taken.text))
        self.spoken_added.set()
        if self.written_speaks:
            self.written_speaks -= 1
            self.speak(taken.place.id, request.principal.sub, ACCEPTED_NAME, taken.text)
            return ToolOutcome(Written(written=True).model_dump_json(), is_error=False)
        if self.sentence_ends_held:
            self.playing.append((taken.place.id, request.principal.sub, taken.text))
        else:
            self.speak(taken.place.id, request.principal.sub, ACCEPTED_NAME, taken.text, spoken=True)
        return ToolOutcome(Spoken(spoken=True).model_dump_json(), is_error=False)

    def _refusal_before_playing(self, taken: SpeakArguments) -> SpeakRefusal | None:
        """Why a sentence is refused before it plays: it names no hold while one holds the member, the hold it names
        does not stand, or it was told fewer changes than the hold counted."""
        if taken.interrupted is None:
            return SpeakRefusal.HELD if taken.place.id in self.holds else None
        self._decision_arrives(taken.place.id, taken.told_changes)
        if self.holds.get(taken.place.id, ("", None))[0] != taken.interrupted:
            return SpeakRefusal.NOT_INTERRUPTED
        return SpeakRefusal.STALE if self._stale(taken.place.id, taken.told_changes) else None

    def _settle(self, place_id: str, *, plays: bool) -> None:
        """The decision that waits at the place plays, or is voided, once."""
        waiting = self.waiting_choices.pop(place_id, None)
        if waiting is not None and not waiting.done():
            waiting.set_result(plays)

    def _decision_arrives(self, place_id: str, told_changes: int | None) -> None:
        """A decision reaches the app: what was said while he decided arrives first."""
        self.hold_told.append(told_changes)
        if self.new_holds_before_decision:
            hold_id, said = self.new_holds_before_decision.pop(0)
            self.hold(place_id, hold_id, said)
        if self.words_before_decision:
            self.talk_over(place_id, *self.words_before_decision.pop(0))

    def _stale(self, place_id: str, told_changes: int | None) -> bool:
        """Whether the decision was told fewer changes than the hold counted."""
        return told_changes is not None and told_changes < self.worded.get(place_id, 0)

    async def _continue_speaking(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Lets the held sentence play to its end, so that the sentence waiting for its turn is answered. A continue
        told too few changes is refused stale; one that plays waits while people talk over him, until they end, and
        answers not interrupted when a new hold, a stop or a later continue voided it."""
        taken = ContinueSpeakingArguments.model_validate(arguments)
        self.hold_calls.append((CONTINUE_SPEAKING_TOOL, taken.interrupted))
        self._decision_arrives(taken.place.id, taken.told_changes)
        if self.hold_outcomes:
            return self.hold_outcomes.pop(0)
        not_interrupted = ContinueSpeakingRefused(refused=ContinueSpeakingRefusal.NOT_INTERRUPTED)
        if self.holds.get(taken.place.id, ("", None))[0] != taken.interrupted:
            return ToolOutcome(not_interrupted.model_dump_json(), is_error=True)
        if self._stale(taken.place.id, taken.told_changes):
            stale = ContinueSpeakingRefused(refused=ContinueSpeakingRefusal.STALE)
            return ToolOutcome(stale.model_dump_json(), is_error=True)
        if self.choices_wait and not await self._plays_once_the_talk_over_ends(taken.place.id):
            return ToolOutcome(not_interrupted.model_dump_json(), is_error=True)
        self._ends_hold(taken.place.id, taken.interrupted, keep_record=True)
        if self.playing:
            self.end_sentence()
        return ToolOutcome(json.dumps({"continued": True}), is_error=False)

    async def _plays_once_the_talk_over_ends(self, place_id: str) -> bool:
        """Whether the decision plays: it replaces the one that waits, which is voided, and waits itself. A call the
        runner gave up leaves it standing."""
        self._settle(place_id, plays=False)
        waiting = asyncio.get_running_loop().create_future()
        self.waiting_choices[place_id] = waiting
        self.choices_waited += 1
        self.choice_waited.set()
        return await asyncio.shield(waiting)

    async def _stop_speaking(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Cuts the held sentence and drops every sentence waiting for its turn; a stop told too few changes is refused
        stale, and a stop voids the decision that waits."""
        taken = StopSpeakingArguments.model_validate(arguments)
        self.hold_calls.append((STOP_SPEAKING_TOOL, taken.interrupted or ""))
        self._decision_arrives(taken.place.id, taken.told_changes)
        if self.hold_outcomes:
            return self.hold_outcomes.pop(0)
        if taken.interrupted is not None and self.holds.get(taken.place.id, ("", None))[0] != taken.interrupted:
            refusal = StopSpeakingRefused(refused=StopSpeakingRefusal.NOT_INTERRUPTED)
            return ToolOutcome(refusal.model_dump_json(), is_error=True)
        if self._stale(taken.place.id, taken.told_changes):
            refusal = StopSpeakingRefused(refused=StopSpeakingRefusal.STALE)
            return ToolOutcome(refusal.model_dump_json(), is_error=True)
        self._settle(taken.place.id, plays=False)
        if taken.interrupted is not None:
            self._ends_hold(taken.place.id, taken.interrupted)
        if not self.playing:
            return ToolOutcome(DroppedSpeaking(dropped=True).model_dump_json(), is_error=False)
        cut = self.playing[0][2]
        self.stop_speaking()
        return ToolOutcome(StoppedSpeaking(cut=cut).model_dump_json(), is_error=False)

    def _ends_hold(self, place_id: str, hold_id: str, *, keep_record: bool = False) -> bool:
        """Whether the hold stands at the place; it ends, and the window says so. A continue keeps the record: people
        go on talking over him without holding him again."""
        if self.holds.get(place_id, ("", None))[0] != hold_id:
            return False
        said = self.holds.pop(place_id)[1]
        self.talked_over.pop(place_id, None)
        if keep_record and said:
            self.talked_over[place_id] = said
        self._signal_window(place_id)
        return True

    def end_sentence(self, *, cut: bool = False) -> None:
        """The playing sentence ends: its own line enters its place, whole or cut, and the next ``speak`` waiting for
        its turn is answered."""
        place_id, author, text = self.playing.pop(0)
        self.speak(place_id, author, ACCEPTED_NAME, text, spoken=True, cut=cut)
        if self.waiting_speaks:
            self.waiting_speaks.pop(0).set_result(True)

    def stop_speaking(self) -> None:
        """The member's own ``stop_speaking``: the playing sentence is cut, its own line entering cut, and every
        ``speak`` waiting for its turn is answered stopped."""
        place_id, author, text = self.playing.pop(0)
        self.speak(place_id, author, ACCEPTED_NAME, text, spoken=True, cut=True)
        for waiting in self.waiting_speaks:
            waiting.set_result(False)
        self.waiting_speaks.clear()
