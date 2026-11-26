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
)
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

    @staticmethod
    def _partial_speeches(said: tuple[tuple[str, str], ...]) -> tuple[PartialSpeech, ...]:
        return tuple(PartialSpeech(name=name, text=text) for name, text in said)

    def hold(self, place_id: str, hold_id: str, *said: tuple[str, str]) -> None:
        """The member is held at the place: the window states the hold and, over it, what each person has said so far
        (a name and its words), and signals the change."""
        self.holds[place_id] = (hold_id, self._partial_speeches(said))
        self._signal_window(place_id)

    def talk_over(self, place_id: str, *said: tuple[str, str]) -> None:
        """What is said over the held member changes, under the same hold."""
        if place_id in self.talked_over:
            self.talked_over[place_id] = self._partial_speeches(said)
            self._signal_window(place_id)
            return
        hold_id, _ = self.holds[place_id]
        self.hold(place_id, hold_id, *said)

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
        if taken.interrupted is None and taken.place.id in self.holds:
            return ToolOutcome(SpeakRefused(refused=SpeakRefusal.HELD).model_dump_json(), is_error=True)
        if taken.interrupted is not None and self.holds.get(taken.place.id, ("", None))[0] != taken.interrupted:
            return ToolOutcome(SpeakRefused(refused=SpeakRefusal.NOT_INTERRUPTED).model_dump_json(), is_error=True)
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

    async def _continue_speaking(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Lets the held sentence play to its end, so that the sentence waiting for its turn is answered."""
        taken = ContinueSpeakingArguments.model_validate(arguments)
        self.hold_calls.append((CONTINUE_SPEAKING_TOOL, taken.interrupted))
        if self.hold_outcomes:
            return self.hold_outcomes.pop(0)
        if not self._ends_hold(taken.place.id, taken.interrupted, keep_record=True):
            refusal = ContinueSpeakingRefused(refused=ContinueSpeakingRefusal.NOT_INTERRUPTED)
            return ToolOutcome(refusal.model_dump_json(), is_error=True)
        if self.playing:
            self.end_sentence()
        return ToolOutcome(json.dumps({"continued": True}), is_error=False)

    async def _stop_speaking(self, request: PersonRequest, arguments: JsonObject) -> ToolOutcome:
        """Cuts the held sentence and drops every sentence waiting for its turn."""
        taken = StopSpeakingArguments.model_validate(arguments)
        self.hold_calls.append((STOP_SPEAKING_TOOL, taken.interrupted or ""))
        if self.hold_outcomes:
            return self.hold_outcomes.pop(0)
        if taken.interrupted is not None and not self._ends_hold(taken.place.id, taken.interrupted):
            refusal = StopSpeakingRefused(refused=StopSpeakingRefusal.NOT_INTERRUPTED)
            return ToolOutcome(refusal.model_dump_json(), is_error=True)
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
