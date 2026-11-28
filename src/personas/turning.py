"""What one turn of an agent in an app does: the tools held, the sessions its requests run on and how the turn ends."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from agents.agent_file import TurnRequest
from agents.backend import TurnPerson, TurnStopped
from agents.member_outcomes import CallFailure
from agents.skills import Skill, SkillRefused
from agents.tools import HandTool, ToolDeclaration, ToolOutcome
from agents.turn import RejectionCode, ToolRound, Turn, TurnMessage, is_within_event_bound
from agents.turn_events import FinalEvent, ToolCallEvent
from agents.voice.reply import REPLY_RULES
from logline import StepTimer

from personas.decision import (
    LOAD_TOOLS_CALLS,
    READ_SKILL_CALLS,
    TURN_ROLE,
    AppPlace,
    act_outcome,
    failure_code,
    turn_call_bound,
)
from personas.entries import ActDone, Room, TurnTold, told_went_on, told_words
from personas.failures import leaves_of, raise_if_cancelled, task_is_being_cancelled
from personas.held_tools import Calls, HeldTools, HoldBack, ToolLoads, TurnOffer
from personas.holds import (
    APP_FAILED_TEXT,
    DECIDING_FIRST_TEXT,
    DECISION_CALLS,
    DECISION_DECLARATIONS,
    HoldDeciding,
    HoldInbox,
    TurnHold,
    Voiced,
    close_when_over,
)
from personas.present import (
    LOAD_TOOLS_DECLARATION,
    NO_SUCH_SKILL_TEXT,
    READ_SKILL_DECLARATION,
    ListedSkill,
    held_told,
    listed_skill_named,
    loaded_by,
    skill_lines,
    skill_text,
    turn_messages,
    user_message,
)
from personas.recall import (
    LOOKUPS_SPENT_TEXT,
    RECALL_DECLARATION,
    STORE_DID_NOT_ANSWER_TEXT,
    line_search,
    lookup_result,
    search_text,
    text_refused_text,
)
from personas.reporting import LOG, Event, LoopReport, TurnTally, agent_app_fields, failure_event, stepping
from personas.sizing import Sizing
from personas.speech import HoldOutcome, Telling
from personas.task_record import (
    RelayFailed,
    ending_loud_when_cancelled,
    ending_with_recorded_defect,
    failure_held_by,
    failures_leaving_with,
)
from personas.tool_lists import Listed, ToolLists
from personas.turn_bound import TurnBound
from personas.turn_sessions import TurnSessions
from personas.voice import DropReporter, Voicing

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
    from contextlib import AbstractAsyncContextManager

    from agents.agent_file import AgentDefinition
    from agents.backend import AgentBackend, TurnEvent
    from agents.mcp_client import MemberSession
    from agents.member import PlaceRef, PlaceWindow
    from agents.turn import FailureCode, JsonObject, Principal

    from personas.admission import Admitted
    from personas.app_session import SessionOpener
    from personas.clock import Clock
    from personas.config import AgentCredentials, AppTarget
    from personas.mind import Mind
    from personas.present import AppSkill
    from personas.recall import LookupBounds
    from personas.remembering import Remembering
    from personas.task_record import TaskRecord
    from personas.tokens import TokenSource
    from personas.voice import Answering

# The backend a turn runs on, which it closes when the turn ends.
type BackendFactory = Callable[[Iterable[HandTool]], AbstractAsyncContextManager[AgentBackend]]

_CALL_BOUND_TEXT: Final = "this app takes no more calls in this turn"
_APART_TEXT: Final = (
    "a hold is decided in a round of its own when one starts: nothing is held now, so there is nothing to decide"
)
_WRITTEN_TURN_TEXT: Final = "this turn is not spoken: continue_speaking and stop_speaking act only in a spoken turn"
_SKILLS_SPENT_TEXT: Final = "no more skills can be read in this turn"
_SKILL_REFUSED_TEXT: Final = "this skill cannot be read"
_SKILL_TOO_LARGE_TEXT: Final = "this skill is too large to read"


async def _final_event_of(events: AsyncIterator[TurnEvent], tally: TurnTally) -> FinalEvent | None:
    """Reads the turn's events to their end, counting its tool calls, and returns its final event, if it had one."""
    final = None
    async for event in events:
        if isinstance(event, ToolCallEvent):
            tally.tool_call_count += 1
        elif isinstance(event, FinalEvent):
            final = event
    return final


@dataclass(frozen=True)
class TurnSetup:
    """What an agent's turn in an app is made of: the persona and the backend that runs it."""

    persona: AgentDefinition
    backend: BackendFactory


class Turning:
    """Runs the turns of one agent in one app, each over the loop's session, holding the tools of every app in the
    configuration's order: the app's own, those of the apps in ``admitted.apps`` and those of the tools-only apps,
    after the lookup when ``[memory]`` is set.

    It holds no state across turns beyond what it records into the agent's mind: each relayed call that passed its
    app's bound, once its request ended, and each tools-only app the realm refuses the agent, which no turn holds
    until its retry is due. A call is held back, once per turn, when something not his own arrived in
    his history since the turn's admission: it is not sent and the model is told what arrived. The token, its use on
    a session and the two reports are the loop's, handed over as callables and read at each call. A lookup reads the
    memory through the loop's ``Remembering``, takes nothing into the agent's history and is never held back.
    """

    def __init__(
        self,
        agent: AgentCredentials,
        app: AppTarget,
        apps: tuple[AppTarget, ...],
        tokens: TokenSource,
        open_session: SessionOpener,
        mind: Mind,
        clock: Clock,
        renew_token: Callable[[], Awaitable[None]],
        report: LoopReport,
        report_pause: Callable[[float, str], None],
        report_tools_left_out: Callable[[AppTarget, int], None],
        report_dropped: DropReporter,
        remembering: Remembering,
    ) -> None:
        self._agent = agent
        self._app = app
        self._apps = tuple((each, each.max_tool_calls) for each in apps if each.max_tool_calls is not None)
        self._app_at = {each.mcp_url: each for each, _ in self._apps}
        self._sessions = TurnSessions(agent, tokens, open_session)
        self._lists = ToolLists(
            agent, app, tuple(each for each, _ in self._apps), self._sessions, mind, clock, renew_token, report
        )
        self._mind = mind
        self._clock = clock
        self._renew_token = renew_token
        self._report = report
        self._report_pause = report_pause
        self._report_tools_left_out = report_tools_left_out
        self._report_dropped = report_dropped
        self._voicing = Voicing(self._sessions, app, report, clock, mind, report_dropped)
        self._left_out_reported: set[str] = set()
        self._remembering = remembering
        self._holds = HoldDeciding(agent, app, self._sessions, mind, report, report_dropped)
        self._aloud_at: PlaceRef | None = None
        self._calls_in_flight = 0
        self._at_safe_point: Callable[[], None] | None = None

    def at_safe_point(self, act: Callable[[], None]) -> None:
        """Runs ``act``, which ends the turn, now when no call it relayed is in flight, else once the last one has
        answered or ended and entered the agent's history, before the call's result goes back to the model."""
        if self._calls_in_flight:
            self._at_safe_point = act
        else:
            act()

    async def take(
        self,
        session: MemberSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        tally: TurnTally,
        gate: Callable[[], Awaitable[None]],
    ) -> None:
        """Runs one turn once its ``gate`` is passed; whatever way it ends, a defect a request of it met ends the
        runner loud. The turn lists the tools of its apps beside the gate, since a listing is a read.

        A session whose close cuts a request short can lose the defect from what leaves, so the turn's record
        joins it. A cancelled turn ends as the cancellation unless a defect was met or nobody cancelled it.
        """
        self._aloud_at = admitted.place if admitted.aloud else None
        try:
            await ending_loud_when_cancelled(
                ending_with_recorded_defect(self._judge_turn(session, setup, admitted, record, tally, gate), record),
                record.loud_end,
            )
        finally:
            self._aloud_at = None
            self._at_safe_point = None
            if admitted.aloud and not tally.answered:
                at = self._at(admitted.place)
                self._report_dropped(self._mind.drop_reason_at_end(), self._mind.unsent_dropped(at))

    @asynccontextmanager
    async def _listing_beside(
        self, session: MemberSession, admitted: Admitted, record: TaskRecord
    ) -> AsyncGenerator[asyncio.Task[tuple[Listed, ...]]]:
        """Lists the tools and skills of the apps in a task beside the block, which awaits it where it needs them.

        The listing is a task of its own, not in a group with the gate, so a listing that fails never cuts the gate
        and its failure ends the turn after the thinking. A turn that ends before it awaited the listing cancels it,
        and a defect the listing met joins the turn's record.
        """
        listing = asyncio.create_task(self._lists.listed(session, admitted, record))
        try:
            yield listing
        finally:
            listing.cancel()
            await asyncio.wait((listing,))
            if not listing.cancelled() and isinstance(error := listing.exception(), Exception):
                record.note(error)

    def _at(self, place: PlaceRef) -> AppPlace:
        return AppPlace(self._app.mcp_url, place)

    async def _judge_turn(
        self,
        session: MemberSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        tally: TurnTally,
        gate: Callable[[], Awaitable[None]],
    ) -> None:
        """Runs one turn over the loop's session.

        A turn the gateway ends short is told to the tally unless a defect was met; a failed relayed call and any other
        error go to the group. Every error of a cancelled turn ends as the cancellation.
        """
        sizing = Sizing()
        try:
            await self._run_turn(session, setup, admitted, record, sizing, tally, gate)
        except Exception as error:
            raise_if_cancelled()
            leaves = list(leaves_of(error))
            relay_stop = next((leaf for leaf in leaves if isinstance(leaf, RelayFailed)), None)
            stopped = relay_stop or next((leaf for leaf in leaves if isinstance(leaf, TurnStopped)), None)
            if stopped is None:
                raise
            # The stop only carries the failure out of the backend; the failure goes to the group, never the stop.
            held = failure_held_by(stopped)
            if held is None and record.defect is None:
                tally.stop_code = stopped.code.value
                self._leave_unanswered(stopped.code, admitted)
            # What left the session beside the stop (a late answer, the failure another stop holds) still goes to the
            # group, to be judged there.
            rest = failures_leaving_with(held, (leaf for leaf in leaves if leaf is not stopped))
            if len(rest) == 1:
                raise rest[0] from None
            if rest:
                raise BaseExceptionGroup("the turn's session failed", rest) from None
        finally:
            tally.sizes = sizing.sizes()

    def _leave_unanswered(self, code: RejectionCode | FailureCode, admitted: Admitted) -> None:
        """A turn that ended without an answer leaves his mind a pause that holds every turn of his, so the lines that
        arrive meanwhile are taken into his history and make one turn after it, with the turn's own place, due again
        from the pause's end. The pause is the mind's, so it outlives this task."""
        now = self._clock.now()
        seconds = self._mind.unanswered(budget_spent=code is RejectionCode.BUDGET_SPENT, now=now)
        self._report_pause(seconds, code.value)
        self._mind.due_again(self._at(admitted.place), admitted.occasion, now + seconds)

    async def _run_turn(
        self,
        session: MemberSession,
        setup: TurnSetup,
        admitted: Admitted,
        record: TaskRecord,
        sizing: Sizing,
        tally: TurnTally,
        gate: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._listing_beside(session, admitted, record) as listing:
            await gate()
            with stepping(
                StepTimer("turn_setup.ended", LOG, **agent_app_fields(self._agent.client_id, self._app.mcp_url))
            ) as hand_off:
                lists = await listing
                turn = TurnTold(
                    admitted.at,
                    self._app.mcp_url,
                    admitted.place,
                    admitted.occasion,
                    aloud=admitted.aloud,
                    through=admitted.history.taken,
                )
                offered = TurnOffer.of((app for app, _ in self._apps), lists, self._tools_left_out)
                told = turn_messages(admitted.facts.name, admitted.present, turn, offered.deferred)
                if admitted.went_on:
                    told = (*told, user_message(told_went_on()))
                if admitted.held is not None:
                    told = (*told, user_message(self._holds.told_standing(admitted.place, admitted.held)))
                bound = TurnBound(self._clock, setup.persona.max_duration_seconds)
                try:
                    await self._run_session_body(
                        setup, admitted, turn, told, session, record, sizing, tally, hand_off, bound, offered
                    )
                except Exception as error:
                    record.note(error)
                    raise
                finally:
                    bound.close()

    async def _run_session_body(
        self,
        setup: TurnSetup,
        admitted: Admitted,
        turn: TurnTold,
        told: tuple[TurnMessage | ToolRound, ...],
        session: MemberSession,
        record: TaskRecord,
        sizing: Sizing,
        tally: TurnTally,
        hand_off: StepTimer,
        bound: TurnBound,
        offered: TurnOffer,
    ) -> None:
        raise_if_cancelled()
        deciding = None if admitted.held is None else TurnHold(admitted.held)
        inbox = self._inbox_of(admitted) if admitted.aloud else None
        place = admitted.place
        speaking, answering = (None, None)
        if inbox is not None:
            speaking, answering = self._voicing.speaking(
                place,
                record,
                tally,
                bound,
                inbox,
                deciding,
                partial(self._holds.hold_decided, place),
                lambda hold: self._holds.end_hold(place, record, hold, HoldOutcome.UNDECIDED, bound),
                partial(self._holds.retold, place, record),
            )
        hold = HoldBack(
            admitted.history,
            inbox,
            deciding,
            went_on_to_tell=partial(self._mind.went_on_to_tell, self._at(admitted.place)),
        )
        held = self._hold_tools(session, record, admitted, bound, offered, told, hold, answering)
        request = self._request_of(setup, admitted, told, held, offered.skills)
        sizing.request_built(held.declarations, held.deferred_names, told)
        person = TurnPerson(admitted.facts.principal, TURN_ROLE, admitted.groups)
        async with (
            setup.backend(tuple(sizing.measuring(each) for each in held.tools)) as backend,
            aclosing(backend.run(request, person)) as events,
        ):
            hand_off.end()
            cut = False
            if inbox is not None and speaking is not None and answering is not None:
                voiced = Voiced(setup, admitted, turn, request, held, speaking, answering, record, inbox, bound)
                final = await self._speak_along(voiced, events, tally)
                cut = speaking.was_cut
            else:
                final = await _final_event_of(events, tally)
        if final is not None or cut:
            tally.answered = True
            tally.usage = None if final is None else final.usage
            self._mind.answered(turn)

    def _request_of(
        self,
        setup: TurnSetup,
        admitted: Admitted,
        told: tuple[TurnMessage | ToolRound, ...],
        held: HeldTools,
        skills: Mapping[str, ListedSkill],
    ) -> TurnRequest:
        front = [setup.persona.prompt, REPLY_RULES, *filter(None, [skill_lines(skills.values())])]
        return TurnRequest(
            agent=setup.persona.name,
            turn=Turn(system="\n\n".join(front), messages=list(told)),
            roles=setup.persona.roles,
            tools=held.declarations,
            deferred_tools=held.deferred_names,
            model=(self._agent.hold_model if admitted.held else None) or setup.persona.model,
            max_duration_seconds=setup.persona.max_duration_seconds,
            max_tool_calls=held.max_tool_calls,
            one_call_per_reply=admitted.aloud,
        )

    async def _speak_along(
        self, voiced: Voiced, events: AsyncIterator[TurnEvent], tally: TurnTally
    ) -> FinalEvent | None:
        """Reads the turn's events while each hold of his is decided beside them; the sentences they release go to the
        mind's record, whose place's sender sends them after the turn too.

        A call is relayed when the events are asked for again, at once unless a hold stands, whatever of his plays. The
        turn ends once its events have. The final event, if the turn had one, is returned; his stop ends the turn
        without one.
        """
        with self._holds.listening(voiced.admitted.place, voiced.inbox):
            async with asyncio.TaskGroup() as group:
                hearing = group.create_task(self._hear_events(events, voiced, tally))
                group.create_task(self._holds.decide_holds(voiced, hearing))
                group.create_task(close_when_over(voiced.inbox, hearing))
        return None if hearing.cancelled() else hearing.result()

    async def _hear_events(
        self, events: AsyncIterator[TurnEvent], voiced: Voiced, tally: TurnTally
    ) -> FinalEvent | None:
        speaking = voiced.speaking
        held = voiced.held
        final = None
        async for event in events:
            if answer := speaking.hear(event):
                await voiced.answering.instead_told(answer)
            if isinstance(event, ToolCallEvent):
                tally.tool_call_count += 1
                voiced.inbox.waiting = event
                if not held.deciding_first:
                    await voiced.inbox.until_none_stands(voiced.bound)
                voiced.inbox.waiting = None
            elif isinstance(event, FinalEvent):
                final = event
        if held.deciding_first:
            speaking.cut()
        return final

    async def hold(self, session: MemberSession, place: PlaceRef, window: PlaceWindow) -> bool:
        """What a read of the place's window says of him. The hold it states stands in the mind's record of the place.
        The spoken turn at the place hears it; with none there, a hold that may be told makes the place due, and a hold
        whose turn ended without deciding it is stopped with its id, the loop's session taking the call, and the unsent
        sentences at the place drop. Whether the hold made the place due."""
        at = self._at(place)
        held = self._mind.hold_read(at, window, self._clock.now())
        if self._holds.heard_at(place, window) or held is None or self._aloud_at == place:
            return False
        if held.telling is Telling.UNDECIDED:
            await self._holds.stop_undecided(session, place, held)
        elif self._mind.hold_makes_due(at):
            return self._mind.due_for_hold(at, window.generation, self._clock.now())
        return False

    def _hold_tools(
        self,
        session: MemberSession,
        record: TaskRecord,
        admitted: Admitted,
        bound: TurnBound,
        offered: TurnOffer,
        told: tuple[TurnMessage | ToolRound, ...],
        hold: HoldBack,
        answering: Answering | None,
    ) -> HeldTools:
        """The lookup when the configuration has a memory, the skills' ``read_skill`` and ``load_tools``, then the tools
        the apps offer, in the configuration's order, each app counting its own calls; those ``offered`` defers are
        sent without their definition until a load of the turn or a round ``told`` loads them.

        Another app whose listing fails, and the place's own app when it does not answer within its call time, is left
        out and reported, and the turn holds the rest.
        """
        deciding = hold.told
        skills, deferred = offered.skills, offered.deferred
        tools: list[HandTool] = []
        bounds: list[int] = []
        lookup = self._remembering.lookup
        if lookup is not None:
            tools.append(HandTool(RECALL_DECLARATION, partial(self._look, lookup, Calls(lookup.calls), admitted, hold)))
            bounds.append(lookup.calls)
        app_calls = {app.mcp_url: Calls(max_calls) for app, max_calls in self._apps}
        at = self._at(admitted.place)
        reading = partial(self._read_skill, skills, app_calls, session, Calls(READ_SKILL_CALLS), hold, record)
        loads = ToolLoads(skills, deferred, set(loaded_by(told)), Calls(LOAD_TOOLS_CALLS), hold, self._enter_load)
        tools.extend((HandTool(READ_SKILL_DECLARATION, reading), HandTool(LOAD_TOOLS_DECLARATION, loads.load)))
        bounds.extend((READ_SKILL_CALLS, LOAD_TOOLS_CALLS))
        for offer in offered.offers:
            if offer.declarations:
                calls = app_calls[offer.app]
                app = self._app_at[offer.app]
                tools.extend(self._hand_tools(app, session, offer.declarations, calls, hold, record, at))
                bounds.append(calls.bound)
        tools.extend(self._decision_tools(admitted, record, bound, deciding, answering))
        bounds.append(DECISION_CALLS)
        return HeldTools(tuple(tools), turn_call_bound(bounds), deferred, hold.inbox, deciding)

    def _decision_tools(
        self,
        admitted: Admitted,
        record: TaskRecord,
        bound: TurnBound,
        deciding: TurnHold | None,
        answering: Answering | None,
    ) -> list[HandTool]:
        """The decision tools: decided at the hold when one stands, else answered apart."""
        if deciding is not None:
            return [
                HandTool(
                    each,
                    partial(self._holds.decide_told, admitted.place, record, deciding, answering, bound, each.name),
                )
                for each in DECISION_DECLARATIONS
            ]
        apart = partial(self._answered_apart, _APART_TEXT if admitted.aloud else _WRITTEN_TURN_TEXT)
        return [HandTool(each, apart) for each in DECISION_DECLARATIONS]

    def _enter_load(self, app: str, skill: str, tools: tuple[str, ...]) -> None:
        self._mind.tools_loaded(app, skill, tools, self._clock.wall())

    def _inbox_of(self, admitted: Admitted) -> HoldInbox:
        """What the spoken turn hears of a hold: the one that stands in the record, told at the turn's start or not."""
        if admitted.held is not None:
            return HoldInbox(admitted.held, told=True)
        return HoldInbox(self._mind.hold_standing(self._at(admitted.place)))

    def _hand_tools(
        self,
        app: AppTarget,
        session: MemberSession,
        offered: tuple[ToolDeclaration, ...],
        calls: Calls,
        hold: HoldBack,
        record: TaskRecord,
        at: AppPlace,
    ) -> list[HandTool]:
        """The tools of ``app`` the turn at ``at`` holds; each call is kept with the room of that place."""
        if app.mcp_url == self._app.mcp_url:
            return [HandTool(each, partial(self._relay, session, each.name, calls, hold, at)) for each in offered]
        return [
            HandTool(each, partial(self._relay_elsewhere, app, each.name, calls, hold, record, at)) for each in offered
        ]

    def _tools_left_out(self, app: AppTarget, count: int) -> None:
        """Reports an app's names left out for an earlier app's, once until the loop's session ends."""
        if count and app.mcp_url not in self._left_out_reported:
            self._left_out_reported.add(app.mcp_url)
            self._report_tools_left_out(app, count)

    def new_session(self) -> None:
        """The loop's app session starts: a name left out is reported again."""
        self._left_out_reported.clear()

    async def _look(
        self,
        lookup: LookupBounds,
        calls: Calls,
        admitted: Admitted,
        hold: HoldBack,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        """Looks up what the agent's own place held before its history: nothing enters the history, and a lookup is
        never held back by what arrived, only by a hold he has not decided."""
        if hold.deciding_first:
            return ToolOutcome(DECIDING_FIRST_TEXT, is_error=True)
        if not calls.take():
            return ToolOutcome(LOOKUPS_SPENT_TEXT, is_error=True)
        text = search_text(arguments, lookup.result_bytes)
        if text is None:
            return ToolOutcome(text_refused_text(lookup.result_bytes), is_error=True)
        search = line_search(
            self._agent.client_id, self._app.mcp_url, admitted.place, admitted.history, text, lookup.result_bytes
        )
        try:
            found = await self._remembering.search(search)
        except Exception as error:
            raise RelayFailed(error) from None
        if found is None:
            return ToolOutcome(STORE_DID_NOT_ANSWER_TEXT, is_error=True)
        return ToolOutcome(lookup_result(found, lookup.result_bytes), is_error=False)

    async def _read_skill(
        self,
        skills: Mapping[str, ListedSkill],
        app_calls: Mapping[str, Calls],
        session: MemberSession,
        calls: Calls,
        hold: HoldBack,
        record: TaskRecord,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        """Reads a skill this turn listed and enters it into his history, so the next turn tells it as a round. It is
        never held back by what arrived, only by a hold he has not decided; its result tells that he went on as a call's
        does. A skill an app refuses, a name no line holds and a body past a result's bound answer an error and enter
        nothing; a read that fails is a failed call to its app."""
        if hold.deciding_first:
            return ToolOutcome(DECIDING_FIRST_TEXT, is_error=True)
        if not calls.take():
            return ToolOutcome(_SKILLS_SPENT_TEXT, is_error=True)
        listed = listed_skill_named(skills, arguments)
        if listed is None:
            return ToolOutcome(NO_SUCH_SKILL_TEXT, is_error=True)
        app = self._app_at[listed.app]
        read = await self._skill_read(app, listed.skill, app_calls[listed.app], session, hold, record)
        return hold.telling_went_on(read if isinstance(read, ToolOutcome) else self._entered(app, read))

    def _entered(self, app: AppTarget, read: Skill) -> ToolOutcome:
        text = skill_text(read.body)
        if not is_within_event_bound(text, None):
            return ToolOutcome(_SKILL_TOO_LARGE_TEXT, is_error=True)
        self._mind.skill_read(app.mcp_url, read.name, read.identity.uri, read.body, self._clock.wall())
        return ToolOutcome(text, is_error=False)

    async def _skill_read(
        self,
        app: AppTarget,
        skill: AppSkill,
        app_calls: Calls,
        session: MemberSession,
        hold: HoldBack,
        record: TaskRecord,
    ) -> Skill | ToolOutcome:
        """The skill, read from its app unless it is the app's implicit skill, whose body its listing kept; or the
        result of a read that failed or was refused."""
        if isinstance(skill, Skill):
            return skill
        if app.mcp_url == self._app.mcp_url:
            answer = await _sent(hold, self._got_renewed(session, skill.uri))
        else:
            answer = await self._asked_elsewhere(
                app,
                app_calls,
                hold,
                partial(self._sessions.ask_elsewhere, app, partial(_got_skill, uri=skill.uri), record),
            )
            if isinstance(answer, ToolOutcome):
                return answer
        if isinstance(answer, SkillRefused):
            self._report(Event.SKILL_REFUSED, answer.refusal.value, app)
            return ToolOutcome(_SKILL_REFUSED_TEXT, is_error=True)
        return answer

    async def _answered_apart(  # NOSONAR(S7503) HandToolHandler is async
        self, text: str, _principal: Principal, _arguments: JsonObject
    ) -> ToolOutcome:
        return ToolOutcome(text, is_error=True)

    def _held(self, hold: HoldBack) -> ToolOutcome | None:
        """The result of a call that is not sent, or ``None`` when the call goes on."""
        if hold.deciding_first:
            return ToolOutcome(DECIDING_FIRST_TEXT, is_error=True)
        arrived = hold.holding(self._mind.history)
        if arrived is not None:
            return ToolOutcome(held_told(arrived), is_error=False)
        if words := hold.said_over_him():
            return ToolOutcome(told_words(words), is_error=False)
        return None

    async def _relay(
        self,
        session: MemberSession,
        name: str,
        calls: Calls,
        hold: HoldBack,
        at: AppPlace,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        if (held := self._held(hold)) is not None:
            return held
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        sent = await _sent(
            hold, self._recorded_as_act(self._app, at, name, arguments, self._invoke(session, name, arguments))
        )
        return hold.telling_went_on(sent)

    async def _invoke(self, session: MemberSession, name: str, arguments: JsonObject) -> ToolOutcome:
        await self._renew_token()
        return await session.invoke(name, arguments)

    async def _got_renewed(self, session: MemberSession, uri: str) -> Skill | SkillRefused:
        await self._renew_token()
        return await _got_skill(session, uri)

    async def _relay_elsewhere(
        self,
        app: AppTarget,
        name: str,
        calls: Calls,
        hold: HoldBack,
        record: TaskRecord,
        at: AppPlace,
        _principal: Principal,
        arguments: JsonObject,
    ) -> ToolOutcome:
        if (held := self._held(hold)) is not None:
            return held
        if not calls.take():
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        answer = await self._asked_elsewhere(
            app,
            calls,
            hold,
            lambda: self._recorded_as_act(
                app,
                at,
                name,
                arguments,
                self._sessions.ask_elsewhere(app, lambda session: session.invoke(name, arguments), record),
            ),
        )
        return hold.telling_went_on(answer)

    async def _asked_elsewhere[Answer](
        self, app: AppTarget, calls: Calls, hold: HoldBack, asking: Callable[[], Awaitable[Answer | CallFailure]]
    ) -> Answer | ToolOutcome:
        """What a call to another app answers, or the error result of a call it refuses or that failed. An app whose
        calls are closed is sent nothing; a failed call closes them for the rest of the turn."""
        if calls.closed:
            return ToolOutcome(_CALL_BOUND_TEXT, is_error=True)
        answer = await _sent(hold, asking())
        if isinstance(answer, CallFailure):
            calls.close()
            self._report(failure_event(app), failure_code(answer), app, answer)
            return ToolOutcome(APP_FAILED_TEXT, is_error=True)
        return answer

    async def _recorded_as_act[Answer: ToolOutcome | CallFailure](
        self, app: AppTarget, at: AppPlace, name: str, arguments: JsonObject, request: Awaitable[Answer]
    ) -> Answer:
        """Awaits the request of a relayed call and records it as an act of the agent however it ends, made in the room
        of the turn's place ``at``.

        Whether the turn is being cancelled is asked once the request and every scope inside it have exited.
        """
        ended: ToolOutcome | BaseException | None = None
        self._calls_in_flight += 1
        try:
            answer = await request
            ended = answer
            return answer
        except Exception as error:
            ended = error
            raise
        except asyncio.CancelledError as cancelled:
            ended = cancelled
            raise
        finally:
            outcome = None if ended is None else act_outcome(ended, cancelling=task_is_being_cancelled())
            if outcome is not None:
                room = Room(at.app, at.place, self._mind.generation_at(at))
                self._mind.take((ActDone(app.mcp_url, name, arguments, outcome, self._clock.wall(), room),))
            self._calls_in_flight -= 1
            if not self._calls_in_flight and (act := self._at_safe_point) is not None:
                self._at_safe_point = None
                act()
                if not isinstance(ended, BaseException):
                    # The cancellation reaches the turn at its next suspension, which can come after the model's next
                    # request went out; the answer of the call does not go back to the model.
                    raise_if_cancelled()


async def _sent[Answer](hold: HoldBack, sending: Awaitable[Answer]) -> Answer:
    """What a call on its way to its app answers while the hold knows it is sent; a failure of it ends the turn as the
    loop's own."""
    try:
        with hold.sending():
            return await sending
    except Exception as error:
        raise RelayFailed(error) from None


async def _got_skill(session: MemberSession, uri: str) -> Skill | SkillRefused:
    """The skill at ``uri``, or the refusal of a file that is not what its entry promised, which is an answer, not a
    defect."""
    try:
        return await session.get_skill(uri)
    except SkillRefused as refused:
        return refused
