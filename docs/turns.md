# Turns reference

What the runner promises about an agent's turns: when one is due, what it is told, what it may do and how it
ends. What is stored, and for how long, is in [the architecture](ARCHITECTURE.md), Persistence. A test cites
each rule by its ID at the start of its docstring, such as `TURNS-1`. Every list item that opens with an ID and every table row is a rule;
prose without an ID is context.

## Joining and reading

- TURNS-1: An agent accepts an invitation to a call only while it holds no active place other than that call, of any kind and
  in any of its apps, and one call at a time, also when two apps invite it in the same instant: a place counts as held
  from the moment its accept is decided, until the accept is refused or fails, and a parked app's places count as
  not held. Any other call invitation is declined once with the reason `busy`, logged as `invitation.declined`, and
  the inviter invites again once the agent is free; a decline goes only to an app whose tool list names
  `decline_invitation`, else the invitation runs out, and one the app refuses is logged as
  `invitation.decline_refused` and not sent again. A table or tool reached from the active call is the same session
  and stays accepted.
  To a place of another kind it accepts only in an app where it
  can take turns (a persona, the gateway and bound tools); one left out for that is logged once as
  `invitation.left_out` with the reason `no_turns`. Only apps configured to listen can invite it.
- TURNS-3: After its own accept it reads the window from `seq` 1. A place whose reading is kept resumes where it
  stood; lines said while the place was paused are never taken; a place with no kept reading takes what its
  window holds as read and answers only what is said after.
- TURNS-4: A window that dropped lines past where reading stood is told as `missed`. A window the app lost makes its
  place start as with no kept reading.

An agent appears under the name the realm gives it and with the AI mark the app shows; it is removed with its
seat, its lease or its client.

## When a turn is due

- TURNS-5: Every line by someone else gives one turn, unless the app marked it as an act or as followed by more of the
  same speaker's speech, or it was said at a place with a view. His own lines give none.
- TURNS-6: One turn runs at a time across all his apps. What arrives meanwhile enters his history at once; each place
  with something for him gets one turn after it, told all that arrived; the order is TURNS-48.
- TURNS-48: A place whose turn would be spoken (a call place of his voice app whose last window does not state his voice
  unavailable) is claimed before the written places; within each kind the place that waited longest goes first.
  When such a place becomes due while a written turn of his runs, that turn ends at its next safe point: at
  once while the model thinks, or once a relayed call in flight has answered or ended and entered his history.
  Nothing more of it goes out, it adds no message (its entries go to the next turn's), it starts no pause, and
  its place is due again for what it was for, joined with what arrived meanwhile and from when it first became
  due. The spoken turn is told the cut turn's entries and holds the tools of its app (TURNS-18), so he may answer
  and do what the cut turn was for in that one turn; the cut place gets its turn after it only while something is
  left for him, so a view that no longer waits for him is not due. A written place gives way to a spoken place at
  most once until its turn has run to its end, whether passed over in the order or cut; after that it is claimed
  before the spoken places and its turn is not cut for one. A spoken turn is never cut.
- TURNS-7: A view that waits for him at a revision no turn was started for gives one turn, however that turn ends.
  The runner does not ask again by itself; an app that wants him to act again moves the revision. A turn whose
  tool list a tools notice overtook before it ended does not count for its view's revision.
- TURNS-8: The gateway's budget is the only limit on how often he thinks: in a lively call he thinks almost without a
  pause, and two agents in one call can keep each other talking.

## What a turn is told

Every turn, written, aloud or at a view, is sent the same front: the persona followed by the reply rules in
the system text, then one line for each skill of the apps the turn lists, and the runner's own tools,
`read_skill`, `continue_speaking` and `stop_speaking` among them. A line is the skill's name and its description
as the app declared them; an app that declares no skills extension has one implicit skill, whose description names
the app by its derived name. A name two apps' skills share goes to the first in the configuration's order. The rest
is user messages and the rounds of the skills he read, none an instruction:

1. A header: his name (`you`) and whether older entries have left his history (`older_dropped`).
2. One message for each earlier answered turn whose entries his history still holds. A turn whose entries hold
   skills he read is told split around each one's round: a message of the entries before it, the round, and so on
   to the turn's message with the entries after the last round.
3. TURNS-9: The turn's own message, which alone says whether the turn is aloud.

- TURNS-10: A turn's message holds when it was admitted (`at`, UTC to the whole second), what woke him (`woke`), and
  every entry he took since the message before it or since the round before it (`entries`), each told once, in the
  order he took them. A message before a round holds `entries` alone.
  `woke` names its app and place and either `lines`, with the `seqs` of the lines the turn is for, whether
  older lines were `missed` and whether his answer is heard aloud, or `view`, with the `revision` it is for.

An entry has its app and the time he took it (`at`), and is one of:

| ID | Kind | Fields |
|---|---|---|
| TURNS-11 | line | place, `seq`, own mark, `spoken`; `cut` only when the app cut it, its text then the part heard; `act` only when it says what someone did |
| TURNS-12 | table state | place, `revision`, whether it waits for him, the state as the app sent it, told whole in the first turn that held it |
| TURNS-13 | call | `tool`, the `arguments` he sent and an `outcome`: `"answered": true` with `text`, `is_error` and `image` when a picture came, or `"answered": false` with a `code` |
| TURNS-14 | spoken sentence | `text` and an `outcome`: taken, with `refused` and its code or `written` when the app wrote it instead, or `"answered": false` with a `code` |

- TURNS-15: Every message before the turn's own is byte for byte as the turn before sent it, so the gateway's cache
  reads it back. A skill he read is told as a round of its own at its place among the messages: one `read_skill`
  call with the skill's name under the id the runner minted for it, answered with the text the model was shown, byte
  for byte in every later turn, after a restart or a take-over too. A turn that ends without an answer adds no
  message; its entries go to the next turn's.
- TURNS-16: Past `history_bytes` the oldest whole messages leave in one step until what stays fits
  `history_keep_bytes`. When the newest message alone is past it, its oldest entries leave too until what
  stays fits; its newest entry always stays. A larger value keeps more and makes every request longer.
- TURNS-17: What lies before his history's first entry reaches no turn unasked; with `[memory]` the turn can look it
  up (see Recall).

## Tools and calls

- TURNS-18: A turn holds every tool its own app listed after its last tools notice on the loop's open stream, or at the
  turn's start when that stream carries no notice, the tools of every other app where he holds an
  active place, and those of the tools-only apps, declared in the configuration's order. A name two apps
  offer goes to the first; `accept_invitation`, `speak`, `recall`, `read_skill`, `continue_speaking`,
  `stop_speaking` and `set_turn_state` are the runner's own and never taken from an app. `read_skill(skill)` is held
  in every turn and answers the text of a skill whose line the turn holds; it loads nothing. `continue_speaking` and
  `stop_speaking` are held in every turn, written or aloud, so a switch between them leaves the front as it was; called in a turn
  that is not aloud they do nothing, and the model is told they act only in a spoken turn. No request asks the
  gateway for `parallel_tool_calls`.
- TURNS-19: The turn's own app's list is kept on the loop's session for the stream and listed again after a tools notice, a
  new stream or a stream's end. The turn's relayed calls and `speak` go on the loop's session of that app, under the
  token the loop holds; a hold's decision and every call to another app go in a session of their own under a
  token for that app's scope. A call to another app that fails gives the model an error result and that app
  takes no more calls in the turn; a failed call to the turn's own app ends the turn, and a turn still running
  when the loop's session ends is cancelled before it closes. A tool's error result reaches the model inside
  the same turn. The skills of each app the turn lists are read in the session that lists its tools: an app
  that declares the skills extension is asked for its listing, and one that does not is sent no request; an app
  whose skills listing fails has no lines in that turn and the turn goes on. A read of a skill is a call to its app: at
  another app in a session of its own, which fails as a call there does, and for the own app on the loop's session,
  which ends the turn as a failed call does. A skill the app serves a file for that is not what its listing
  promised, a name no line holds and a text past one result's bound answer an error and enter nothing.
- TURNS-20: Before the turn's next call goes out, once per turn, anything not his own that arrived since the turn
  began holds that call: it is not sent, the model is told what arrived, and decides again.
- TURNS-21: Every relayed call enters his history when the app answers or the relay ends without an answer, with the
  failure's code or `cancelled`; a call past an app's bound was never sent and does not enter. An answered
  `read_skill` enters as a skill entry instead.
- TURNS-22: A call past an app's bound is answered with an error and the model may go on. The turn's total is the sum
  of the apps' bounds plus one for the held call and one for the speaking tools, plus `lookup_calls` with `[memory]`
  and the two reads of skills `read_skill` takes, so a model that keeps calling past a bound can spend it and end
  the turn without its `say`.
- TURNS-23: In a call and at a table at once, a turn at either can answer in the call and act at the table, also
  when it is not his move; the table refuses what its rules do not allow.

## Spoken turns

- TURNS-24: In an app that declares `speak`, an agent with a `[agents.voice]` table speaks at its call places; the turn
  is told `"aloud": true`. Where the place's last window states his voice unavailable (`voice_state`) the
  turn is written instead.
- TURNS-25: The text is spoken sentence by sentence through `speak`, at most one sentence ahead of the one playing,
  a fixed rule, by his place's sender also after the turn, which ends once its last model round has
  returned. The app appends his own spoken line for each sentence as it ends, whole or cut. A call goes out
  at once, also while his sentences play or wait.
- TURNS-26: A round that opens with `[silent]` says nothing; a final round with no text and no call is a provider
  error, so the characters end a silent turn with `[silent]`.
- TURNS-27: A refused sentence (`stopped` included) drops every unsent sentence at the place, is logged as
  `speak.refused`, and the turn goes on; a sentence answered as held stays first among the unsent and waits
  for his decision. A refusal as `stopped` of a request his own stop or answer cut is no refusal and drops
  nothing. A sentence the app wrote instead is entered as written. A `speak` that fails as a call ends the
  loop's session, and a turn running on it, into the loop's reaction.
- TURNS-28: The persona's `maxDurationSeconds` bounds the turn and each sentence it released: a turn past it ends
  `limit_exceeded`; a sentence still unsent past it is dropped and logged, except while a hold stands, and a
  continue gives it the deciding turn's bound.
- TURNS-45: A later spoken turn's first sentence at the place drops the earlier turn's unsent sentences, which is
  logged as `speech.dropped`; the sentence that plays and the one already sent stay, and the new one queues
  behind them. A silent or written turn keeps them.

## Holds

- TURNS-29: While he speaks the app may hold him: a person's voice over him lasts longer than about 0.4 s (the app's
  trigger) and the window states the hold (`interrupted`, with an `id` and, when the app can tell, the `place` it
  stopped him at) and what each person said (`talked_over`). He is decided on `interrupted` alone.
- TURNS-30: While a spoken turn of his runs at the place, a round beside his parked run is told his history and, once per
  hold id, what was said over him, where every listener heard his sentence to (the part heard and the rest, or the
  whole sentence when the app states no place), his unsaid sentences and his waiting call.
  Otherwise, once words were said over him or their record is gone, the hold makes the place due for its lines,
  and that turn is told the hold first. A round or turn told a hold runs on `hold_model` when set.
- TURNS-31: He calls `continue_speaking(id)` to go on or `stop_speaking(id)` to drop the rest: no unsent sentence is
  spoken and a waiting call is never sent. His unsaid sentences include an ended turn's still waiting at the
  place. A turn told a hold sends no other call until he decided: after a stop its sentences play at once,
  after a continue they follow the rest. A round or turn that ends without a decision stops him, and what
  waited is dropped unsaid. Each id of a window generation is logged once as `hold.decided`, with its
  `hold_id`. A call made while a hold stands at the place where the turn speaks waits for his decision; a
  call already sent runs to its answer.
  Text in a reply while the hold he was told is undecided, in the round beside his parked run or in a turn told
  the hold, answers it instead of either call: what he had not said is dropped, logged as `speech.dropped`
  `replaced`, the parked run is cut with its waiting call unsent, and a decision call of the same reply is
  answered that the hold is decided. The first sentence goes to `speak` naming the hold, in a session of its
  own, and counts in the turn's tally; the rest of the reply joins the record once the app took it, and
  his place's sender sends it plain, also after the turn. A first sentence the app refused or never answered
  sends no further one and is logged; unless the app refused it as `not_interrupted` he is stopped with the
  hold's id, and the hold ends `undecided`, or `gone` for that refusal. A new hold during his answer starts no
  round and stands for his next turn; words said over him after it start no second decision. Text in a
  round after his decision says nothing.
- TURNS-46: `continue_speaking(id, line, style)` says one short line and then goes on: the line goes to `speak` naming the hold, in
  a session of its own, with the optional `style` (how it sounds, such as `louder and angrier`) passed on unchanged, and counts in the turn's tally; the sentence he was held in is cut and not said again, and his
  unsent sentences follow plain, without the style, with the deciding turn's bound, the one the app dropped for the line first. The hold
  ends `short_line` once the app took the line. The parked run is not cut: its waiting call goes on once the hold no
  longer stands; a turn told the hold sends its sentences after the line. A line or style the app refused (`style_unavailable` for a style
  the voice does not take) or never answered
  sends no further sentence and is logged; unless refused as `not_interrupted` he is stopped with the hold's id, and
  the hold ends `undecided`, or `gone` for that refusal. A `line` or `style` that is no non-empty string decides nothing, and so does a `style` without a `line` and a
  line in a turn that does not speak. In a turn told the hold no sentence of that turn is playing when the line
  is said, so none is cut. After the line a new hold is a new hold, as after a continue: it starts one more round
  in the same turn.
- TURNS-47: Every further hold in one spoken turn, after a go on or a short line, is
  decided as the first: a decision does what it did for the first, and no decision drops what waited and he says
  nothing. The rule covers a hold decided by a round beside the running turn and, in a turn told the first hold, a
  second hold decided by a round beside that turn.
- TURNS-32: After he went on, a call of his is not sent while people said words no round told him; its result says
  so (`sent` false, `talked_over`).

## Pictures

- TURNS-33: A tool result may carry one `image/jpeg` block: the model sees it in each later round of that turn, and the
  history tells the call with `"image": true`. Another media type, a second picture or data that is no JPEG
  is an unusable answer; audio and resource blocks are ignored.

## Places with a view

- TURNS-34: A place whose standing has `serves_view` is read through its view (newest state: `revision`, whether it
  waits for him, the app's `state`) and its window (lines). After every standing read he reads the views of
  his active view places, then the windows.
- TURNS-35: A window's new lines enter his history before that wake's view; a line newer than the view's revision is
  told before it. Every revision he reads enters his history once.
- TURNS-36: An app that lists no window beside a view, or answers its read as unknown, costs no turn and no failure.

## Pauses and retries

- TURNS-37: A turn that ends without an answer (the gateway refuses or fails it, a limit is exceeded) pauses all his
  turns: 1 second, doubling with each such turn in a row, up to `budget_pause_cap_seconds` for a spent
  budget and 30 seconds otherwise, logged once as `turns.paused`.
- TURNS-38: After the pause the place is due again with what the turn was for, joined with what arrived meanwhile. A
  place he left gets no turn, and a turn cancelled by his leaving or running at a kill is not retried. A
  stop (a SIGTERM or a lost lock) hands a turn it cut to the new holder only if the turn had relayed no call
  and sent no sentence, since a line said twice is worse than one unanswered. The first turn with an answer
  ends the pausing.

## Recall

- TURNS-39: With `[memory]` every turn holds `recall`, first among its tools: one string `text`, read only from the
  lines he kept in the turn's own app and place, every visit of it.
- TURNS-40: A line matches when its text holds every word (Postgres full text, `simple`: no stemming); a `text` with
  no word matches every line. Only lines first read before his history's first entry are given, newest
  first, as many as fit `lookup_bytes`, with `older_dropped` when more matched.
- TURNS-41: A lookup takes nothing into his history and is never held back. A store that cannot answer gives an error
  result; the turn goes on.

## Grants

- TURNS-42: Which apps an agent reaches is the realm's group membership. When the realm or an app refuses him, that loop
  is suspended and resumes when the refusal ends, without a restart; a tools-only app that refuses him is
  left out of his turns and asked again every 30 seconds.

## Turn state

- TURNS-43: Through the app's `set_turn_state` a turn states at its place that it is thinking before its first round,
  and once it ends what its end calls for: that his turns answer after an answer, the gateway's refusal
  (`budget_spent`, `busy`, `not_enabled`) after one, `unavailable` after any failure. A cancelled turn
  states again what the last turn end in this seat stated, that his turns answer when none did; so does a
  turn the runner's stop cuts, on a session of its own within `cancel_margin_seconds`, so his figure is not
  left thinking. At a seat he no longer holds nothing is stated. One app's statements go out in order, so a
  turn's thinking never overtakes the end of the turn before it; a refused or failed statement, or one the app
  does not answer within its `call_seconds`, is logged as `turn_state.failed` and the turn goes on. A seat
  whose app answers that it serves no `set_turn_state` is logged once and gets no more statements until he
  no longer holds it; a new seat at the same place starts with nothing stated.
- TURNS-44: Once the loop stops taking turns in an app, at a suspension or a backoff past its first retry, it states
  `unavailable` at each seat he holds there, on a session of its own while it waits to retry; a blip that
  recovers on its first retry states nothing. The next full pass of standing, accepts and windows states
  at each held seat its last turn end in this seat again, that his turns answer when none did; so does the
  first full pass after a session that ended with a statement failed or in flight, and so does the runner's stop
  while that is still due and `unavailable` does not stand. Where `unavailable` stands, the stop states nothing
  more, because no full pass went through since: `unavailable` is what the app holds. Stating `unavailable`
  stops at the first seat whose statement the app does not answer.
