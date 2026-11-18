# Turns reference

What the runner promises about an agent's turns: when one is due, what it is told, what it may do and how it
ends. What is stored, and for how long, is in [the architecture](ARCHITECTURE.md), Persistence.

## Joining and reading

- An agent accepts every invitation to a call. To a place of another kind it accepts only in an app where it
  can take turns (a persona, the gateway and bound tools); one left out for that is logged once as
  `invitation.left_out` with the reason `no_turns`. Only apps configured to listen can invite it.
- It appears under the name the realm gives it, with the AI mark, and never leaves on its own: it is removed
  with its seat, its lease or its client.
- After its own accept it reads the window from `seq` 1. A place whose reading is kept resumes where it
  stood; lines said while the place was paused are never taken; a place with no kept reading takes what its
  window holds as read and answers only what is said after.
- A window that dropped lines past where reading stood is told as `missed`. A window the app lost makes its
  place start as with no kept reading.

## When a turn is due

- Every line by someone else gives one turn, unless the app marked it as an act or it was said at a place
  with a view. His own lines give none.
- One turn runs at a time across all his apps. What arrives meanwhile enters his history at once; each place
  with something for him gets one turn after it, the place that waited longest first, told all that arrived.
- A view that waits for him at a revision no turn was started for gives one turn, however that turn ends.
  The runner does not ask again by itself; an app that wants him to act again moves the revision.
- The gateway's budget is the only limit on how often he thinks: in a lively call he thinks almost without a
  pause, and two agents in one call can keep each other talking.

## What a turn is told

Every turn, written, aloud or at a view, is sent the persona and then user messages, none an instruction:

1. A header: his name (`you`) and whether older entries have left his history (`older_dropped`).
2. One message for each earlier answered turn whose entries his history still holds.
3. The turn's own message.

A turn's message holds when it was admitted (`at`, UTC to the whole second), what woke him (`woke`), and
every entry he took since the message before it (`entries`), each told once, in the order he took them.
`woke` names its app and place and either `lines`, with the `seqs` of the lines the turn is for, whether
older lines were `missed` and whether his answer is heard aloud, or `view`, with the `revision` it is for.

An entry has its app and the time he took it (`at`), and is one of:

| Kind | Fields |
|---|---|
| line | place, `seq`, own mark, `spoken`; `cut` only when the app cut it, its text then the part heard; `act` only when it says what someone did |
| table state | place, `revision`, whether it waits for him, the state as the app sent it, told whole in the first turn that held it |
| call | `tool`, the `arguments` he sent and an `outcome`: `"answered": true` with `text`, `is_error` and `image` when a picture came, or `"answered": false` with a `code` |
| spoken sentence | `text` and an `outcome`: taken, with `refused` and its code or `written` when the app wrote it instead, or `"answered": false` with a `code` |

- Every message before the turn's own is byte for byte as the turn before sent it, so the gateway's cache
  reads it back. A turn that ends without an answer adds no message; its entries go to the next turn's.
- Past `history_bytes` the oldest whole messages leave in one step until what stays fits
  `history_keep_bytes`; the newest message and its newest entry always stay. A larger value keeps more and
  makes every request longer.
- What lies before his history's first entry reaches no turn unasked; with `[memory]` the turn can look it
  up (see Recall).

## Tools and calls

- A turn holds every tool its own app lists when it starts, the tools of every other app where he holds an
  active place, and those of the tools-only apps, declared in the configuration's order. A name two apps
  offer goes to the first; `accept_invitation`, `speak`, `recall`, `continue_speaking` and `stop_speaking`
  are the runner's own and never taken from an app.
- A call goes to its app in a session of its own under a token for that app's scope. A call to another app
  that fails gives the model an error result and that app takes no more calls in the turn; a failed call to
  the turn's own app ends the turn. A tool's error result reaches the model inside the same turn.
- Before the turn's next call goes out, once per turn, anything not his own that arrived since the turn
  began holds that call: it is not sent, the model is told what arrived, and decides again.
- Every relayed call enters his history when the app answers or the relay ends without an answer, with the
  failure's code or `cancelled`; a call past an app's bound was never sent and does not enter.
- A call past an app's bound is answered with an error and the model may go on. The turn's total is the sum
  of the apps' bounds plus one for the held call, plus `lookup_calls` with `[memory]`, so a model that keeps
  calling past a bound can spend it and end the turn without its `say`.
- In a call and at a table at once, a turn at either can answer in the call and act at the table, also
  when it is not his move; the table refuses what its rules do not allow.

## Spoken turns

- In an app that declares `speak`, an agent with a `[agents.voice]` table speaks at its call places; the turn
  is told `"aloud": true`. Where the place's last window states his voice unavailable (`voice_state`) the
  turn is written instead.
- The text is spoken sentence by sentence through `speak`, at most one sentence ahead of the one playing,
  a fixed rule. The app appends his own spoken line for each sentence as it ends, whole or cut; a call goes
  out, and the turn ends, only after the line of every sentence the app took has entered.
- A round that opens with `[silent]` says nothing; a final round with no text and no call is a provider
  error, so the characters end a silent turn with `[silent]`.
- A refused sentence (`stopped` included) ends the turn's speech, is logged as `speak.refused`, and the turn
  goes on. A sentence the app wrote instead is entered as written. A `speak` that fails as a call ends the
  turn.
- The persona's `maxDurationSeconds` bounds the turn with its speech: past it nothing more goes out and the
  turn ends with the reason `limit_exceeded`.

## Holds

- While he speaks the app may hold him: someone talks over him and the window states the hold
  (`interrupted`, with an `id`) and what each person said (`talked_over`).
- A round beside his parked run is told his history and, once per hold id, what was said over him, the
  sentence he was in, his unsaid sentences and his waiting call. It runs on `hold_model` when set.
- He calls `continue_speaking(id)` to go on or `stop_speaking(id)` to drop the rest: no unsent sentence is
  spoken, a waiting call is never sent and the turn ends. Any other ending, a turn already over, or a hold
  where no spoken turn of his runs stops him too. Each id is logged once as `hold.decided`.
- After he went on, a call of his is not sent while people said words no round told him; its result says
  so (`sent` false, `talked_over`).

## Pictures

A tool result may carry one `image/jpeg` block: the model sees it in each later round of that turn, and the
history tells the call with `"image": true`. Another media type, a second picture or data that is no JPEG
is an unusable answer; audio and resource blocks are ignored.

## Places with a view

- A place whose standing has `serves_view` is read through its view (newest state: `revision`, whether it
  waits for him, the app's `state`) and its window (lines). After every standing read he reads the views of
  his active view places, then the windows.
- A window's new lines enter his history before that wake's view; a line newer than the view's revision is
  told before it. Every revision he reads enters his history once.
- An app that lists no window beside a view, or answers its read as unknown, costs no turn and no failure.

## Pauses and retries

- A turn that ends without an answer (the gateway refuses or fails it, a limit is exceeded) pauses all his
  turns: 1 second, doubling with each such turn in a row, up to `budget_pause_cap_seconds` for a spent
  budget and 30 seconds otherwise, logged once as `turns.paused`.
- After the pause the place is due again with what the turn was for, joined with what arrived meanwhile. A
  place he left gets no turn, and a turn cancelled by his leaving or running at a kill is not retried. A
  SIGTERM hands a turn it cut to the new holder only if the turn had relayed no call and sent no sentence,
  since a line said twice is worse than one unanswered. The first turn with an answer ends the pausing.

## Recall

- With `[memory]` every turn holds `recall`, first among its tools: one string `text`, read only from the
  lines he kept in the turn's own app and place, every visit of it.
- A line matches when its text holds every word (Postgres full text, `simple`: no stemming); a `text` with
  no word matches every line. Only lines first read before his history's first entry are given, newest
  first, as many as fit `lookup_bytes`, with `older_dropped` when more matched.
- A lookup takes nothing into his history and is never held back. A store that cannot answer gives an error
  result; the turn goes on.

## Grants

Which apps an agent reaches is the realm's group membership. When the realm or an app refuses him, that loop
is suspended and resumes when the refusal ends, without a restart; a tools-only app that refuses him is
left out of his turns and asked again every 30 seconds.
