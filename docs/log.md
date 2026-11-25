# Log reference

What the runner writes to stdout, for whoever reads or alerts on it. `src/personas/reporting.py` owns the
events and their fields. A test cites each rule by its ID at the start of its docstring, such as
`LOG-1`. Every list item that opens with an ID and every table row is a rule; prose without an ID is context.

## Shape

- LOG-1: Every line is one JSON object through [logline](https://github.com/overnightworks/logline), with
  `service` `personas`. `LOG_LEVEL` sets the level (default `info`) on every command, `SERVICE_VERSION` the `version`.
  alembic, which checks the memory's schema at the start, writes only its warnings and errors. The MCP SDK's client
  logger, which agents' member client uses, writes an ERROR for a stream that closed under a request, which a
  recovered blip causes; it is written at info, and any other error of that client stays at error. An uncaught
  exception of any command writes logline's `process.crashed` line at level
  critical, with `error_type` and `exception`, Python's traceback with its frames and messages; a configuration it
  refuses before it starts is one plain line on stderr.
- LOG-2: An event is `<noun>.<verb_past>`, written once per state change. Its ids are `agent_id` and `app_id`, the
  host of the app's URL; a `reason` is a closed code.
- LOG-3: A line holds ids, numbers named with their unit and closed codes: never text a person or an app wrote,
  a table state, a token or a secret. The one exception is the crash line's `exception`, whose messages may
  quote such text; `secret.Secret` keeps every secret and token out of an error message.
- LOG-4: A turn writes every line it causes in one trace (`trace_id`): that of the newest line that made it due,
  or of the view revision, or its own when the app gives none or one that cannot be read. Every request to
  an app, the realm and the gateway carries that trace's `traceparent`. agents adds `model_request.ended`
  and `tool_call.ended` in the same trace; its `docs/connector.md` lists them. Every step line (`turn.ended`,
  `turn_wait.ended` and the other timed lines) also carries `span_id` and `parent_span_id`, so a turn's
  lines form a tree.
- LOG-5: `personas migrate` and `personas memory` write no JSON line. `personas memory` prints its result on stdout;
  `personas migrate` prints nothing on success and exits 0, which is its result.

## Events

| ID | Event | Written when | Fields beyond the ids |
|---|---|---|---|
| LOG-6 | `loop.started` | a loop starts | `model` when the agent can take turns |
| LOG-7 | `loop.suspended` | the realm or an app refuses the agent | `reason`; `status` and `mcp_method` when a call failed |
| LOG-8 | `loop.resumed` | a full pass of standing, accepts and windows went through after a suspension or a backoff, so a retry whose window reads still fail logs none | none |
| LOG-9 | `loop.backed_off` | a network failure, a lost stream or a 5xx | `reason`; `status` and `mcp_method` when a call failed |
| LOG-10 | `invitation.accepted`, `invitation.refused` | an accept answered | `reason` for a refusal |
| LOG-11 | `invitation.left_out` | an invitation the agent cannot take turns at | `place_kind`, `reason` `no_turns` |
| LOG-11 | `invitation.declined` | an invitation to a call declined as busy while the agent held an active place | `place_kind`, `reason` `busy` |
| LOG-11 | `invitation.decline_refused` | a decline of a call invitation the app refused | `place_kind`, `reason` the refusal's code, or `unrecognised` for an error answer that is none of the app's refusals |
| LOG-37 | `turn.fell_due` | a pass made a place due that was not due | `occasion` (`line`, `view` or `hold`), `spoken` (whether its turn would be spoken), `read_ms` |
| LOG-12 | `turn.ended` | every turn, however it ends | see below |
| LOG-13 | `turn_wait.ended`, `claim_write.ended`, `thinking.ended`, `session_open.ended`, `tool_list.ended`, `skill_list.ended`, `turn_setup.ended` | each step of a turn | `duration_ms`, `outcome`; `reason` and `error_type` of a failed `session_open`, `tool_list`, `skill_list` or `turn_setup`, and `entry`, the index of the entry a malformed skills listing broke a rule at, never its text (`session_open` is a session the turn opens to another app, since the turn's own app is served on the loop's session; a place's own app that does not list its tools within its `call_seconds` fails the `tool_list` step and the turn goes on without them, and `tool_list` is written only for a listing that was sent, so a turn that uses the list its app's stream kept writes none; `skill_list` is written only for a `skills/list` that was sent, so an app that declares no skills extension writes none, and an app whose listing fails keeps its tools and has no skill lines in that turn; `turn_wait` ends when the turn starts and `claim_write` only waits for the store's writer, so neither fails; `thinking` is the statement that the turn is thinking, answered by the app; `turn_setup` is what the turn still waited for after its thinking was answered); `wait_reason` on `turn_wait.ended` |
| LOG-14 | `token_request.ended` | each token request | `duration_ms`, `outcome`; `reason` and `error_type` of a failed request |
| LOG-15 | `turns.paused` | a turn without an answer pauses his turns | `reason` (the turn's code, `budget_spent` among them), `pause_ms` |
| LOG-16 | `speak.refused` | the app refused a sentence, which drops the unsent ones at its place; his own stop's or answer's refusal of a request it cut is not logged | `reason` (the app's code) |
| LOG-36 | `speech.dropped` | sentences were dropped unsent, once per drop | `reason`: `replaced` (a later spoken turn's first sentence, or his answer instead of a hold), `limit_exceeded` (unsent at its bound, or a request not answered by it), `stopped` (a hold's stop or undecided end, a runner stop) or `failed` (its turn ended without an answer, a `speak` failed as a call, the session ended); `sentence_count` |
| LOG-17 | `hold.decided` | once per hold id | `reason`: `continued`, `short_line` (the app took his short line, then his unsent sentences follow), `stopped`, `answered` (the app took the first sentence of his answer instead), `undecided`, or `gone` when the app refused the decision because no such hold stood; `hold_id`, the app's id for the hold, which comes again in a new window generation |
| LOG-40 | `skill.refused` | the app served a skill file that is not what its listing promised, so `read_skill` answered an error | `reason` (the refusal's code), never a name or text |
| LOG-18 | `tools.left_out` | an app lost tool names to an earlier app, once per session | `tool_count`, never a name |
| LOG-39 | `argument.left_out` | a call went out without an optional argument its tool's declared schema does not name (`figure`, `voices`, `looks` and `looks[].room`, the `room` of each entry of `looks`, of `accept_invitation`, `style` and `interrupted` of `speak`, `thinking` of `set_turn_state`), once per tool and argument per app session, so an app that renamed an argument shows instead of losing the feature quietly; a required argument is never left out; an `accept_invitation` that carries `figure`, `voices` or `looks` on a session with no list yet asks the app for its tools first, so an app whose list fails fails that accept with the list's failure; logged at info | `tool`, `argument`, never a value |
| LOG-19 | `held_app.failed`, `tools_only_app.failed` | a call to another app failed, a read of a skill included | `status`, `mcp_method`, `reason` |
| LOG-20 | `store.failed`, `store.resumed` | once per store outage, and when it answers again | `reason` `store_unreachable` or `store_timeout` |
| LOG-21 | `lines.swept` | a sweep that erased lines | `line_count`; no agent or app |
| LOG-22 | `agent.taken` | a pod holds an agent's lock | the agent only |
| LOG-23 | `agent.lost` | a pod stops an agent it served | `reason`: `hold_lost`, `generation_stale` or the store's code |
| LOG-24 | `agent.stop_overran` | an agent took longer than `cancel_margin_seconds` to stop | the agent only |
| LOG-25 | `runner.stop_cut` | the runner had not stopped `stop_seconds` after a SIGTERM and ends at once | none; no agent or app |
| LOG-34 | `turn_state.failed` | the app refused or failed a statement of his turns at a place, or a stop's statements did not go out within its margin | `reason` (the app's refusal code, the failed call's reason, `invalid_params` for a tool the app does not serve, `network` for a stop's margin); `status` and `mcp_method` when a call failed |
| LOG-35 | `turn_state.lost` | a statement whose answer was lost when its session ended on another call's failure or on the runner's stop; the runner recovers on its own, since the next full pass states it again, and the stop states each seat's last end again within its margin, so it logs at info, once for the statement and not again for that restatement | none |

- LOG-26: A `status` is the integer of an error status, absent for a JSON-RPC error sent with a success status. A
  `reason` of a failed call is a standard JSON-RPC error or a known media type by name, else one fixed word,
  never the app's own value.

- LOG-27: A `turn_wait.ended` runs to the turn's start from the earliest arrival of a line the turn answers, or at a
  place with a view from the moment the loop saw the revision that waits for the agent. A line that a turn
  without an answer left counts from the end of the pause that turn caused, and one that arrived during the
  pause keeps its arrival. A place a pod resumes at a takeover or a restart counts from that resume. Every
  moment is read on the monotonic scale of the clock boundary (`Clock.now` in `clock.py`), and `due.py` keeps
  the earliest through a join or a replacement.

- LOG-32: A `turn_wait.ended` names in `wait_reason` what first stood directly in front of the place once its wait
  began: `older_due` for a due place of his that is claimed before it, else `turn_here` for his running turn at the
  place's app or `turn_elsewhere` for one at another app, else `paused` for a pause of his turns; `none` when its
  turn could start at once. It is the first reason, not the longest: a loop reads it only when it tries to claim a turn,
  so what holds the place between two tries is not measured. A join keeps the reason; a place a pod resumes
  starts with none.

- LOG-33: A `claim_write.ended` runs from the turn's start until the store keeps its claim. `turn_wait`, `claim_write`,
  `thinking` and `turn_setup` run in sequence and add up to the time from due to the model request. `tool_list`
  and `session_open` may run beside `claim_write` and `thinking`: their overlap is read from each line's time
  minus its duration.

- LOG-37: A `turn.fell_due` is written once for a place that a pass makes due while it was not due: by a line that wakes
  him, by a view revision that waits for him, or by a hold that may be told. `read_ms` runs from the pass's start
  of reading the standing to that moment, on the monotonic scale of `Clock.now`. It stands in the trace of the
  line or revision that made the place due, so it joins that place's `turn_wait.ended`; a hold carries no trace, so
  its event has none. A line that joins a place already due, a wake that makes nothing due, a place due again after
  a turn without an answer and a place a pod resumes write none. The event carries no line's kind, text or
  name.

## `turn.ended`

- LOG-28: `duration_ms` from the turn's start to its last model round's end (its sentences' sending is no part of
  it), and an `outcome`: `ok` for an answer, `refused` by the gateway,
  `failed` at level error, `cancelled`. `refused` and `failed` carry a `reason`: the gateway's code, the
  reason of the failed call, `defect`, `limit_exceeded` or `no_final_event`.
- LOG-38: A written turn that ended at a safe point for a spoken place (TURNS-48) is `cancelled` and carries `cut`
  with the value `spoken`; no other turn carries `cut`.
- LOG-29: `sentence_count` (released by the turn's reply, whether sent or not), `first_output_ms` (absent when none
  was released), `silent_count` (rounds that opened with `[silent]`) and `tool_call_count`.
- LOG-30: Of a turn that sent a model request, its input bytes: `tools_bytes`, `told_bytes` (the messages and the
  calls and results of the rounds it was told), `results_bytes` and `images_bytes` (decoded picture bytes, apart from
  `results_bytes`). The skill lines in the system text count in none.
- LOG-31: Of a turn that got the final answer, when the gateway reported them for every round, the sums
  `input_tokens` (cached included; a picture counts once per round), `cached_input_tokens`,
  `cache_write_tokens` and `output_tokens`, and `model_request_count`; otherwise all five are absent, never
  zero.

The counts change nothing the runner does.
