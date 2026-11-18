# Log reference

What the runner writes to stdout, for whoever reads or alerts on it. `src/personas/reporting.py` owns the
events and their fields. A test cites each rule by its ID at the start of its docstring, such as
`LOG-1`.

## Shape

- Every line is one JSON object through [logline](https://github.com/overnightworks/logline), with
  `service` `personas`. `LOG_LEVEL` sets the level (default `info`), `SERVICE_VERSION` the `version`.
  alembic, which checks the memory's schema at the start, writes only its warnings and errors. (LOG-1)
- An event is `<noun>.<verb_past>`, written once per state change. Its ids are `agent_id` and `app_id`, the
  host of the app's URL; a `reason` is a closed code. (LOG-2)
- A line holds ids, numbers named with their unit and closed codes: never text a person or an app wrote,
  a table state, a token or a secret. (LOG-3)
- A turn writes every line it causes in one trace (`trace_id`): that of the newest line that made it due,
  or of the view revision, or its own when the app gives none or one that cannot be read. Every request to
  an app, the realm and the gateway carries that trace's `traceparent`. agents adds `model_request.ended`
  and `tool_call.ended` in the same trace; its `docs/connector.md` lists them. (LOG-4)
- `personas migrate` and `personas memory` write no JSON line. `personas memory` prints its result on stdout;
  `personas migrate` prints nothing on success and exits 0, which is its result. (LOG-5)

## Events

| Event | Written when | Fields beyond the ids |
|---|---|---|
| `loop.started` | a loop starts | `model` when the agent has a persona (LOG-6) |
| `loop.suspended` | the realm or an app refuses the agent | `reason`; `status` and `mcp_method` when a call failed (LOG-7) |
| `loop.resumed` | the refusal ended | none (LOG-8) |
| `loop.backed_off` | a network failure, a lost stream or a 5xx | `reason`; `status` and `mcp_method` when a call failed (LOG-9) |
| `invitation.accepted`, `invitation.refused` | an accept answered | `reason` for a refusal (LOG-10) |
| `invitation.left_out` | an invitation the agent cannot take turns at | `place_kind`, `reason` `no_turns` (LOG-11) |
| `turn.ended` | every turn, however it ends | see below (LOG-12) |
| `turn_wait.ended`, `token_hold.ended`, `session_open.ended`, `tool_list.ended`, `turn_setup.ended` | each step of a turn | `duration_ms`, `outcome`, `reason` of a failed step (LOG-13) |
| `token_request.ended` | each token request | `duration_ms`, `outcome` (LOG-14) |
| `turns.paused` | a turn without an answer pauses his turns | `reason` (the turn's code, `budget_spent` among them), `pause_ms` (LOG-15) |
| `speak.refused` | the app refused a sentence | `reason` (the app's code) (LOG-16) |
| `hold.decided` | once per hold id | `reason`: `continued`, `stopped`, `undecided`, `no_turn` (LOG-17) |
| `tools.left_out` | an app lost tool names to an earlier app, once per session | `tool_count`, never a name (LOG-18) |
| `held_app.failed`, `tools_only_app.failed` | a call to another app failed | `status`, `mcp_method`, `reason` (LOG-19) |
| `store.failed`, `store.resumed` | once per store outage, and when it answers again | `reason` `store_unreachable` or `store_timeout` (LOG-20) |
| `lines.swept` | each sweep | `line_count`; no agent or app (LOG-21) |
| `agent.taken` | a pod holds an agent's lock | the agent only (LOG-22) |
| `agent.lost` | a pod stops an agent it served | `reason`: `hold_lost`, `generation_stale` or the store's code (LOG-23) |
| `agent.stop_overran` | an agent took longer than `cancel_margin_seconds` to stop | the agent only (LOG-24) |
| `runner.stop_cut` | the runner had not stopped `stop_seconds` after a SIGTERM and ends at once | none; no agent or app (LOG-25) |

A `status` is the integer of an error status, absent for a JSON-RPC error sent with a success status. A
`reason` of a failed call is a standard JSON-RPC error or a known media type by name, else one fixed word,
never the app's own value. (LOG-26)

A `turn_wait.ended` runs to the turn's start from the earliest arrival of a line the turn answers, or at a
place with a view from the moment the loop saw the revision that waits for the agent. A line that a turn
without an answer left counts from the end of the pause that turn caused, and one that arrived during the
pause keeps its arrival. A place a pod resumes at a takeover or a restart counts from that resume. Every
moment is read on the monotonic scale of the clock boundary (`Clock.now` in `clock.py`), and `due.py` keeps
the earliest through a join or a replacement. (LOG-27)

## `turn.ended`

- `duration_ms` from the turn's start, and an `outcome`: `ok` for an answer, `refused` by the gateway,
  `failed` at level error, `cancelled`. `refused` and `failed` carry a `reason`: the gateway's code, the
  reason of the failed call, `defect`, `limit_exceeded` or `no_final_event`. (LOG-28)
- `sentence_count` (sent to `speak`, refused and cut ones included), `first_output_ms` (absent when none
  was sent), `silent_count` (rounds that opened with `[silent]`) and `tool_call_count`. (LOG-29)
- Of a turn that sent a model request, its input bytes: `tools_bytes`, `told_bytes`, `results_bytes` and
  `images_bytes` (decoded picture bytes, apart from `results_bytes`). (LOG-30)
- Of a turn that got the final answer, when the gateway reported them for every round, the sums
  `input_tokens` (cached included; a picture counts once per round), `cached_input_tokens`,
  `cache_write_tokens` and `output_tokens`, and `model_request_count`; otherwise all five are absent, never
  zero. (LOG-31)

The counts change nothing the runner does.
