# Configuration reference

- CONFIG-1: The runner reads one TOML file named by `PERSONAS_CONFIG`. `src/personas/config.py` owns every field and
  refusal; every value is checked at start, an unknown key is refused, and an error names the field or path,
  never a secret.

What the stores keep is in [the architecture](ARCHITECTURE.md), Persistence. A test
cites each rule by its ID at the start of its docstring, such as `CONFIG-1`. Every list item that opens with an
ID is a rule; prose without an ID is context.

## Top level

- CONFIG-2: `token_endpoint`; type: URL; default: required; meaning: the realm's token endpoint
- CONFIG-3: `ready_file`; type: absolute path; default: required; meaning: the readiness marker
- CONFIG-4: `history_bytes`; type: integer above 0; default: 49152; meaning: the history each agent keeps and tells every turn, counted in the JSON bytes of each entry as told
- CONFIG-5: `history_keep_bytes`; type: integer above 0, at most `history_bytes`; default: 32768; meaning: what stays after a cut; see [the turns reference](turns.md), What a turn is told
- CONFIG-6: `budget_pause_cap_seconds`; type: number, at least 1; default: 300; meaning: the longest pause between his turns while the gateway refuses for a spent budget

At the default history, a skat game (40 to 50 states of about 1.5 KB) fills it after about half a game.

## `[[apps]]`

- CONFIG-7: `mcp_url`; type: URL, once per file; default: required; meaning: the app's MCP endpoint; it names the app in the history and the lines
- CONFIG-8: `scope`; type: string; default: required; meaning: the token scope for this app
- CONFIG-9: `max_tool_calls`; type: integer above 0; default: none; meaning: binds every tool the app offers to its agents' turns, that many calls per turn
- CONFIG-10: `listens`; type: boolean; default: true; meaning: `false` serves tools only: no loop runs, its tools join the turns of the other apps deferred, shown to the model once `load_tools` loads them; needs `max_tool_calls`
- CONFIG-11: `call_seconds`; type: number above 0; default: 30; meaning: bounds each request a turn of another app makes to it, and the tool list and turn state requests of a turn at its own places; keep it below the persona's `maxDurationSeconds`

- CONFIG-12: At least one app listens. Apps are declared to a turn in the file's order, so list the apps whose places
  come and go last. An app with `max_tool_calls` requires `[gateway]` and a `persona_file` for every agent.

## `[[agents]]`

- CONFIG-13: `client_id`; type: string, once per file; default: required; meaning: the agent's realm client; two whose lock keys collide are refused
- CONFIG-14: `secret_file`; type: absolute path; default: required; meaning: the file holding its client secret
- CONFIG-15: `persona_file`; type: absolute path; default: none; meaning: its agent file `<name>.md`; the image holds them at `/app/characters/`
- CONFIG-16: `model`; type: `text/<model>`; default: the agent file's; meaning: the gateway model his turns run on; needs `persona_file`
- CONFIG-17: `hold_model`; type: `text/<model>`; default: the agent's model; meaning: the model of the round or turn that decides a hold; needs `persona_file`
- CONFIG-18: `[agents.figure]`; type: `url` (absolute `https`, no user, query or fragment), `sha256` (64 lowercase hex); default: none; meaning: carried by every accept; an app that shows figures and allows the address shows him with it
  - `[agents.room]`; type: `url`, `sha256` as `[agents.figure]`; needs `[agents.figure]`; default: none; meaning: the agent's own room, drawn behind his figure; every accept carries it as the `room` of his figure's look, beside the `figure`, and an app that does not declare `looks` or `looks[].room` gets the accept without it
- CONFIG-19: `[agents.voice]`; type: table, optional `names`: 1 to 8 distinct voice ids, first choice first; default: none; meaning: turns spoken turns on; every accept carries `names` as `voices`

- CONFIG-20: The agent file names no tools and no call bound, offers the role `user`, and has no `voice` section. The
  token's `agent_name` claim is the name a line must carry to address the agent. A change of a figure or a
  model needs a restart; the runner fetches and checks nothing about a figure's address or bytes.

## `[gateway]`

- CONFIG-21: The table holds the fields of the agents library's `AgentConfig` that reach the model gateway: `api_group`
  and `[gateway.gateway]` with `origin`, `key_file` and, for HTTPS, `ca_file`. It is required
  when an app sets `max_tool_calls`. What an agent may spend is bounded at the gateway's budget; the runner
  sets no limit of its own.

## `[database]`, required

- CONFIG-22: `host`, `dbname`, `user`; type: strings; default: required; meaning: the Postgres every pod shares
  - `port`; type: integer, 1 to 65535; default: required; meaning: the Postgres every pod shares
- CONFIG-23: `password_file`; type: absolute path; default: required; meaning: the file holding the password
- CONFIG-24: `write_seconds`; type: number above 0, or `inf`; default: 2; meaning: the bound on one write or read against a database that answers

- CONFIG-25: A connection that goes dark mid-write is given up only after the driver's own cancel wait, about five
  seconds more. A write runs inside the listen's deadline, so a bound from about 40 seconds on, or `inf`, lets
  a dark database hold that loop until the deadline and cancel the running turn, with no `store.failed`.

## `[ownership]`

- CONFIG-26: `contend_seconds`; default: 5; meaning: how often a pod takes free locks and confirms its own
- CONFIG-27: `heartbeat_timeout_seconds`; default: 3; meaning: how long one such round may take
- CONFIG-28: `cancel_margin_seconds`; default: 2; meaning: the time a pod that lost an agent has to stop him
- CONFIG-29: `grace_seconds`; default: 12; meaning: how long a pod that took an agent waits before it starts him
- CONFIG-30: `stop_seconds`; default: 10; meaning: how long after a SIGTERM a pod may take to stop before it ends at once

- CONFIG-31: Each is a finite number above 0. A grace at or below the first three together is refused, and so is a
  stop at or below `cancel_margin_seconds`, the time each agent's last write has. A handover takes at most
  `contend_seconds + heartbeat_timeout_seconds + grace_seconds`, 20 seconds by default, inside agents' member
  lease of 120 seconds.

- CONFIG-32: A stop that reaches `stop_seconds` is logged as `runner.stop_cut` and exits 5, whatever still hangs then;
  with `terminationGracePeriodSeconds` above it, Kubernetes sees that exit instead of killing the pod.

## `[memory]`

- CONFIG-33: Its presence keeps every line the agents read, sweeps them and gives every turn the lookup `recall`; removing it
  stops all three.

What then happens to lines kept before is in [the architecture](ARCHITECTURE.md),
Persistence.

- CONFIG-34: `lines_keep_days`; type: number above 0, or `inf`; default: 30; meaning: how long after its first read a line becomes due for the next sweep; `inf` keeps every line
- CONFIG-35: `sweep_minutes`; type: number above 0, or `inf`; default: 60; meaning: how often the sweep runs after its first at start; `inf` sweeps once
- CONFIG-36: `lookup_calls`; type: integer above 0; default: 2; meaning: lookups one turn may make; they join the turn's call total
- CONFIG-37: `lookup_bytes`; type: integer, at least an empty result's size; default: 8192; meaning: the bytes of one result and the longest `text` a lookup takes

## Connection budget

- CONFIG-38: A pod holds 1 lock connection, the history store's pool of 5 with an overflow of 10, and as many again for
  the line store with `[memory]`: 31 with a memory, 16 without. A rolling update with two replicas and a surge
  of one runs three pods. `personas migrate` holds 1 and each `personas memory` command 2. Set the server's
  `max_connections` above the sum of these and everything else that uses the database.
