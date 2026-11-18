# Configuration reference

The runner reads one TOML file named by `PERSONAS_CONFIG`. `src/personas/config.py` owns every field and
refusal; every value is checked at start, an unknown key is refused, and an error names the field or path,
never a secret (CONFIG-1). What the stores keep is in [the architecture](ARCHITECTURE.md), Persistence. A test
cites each rule by its ID at the start of its docstring, such as `CONFIG-1`. Every bullet and table row is a
rule; prose without an ID is context.

## Top level

| Name | Type | Default | Meaning |
|---|---|---|---|
| `token_endpoint` | URL | required | the realm's token endpoint (CONFIG-2) |
| `ready_file` | absolute path | required | the readiness marker (CONFIG-3) |
| `history_bytes` | integer above 0 | 49152 | the history each agent keeps and tells every turn, counted in the JSON bytes of each entry as told (CONFIG-4) |
| `history_keep_bytes` | integer above 0, at most `history_bytes` | 32768 | what stays after a cut; see [the turns reference](turns.md), What a turn is told (CONFIG-5) |
| `budget_pause_cap_seconds` | number, at least 1 | 300 | the longest pause between his turns while the gateway refuses for a spent budget (CONFIG-6) |

At the default history, a skat game (40 to 50 states of about 1.5 KB) fills it after about half a game.

## `[[apps]]`

| Name | Type | Default | Meaning |
|---|---|---|---|
| `mcp_url` | URL, once per file | required | the app's MCP endpoint; it names the app in the history and the lines (CONFIG-7) |
| `scope` | string | required | the token scope for this app (CONFIG-8) |
| `max_tool_calls` | integer above 0 | none | binds every tool the app offers to its agents' turns, that many calls per turn (CONFIG-9) |
| `listens` | boolean | true | `false` serves tools only: no loop runs, its tools join the turns of the other apps; needs `max_tool_calls` (CONFIG-10) |
| `call_seconds` | number above 0 | 30 | bounds each request a turn of another app makes to it; keep it below the persona's `maxDurationSeconds` (CONFIG-11) |

At least one app listens. Apps are declared to a turn in the file's order, so list the apps whose places
come and go last. An app with `max_tool_calls` requires `[gateway]` and a `persona_file` for every agent. (CONFIG-12)

## `[[agents]]`

| Name | Type | Default | Meaning |
|---|---|---|---|
| `client_id` | string, once per file | required | the agent's realm client; two whose lock keys collide are refused (CONFIG-13) |
| `secret_file` | absolute path | required | the file holding its client secret (CONFIG-14) |
| `persona_file` | absolute path | none | its agent file `<name>.md`; the image holds them at `/app/characters/` (CONFIG-15) |
| `model` | `text/<model>` | the agent file's | the gateway model his turns run on; needs `persona_file` (CONFIG-16) |
| `hold_model` | `text/<model>` | the agent's model | the model of the round that decides a hold; needs `persona_file` (CONFIG-17) |
| `[agents.figure]` | `url` (absolute `https`, no user, query or fragment), `sha256` (64 lowercase hex) | none | carried by every accept; an app that shows figures and allows the address shows him with it (CONFIG-18) |
| `[agents.voice]` | table, optional `names`: 1 to 8 distinct voice ids, first choice first | none | turns spoken turns on; every accept carries `names` as `voices` (CONFIG-19) |

The agent file names no tools and no call bound, offers the role `user`, and has no `voice` section. The
token's `agent_name` claim is the name a line must carry to address the agent. A change of a figure or a
model needs a restart; the runner fetches and checks nothing about a figure's address or bytes. (CONFIG-20)

## `[gateway]`

The table holds the fields of the agents library's `AgentConfig` that reach the model gateway: `api_group`
and `[gateway.gateway]` with `origin`, `key_file` and, for HTTPS, `ca_file`. It is required
when an app sets `max_tool_calls`. What an agent may spend is bounded at the gateway's budget; the runner
sets no limit of its own. (CONFIG-21)

## `[database]`, required

| Name | Type | Default | Meaning |
|---|---|---|---|
| `host`, `dbname`, `user` | strings | required | the Postgres every pod shares (CONFIG-22) |
| `port` | integer, 1 to 65535 | required | the Postgres every pod shares (CONFIG-22) |
| `password_file` | absolute path | required | the file holding the password (CONFIG-23) |
| `write_seconds` | number above 0, or `inf` | 2 | the bound on one write or read against a database that answers (CONFIG-24) |

A connection that goes dark mid-write is given up only after the driver's own cancel wait, about five
seconds more. A write runs inside the listen's deadline, so a bound from about 40 seconds on, or `inf`, lets
a dark database hold that loop until the deadline and cancel the running turn, with no `store.failed`. (CONFIG-25)

## `[ownership]`

| Name | Default | Meaning |
|---|---|---|
| `contend_seconds` | 5 | how often a pod takes free locks and confirms its own (CONFIG-26) |
| `heartbeat_timeout_seconds` | 3 | how long one such round may take (CONFIG-27) |
| `cancel_margin_seconds` | 2 | the time a pod that lost an agent has to stop him (CONFIG-28) |
| `grace_seconds` | 12 | how long a pod that took an agent waits before it starts him (CONFIG-29) |
| `stop_seconds` | 10 | how long after a SIGTERM a pod may take to stop before it ends at once (CONFIG-30) |

Each is a finite number above 0. A grace at or below the first three together is refused, and so is a
stop at or below `cancel_margin_seconds`, the time each agent's last write has. A handover takes at most
`contend_seconds + heartbeat_timeout_seconds + grace_seconds`, 20 seconds by default, inside agents' member
lease of 120 seconds. (CONFIG-31)

A stop that reaches `stop_seconds` is logged as `runner.stop_cut` and exits 5, whatever still hangs then;
with `terminationGracePeriodSeconds` above it, Kubernetes sees that exit instead of killing the pod. (CONFIG-32)

## `[memory]`

Its presence keeps every line the agents read, sweeps them and gives every turn the lookup `recall`; removing it
stops all three (CONFIG-33). What then happens to lines kept before is in [the architecture](ARCHITECTURE.md),
Persistence.

| Name | Type | Default | Meaning |
|---|---|---|---|
| `lines_keep_days` | number above 0, or `inf` | 30 | how long after its first read a line becomes due for the next sweep; `inf` keeps every line (CONFIG-34) |
| `sweep_minutes` | number above 0, or `inf` | 60 | how often the sweep runs after its first at start; `inf` sweeps once (CONFIG-35) |
| `lookup_calls` | integer above 0 | 2 | lookups one turn may make; they join the turn's call total (CONFIG-36) |
| `lookup_bytes` | integer, at least an empty result's size | 8192 | the bytes of one result and the longest `text` a lookup takes (CONFIG-37) |

## Connection budget

A pod holds 1 lock connection, the history store's pool of 5 with an overflow of 10, and as many again for
the line store with `[memory]`: 31 with a memory, 16 without. A rolling update with two replicas and a surge
of one runs three pods. `personas migrate` holds 1 and each `personas memory` command 2. Set the server's
`max_connections` above the sum of these and everything else that uses the database. (CONFIG-38)
