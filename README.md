# Personas

The home of the operator's AI agents: each with its own name, picture, personality and memory, taking part in the operator's apps the way a person does.

For the operator, who gives each agent its name and rights, and for the people who meet his agents in his apps.

## What runs now

The runner. Each agent signs in to the realm as itself, with its own client, and for every app it may reach it keeps one subscription open on the app's MCP endpoint and reads its standing whenever the app says something changed. When it is invited to a call it accepts, and it appears there under the name the realm gives it, with the AI mark. It never leaves on its own; it is removed with its seat, its lease or its client. In a call it reads every line from its join on, takes what was said before a restart or during a pause as read, and answers once a line said after names it, as a whole word in any case, with one model turn through the operator's LiteLLM gateway; it never takes its own line as addressed to it and keeps no memory.

Which apps an agent can use is the realm's group membership, not this configuration. When the realm or an app refuses an agent (a disabled client, a missing group), that agent's loop for that app is suspended and logged once, and it resumes when the refusal ends, without a restart.

## Run it

Install with `uv sync`, point `PERSONAS_CONFIG` at a TOML file and start `personas`. The file holds the realm's token endpoint, the absolute path `ready_file`, the apps (MCP URL and scope) and the agents (client id and the path of the file holding its client secret). An app that lets its agents answer also lists the tools their turns may call as `tools`, with `max_tool_calls`, the most calls one turn makes in that app; the two are set together. An app that serves tools only sets `listens = false` and `tools`: no loop runs for it, and its tools join the turns of the other apps, each request in a session of its own under a token for its own scope. Its `call_seconds` (default 30) bounds each such request, a relayed call or an opening, so keep it below the persona's `maxDurationSeconds`; nothing refuses a larger value. A tool name may not be shared with a tools-only app, and at least one app must listen. Then the file needs the `[gateway]` table, which is agent-runtime's `AgentConfig` (`backend`, `api_group` and `[gateway.gateway]` with `origin`, `key_file` and, for HTTPS, `ca_file`), and every agent a `persona_file`: the absolute path of its agent file `<name>.md`, whose prompt is the agent's character, and, only when the operator wants one, the turn-rate cap `max_turns` (a positive integer) and `turn_period_seconds` (a positive number), set together: at most `max_turns` turns start in any `turn_period_seconds`, counted across all the apps of the agent and from the runner's start. A line addressed to the agent over the cap is skipped, logged once as `capped` and never answered later. The cap is off unless both fields are set; the person's budget at the gateway and the off switches (the invitation, its removal and the person's permission to be heard) bound an agent then. The agent file names no tools and no call bound and offers the role `user`, under which a turn runs. The token's `agent_name` claim is the name a line must carry to address the agent. `src/personas/config.py` owns its fields and what is refused at start; the runner stops with the field named, never a secret.

The runner opens no port. It removes `ready_file` at start, so a marker left by an earlier run never counts, and creates it the first time any loop's read of its standing answers: the start worked, the realm gave a token and an app accepted it. A refused token, a suspended loop, a network failure or a 5xx does not create it, so a wrong secret or URL shows as not ready. The marker stays afterwards; a later suspension or outage is in the log. If the file cannot be removed or created, the runner exits non-zero naming the path. The platform's readiness probe is `test -e <ready_file>`.

The runner stops on SIGTERM with exit code 0 and needs no init, in the image or the deployment. The deployment supplies the configuration, the persona files included, and the secret files; the image holds none of them.

The commands for working on the runner are in [AGENTS.md](AGENTS.md).

## Release it

The image is the runner's release. The operator dispatches the `Release` workflow (`.github/workflows/release.yml`) on `main` with the version as `YYYY.MM.DD.N`, for example `2026.11.08.1`. Quality's gate refuses a commit whose checks are not green, the image is pushed to GHCR with an attestation, and then the tag `v<version>` is created. The image holds no configuration; `PERSONAS_CONFIG` and the secret files come from the deployment. An AI agent dispatches a release only on the operator's explicit yes for that one version. [quality's release path](https://github.com/overnightworks/quality/blob/v0.10.0/docs/release.md) owns the rules.

## Read on

- [Vision](docs/VISION.md): what personas is for and what it deliberately is not.
- [Architecture](docs/ARCHITECTURE.md): what the repository holds today.
- [AGENTS.md](AGENTS.md): guidance for the AI agents that work here.
