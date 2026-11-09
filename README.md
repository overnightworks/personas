# Agents

The home of the operator's AI agents: each with its own name, picture, personality and memory, taking part in the operator's apps the way a person does.

For the operator, who gives each agent its name and rights, and for the people who meet his agents in his apps.

## What runs now

The runner. Each agent signs in to the realm as itself, with its own client, and for every app it may reach it keeps one subscription open on the app's MCP endpoint and reads its standing whenever the app says something changed. When it is invited to a call it accepts, and it appears there under the name the realm gives it, with the AI mark. It never leaves on its own; it is removed with its seat, its lease or its client. It takes no model turn yet and keeps no memory.

Which apps an agent can use is the realm's group membership, not this configuration. When the realm or an app refuses an agent (a disabled client, a missing group), that agent's loop for that app is suspended and logged once, and it resumes when the refusal ends, without a restart.

## Run it

Install with `uv sync`, point `AGENTS_CONFIG` at a TOML file and start `agents`. The file holds the realm's token endpoint, the absolute path `ready_file`, the apps (MCP URL and scope) and the agents (client id and the path of the file holding its client secret). `src/agents/config.py` owns its fields and what is refused at start; the runner stops with the field named, never a secret.

The runner opens no port. It removes `ready_file` at start, so a marker left by an earlier run never counts, and creates it the first time any loop's read of its standing answers: the start worked, the realm gave a token and an app accepted it. A refused token, a suspended loop, a network failure or a 5xx does not create it, so a wrong secret or URL shows as not ready. The marker stays afterwards; a later suspension or outage is in the log. If the file cannot be removed or created, the runner exits non-zero naming the path. The platform's readiness probe is `test -e <ready_file>`.

The commands for working on the runner are in [AGENTS.md](AGENTS.md).

## Release it

The image is the runner's release. The operator dispatches the `Release` workflow (`.github/workflows/release.yml`) on `main` with the version as `YYYY.MM.DD.N`, for example `2026.11.08.1`. Quality's gate refuses a commit whose checks are not green, the image is pushed to GHCR with an attestation, and then the tag `v<version>` is created. The image holds no configuration; `AGENTS_CONFIG` and the secret files come from the deployment. An AI agent dispatches a release only on the operator's explicit yes for that one version. [quality's release path](https://github.com/overnightworks/quality/blob/v0.10.0/docs/release.md) owns the rules.

## Read on

- [Vision](docs/VISION.md): what agents is for and what it deliberately is not.
- [Architecture](docs/ARCHITECTURE.md): what the repository holds today.
- [AGENTS.md](AGENTS.md): guidance for the AI agents that work here.
