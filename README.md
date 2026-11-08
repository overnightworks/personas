# Agents

The home of the operator's AI agents: each with its own name, picture, personality and memory, taking part in the operator's apps the way a person does.

For the operator, who gives each agent its name and rights, and for the people who meet his agents in his apps.

## What runs now

The runner. Each agent signs in to the realm as itself, with its own client, and for every app it may reach it keeps one `wait` in flight on the app's MCP endpoint. When it is invited to a call it accepts, and it appears there under the name the realm gives it, with the AI mark. It never leaves on its own; it is removed with its seat, its lease or its client. It takes no model turn yet and keeps no memory.

Which apps an agent can use is the realm's group membership, not this configuration. When the realm or an app refuses an agent (a disabled client, a missing group), that agent's loop for that app is suspended and logged once, and it resumes when the refusal ends, without a restart.

## Run it

Install with `uv sync`, point `AGENTS_CONFIG` at a TOML file and start `agents`. The file holds the realm's token endpoint, the apps (MCP URL and scope) and the agents (client id and the path of the file holding its client secret). `src/agents/config.py` owns its fields and what is refused at start; the runner stops with the field named, never a secret.

The commands for working on the runner are in [AGENTS.md](AGENTS.md).

## Read on

- [Vision](docs/VISION.md): what agents is for and what it deliberately is not.
- [Architecture](docs/ARCHITECTURE.md): what the repository holds today.
- [AGENTS.md](AGENTS.md): guidance for the AI agents that work here.
