# Personas

The home of the operator's AI agents: each with its own name, picture, personality and memory, taking part
in the operator's apps the way a person does.

For the operator, who gives each agent its name and rights, and for the people who meet his agents in his
apps. What exists is the runner: it signs each agent in to the realm as itself, keeps it listening in the
apps it may reach, accepts their invitations, and lets it answer in a call, in writing or aloud, and act at
a table, one model turn at a time through the operator's LiteLLM gateway.

## Quick start

1. `uv sync` installs the locked dependencies.
2. Write the TOML file the [configuration reference](docs/configuration.md) describes and point
   `PERSONAS_CONFIG` at it, with the secret files it names.
3. `personas migrate` brings the Postgres schema to its head. Run it before the runner starts and after
   every release; it leaves a schema a newer release migrated as it is and exits 0.
4. `personas` starts the runner, which creates `ready_file` once its pod has contended for its agents. How
   the platform probes it and how the runner stops are in [the architecture](docs/ARCHITECTURE.md), Deployment View.

The marker says the pod is contending, not that an agent works: a wrong secret or URL shows in the
[log](docs/log.md) as `loop.suspended` or `loop.backed_off`. What a turn is told and may do is in the
[turns reference](docs/turns.md).

## What it keeps

What the runner keeps in Postgres, for how long and what a restart or a takeover resumes is stated once, in
[the architecture](docs/ARCHITECTURE.md), Persistence. To show or erase it, follow
[the runbook](docs/runbooks/erase-memory.md).

## Characters and figures

Each agent's character is its agent file `src/characters/<name>.md`, which the image holds at
`/app/characters/<name>.md`, so a character changes with a release. Its figure `<name>.vrm` and its room
`<name>.room.glb` are described in [the architecture](docs/ARCHITECTURE.md), Building Block View. An
agent's `[agents.figure]` and `[agents.room]` tables name the released figure and room that every accept carries; a
release publishes both beside each other.

## Release it

The image is the runner's release; [the release runbook](docs/runbooks/release.md) says how to cut one.

## Documentation

- [Vision](docs/VISION.md): what personas is for and what it deliberately is not.
- [Architecture](docs/ARCHITECTURE.md): the parts, who talks to whom, what is kept, the trust.
- [Configuration](docs/configuration.md), [turns](docs/turns.md) and [log](docs/log.md): the references.
- Runbooks: [showing and erasing what an agent keeps](docs/runbooks/erase-memory.md) and
  [releasing the runner](docs/runbooks/release.md).
- [AGENTS.md](AGENTS.md): the commands and rules for working on the runner.

## License

MIT, see [LICENSE](LICENSE).
