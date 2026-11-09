# Agents - Agent Guidance

This file holds the repository's own facts on top of the global contract.
`CLAUDE.md` imports it.

## State

The repository holds the runner (`src/agents/`, console script `agents`) and
its documents. Python 3.14 with uv: `pyproject.toml` and `uv.lock` own the
dependencies, hatchling is pinned as in skat, and `uv sync` installs.

## Commands

- Lint: `uv run ruff check .` and `uv run ruff format --check src tests`.
- Types (strict): `uv run pyright`. Dead code: `uv run vulture`.
- Tests: name the module, as in `uv run pytest tests/test_loop.py --no-cov`.
  Async tests carry `pytestmark = pytest.mark.anyio`. Coverage is branch
  coverage over `src`, set in `pyproject.toml`.
- Gate rules: `quality check` at the release `ci.yml` pins.

## Where things live

- `docs/VISION.md` is the operator-blessed vision text. Change its words only
  with his yes.
- `src/agents/decision.py` decides everything that needs no process, network
  or clock; `loop.py` performs it through the boundaries `tokens.py`,
  `app_session.py`, `clock.py` and `reporting.py` and the model gateway
  (`Boundaries.backend`); `config.py` owns the TOML.
  Extend the owner of a question, never a second decider.
- A secret or token lives in `secret.Secret` and reaches no log line, error
  message, repr or test literal; tests generate theirs.
- The persona and memory shape is not ruled yet. `docs/ARCHITECTURE.md` states
  only what exists.

## Checks

- `Documents` (`.github/workflows/ci.yml`) runs lychee offline over every
  tracked Markdown file and fails on a relative link or anchor that points
  nowhere. Its arguments live in the workflow, so the root holds no lychee
  config.
- `Lint and test` runs ruff, pyright, vulture and pytest with coverage over
  `src`, installing only what `uv.lock` resolved, and uploads `coverage.xml`
  for Sonar.
- `Image` builds the `Dockerfile` on every pull request and push without
  pushing it, and checks that the image refuses to start with no
  `AGENTS_CONFIG`. `release.yml` is the only path that pushes an image.
- `quality/config.toml` configures the shared gate: `mandatory_jobs` names `Documents`,
  `Lint and test` and `Image`, and there is no `legacy` key. The required check is
  `quality / verdict`; overnightworks/quality's README owns the gate's semantics. A new check
  joins `mandatory_jobs` with its job.
- The root holds only what the gate's `root-layout` rule allows; a new root
  entry needs a `[[root.allow]]` entry with its reason.
- `sonar.sources` names `src/agents`, `.github` whole and `Dockerfile`; add each directory of
  code with the slice that brings it.
