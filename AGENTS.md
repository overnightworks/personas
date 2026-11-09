# personas

## Commands

- `uv sync`: installs the locked dependencies; run once in a new worktree, then use `uv run --no-sync`.
- `uv run ruff check .` and `uv run ruff format --check src tests`: lint and format.
- `uv run pyright`: strict types.
- `uv run vulture`: dead code.
- `uv run pytest tests/<module>.py --no-cov`: one test module; never the whole suite.
- `uvx --python 3.14 --from "overnightworks-quality @ git+https://github.com/overnightworks/quality@<sha>" quality check`: the shared gate's rules, at the sha `.github/workflows/ci.yml` pins on `uses: overnightworks/quality/...`; run after committing, because it reads tracked files only.

## Rules

- A decision that needs no process, network or clock goes into `src/personas/decision.py`; `loop.py` performs it through the boundaries, and `config.py` alone owns the TOML's fields and refusals. Extend the owner of a question, never a second decider.
- A secret or token is held in `secret.Secret` and reaches no log line, error message, repr or test literal; tests generate theirs.
- Every `except asyncio.CancelledError` re-raises (Sonar S7497); decide what follows in a `finally` or outside the handler.
- Async tests carry `pytestmark = pytest.mark.anyio`.
- A test module's name starts, at a word boundary, with the name of the `src/personas` module it tests (quality's `test-layout` rule).
- A new CI job that must pass joins `mandatory_jobs` in `quality/config.toml`.
- A new root entry needs a `[[root.allow]]` entry with its reason in `quality/config.toml`.
- A new directory of code joins `sonar.sources` in `sonar-project.properties` in the change that brings it.
- lychee's arguments live in `.github/workflows/ci.yml`; the root holds no lychee config.
- Only `.github/workflows/release.yml` pushes an image.

## Owners

- The operator: the words of `docs/VISION.md`, and a yes for each release version before an agent dispatches `release.yml`.
- overnightworks/agent-runtime: the client library and its wire contract; personas takes its releases.
- overnightworks/quality: the shared gate (`quality / verdict`) and its rules.
- overnightworks/hal: the realm and the model gateway.
- overnightworks/platform: the runner's deployment.

## Documents

- `README.md`: read when configuring, starting or releasing the runner.
- `docs/ARCHITECTURE.md`: read when changing the runner's parts, their boundaries or the checks.
- `docs/VISION.md`: read when judging whether a change serves what personas is for.
