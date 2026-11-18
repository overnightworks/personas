# Runbook: release the runner

Task: a change on `main` should reach the deployment. Impact: the image is the runner's release, so a
release changes the runner, its characters and its figures together. The rules it follows are owned by
[quality's release path](https://github.com/overnightworks/quality/blob/v0.21.1/docs/release.md).

## Check

1. `main`'s CI is green on the commit to release.
2. A release whose agents bump changes the member-tool wire waits until every app the runner serves pins
   that agents version or a newer one. Read the agents release notes from the version in `pyproject.toml`
   at the newest `v<version>` tag up to the one on `main`. When one changes the wire, take the served apps
   from the deployment's runner configuration (`apps/personas/personas.toml` in platform) and read each
   app's `main` for its agents pin.

## Fix

1. Dispatch the `Release` workflow (`.github/workflows/release.yml`) on `main` with the version as
   `YYYY.MM.DD.N`, for example `2026.11.08.1`.
2. Expect quality's gate to refuse a commit whose checks are not green. Otherwise the image is pushed to
   GHCR with an attestation, and the tag `v<version>` is created with the figures `src/characters/*.vrm`
   as assets.
3. The release notes open with the configuration changes, written by
   `scripts/configuration_release_notes.sh`: each field of the runner's file added, removed or newly
   required since the newest tag the commit reaches, with the TOML it needs, or the sentence that none
   changed. They are derived from `src/personas/config.py` at both tags, so no change needs a note of
   its own, and they say what a `<kind>` placeholder asks for. The deployment's bump applies them all
   to its runner configuration.
4. Then come one `figure` line per agent, written by `scripts/figure_release_notes.sh`: paste it,
   without the name and its colon, into that agent's `[[agents]]` entry. An app that shows
   figures fetches them only from the prefix its platform allows (hopin accepts
   `https://github.com/overnightworks/personas/releases/download/`).

Rollback: none here; a pushed image and its tag stay, and the deployment in overnightworks/platform
chooses which release runs.

## Escalate

A gate refusal is fixed on `main` first and dispatched again. A served app that pins an older agents
version than a wire-changing bump needs: the release waits, and the operator hears which app holds it.
