# Runbook: release the runner

Task: a change on `main` should reach the deployment. Impact: the image is the runner's release, so a
release changes the runner, its characters and its figures together. The rules it follows are owned by
[quality's release path](https://github.com/overnightworks/quality/blob/v0.21.1/docs/release.md).

## Check

1. `main`'s CI is green on the commit to release.
2. A release waits when its agents bump changes or removes a tool, record or field that a served app
   uses, or adds a tool that `docs/calling.md` requires an app to serve, until every such app pins that
   agents version or a newer one. A tool that `docs/calling.md` lets an app leave unserved does not hold
   a release, since that app answers "unknown tool". A record or field that a release only adds holds
   nothing, an optional argument of a tool included, since the runner leaves out an optional argument the app's
   declared schema does not name and logs `argument.left_out`: the added-tool clause is the only addition that
   holds. A removed, renamed or newly required argument still holds. Read each agents release's `Member-tool wire:` line,
   from the version in `pyproject.toml` at the newest `v<version>` tag up to the one on `main`. Its names
   are changes, removals or additions: a name is an addition when `docs/calling.md` at the previous
   release does not list it (compare the two files), and otherwise a change or removal. A line that says
   `unchanged.` holds nothing. A line that says `changed` and names nothing (its change is written in
   prose) is read by hand: it holds when a kept app uses the change its prose describes, judged against
   the previous release's notes. For each name that holds, keep the served apps from the deployment's
   runner configuration (`apps/personas/personas.toml` in platform) that use the changed or removed
   tool, record or field, or every served app for an added tool, then read each kept app's `main` for
   its agents pin.

## Fix

1. Dispatch the `Release` workflow (`.github/workflows/release.yml`) on `main` with the version as
   `YYYY.MM.DD.N`, for example `2026.11.08.1`.
2. Expect quality's gate to refuse a commit whose checks are not green. Otherwise the image is pushed to
   GHCR with an attestation, and the tag `v<version>` is created with the figures `src/characters/*.vrm`
   as assets.
3. The release notes open with the configuration changes, written by
   `scripts/configuration_release_notes.sh`: each field of the runner's file added, removed, newly
   required or changed in what it takes (its type, its bounds, a map's key or value type) or in its
   default, a table's included, since the newest tag the commit reaches, with the TOML it needs and
   where that goes in the file as it is, and each new rule across fields, or the sentence that none
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
version than a bump that holds a release under Check 2 (a change or removal of a tool, record or field
the app uses, or an added tool `docs/calling.md` requires it to serve) needs: the release waits, and
the operator hears which app holds it.
