#!/usr/bin/env bash
# Print the release notes' configuration lines: scripts/configuration_release_notes.sh <version>
#
# Compares the configuration file's schema at the newest tag the checkout reaches with the checkout's own, each read
# by that tree's personas.config in that tree's locked environment, and prints what a deployer must change. The
# checkout's environment is the one `uv sync` installed; the tag's is installed into a temporary directory the same
# way CI installs, wheels only except the overnightworks libraries.
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9] ]]; then
    echo "usage: $0 <version>: the release's version, without a v" >&2
    exit 2
fi
version="$1"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
renderer="${PWD}/scripts/configuration_release_notes.py"
previous="$(git describe --tags --abbrev=0 --match 'v[0-9]*' HEAD)"

work="$(mktemp -d)"
trap 'rm -rf "${work:?}"' EXIT
mkdir "${work}/before"
git archive "${previous}" | tar -x -C "${work}/before"
env -u UV_PROJECT_ENVIRONMENT -u VIRTUAL_ENV \
    uv sync --quiet --project "${work}/before" --locked --no-install-project --no-build --no-dev
PYTHONPATH="${work}/before/src" "${work}/before/.venv/bin/python" "${renderer}" schema > "${work}/before.json"
PYTHONPATH="${PWD}/src" uv run --no-sync python "${renderer}" notes "${previous}" "${version}" < "${work}/before.json"
