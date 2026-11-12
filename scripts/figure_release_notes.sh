#!/usr/bin/env bash
# Print the release notes' figure lines: scripts/figure_release_notes.sh <version>
#
# One line per src/characters/*.vrm in name order, ready for the runner's TOML: the address the file has in the release
# v<version> and the sha256 of the file as checked out. The release workflow publishes those very files.
set -euo pipefail
shopt -s nullglob

RELEASE_URL='https://github.com/overnightworks/personas/releases/download'

if [[ "$#" -ne 1 || -z "$1" ]]; then
    echo "usage: $0 <version>: the release's version is missing" >&2
    exit 2
fi
version="$1"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
figures=(src/characters/*.vrm)
# A directory without a figure is a failure, not an empty loop.
if [[ "${#figures[@]}" -eq 0 ]]; then
    echo "no figure file under src/characters" >&2
    exit 1
fi

echo "## Agent figures"
echo
echo "Each line is an agent's \`figure\` for the runner's configuration (its \`[agents.figure]\` table):"
echo
for figure in "${figures[@]}"; do
    name="$(basename "${figure}" .vrm)"
    digest="$(sha256sum "${figure}")"
    echo "${name}: figure = { url = \"${RELEASE_URL}/v${version}/${name}.vrm\", sha256 = \"${digest%% *}\" }"
done
