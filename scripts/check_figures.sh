#!/usr/bin/env bash
# Check the figures under src/characters: scripts/check_figures.sh
#
# The generator's output must equal the tracked .vrm files, none more and none fewer. Then `figure check` from figure's
# release runs on each file, after the tarball's sha512 has matched the integrity its release notes print. Needs uv,
# node (24 or newer, as the release asks), curl, openssl and tar. Bump the version and the integrity together, to the
# release the apps' figure tile runs.
set -euo pipefail
# A directory without a figure is a failure, not an empty loop.
shopt -s failglob

VERSION='0.2.0'
INTEGRITY='sha512-4YGqA8NQbEEl6+Q3x8uj3p8ZHDW4mpUlxiEhnkux9upt8ESPh9cY7jFqNXYcTHmyqvAy80vFrhqf5Z13LYr9ug=='
TARBALL_URL="https://github.com/overnightworks/figure/releases/download/v${VERSION}/overnightworks-figure-${VERSION}.tgz"

cd "$(git rev-parse --show-toplevel)"
work="$(mktemp --directory)"
trap 'rm -rf "${work}"' EXIT

uv run --no-project --python 3.14 scripts/write_figures.py "${work}/generated"
if ! diff --recursive --brief --exclude='*.md' "${work}/generated" src/characters; then
    echo "src/characters/*.vrm differ from the generator's output: run scripts/write_figures.py src/characters" >&2
    exit 1
fi

untracked="$(git ls-files --others --exclude-standard src/characters)"
if [[ -n "${untracked}" ]]; then
    echo "untracked files under src/characters: ${untracked}" >&2
    exit 1
fi

curl --proto '=https' --proto-redir '=https' --fail --silent --show-error --location \
    --output "${work}/figure.tgz" "${TARBALL_URL}"
actual="sha512-$(openssl dgst -sha512 -binary "${work}/figure.tgz" | base64 --wrap=0)"
if [[ "${actual}" != "${INTEGRITY}" ]]; then
    echo "figure ${VERSION}: the tarball's integrity is ${actual}, not the pinned one" >&2
    exit 1
fi

mkdir "${work}/release"
tar --extract --gzip --file "${work}/figure.tgz" --directory "${work}/release"
for figure in src/characters/*.vrm; do
    node "${work}/release/package/dist/bin.js" check "${figure}"
done
