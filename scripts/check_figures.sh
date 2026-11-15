#!/usr/bin/env bash
# Check the figures and rooms under src/characters: scripts/check_figures.sh
#
# The generator's output must equal the tracked .vrm and .room.glb files, none more and none fewer. Then `figure check`
# from figure's release runs on each figure, and `figure check --as room` on each room, after the tarball's sha512 has
# matched the integrity its release notes print. Needs uv, node (24 or newer, as the release asks), curl, openssl and
# tar. Bump the version and the integrity together, to the release the apps' figure tile runs.
set -euo pipefail
# A directory without a figure or a room is a failure, not an empty loop.
shopt -s failglob

VERSION='0.8.0'
INTEGRITY='sha512-DwM8bSy/QiFjJgif7jCJ2XGHg/9Fuag72auwn780wmaXanKpWNeLC8ixLz6lxObw6KLu0aXThrEpg/Zx6zRqlQ=='
TARBALL_URL="https://github.com/overnightworks/figure/releases/download/v${VERSION}/overnightworks-figure-${VERSION}.tgz"

cd "$(git rev-parse --show-toplevel)"
work="$(mktemp --directory)"
trap 'rm -rf "${work}"' EXIT

uv run --no-project --python 3.14 scripts/write_figures.py "${work}/generated"
if ! diff --recursive --brief --exclude='*.md' "${work}/generated" src/characters; then
    echo "src/characters/*.vrm and *.room.glb differ from the generator's output: run scripts/write_figures.py src/characters" >&2
    exit 1
fi

untracked="$(git ls-files --others src/characters)"
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
for room in src/characters/*.room.glb; do
    node "${work}/release/package/dist/bin.js" check --as room "${room}"
done
