#!/usr/bin/env bash
# Run a command against a throwaway Postgres: scripts/test_postgres.sh <command> [argument...]
#
# The command finds the server in PERSONAS_TEST_POSTGRES_URL. The container is removed when the command
# ends, is stopped, or the caller dies. Keep the image equal to the one CI's service runs.
set -euo pipefail

IMAGE='postgres:18.6-alpine3.24@sha256:123b17998ccaf47e53b502e8c6a1e79afd8f579c6f3eec83c563da026f957395'
# The cap and its 30 s grace are the repository-wide rule; the container stops within the grace.
CAP_SECONDS=1800
GRACE_SECONDS=30
LOAD_WAIT_SECONDS=1800
READY_WAIT_SECONDS=60

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 <command> [argument...]" >&2
    exit 2
fi

# Re-run this script under the cap, so that the trap below also runs when the cap or the caller's death ends it.
if [[ -z "${PERSONAS_TEST_POSTGRES_CAPPED:-}" ]]; then
    PERSONAS_TEST_POSTGRES_CAPPED=1 exec setpriv --pdeathsig TERM timeout -k "${GRACE_SECONDS}" "${CAP_SECONDS}" "$0" "$@"
fi

wait_for_low_load() {
    local limit waited=0 load
    limit=$(awk -v cores="$(nproc)" 'BEGIN { print 1.5 * cores }')
    while load=$(cut -d' ' -f1 /proc/loadavg) && awk -v load="${load}" -v limit="${limit}" 'BEGIN { exit !(load > limit) }'; do
        if [[ "${waited}" -ge "${LOAD_WAIT_SECONDS}" ]]; then
            echo "load ${load} stayed above ${limit}" >&2
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
}

wait_for_low_load

container="personas-test-postgres-$$"
trap 'docker rm --force "${container}" >/dev/null 2>&1 || true' EXIT TERM
docker run --quiet --detach --name "${container}" \
    --env POSTGRES_HOST_AUTH_METHOD=trust \
    --publish 127.0.0.1::5432 \
    --tmpfs /var/lib/postgresql \
    "${IMAGE}" >/dev/null

for _ in $(seq "${READY_WAIT_SECONDS}"); do
    docker exec "${container}" pg_isready --username postgres >/dev/null 2>&1 && break
    sleep 1
done
docker exec "${container}" pg_isready --username postgres >/dev/null

port=$(docker port "${container}" 5432/tcp | head -n 1 | sed 's/.*://')
# Run as a child, not exec'd, so that the trap above survives the command.
PERSONAS_TEST_POSTGRES_URL="postgresql+psycopg://postgres@127.0.0.1:${port}/postgres" "$@"
