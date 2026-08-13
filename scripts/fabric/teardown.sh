#!/usr/bin/env bash
# Stop every local Fabric process and (optionally) delete the generated state.
#
# Usage:
#   bash scripts/fabric/teardown.sh          # stop processes, keep the ledger
#   bash scripts/fabric/teardown.sh --purge  # also delete .fabric/ entirely
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$REPO/.fabric"

# SIGTERM and then *wait*. A previous teardown only signalled and slept, so an
# orderer that had not finished shutting down was still writing while the next
# setup recreated its ledger directory — the new network then died with
# "unexpected Previous block hash", which reads like ledger corruption rather
# than a shutdown race.
stop() {
  local name="$1" pid="$2"
  kill "$pid" 2>/dev/null || return 0
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || { echo "stopped $name (pid $pid)"; return 0; }
    sleep 0.5
  done
  kill -9 "$pid" 2>/dev/null && echo "force-killed $name (pid $pid)"
  sleep 1
}

for name in gateway chaincode peer orderer; do
  pidfile="$RUN/$name.pid"
  if [ -f "$pidfile" ]; then
    stop "$name" "$(cat "$pidfile")"
    rm -f "$pidfile"
  fi
done

# Belt and braces: anything still holding the ports would break a fresh setup.
pkill -f "privchain-cc-server" 2>/dev/null && echo "stopped stray chaincode" || true
pkill -f "privchain-gateway" 2>/dev/null && echo "stopped stray gateway" || true
pkill -f "peer node start" 2>/dev/null && echo "stopped stray peer" || true
pkill -x orderer 2>/dev/null && echo "stopped stray orderer" || true

# Do not return until the ports are actually free, or the next setup races the
# processes it is replacing.
for _ in $(seq 1 30); do
  busy=""
  for port in 7050 7051 7052 8801 9999; do
    (echo > "/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1 && busy="$busy $port"
  done
  [ -z "$busy" ] && break
  sleep 1
done
[ -n "${busy:-}" ] && echo "warning: ports still in use:$busy"

if [ "${1:-}" = "--purge" ]; then
  rm -rf "$RUN"
  echo "purged $RUN (crypto material, ledger data and logs)"
fi

echo "TEARDOWN_OK"
