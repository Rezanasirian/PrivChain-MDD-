#!/usr/bin/env bash
# Prove the ledger is real, not a cache, and that the chaincode enforces the
# invariants MockLedger has been promising (Phase 5 DoD, ADR-0022).
#
# Three things are checked, in order of how easily they could be faked:
#   1. a write is readable back;
#   2. it survives a **peer restart** — that is what distinguishes committed
#      ledger state from an in-memory cache;
#   3. the append-only and immutability rules are enforced by the real chaincode,
#      not merely by the Python mock.
#
# Usage:  bash scripts/fabric/verify_ledger.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FABRIC_HOME="${FABRIC_HOME:-/workspace/fabric}"
RUN="$REPO/.fabric"
CRYPTO="$RUN/crypto-config"
CHANNEL="${CHANNEL:-privchain-channel}"
CC_NAME="${CC_NAME:-privchain-cc}"

export PATH="$FABRIC_HOME/bin:$PATH"
ORDERER_HOME="$CRYPTO/ordererOrganizations/privchain.local/orderers/orderer.privchain.local"
PEER_HOME="$CRYPTO/peerOrganizations/org1.privchain.local/peers/peer0.org1.privchain.local"
ADMIN_MSP="$CRYPTO/peerOrganizations/org1.privchain.local/users/Admin@org1.privchain.local/msp"

peer_as_admin() {
  env FABRIC_CFG_PATH="$RUN/peercfg" \
    CORE_PEER_ADDRESS=localhost:7051 CORE_PEER_LOCALMSPID=Org1MSP \
    CORE_PEER_MSPCONFIGPATH="$ADMIN_MSP" CORE_PEER_TLS_ENABLED=true \
    CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" peer "$@"
}
invoke() {
  peer_as_admin chaincode invoke -o 127.0.0.1:7050 \
    --ordererTLSHostnameOverride orderer.privchain.local \
    --tls --cafile "$ORDERER_HOME/tls/ca.crt" \
    --channelID "$CHANNEL" --name "$CC_NAME" -c "$1" --waitForEvent 2>&1
}
query() {
  peer_as_admin chaincode query --channelID "$CHANNEL" --name "$CC_NAME" -c "$1" 2>&1
}
log() { printf '\n=== %s ===\n' "$1"; }
fail() { echo "VERIFY_FAILED: $1" >&2; exit 1; }

# Subgraph rounds are immutable, so a fixed round number makes this script
# single-use against any given ledger. Derive both from the PID instead.
CID="verify-client-$$"
ROUND="$(( (($$ % 9000) + 1000) ))"

# ── 1. A write is readable back ──────────────────────────────────────────────
log "RegisterClient $CID"
invoke "{\"Args\":[\"RegisterClient\",\"$CID\",\"1\",\"0\",\"1\"]}" | tail -1
log "GetClient $CID"
BEFORE="$(query "{\"Args\":[\"GetClient\",\"$CID\"]}")"
echo "$BEFORE"
echo "$BEFORE" | grep -q "$CID" || fail "the registered client did not read back"

log "LogPrivacyBudget round 1"
invoke "{\"Args\":[\"LogPrivacyBudget\",\"$CID\",\"audio\",\"1\",\"0.5\"]}" | tail -1

# ── 2. It survives a peer restart ────────────────────────────────────────────
# A cache does not. This is the step that makes the claim "real ledger" rather
# than "the process remembered".
log "restarting the peer"
kill "$(cat "$RUN/peer.pid")" 2>/dev/null || true
sleep 4
env FABRIC_CFG_PATH="$RUN/peercfg" \
  CORE_PEER_ID=peer0.org1.privchain.local \
  CORE_PEER_ADDRESS=localhost:7051 CORE_PEER_LISTENADDRESS=0.0.0.0:7051 \
  CORE_PEER_CHAINCODEADDRESS=localhost:7052 CORE_PEER_CHAINCODELISTENADDRESS=0.0.0.0:7052 \
  CORE_PEER_LOCALMSPID=Org1MSP CORE_PEER_MSPCONFIGPATH="$PEER_HOME/msp" \
  CORE_PEER_TLS_ENABLED=true \
  CORE_PEER_TLS_CERT_FILE="$PEER_HOME/tls/server.crt" \
  CORE_PEER_TLS_KEY_FILE="$PEER_HOME/tls/server.key" \
  CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" \
  CORE_PEER_FILESYSTEMPATH="$RUN/peer/data" \
  CORE_OPERATIONS_LISTENADDRESS=127.0.0.1:9444 CORE_METRICS_PROVIDER=disabled \
  CORE_LEDGER_STATE_STATEDATABASE=goleveldb \
  peer node start >> "$RUN/logs/peer.log" 2>&1 &
echo $! > "$RUN/peer.pid"
for _ in $(seq 1 40); do (echo > /dev/tcp/127.0.0.1/7051) >/dev/null 2>&1 && break; sleep 1; done
sleep 3

log "GetClient $CID after restart"
AFTER="$(query "{\"Args\":[\"GetClient\",\"$CID\"]}")"
echo "$AFTER"
echo "$AFTER" | grep -q "$CID" || fail "state did not survive a peer restart"

# ── 3. The real chaincode enforces the invariants ────────────────────────────
# `peer` exits non-zero when the chaincode rejects a transaction, which is the
# outcome we are asserting. Piping it into grep under `set -o pipefail` returns
# the peer's failure even when grep matched, so a correct rejection reads as a
# failed check — capture the output first instead.
expect_rejection() {
  local args="$1" want="$2" what="$3" out
  out="$(invoke "$args" || true)"
  echo "$out" | tail -1
  echo "$out" | grep -qi "$want" || fail "$what"
}

log "LogPrivacyBudget for the same (client, modality, round) must be REJECTED"
expect_rejection \
  "{\"Args\":[\"LogPrivacyBudget\",\"$CID\",\"audio\",\"1\",\"0.9\"]}" \
  "append-only" \
  "duplicate budget write was accepted -- append-only is NOT enforced"
echo "correctly rejected (append-only holds on the real ledger)"

log "budget history is intact"
query "{\"Args\":[\"GetBudgetHistory\",\"$CID\",\"audio\"]}"

log "PublishSubgraph round $ROUND, then the same round again"
invoke "{\"Args\":[\"PublishSubgraph\",\"$ROUND\",\"$CID\"]}" | tail -1
expect_rejection \
  "{\"Args\":[\"PublishSubgraph\",\"$ROUND\",\"$CID\"]}" \
  "immutable\|already" \
  "republishing a subgraph was accepted -- immutability is NOT enforced"
echo "correctly rejected (subgraph immutability holds on the real ledger)"

log "channel height"
peer_as_admin channel getinfo -c "$CHANNEL"

echo "LEDGER_VERIFIED"
