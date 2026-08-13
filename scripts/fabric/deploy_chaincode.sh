#!/usr/bin/env bash
# Deploy privchain-cc to the local network as a Chaincode-as-a-Service (ADR-0022).
#
# The peer never builds an image: the chaincode is compiled here, started as an
# ordinary process, and the package merely tells the peer which address to dial.
# Everything else — install, approve, commit, init — is the standard Fabric 2.x
# lifecycle, unchanged.
#
# Usage:  bash scripts/fabric/deploy_chaincode.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FABRIC_HOME="${FABRIC_HOME:-/workspace/fabric}"
RUN="$REPO/.fabric"
CRYPTO="$RUN/crypto-config"
CHANNEL="${CHANNEL:-privchain-channel}"
CC_NAME="${CC_NAME:-privchain-cc}"
CC_VERSION="${CC_VERSION:-1.0}"
CC_ADDRESS="${CC_ADDRESS:-127.0.0.1:9999}"
COORDINATOR_MSP="${COORDINATOR_MSP:-Org1MSP}"

export PATH="$FABRIC_HOME/bin:$FABRIC_HOME/go/bin:$PATH"
export GOPATH="${GOPATH:-/workspace/gopath}"
export GOCACHE="${GOCACHE:-/workspace/gocache}"

ORDERER_HOME="$CRYPTO/ordererOrganizations/privchain.local/orderers/orderer.privchain.local"
PEER_HOME="$CRYPTO/peerOrganizations/org1.privchain.local/peers/peer0.org1.privchain.local"
ADMIN_MSP="$CRYPTO/peerOrganizations/org1.privchain.local/users/Admin@org1.privchain.local/msp"

# Every peer CLI call runs as the org admin against the local peer.
peer_as_admin() {
  env \
    FABRIC_CFG_PATH="$RUN/peercfg" \
    CORE_PEER_ADDRESS=localhost:7051 \
    CORE_PEER_LOCALMSPID=Org1MSP \
    CORE_PEER_MSPCONFIGPATH="$ADMIN_MSP" \
    CORE_PEER_TLS_ENABLED=true \
    CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" \
    peer "$@"
}

log() { printf '\n=== %s ===\n' "$1"; }

# ── 1. Compile ───────────────────────────────────────────────────────────────
log "building the chaincode binary"
(cd "$REPO/chaincode/privchain-cc" && GOFLAGS=-mod=mod go build -o "$RUN/privchain-cc-server" .)

# ── 2. Package ───────────────────────────────────────────────────────────────
# A ccaas package carries no code, only the address the peer should dial.
log "packaging as chaincode-as-a-service"
PKG="$RUN/ccaas-pkg"
rm -rf "$PKG" && mkdir -p "$PKG/src"
cat > "$PKG/src/connection.json" <<JSON
{
  "address": "$CC_ADDRESS",
  "dial_timeout": "10s",
  "tls_required": false
}
JSON
cat > "$PKG/metadata.json" <<JSON
{
  "type": "ccaas",
  "label": "${CC_NAME}_${CC_VERSION}"
}
JSON
tar -czf "$PKG/code.tar.gz" -C "$PKG/src" connection.json
tar -czf "$RUN/$CC_NAME.tgz" -C "$PKG" metadata.json code.tar.gz

log "installing the package"
# Take the identifier from *this* install, not from `queryinstalled`: that lists
# every package ever installed on the peer, so picking the first match silently
# deploys an older build whenever the chaincode has been packaged more than once.
INSTALL_OUT="$(peer_as_admin lifecycle chaincode install "$RUN/$CC_NAME.tgz" 2>&1)"
echo "$INSTALL_OUT" | tail -1
PACKAGE_ID="$(echo "$INSTALL_OUT" \
  | sed -n "s/.*Chaincode code package identifier: \(${CC_NAME}_${CC_VERSION}:[a-f0-9]*\).*/\1/p" \
  | tail -1)"
if [ -z "$PACKAGE_ID" ]; then
  echo "could not determine the installed package ID" >&2
  exit 1
fi
echo "package ID: $PACKAGE_ID"

# ── 3. Start the chaincode process ───────────────────────────────────────────
# It must be listening before the peer is asked to commit, because commit
# triggers a connection.
log "starting the chaincode process"
# `setsid`, for the same reason as the orderer and peer: a background job of this
# script is killed by SIGHUP when the script returns, and a chaincode that dies
# after deployment presents as "no peers available to evaluate chaincode".
setsid env CHAINCODE_ID="$PACKAGE_ID" CHAINCODE_SERVER_ADDRESS="0.0.0.0:${CC_ADDRESS##*:}" \
  "$RUN/privchain-cc-server" > "$RUN/logs/chaincode.log" 2>&1 &
echo $! > "$RUN/chaincode.pid"
for _ in $(seq 1 30); do
  if (echo > "/dev/tcp/127.0.0.1/${CC_ADDRESS##*:}") >/dev/null 2>&1; then break; fi
  sleep 1
done

# ── 4. Lifecycle ─────────────────────────────────────────────────────────────
log "approving for Org1MSP"
peer_as_admin lifecycle chaincode approveformyorg \
  -o 127.0.0.1:7050 --ordererTLSHostnameOverride orderer.privchain.local \
  --tls --cafile "$ORDERER_HOME/tls/ca.crt" \
  --channelID "$CHANNEL" --name "$CC_NAME" --version "$CC_VERSION" \
  --package-id "$PACKAGE_ID" --sequence 1 --init-required

log "committing to the channel"
peer_as_admin lifecycle chaincode commit \
  -o 127.0.0.1:7050 --ordererTLSHostnameOverride orderer.privchain.local \
  --tls --cafile "$ORDERER_HOME/tls/ca.crt" \
  --channelID "$CHANNEL" --name "$CC_NAME" --version "$CC_VERSION" \
  --sequence 1 --init-required

log "initialising with coordinator MSP $COORDINATOR_MSP"
peer_as_admin chaincode invoke \
  -o 127.0.0.1:7050 --ordererTLSHostnameOverride orderer.privchain.local \
  --tls --cafile "$ORDERER_HOME/tls/ca.crt" \
  --channelID "$CHANNEL" --name "$CC_NAME" --isInit \
  -c "{\"Args\":[\"Init\",\"$COORDINATOR_MSP\"]}" --waitForEvent

log "committed chaincode on the channel"
peer_as_admin lifecycle chaincode querycommitted --channelID "$CHANNEL" --name "$CC_NAME"

echo "CHAINCODE_DEPLOYED"
