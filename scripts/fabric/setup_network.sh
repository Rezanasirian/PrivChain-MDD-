#!/usr/bin/env bash
# Bring up a real single-org Hyperledger Fabric network WITHOUT Docker (ADR-0022).
#
# Docker is unavailable in the deployment container (no CAP_SYS_ADMIN, and the
# seccomp profile blocks `unshare`, so rootless engines are out too). Fabric does
# not actually need it: peer and orderer are static binaries, and chaincode runs
# as an ordinary process via Chaincode-as-a-Service. Only the chaincode *launcher*
# differs from the Docker test network; consensus, endorsement, the ledger and
# the lifecycle are identical.
#
# Everything it generates — MSP material, TLS keys, ledger data — lands in
# `.fabric/`, which is git-ignored. No secret is ever committed (CLAUDE.md §7).
#
# Usage:  bash scripts/fabric/setup_network.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FABRIC_HOME="${FABRIC_HOME:-/workspace/fabric}"
RUN="$REPO/.fabric"
CRYPTO="$RUN/crypto-config"
CHANNEL="${CHANNEL:-privchain-channel}"

export PATH="$FABRIC_HOME/bin:$FABRIC_HOME/go/bin:$PATH"
export FABRIC_CFG_PATH="$HERE"

ORDERER_HOME="$CRYPTO/ordererOrganizations/privchain.local/orderers/orderer.privchain.local"
PEER_HOME="$CRYPTO/peerOrganizations/org1.privchain.local/peers/peer0.org1.privchain.local"
ADMIN_MSP="$CRYPTO/peerOrganizations/org1.privchain.local/users/Admin@org1.privchain.local/msp"

log() { printf '\n=== %s ===\n' "$1"; }

# ── 1. Identity ──────────────────────────────────────────────────────────────
log "generating crypto material"
rm -rf "$RUN"
mkdir -p "$RUN"
cryptogen generate --config="$HERE/crypto-config.yaml" --output="$CRYPTO"

# ── 2. Channel genesis block ─────────────────────────────────────────────────
# Fabric 2.4+ has no system channel: configtxgen writes the application
# channel's genesis block directly and the orderer joins it via the admin API.
log "building the channel genesis block"
mkdir -p "$RUN/channel-artifacts"
configtxgen -profile PrivChainChannel \
  -outputBlock "$RUN/channel-artifacts/$CHANNEL.block" \
  -channelID "$CHANNEL"

# ── 3. Orderer ───────────────────────────────────────────────────────────────
# The raft write-ahead log and snapshots must live inside $RUN alongside the
# block ledger. Their stock location is /var/hyperledger/production, which
# survives a purge — so a second run replayed a previous network's raft entries
# against a freshly generated genesis block and the orderer panicked with
# "unexpected Previous block hash". That is why the first run always worked and
# every rerun did not.
log "starting the orderer"
mkdir -p "$RUN/orderer" "$RUN/logs"
# `setsid` detaches the node from this script's session. Without it the orderer
# and peer are killed by SIGHUP the moment the script returns, which looks like a
# network that came up fine and then refused connections at the next step.
setsid env \
  FABRIC_CFG_PATH="$FABRIC_HOME/config" \
  ORDERER_GENERAL_LISTENADDRESS=0.0.0.0 \
  ORDERER_GENERAL_LISTENPORT=7050 \
  ORDERER_GENERAL_LOCALMSPID=OrdererMSP \
  ORDERER_GENERAL_LOCALMSPDIR="$ORDERER_HOME/msp" \
  ORDERER_GENERAL_TLS_ENABLED=true \
  ORDERER_GENERAL_TLS_PRIVATEKEY="$ORDERER_HOME/tls/server.key" \
  ORDERER_GENERAL_TLS_CERTIFICATE="$ORDERER_HOME/tls/server.crt" \
  ORDERER_GENERAL_TLS_ROOTCAS="[$ORDERER_HOME/tls/ca.crt]" \
  ORDERER_GENERAL_BOOTSTRAPMETHOD=none \
  ORDERER_CHANNELPARTICIPATION_ENABLED=true \
  ORDERER_ADMIN_LISTENADDRESS=0.0.0.0:7053 \
  ORDERER_ADMIN_TLS_ENABLED=true \
  ORDERER_ADMIN_TLS_PRIVATEKEY="$ORDERER_HOME/tls/server.key" \
  ORDERER_ADMIN_TLS_CERTIFICATE="$ORDERER_HOME/tls/server.crt" \
  ORDERER_ADMIN_TLS_ROOTCAS="[$ORDERER_HOME/tls/ca.crt]" \
  ORDERER_ADMIN_TLS_CLIENTROOTCAS="[$ORDERER_HOME/tls/ca.crt]" \
  ORDERER_ADMIN_TLS_CLIENTAUTHREQUIRED=true \
  ORDERER_FILELEDGER_LOCATION="$RUN/orderer/ledger" \
  ORDERER_CONSENSUS_WALDIR="$RUN/orderer/etcdraft/wal" \
  ORDERER_CONSENSUS_SNAPDIR="$RUN/orderer/etcdraft/snapshot" \
  ORDERER_OPERATIONS_LISTENADDRESS=127.0.0.1:9443 \
  ORDERER_METRICS_PROVIDER=disabled \
  orderer > "$RUN/logs/orderer.log" 2>&1 &
echo $! > "$RUN/orderer.pid"

# The admin endpoint is the first thing to come up; poll it rather than sleeping
# a fixed amount and hoping.
for _ in $(seq 1 30); do
  if osnadmin channel list -o localhost:7053 \
      --ca-file "$ORDERER_HOME/tls/ca.crt" \
      --client-cert "$ORDERER_HOME/tls/server.crt" \
      --client-key "$ORDERER_HOME/tls/server.key" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

log "joining the orderer to $CHANNEL"
osnadmin channel join --channelID "$CHANNEL" \
  --config-block "$RUN/channel-artifacts/$CHANNEL.block" \
  -o localhost:7053 \
  --ca-file "$ORDERER_HOME/tls/ca.crt" \
  --client-cert "$ORDERER_HOME/tls/server.crt" \
  --client-key "$ORDERER_HOME/tls/server.key"

# ── 4. Peer ──────────────────────────────────────────────────────────────────
# The peer's core.yaml is the stock one plus an `externalBuilders` entry. That
# entry is what makes a Docker-free deployment possible: it tells the peer to
# hand chaincode packages to our scripts instead of building an image.
log "preparing peer config with the external builder"
mkdir -p "$RUN/peercfg" "$RUN/peer"
cp "$FABRIC_HOME/config/core.yaml" "$RUN/peercfg/core.yaml"
python3 - "$RUN/peercfg/core.yaml" "$HERE/external_builder" <<'PY'
import re
import sys

config_path, builder_path = sys.argv[1], sys.argv[2]
text = open(config_path, encoding="utf-8").read()
block = (
    "    externalBuilders:\n"
    "        - name: ccaas_builder\n"
    f"          path: {builder_path}\n"
    "          propagateEnvironment:\n"
    "              - CHAINCODE_AS_A_SERVICE_BUILDER_CONFIG\n"
)
# Replace the commented-out stock stanza; fall back to appending under chaincode.
pattern = re.compile(r"^[ \t]*#?[ \t]*externalBuilders:.*?(?=^\s{4}\w)", re.M | re.S)
text, count = pattern.subn(block, text, count=1)
if count == 0:
    raise SystemExit("could not find the externalBuilders stanza in core.yaml")
open(config_path, "w", encoding="utf-8").write(text)
print("external builder registered:", builder_path)
PY
# Fabric requires the builder's scripts under a `bin/` subdirectory of the
# configured path. With them one level up the peer finds no `detect`, silently
# falls back to its Docker builder, and fails with a confusing docker.sock error.
chmod +x "$HERE/external_builder/bin/"*

log "starting the peer"
setsid env \
  FABRIC_CFG_PATH="$RUN/peercfg" \
  CORE_PEER_ID=peer0.org1.privchain.local \
  CORE_PEER_ADDRESS=localhost:7051 \
  CORE_PEER_LISTENADDRESS=0.0.0.0:7051 \
  CORE_PEER_CHAINCODEADDRESS=localhost:7052 \
  CORE_PEER_CHAINCODELISTENADDRESS=0.0.0.0:7052 \
  CORE_PEER_LOCALMSPID=Org1MSP \
  CORE_PEER_MSPCONFIGPATH="$PEER_HOME/msp" \
  CORE_PEER_TLS_ENABLED=true \
  CORE_PEER_TLS_CERT_FILE="$PEER_HOME/tls/server.crt" \
  CORE_PEER_TLS_KEY_FILE="$PEER_HOME/tls/server.key" \
  CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" \
  CORE_PEER_FILESYSTEMPATH="$RUN/peer/data" \
  CORE_OPERATIONS_LISTENADDRESS=127.0.0.1:9444 \
  CORE_METRICS_PROVIDER=disabled \
  CORE_LEDGER_STATE_STATEDATABASE=goleveldb \
  peer node start > "$RUN/logs/peer.log" 2>&1 &
echo $! > "$RUN/peer.pid"

for _ in $(seq 1 30); do
  if (echo > /dev/tcp/127.0.0.1/7051) >/dev/null 2>&1; then break; fi
  sleep 1
done

log "joining the peer to $CHANNEL"
env \
  FABRIC_CFG_PATH="$RUN/peercfg" \
  CORE_PEER_ADDRESS=localhost:7051 \
  CORE_PEER_LOCALMSPID=Org1MSP \
  CORE_PEER_MSPCONFIGPATH="$ADMIN_MSP" \
  CORE_PEER_TLS_ENABLED=true \
  CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" \
  peer channel join -b "$RUN/channel-artifacts/$CHANNEL.block"

log "channels the peer has joined"
env \
  FABRIC_CFG_PATH="$RUN/peercfg" \
  CORE_PEER_ADDRESS=localhost:7051 \
  CORE_PEER_LOCALMSPID=Org1MSP \
  CORE_PEER_MSPCONFIGPATH="$ADMIN_MSP" \
  CORE_PEER_TLS_ENABLED=true \
  CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" \
  peer channel list

echo "NETWORK_UP"
