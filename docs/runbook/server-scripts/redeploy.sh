set -euo pipefail
cd /workspace/PrivChain-MDD-
chmod +x scripts/fabric/*.sh scripts/fabric/external_builder/bin/*
# Restart the peer so it re-reads core.yaml and finds the builder in bin/.
[ -f .fabric/peer.pid ] && kill "$(cat .fabric/peer.pid)" 2>/dev/null || true
sleep 3
FABRIC_HOME=/workspace/fabric
export PATH="$FABRIC_HOME/bin:$FABRIC_HOME/go/bin:$PATH"
CRYPTO=.fabric/crypto-config
PEER_HOME="$PWD/$CRYPTO/peerOrganizations/org1.privchain.local/peers/peer0.org1.privchain.local"
env FABRIC_CFG_PATH="$PWD/.fabric/peercfg" \
  CORE_PEER_ID=peer0.org1.privchain.local \
  CORE_PEER_ADDRESS=localhost:7051 CORE_PEER_LISTENADDRESS=0.0.0.0:7051 \
  CORE_PEER_CHAINCODEADDRESS=localhost:7052 CORE_PEER_CHAINCODELISTENADDRESS=0.0.0.0:7052 \
  CORE_PEER_LOCALMSPID=Org1MSP CORE_PEER_MSPCONFIGPATH="$PEER_HOME/msp" \
  CORE_PEER_TLS_ENABLED=true \
  CORE_PEER_TLS_CERT_FILE="$PEER_HOME/tls/server.crt" \
  CORE_PEER_TLS_KEY_FILE="$PEER_HOME/tls/server.key" \
  CORE_PEER_TLS_ROOTCERT_FILE="$PEER_HOME/tls/ca.crt" \
  CORE_PEER_FILESYSTEMPATH="$PWD/.fabric/peer/data" \
  CORE_OPERATIONS_LISTENADDRESS=127.0.0.1:9444 CORE_METRICS_PROVIDER=disabled \
  CORE_LEDGER_STATE_STATEDATABASE=goleveldb \
  peer node start > .fabric/logs/peer.log 2>&1 &
echo $! > .fabric/peer.pid
for _ in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/7051) >/dev/null 2>&1 && break; sleep 1; done
bash scripts/fabric/deploy_chaincode.sh 2>&1 | tail -28
