set -euo pipefail
cd /workspace/PrivChain-MDD-
export PATH=/workspace/fabric/go/bin:$PATH GOPATH=/workspace/gopath GOCACHE=/workspace/gocache GOFLAGS=-mod=mod
RUN=$PWD/.fabric
CRYPTO=$RUN/crypto-config
PH=$CRYPTO/peerOrganizations/org1.privchain.local/peers/peer0.org1.privchain.local
AM=$CRYPTO/peerOrganizations/org1.privchain.local/users/Admin@org1.privchain.local/msp
mkdir -p "$RUN/logs"
cd scripts/fabric/gateway
go build -o "$RUN/privchain-gateway" . 2>&1 | tail -5
[ -f "$RUN/gateway.pid" ] && kill "$(cat "$RUN/gateway.pid")" 2>/dev/null || true
sleep 1
setsid env GATEWAY_LISTEN=127.0.0.1:8801 PEER_ENDPOINT=localhost:7051 \
    PEER_HOST_ALIAS=peer0.org1.privchain.local \
    PEER_TLS_CA="$PH/tls/ca.crt" MSP_DIR="$AM" MSP_ID=Org1MSP \
    "$RUN/privchain-gateway" > "$RUN/logs/gateway.log" 2>&1 &
echo $! > "$RUN/gateway.pid"
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:8801/healthz >/dev/null 2>&1 && break; sleep 1; done
curl -s http://127.0.0.1:8801/healthz; echo
echo "GATEWAY_UP"
