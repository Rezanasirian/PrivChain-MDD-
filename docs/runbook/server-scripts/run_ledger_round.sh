set -euo pipefail
cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
# A live-network config, kept out of the repo so the committed default stays `mock`.
cat > /workspace/blockchain_live.yaml <<'YAML'
ledger:
  backend: fabric_rest
  channel: privchain-channel
  chaincode: privchain-cc
  gateway_url: http://127.0.0.1:8801
  timeout_seconds: 30.0
YAML
python scripts/run_federated_with_ledger.py \
  --blockchain-config /workspace/blockchain_live.yaml \
  --rounds 3 --num-clients 4 2>&1 | tail -40
