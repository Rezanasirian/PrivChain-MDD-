#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/PrivChain-MDD-}"
ARTIFACT_DIR="${CTD_ARTIFACT_DIR:-/workspace/depression-detection-ctd/src/ctd/outputs/fusion}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
SEEDS=(42 43 44)

cd "$PROJECT_ROOT"
rm -f /workspace/ctd_reference_campaign.done /workspace/ctd_reference_campaign.failed

for seed in "${SEEDS[@]}"; do
  output="experiments/ctd_reference_campaign/seed_${seed}"
  if [[ -f "$output/comparison.json" ]]; then
    echo "[skip] seed $seed already complete"
    continue
  fi
  echo "[run] CTD Central/FedAvg/DP seed=$seed"
  PYTHONPATH=src "$PYTHON_BIN" scripts/run_ctd_privacy_comparison.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --output-dir "$output" \
    --seed "$seed" \
    --epochs 200 \
    --rounds 120 \
    --num-clients 10 \
    --clients-per-round 10 \
    --local-epochs 1 \
    --batch-size 16 \
    --epsilon 8 \
    --device cuda
done

touch /workspace/ctd_reference_campaign.done
