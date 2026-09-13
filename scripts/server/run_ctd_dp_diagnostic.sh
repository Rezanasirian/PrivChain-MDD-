#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/workspace/PrivChain-MDD-}"
ARTIFACTS="${CTD_ARTIFACT_DIR:-/workspace/depression-detection-ctd/src/ctd/outputs/fusion}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

cd "$ROOT"
rm -f /workspace/ctd_dp_diagnostic.done /workspace/ctd_dp_diagnostic.failed

for batch_size in 2 4; do
  for local_epochs in 1 2; do
    name="batch_${batch_size}_epochs_${local_epochs}"
    output="experiments/ctd_dp_diagnostic_wandb/$name"
    echo "[run] $name"
    PYTHONPATH=src "$PYTHON_BIN" scripts/run_ctd_privacy_comparison.py \
      --artifact-dir "$ARTIFACTS" \
      --output-dir "$output" \
      --seed 43 \
      --rounds 120 \
      --num-clients 10 \
      --clients-per-round 10 \
      --batch-size 16 \
      --learning-rate 0.01 \
      --optimizer sgd \
      --class-weight-mode per_shard \
      --epsilon 8 \
      --dp-batch-size "$batch_size" \
      --dp-local-epochs "$local_epochs" \
      --arms fedavg_dp \
      --wandb-project privchain-mdd-ctd-dp \
      --wandb-run-name "ctd_dp_${name}_seed43" \
      --device cuda
  done
done

touch /workspace/ctd_dp_diagnostic.done
