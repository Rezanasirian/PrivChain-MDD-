#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/PrivChain-MDD-}"
ARTIFACT_DIR="${CTD_ARTIFACT_DIR:-/workspace/depression-detection-ctd/src/ctd/outputs/fusion}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/ctd_dp_dev_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-privchain-mdd-ctd-dp-dev-sweep}"

# rounds learning-rate clip-norm
CONFIGS=(
  "120 0.01 0.05"
  "120 0.01 0.1"
  "120 0.01 0.5"
  "120 0.01 1.0"
  "300 0.01 0.1"
  "300 0.01 0.5"
  "300 0.03 0.1"
)
SEEDS=(42 43 44)

cd "$PROJECT_ROOT"
for config in "${CONFIGS[@]}"; do
  read -r rounds learning_rate clip_norm <<<"$config"
  config_name="r${rounds}_lr${learning_rate}_c${clip_norm}"
  for seed in "${SEEDS[@]}"; do
    output_dir="$OUTPUT_ROOT/$config_name/seed_$seed"
    if [[ -f "$output_dir/comparison.json" ]]; then
      echo "[skip] $config_name seed=$seed"
      continue
    fi
    echo "[run] $config_name seed=$seed"
    PYTHONPATH=src "$PYTHON_BIN" scripts/run_ctd_privacy_comparison.py \
      --artifact-dir "$ARTIFACT_DIR" \
      --output-dir "$output_dir" \
      --seed "$seed" \
      --rounds "$rounds" \
      --num-clients 10 \
      --clients-per-round 10 \
      --dp-local-epochs 1 \
      --dp-batch-size 16 \
      --batch-size 16 \
      --learning-rate "$learning_rate" \
      --weight-decay 0.001 \
      --optimizer sgd \
      --selection-metric roc_auc \
      --class-weight-mode aggregate_counts \
      --epsilon 8 \
      --max-grad-norm "$clip_norm" \
      --selection-only \
      --arms fedavg_dp \
      --device cuda \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "$config_name-seed$seed"
  done
done

