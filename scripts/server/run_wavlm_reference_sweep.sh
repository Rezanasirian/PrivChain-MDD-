#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/depression-detection-ctd/src/acoustic-depr-wavlm}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PROJECT="${WANDB_PROJECT:-acoustic-depr-daic}"
SEEDS=(42 43 44 71 72 73)
LOG_DIR="$ROOT/checkpoints/seed_sweep_logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"
rm -f /workspace/wavlm_sweep.done /workspace/wavlm_sweep.failed

for seed in "${SEEDS[@]}"; do
  checkpoint_dir="checkpoints/seed_${seed}"
  if [[ -f "$checkpoint_dir/best.pt" ]]; then
    echo "[skip] seed $seed already has $checkpoint_dir/best.pt"
  else
    echo "[train] seed=$seed project=$PROJECT"
    "$PYTHON_BIN" train.py \
      --seed "$seed" \
      --checkpoint-dir "$checkpoint_dir" \
      --wandb-project "$PROJECT" \
      --wandb-mode online \
      --run-name "wavlm_seed${seed}" \
      2>&1 | tee "$LOG_DIR/train_seed${seed}.log"
  fi

  "$PYTHON_BIN" evaluate.py \
    --checkpoint "$checkpoint_dir/best.pt" \
    --split test \
    --out "results_test_seed${seed}.json" \
    2>&1 | tee "$LOG_DIR/eval_seed${seed}.log"
done

touch /workspace/wavlm_sweep.done
