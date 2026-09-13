#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/PrivChain-MDD-}"
ARTIFACT_DIR="${CTD_ARTIFACT_DIR:-/workspace/depression-detection-ctd/src/ctd/outputs/fusion}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/ctd_train_cv_confirmatory}"

cd "$PROJECT_ROOT"
if [[ ! -f "$ARTIFACT_DIR/ctd_train.npz" ]]; then
  echo "missing CTD train artifact: $ARTIFACT_DIR/ctd_train.npz" >&2
  exit 2
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/run_ctd_train_cv.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seeds 42 43 44 45 46 \
  --folds 5 \
  --selection-fraction 0.2 \
  --num-clients 10 \
  --batch-size 16 \
  --local-epochs 1 \
  --rounds 300 \
  --patience 60 \
  --bootstrap-resamples 2000

touch /workspace/ctd_train_cv.done
