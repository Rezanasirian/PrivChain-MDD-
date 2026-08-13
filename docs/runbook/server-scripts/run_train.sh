#!/bin/bash
# Launch one clean Phase 1 baseline run on the real corpus.
cd /workspace/PrivChain-MDD- || exit 1
source /venv/main/bin/activate
rm -f /workspace/train1.log
setsid nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python scripts/train_baseline.py --daic-config configs/daic_woz.yaml \
  > /workspace/train1.log 2>&1 < /dev/null &
echo "started pid=$!"
