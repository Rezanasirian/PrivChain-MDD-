#!/bin/bash
# Launch the Phase 3 per-modality DP sweep on the real corpus.
cd /workspace/PrivChain-MDD- || exit 1
source /venv/main/bin/activate
rm -f /workspace/dp.log
setsid nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python scripts/run_dp_sweep.py --daic-config configs/daic_woz.yaml \
  > /workspace/dp.log 2>&1 < /dev/null &
echo "started pid=$!"
