#!/bin/bash
# Full Chapter-4 grid, then Phase 3 re-run under the new inverse_risk default.
cd /workspace/PrivChain-MDD- || exit 1
# On the vast.ai PyTorch image `python3` is the system interpreter and has no
# torch; the stack lives in /venv/main. Activate it rather than assuming PATH.
source /venv/main/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
echo "### phase7 full $(date -u +%H:%M:%S)"
python scripts/run_final_evaluation.py --daic-config configs/daic_woz.yaml
echo "### phase3 allocation (inverse_risk) $(date -u +%H:%M:%S)"
python scripts/run_allocation_comparison.py --daic-config configs/daic_woz.yaml
echo "### all done $(date -u +%H:%M:%S)"
