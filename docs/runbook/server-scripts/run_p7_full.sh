#!/bin/bash
# Full Chapter-4 grid, then Phase 3 re-run under the new inverse_risk default.
cd /workspace/PrivChain-MDD-
echo "### phase7 full $(date -u +%H:%M:%S)"
python3 scripts/run_final_evaluation.py --daic-config configs/daic_woz.yaml
echo "### phase3 allocation (inverse_risk) $(date -u +%H:%M:%S)"
python3 scripts/run_allocation_comparison.py --daic-config configs/daic_woz.yaml
echo "### all done $(date -u +%H:%M:%S)"
