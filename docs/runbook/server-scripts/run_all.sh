#!/bin/bash
cd /workspace/PrivChain-MDD- || exit 1
source /venv/main/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
echo "########## MODALITY ABLATION ##########"
python scripts/run_modality_ablation.py --daic-config configs/daic_woz.yaml 2>&1 | grep -v "it/s\]\|HF Hub"
echo "########## DP SWEEP ##########"
python scripts/run_dp_sweep.py --daic-config configs/daic_woz.yaml 2>&1 | grep -v "it/s\]\|HF Hub\|UserWarning\|warnings.warn"
echo "########## DONE ##########"
