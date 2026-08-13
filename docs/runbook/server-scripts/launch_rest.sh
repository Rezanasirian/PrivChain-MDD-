cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
echo "########## REID RISK normalization=none (fair estimate) ##########"
python scripts/run_reid_risk.py --daic-config configs/daic_woz.yaml --normalization none
echo "########## BASELINE (with CI) ##########"
python scripts/train_baseline.py --daic-config configs/daic_woz.yaml
echo "########## DP SWEEP (with CI) ##########"
python scripts/run_dp_sweep.py --daic-config configs/daic_woz.yaml
echo "########## ALLOCATION COMPARISON (paired) ##########"
python scripts/run_allocation_comparison.py --daic-config configs/daic_woz.yaml
echo "########## DONE ##########"
