cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
echo "########## REGRESSION: dp sweep after refactor ##########"
python scripts/run_dp_sweep.py --daic-config configs/daic_woz.yaml
echo "########## ALLOCATION COMPARISON ##########"
python scripts/run_allocation_comparison.py --daic-config configs/daic_woz.yaml
echo "########## DONE ##########"
