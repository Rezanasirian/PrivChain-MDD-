cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
python scripts/run_reid_risk.py --daic-config configs/daic_woz.yaml > /workspace/reid_risk.log 2>&1
