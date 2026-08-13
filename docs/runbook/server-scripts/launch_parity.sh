cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/flwr-libs
python scripts/run_flower_parity.py --daic-config configs/daic_woz.yaml --rounds 10
echo "PARITY_EXIT=$?"
