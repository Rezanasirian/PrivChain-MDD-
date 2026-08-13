cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
for MODE in session corpus none; do
  echo "########## ABLATION normalization=$MODE ##########"
  python scripts/run_modality_ablation.py --daic-config configs/daic_woz.yaml --normalization "$MODE"
done
echo "########## DONE ##########"
