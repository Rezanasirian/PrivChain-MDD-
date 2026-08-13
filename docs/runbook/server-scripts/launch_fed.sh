cd /workspace/PrivChain-MDD-
. /venv/main/bin/activate
export PYTHONUNBUFFERED=1
for MODE in iid dirichlet; do
  echo "########## FEDERATED partition=$MODE ##########"
  python scripts/run_federated_comparison.py --daic-config configs/daic_woz.yaml --partition "$MODE"
done
echo "########## CLIENT-COUNT SENSITIVITY (5 clients, iid) ##########"
python scripts/run_federated_comparison.py --daic-config configs/daic_woz.yaml --partition iid --num-clients 5
echo "########## DONE ##########"
