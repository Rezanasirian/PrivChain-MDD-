set -uo pipefail
cd /workspace/PrivChain-MDD-
export FABRIC_HOME=/workspace/fabric
echo "########## TEARDOWN (fresh ledger) ##########"
bash scripts/fabric/teardown.sh --purge 2>&1 | tail -3
echo "########## NETWORK ##########"
bash scripts/fabric/setup_network.sh 2>&1 | tail -6
echo "########## CHAINCODE ##########"
bash scripts/fabric/deploy_chaincode.sh 2>&1 | tail -5
echo "########## GATEWAY ##########"
bash /workspace/start_gateway.sh 2>&1 | tail -3
echo "########## FEDERATED ROUND ON THE REAL LEDGER ##########"
bash /workspace/run_ledger_round.sh 2>&1 | tail -32
echo "DEMO_EXIT=$?"
