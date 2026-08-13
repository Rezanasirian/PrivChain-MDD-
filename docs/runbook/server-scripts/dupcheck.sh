set -uo pipefail
cd /workspace/PrivChain-MDD-
export PATH=/workspace/fabric/bin:$PATH
RUN=$PWD/.fabric; CRYPTO=$RUN/crypto-config
OH=$CRYPTO/ordererOrganizations/privchain.local/orderers/orderer.privchain.local
PH=$CRYPTO/peerOrganizations/org1.privchain.local/peers/peer0.org1.privchain.local
AM=$CRYPTO/peerOrganizations/org1.privchain.local/users/Admin@org1.privchain.local/msp
pa() { env FABRIC_CFG_PATH=$RUN/peercfg CORE_PEER_ADDRESS=localhost:7051 \
  CORE_PEER_LOCALMSPID=Org1MSP CORE_PEER_MSPCONFIGPATH=$AM CORE_PEER_TLS_ENABLED=true \
  CORE_PEER_TLS_ROOTCERT_FILE=$PH/tls/ca.crt peer "$@"; }
CID=dup-test-$$
echo "--- register $CID ---"
pa chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.privchain.local --tls --cafile $OH/tls/ca.crt --channelID privchain-channel --name privchain-cc -c "{\"Args\":[\"RegisterClient\",\"$CID\",\"1\",\"0\",\"1\"]}" --waitForEvent 2>&1 | tail -1
echo "--- first budget write ---"
pa chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.privchain.local --tls --cafile $OH/tls/ca.crt --channelID privchain-channel --name privchain-cc -c "{\"Args\":[\"LogPrivacyBudget\",\"$CID\",\"audio\",\"1\",\"0.5\"]}" --waitForEvent 2>&1 | tail -1
echo "--- history after first ---"
pa chaincode query --channelID privchain-channel --name privchain-cc -c "{\"Args\":[\"GetBudgetHistory\",\"$CID\"]}" 2>&1
echo "--- DUPLICATE write (raw output) ---"
pa chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.privchain.local --tls --cafile $OH/tls/ca.crt --channelID privchain-channel --name privchain-cc -c "{\"Args\":[\"LogPrivacyBudget\",\"$CID\",\"audio\",\"1\",\"0.9\"]}" --waitForEvent 2>&1 | tail -3
echo "--- history after duplicate ---"
pa chaincode query --channelID privchain-channel --name privchain-cc -c "{\"Args\":[\"GetBudgetHistory\",\"$CID\"]}" 2>&1
