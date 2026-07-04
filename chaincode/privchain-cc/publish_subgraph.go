package main

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// PublishSubgraph records the set of clients aggregated together in a round.
// A round's subgraph is immutable once published (see ADR-0006), so a second
// publish for the same round is rejected.
//
// Args: round, clientID1, clientID2, ... (at least one client).
func (s *SmartContract) PublishSubgraph(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) < 2 {
		return shim.Error("PublishSubgraph expects: round, clientID1, [clientID2, ...]")
	}
	round, err := strconv.Atoi(args[0])
	if err != nil || round < 0 {
		return shim.Error("round must be a non-negative integer")
	}
	clientIDs := args[1:]
	for _, id := range clientIDs {
		if id == "" {
			return shim.Error("subgraph client IDs must not be empty")
		}
	}

	key, err := stub.CreateCompositeKey(subgraphObjectType, []string{strconv.Itoa(round)})
	if err != nil {
		return shim.Error(err.Error())
	}
	existing, err := stub.GetState(key)
	if err != nil {
		return shim.Error(err.Error())
	}
	if existing != nil {
		return shim.Error(fmt.Sprintf("subgraph for round %d already published; it is immutable", round))
	}

	blob, err := json.Marshal(SubgraphRecord{Round: round, ClientIDs: clientIDs})
	if err != nil {
		return shim.Error(err.Error())
	}
	if err := stub.PutState(key, blob); err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(blob)
}

// GetSubgraph returns the subgraph published for a round. Args: round.
func (s *SmartContract) GetSubgraph(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 1 {
		return shim.Error("GetSubgraph expects 1 arg: round")
	}
	round, err := strconv.Atoi(args[0])
	if err != nil || round < 0 {
		return shim.Error("round must be a non-negative integer")
	}
	key, err := stub.CreateCompositeKey(subgraphObjectType, []string{strconv.Itoa(round)})
	if err != nil {
		return shim.Error(err.Error())
	}
	blob, err := stub.GetState(key)
	if err != nil {
		return shim.Error(err.Error())
	}
	if blob == nil {
		return shim.Error(fmt.Sprintf("no subgraph published for round %d", round))
	}
	return shim.Success(blob)
}
