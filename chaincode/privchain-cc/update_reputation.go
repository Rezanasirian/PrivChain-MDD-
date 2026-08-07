package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// UpdateReputation sets a client's latest per-modality reputation score.
// Reputation is designed to evolve across rounds, so — unlike the privacy
// budget — this record is intentionally updated in place; the round it was set
// at is retained for audit.
//
// Args: clientID, modality, score, round. Score must be in [0, 1].
func (s *SmartContract) UpdateReputation(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 4 {
		return shim.Error("UpdateReputation expects 4 args: clientID, modality, score, round")
	}
	clientID := args[0]
	modality := args[1]
	if clientID == "" {
		return shim.Error("clientID must not be empty")
	}
	if !isKnownModality(modality) {
		return shim.Error(fmt.Sprintf("unknown modality %q", modality))
	}
	score, err := strconv.ParseFloat(args[2], 64)
	if err != nil || math.IsNaN(score) || score < 0 || score > 1 {
		return shim.Error("score must be a number in [0, 1]")
	}
	round, err := strconv.Atoi(args[3])
	if err != nil || round < 0 {
		return shim.Error("round must be a non-negative integer")
	}

	// Coordinator-only: a client must never be able to raise its own reputation,
	// which directly sets its aggregation weight (ADR-0006).
	if _, err := requireCoordinator(stub); err != nil {
		return shim.Error(err.Error())
	}
	if err := s.assertClientExists(stub, clientID); err != nil {
		return shim.Error(err.Error())
	}

	key, err := stub.CreateCompositeKey(reputationObjectType, []string{clientID, modality})
	if err != nil {
		return shim.Error(err.Error())
	}
	blob, err := json.Marshal(ReputationRecord{
		ClientID: clientID,
		Modality: modality,
		Score:    score,
		Round:    round,
	})
	if err != nil {
		return shim.Error(err.Error())
	}
	if err := stub.PutState(key, blob); err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(blob)
}

// GetReputation returns a client's latest reputation for a modality.
// Args: clientID, modality.
func (s *SmartContract) GetReputation(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 2 {
		return shim.Error("GetReputation expects 2 args: clientID, modality")
	}
	if !isKnownModality(args[1]) {
		return shim.Error(fmt.Sprintf("unknown modality %q", args[1]))
	}
	key, err := stub.CreateCompositeKey(reputationObjectType, []string{args[0], args[1]})
	if err != nil {
		return shim.Error(err.Error())
	}
	blob, err := stub.GetState(key)
	if err != nil {
		return shim.Error(err.Error())
	}
	if blob == nil {
		return shim.Error(fmt.Sprintf("no reputation for client %q modality %q", args[0], args[1]))
	}
	return shim.Success(blob)
}
