package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// LogPrivacyBudget appends a per-modality privacy-budget entry for a client in a
// round. Consumed epsilon is append-only and MUST NOT be overwritten
// (CLAUDE.md §7), so a second write for the same (clientID, modality, round) is
// rejected.
//
// Args: clientID, group, round, epsilonIncremental, epsilonCumulative.
func (s *SmartContract) LogPrivacyBudget(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 4 && len(args) != 5 {
		return shim.Error("LogPrivacyBudget expects 5 args: clientID, group, round, epsilonIncremental, epsilonCumulative")
	}
	clientID := args[0]
	modality := args[1]
	if clientID == "" {
		return shim.Error("clientID must not be empty")
	}
	if !isKnownPrivacyGroup(modality) {
		return shim.Error(fmt.Sprintf("unknown privacy group %q", modality))
	}
	round, err := strconv.Atoi(args[2])
	if err != nil || round < 0 {
		return shim.Error("round must be a non-negative integer")
	}
	incremental, err := strconv.ParseFloat(args[3], 64)
	if err != nil || math.IsNaN(incremental) || math.IsInf(incremental, 0) || incremental < 0 {
		return shim.Error("epsilonIncremental must be a finite, non-negative number")
	}
	cumulative := incremental
	if len(args) == 5 {
		cumulative, err = strconv.ParseFloat(args[4], 64)
	}
	if err != nil || math.IsNaN(cumulative) || math.IsInf(cumulative, 0) || cumulative < incremental {
		return shim.Error("epsilonCumulative must be finite and at least epsilonIncremental")
	}

	// Only the identity that registered this client (or the coordinator) may
	// write its consumed epsilon — otherwise any peer could forge another
	// client's privacy accounting.
	if _, err := requireOwnerOrCoordinator(stub, clientID); err != nil {
		return shim.Error(err.Error())
	}

	key, err := stub.CreateCompositeKey(budgetObjectType, []string{clientID, modality, roundKey(round)})
	if err != nil {
		return shim.Error(err.Error())
	}
	existing, err := stub.GetState(key)
	if err != nil {
		return shim.Error(err.Error())
	}
	if existing != nil {
		return shim.Error(fmt.Sprintf(
			"privacy budget already logged for client %q modality %q round %d; "+
				"consumed epsilon is append-only and must not be overwritten",
			clientID, modality, round))
	}

	blob, err := json.Marshal(BudgetRecord{
		ClientID:           clientID,
		Modality:           modality,
		Round:              round,
		EpsilonIncremental: incremental,
		EpsilonCumulative:  cumulative,
		EpsilonSpent:       cumulative,
	})
	if err != nil {
		return shim.Error(err.Error())
	}
	if err := stub.PutState(key, blob); err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(blob)
}

// GetBudgetHistory returns every budget entry for a client+modality in
// ascending round order (the round is zero-padded in the composite key, so the
// ledger's lexicographic iteration is also numeric order).
// Args: clientID, modality.
func (s *SmartContract) GetBudgetHistory(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 2 {
		return shim.Error("GetBudgetHistory expects 2 args: clientID, modality")
	}
	clientID := args[0]
	modality := args[1]
	if !isKnownModality(modality) {
		return shim.Error(fmt.Sprintf("unknown modality %q", modality))
	}

	iterator, err := stub.GetStateByPartialCompositeKey(budgetObjectType, []string{clientID, modality})
	if err != nil {
		return shim.Error(err.Error())
	}
	defer iterator.Close()

	records := make([]BudgetRecord, 0)
	for iterator.HasNext() {
		kv, err := iterator.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		var record BudgetRecord
		if err := json.Unmarshal(kv.Value, &record); err != nil {
			return shim.Error(err.Error())
		}
		records = append(records, record)
	}

	blob, err := json.Marshal(records)
	if err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(blob)
}

// assertClientExists returns an error if clientID has not been registered.
func (s *SmartContract) assertClientExists(stub shim.ChaincodeStubInterface, clientID string) error {
	_, err := readClient(stub, clientID)
	return err
}
