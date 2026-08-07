package main

import (
	"fmt"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// SmartContract implements the PrivChain-MDD auditability chaincode.
//
// It uses the low-level shim.Chaincode interface (Init/Invoke) rather than the
// contract API so it can be unit-tested with shimtest.MockStub, as mandated by
// CLAUDE.md §4.
type SmartContract struct{}

// Init is invoked on instantiation/upgrade. It records which MSP acts as the
// federation coordinator; that MSP is the only one allowed to update reputation
// or publish subgraphs (see ADR-0006). The value comes from the instantiation
// arguments rather than being hardcoded, so the same chaincode serves any
// network (CLAUDE.md §3).
//
// Args: coordinatorMSPID — passed after the function name, as a deployment
// does: `-c '{"Args":["init","CoordinatorMSP"]}'`. On upgrade the argument may
// be omitted to keep the value already stored.
func (s *SmartContract) Init(stub shim.ChaincodeStubInterface) peer.Response {
	_, args := stub.GetFunctionAndParameters()
	if len(args) == 0 {
		if _, err := getCoordinatorMSP(stub); err != nil {
			return shim.Error(
				"Init expects 1 arg: coordinatorMSPID (the MSP allowed to update " +
					"reputation and publish subgraphs)")
		}
		return shim.Success(nil)
	}
	if len(args) != 1 {
		return shim.Error("Init expects exactly 1 arg: coordinatorMSPID")
	}
	if err := setCoordinatorMSP(stub, args[0]); err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(nil)
}

// GetCoordinatorMSP returns the configured coordinator MSP ID. Args: none.
func (s *SmartContract) GetCoordinatorMSP(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 0 {
		return shim.Error("GetCoordinatorMSP expects no arguments")
	}
	mspID, err := getCoordinatorMSP(stub)
	if err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success([]byte(mspID))
}

// Invoke routes a transaction to one of the audit functions by name. Unknown
// function names are rejected with an explicit error (never a panic).
func (s *SmartContract) Invoke(stub shim.ChaincodeStubInterface) peer.Response {
	function, args := stub.GetFunctionAndParameters()
	switch function {
	case "RegisterClient":
		return s.RegisterClient(stub, args)
	case "LogPrivacyBudget":
		return s.LogPrivacyBudget(stub, args)
	case "UpdateReputation":
		return s.UpdateReputation(stub, args)
	case "PublishSubgraph":
		return s.PublishSubgraph(stub, args)
	case "GetClient":
		return s.GetClient(stub, args)
	case "GetReputation":
		return s.GetReputation(stub, args)
	case "GetSubgraph":
		return s.GetSubgraph(stub, args)
	case "GetBudgetHistory":
		return s.GetBudgetHistory(stub, args)
	case "GetCoordinatorMSP":
		return s.GetCoordinatorMSP(stub, args)
	default:
		return shim.Error(fmt.Sprintf("unknown function %q", function))
	}
}
