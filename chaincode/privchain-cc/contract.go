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

// Init is invoked on instantiation/upgrade. The ledger needs no bootstrap
// state, so this is a no-op that always succeeds.
func (s *SmartContract) Init(stub shim.ChaincodeStubInterface) peer.Response {
	return shim.Success(nil)
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
	default:
		return shim.Error(fmt.Sprintf("unknown function %q", function))
	}
}
