package main

import (
	"encoding/json"
	"testing"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/shimtest"
	"github.com/hyperledger/fabric-protos-go/peer"
)

func newStub() *shimtest.MockStub {
	return shimtest.NewMockStub("privchain-cc", new(SmartContract))
}

// invoke is a small helper turning string args into the [][]byte MockInvoke wants.
func invoke(stub *shimtest.MockStub, tx string, args ...string) peer.Response {
	raw := make([][]byte, len(args))
	for i, a := range args {
		raw[i] = []byte(a)
	}
	return stub.MockInvoke(tx, raw)
}

func mustRegister(t *testing.T, stub *shimtest.MockStub, id, a, v, x string) {
	t.Helper()
	if res := invoke(stub, "reg-"+id, "RegisterClient", id, a, v, x); res.Status != shim.OK {
		t.Fatalf("RegisterClient(%s) failed: %s", id, res.Message)
	}
}

func TestRegisterClient(t *testing.T) {
	stub := newStub()

	mustRegister(t, stub, "client-0", "1", "0", "1")

	// Duplicate registration is rejected.
	if res := invoke(stub, "dup", "RegisterClient", "client-0", "1", "1", "1"); res.Status == shim.OK {
		t.Fatal("expected duplicate RegisterClient to fail")
	}
	// A client with no modalities is rejected.
	if res := invoke(stub, "empty", "RegisterClient", "client-1", "0", "0", "0"); res.Status == shim.OK {
		t.Fatal("expected all-zero capability to fail")
	}
	// Non-binary flags are rejected.
	if res := invoke(stub, "bad", "RegisterClient", "client-2", "2", "0", "0"); res.Status == shim.OK {
		t.Fatal("expected non-binary capability flag to fail")
	}
	// Wrong arity is rejected.
	if res := invoke(stub, "arity", "RegisterClient", "client-3", "1"); res.Status == shim.OK {
		t.Fatal("expected wrong arity to fail")
	}

	res := invoke(stub, "get", "GetClient", "client-0")
	if res.Status != shim.OK {
		t.Fatalf("GetClient failed: %s", res.Message)
	}
	var record ClientRecord
	if err := json.Unmarshal(res.Payload, &record); err != nil {
		t.Fatalf("bad payload: %v", err)
	}
	if record.Capability != (CapabilityVector{1, 0, 1}) {
		t.Fatalf("unexpected capability: %v", record.Capability)
	}
}

func TestLogPrivacyBudgetIsAppendOnly(t *testing.T) {
	stub := newStub()
	mustRegister(t, stub, "client-0", "1", "0", "1")

	if res := invoke(stub, "b1", "LogPrivacyBudget", "client-0", "audio", "1", "0.5"); res.Status != shim.OK {
		t.Fatalf("first LogPrivacyBudget failed: %s", res.Message)
	}
	// Overwriting the same (client, modality, round) must be rejected.
	if res := invoke(stub, "b2", "LogPrivacyBudget", "client-0", "audio", "1", "0.9"); res.Status == shim.OK {
		t.Fatal("expected append-only violation to fail")
	}
	// A later round for the same modality is allowed.
	if res := invoke(stub, "b3", "LogPrivacyBudget", "client-0", "audio", "2", "0.8"); res.Status != shim.OK {
		t.Fatalf("second-round LogPrivacyBudget failed: %s", res.Message)
	}
	// Unknown modality, negative epsilon, and unregistered client are rejected.
	if res := invoke(stub, "b4", "LogPrivacyBudget", "client-0", "smell", "1", "0.5"); res.Status == shim.OK {
		t.Fatal("expected unknown modality to fail")
	}
	if res := invoke(stub, "b5", "LogPrivacyBudget", "client-0", "audio", "3", "-1"); res.Status == shim.OK {
		t.Fatal("expected negative epsilon to fail")
	}
	if res := invoke(stub, "b6", "LogPrivacyBudget", "ghost", "audio", "1", "0.5"); res.Status == shim.OK {
		t.Fatal("expected unregistered client to fail")
	}

	res := invoke(stub, "hist", "GetBudgetHistory", "client-0", "audio")
	if res.Status != shim.OK {
		t.Fatalf("GetBudgetHistory failed: %s", res.Message)
	}
	var history []BudgetRecord
	if err := json.Unmarshal(res.Payload, &history); err != nil {
		t.Fatalf("bad payload: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected 2 budget entries, got %d", len(history))
	}
}

func TestUpdateReputation(t *testing.T) {
	stub := newStub()
	mustRegister(t, stub, "client-0", "1", "0", "1")

	if res := invoke(stub, "r1", "UpdateReputation", "client-0", "audio", "0.7", "1"); res.Status != shim.OK {
		t.Fatalf("UpdateReputation failed: %s", res.Message)
	}
	// Reputation is intentionally updatable in place.
	if res := invoke(stub, "r2", "UpdateReputation", "client-0", "audio", "0.9", "2"); res.Status != shim.OK {
		t.Fatalf("reputation update failed: %s", res.Message)
	}
	// Out-of-range score is rejected.
	if res := invoke(stub, "r3", "UpdateReputation", "client-0", "audio", "1.5", "3"); res.Status == shim.OK {
		t.Fatal("expected out-of-range score to fail")
	}

	res := invoke(stub, "rget", "GetReputation", "client-0", "audio")
	if res.Status != shim.OK {
		t.Fatalf("GetReputation failed: %s", res.Message)
	}
	var record ReputationRecord
	if err := json.Unmarshal(res.Payload, &record); err != nil {
		t.Fatalf("bad payload: %v", err)
	}
	if record.Score != 0.9 || record.Round != 2 {
		t.Fatalf("unexpected reputation: %+v", record)
	}
}

func TestPublishSubgraphIsImmutable(t *testing.T) {
	stub := newStub()

	if res := invoke(stub, "s1", "PublishSubgraph", "1", "client-0", "client-1"); res.Status != shim.OK {
		t.Fatalf("PublishSubgraph failed: %s", res.Message)
	}
	// Re-publishing the same round must be rejected.
	if res := invoke(stub, "s2", "PublishSubgraph", "1", "client-2"); res.Status == shim.OK {
		t.Fatal("expected immutable subgraph re-publish to fail")
	}
	// A different round is fine.
	if res := invoke(stub, "s3", "PublishSubgraph", "2", "client-0"); res.Status != shim.OK {
		t.Fatalf("second-round PublishSubgraph failed: %s", res.Message)
	}
	// Missing client list is rejected.
	if res := invoke(stub, "s4", "PublishSubgraph", "3"); res.Status == shim.OK {
		t.Fatal("expected empty subgraph to fail")
	}

	res := invoke(stub, "sget", "GetSubgraph", "1")
	if res.Status != shim.OK {
		t.Fatalf("GetSubgraph failed: %s", res.Message)
	}
	var record SubgraphRecord
	if err := json.Unmarshal(res.Payload, &record); err != nil {
		t.Fatalf("bad payload: %v", err)
	}
	if len(record.ClientIDs) != 2 {
		t.Fatalf("expected 2 clients in subgraph, got %d", len(record.ClientIDs))
	}
}

func TestUnknownFunction(t *testing.T) {
	stub := newStub()
	if res := invoke(stub, "u1", "NoSuchFunction"); res.Status == shim.OK {
		t.Fatal("expected unknown function to fail")
	}
}
