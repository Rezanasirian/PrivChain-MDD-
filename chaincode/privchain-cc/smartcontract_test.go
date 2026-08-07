package main

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/shimtest"
	"github.com/hyperledger/fabric-protos-go/msp"
	"github.com/hyperledger/fabric-protos-go/peer"
)

const (
	coordinatorMSP = "CoordinatorMSP"
	clientMSP      = "ClientOrgMSP"
	otherMSP       = "OtherOrgMSP"
)

// asIdentity sets the stub's transaction creator to a serialized identity, the
// way a real peer would. shimtest exposes Creator directly, so access control
// can be tested without minting x509 certificates.
func asIdentity(stub *shimtest.MockStub, mspID, certBody string) {
	creator, err := proto.Marshal(&msp.SerializedIdentity{
		Mspid:   mspID,
		IdBytes: []byte(certBody),
	})
	if err != nil {
		panic(err)
	}
	stub.Creator = creator
}

// initArgs builds the instantiation arguments as a real deployment passes them:
// the function name first, then the parameters
// (`-c '{"Args":["init","CoordinatorMSP"]}'`).
func initArgs(params ...string) [][]byte {
	raw := [][]byte{[]byte("init")}
	for _, p := range params {
		raw = append(raw, []byte(p))
	}
	return raw
}

// newStub builds a stub initialized with the coordinator MSP, acting by default
// as a member of that MSP.
func newStub() *shimtest.MockStub {
	stub := shimtest.NewMockStub("privchain-cc", new(SmartContract))
	asIdentity(stub, coordinatorMSP, "coordinator-cert")
	stub.MockInit("init", initArgs(coordinatorMSP))
	return stub
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

func TestInitRequiresCoordinatorMSP(t *testing.T) {
	stub := shimtest.NewMockStub("privchain-cc", new(SmartContract))
	asIdentity(stub, coordinatorMSP, "coordinator-cert")

	if res := stub.MockInit("init-empty", initArgs()); res.Status == shim.OK {
		t.Fatal("expected Init without a coordinator MSP to fail")
	}
	if res := stub.MockInit("init-ok", initArgs(coordinatorMSP)); res.Status != shim.OK {
		t.Fatalf("Init failed: %s", res.Message)
	}
	res := invoke(stub, "cfg", "GetCoordinatorMSP")
	if res.Status != shim.OK || string(res.Payload) != coordinatorMSP {
		t.Fatalf("unexpected coordinator MSP: %q (%s)", res.Payload, res.Message)
	}
	// Re-init without args keeps the stored value (upgrade path).
	if res := stub.MockInit("init-again", initArgs()); res.Status != shim.OK {
		t.Fatalf("re-Init without args should keep the stored MSP: %s", res.Message)
	}
}

func TestRegisterClient(t *testing.T) {
	stub := newStub()

	asIdentity(stub, clientMSP, "client-0-cert")
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
	// Registration binds the client to its submitter.
	if record.OwnerMSPID != clientMSP || record.OwnerFingerprint == "" {
		t.Fatalf("client was not bound to its registering identity: %+v", record)
	}
}

func TestRegisterClientRejectsMissingCreator(t *testing.T) {
	stub := newStub()
	stub.Creator = nil
	if res := invoke(stub, "anon", "RegisterClient", "client-x", "1", "0", "0"); res.Status == shim.OK {
		t.Fatal("expected an unauthenticated RegisterClient to fail")
	}
}

func TestLogPrivacyBudgetIsAppendOnly(t *testing.T) {
	stub := newStub()
	asIdentity(stub, clientMSP, "client-0-cert")
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

func TestLogPrivacyBudgetRequiresOwnership(t *testing.T) {
	stub := newStub()
	asIdentity(stub, clientMSP, "client-0-cert")
	mustRegister(t, stub, "client-0", "1", "0", "1")

	// A different organization must not be able to forge this client's spend.
	asIdentity(stub, otherMSP, "attacker-cert")
	if res := invoke(stub, "forge-msp", "LogPrivacyBudget", "client-0", "audio", "1", "9.9"); res.Status == shim.OK {
		t.Fatal("expected a foreign MSP to be rejected")
	}

	// Nor may a different identity inside the same organization.
	asIdentity(stub, clientMSP, "client-1-cert")
	if res := invoke(stub, "forge-cert", "LogPrivacyBudget", "client-0", "audio", "1", "9.9"); res.Status == shim.OK {
		t.Fatal("expected a different identity in the same MSP to be rejected")
	}

	// The owner may.
	asIdentity(stub, clientMSP, "client-0-cert")
	if res := invoke(stub, "own", "LogPrivacyBudget", "client-0", "audio", "1", "0.5"); res.Status != shim.OK {
		t.Fatalf("owner should be allowed to log its own budget: %s", res.Message)
	}

	// And so may the coordinator (backfill path).
	asIdentity(stub, coordinatorMSP, "coordinator-cert")
	if res := invoke(stub, "coord", "LogPrivacyBudget", "client-0", "audio", "2", "0.6"); res.Status != shim.OK {
		t.Fatalf("coordinator should be allowed to log a budget: %s", res.Message)
	}
}

func TestBudgetHistoryIsOrderedByRound(t *testing.T) {
	// Rounds are stored zero-padded precisely so that lexicographic ledger
	// iteration is numeric: a bare decimal key would return round 10 before 2.
	stub := newStub()
	asIdentity(stub, clientMSP, "client-0-cert")
	mustRegister(t, stub, "client-0", "1", "0", "1")

	for round := 1; round <= 12; round++ {
		tx := fmt.Sprintf("b%d", round)
		epsilon := fmt.Sprintf("%.2f", 0.1*float64(round))
		if res := invoke(stub, tx, "LogPrivacyBudget", "client-0", "audio", fmt.Sprint(round), epsilon); res.Status != shim.OK {
			t.Fatalf("LogPrivacyBudget(round=%d) failed: %s", round, res.Message)
		}
	}

	res := invoke(stub, "hist", "GetBudgetHistory", "client-0", "audio")
	if res.Status != shim.OK {
		t.Fatalf("GetBudgetHistory failed: %s", res.Message)
	}
	var history []BudgetRecord
	if err := json.Unmarshal(res.Payload, &history); err != nil {
		t.Fatalf("bad payload: %v", err)
	}
	if len(history) != 12 {
		t.Fatalf("expected 12 budget entries, got %d", len(history))
	}
	for i, record := range history {
		if record.Round != i+1 {
			t.Fatalf("history is out of order at index %d: got round %d", i, record.Round)
		}
	}
}

func TestUpdateReputation(t *testing.T) {
	stub := newStub()
	asIdentity(stub, clientMSP, "client-0-cert")
	mustRegister(t, stub, "client-0", "1", "0", "1")
	asIdentity(stub, coordinatorMSP, "coordinator-cert")

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

func TestUpdateReputationIsCoordinatorOnly(t *testing.T) {
	// The whole point: a client must not be able to raise its own aggregation
	// weight by writing its own reputation.
	stub := newStub()
	asIdentity(stub, clientMSP, "client-0-cert")
	mustRegister(t, stub, "client-0", "1", "0", "1")

	if res := invoke(stub, "self", "UpdateReputation", "client-0", "audio", "1.0", "1"); res.Status == shim.OK {
		t.Fatal("expected a client to be barred from setting its own reputation")
	}
	asIdentity(stub, otherMSP, "attacker-cert")
	if res := invoke(stub, "other", "UpdateReputation", "client-0", "audio", "0.0", "1"); res.Status == shim.OK {
		t.Fatal("expected a foreign MSP to be barred from setting reputation")
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

func TestPublishSubgraphIsCoordinatorOnly(t *testing.T) {
	stub := newStub()
	asIdentity(stub, clientMSP, "client-0-cert")
	if res := invoke(stub, "s1", "PublishSubgraph", "1", "client-0"); res.Status == shim.OK {
		t.Fatal("expected a client to be barred from publishing a subgraph")
	}
}

func TestSubgraphRoundsAreRetrievableBeyondTen(t *testing.T) {
	// Guards the same zero-padding fix on the subgraph key space.
	stub := newStub()
	for round := 1; round <= 12; round++ {
		tx := fmt.Sprintf("s%d", round)
		if res := invoke(stub, tx, "PublishSubgraph", fmt.Sprint(round), "client-0"); res.Status != shim.OK {
			t.Fatalf("PublishSubgraph(round=%d) failed: %s", round, res.Message)
		}
	}
	for _, round := range []string{"2", "10", "12"} {
		res := invoke(stub, "get"+round, "GetSubgraph", round)
		if res.Status != shim.OK {
			t.Fatalf("GetSubgraph(%s) failed: %s", round, res.Message)
		}
		var record SubgraphRecord
		if err := json.Unmarshal(res.Payload, &record); err != nil {
			t.Fatalf("bad payload: %v", err)
		}
		if fmt.Sprint(record.Round) != round {
			t.Fatalf("GetSubgraph(%s) returned round %d", round, record.Round)
		}
	}
}

func TestUnknownFunction(t *testing.T) {
	stub := newStub()
	if res := invoke(stub, "u1", "NoSuchFunction"); res.Status == shim.OK {
		t.Fatal("expected unknown function to fail")
	}
}
