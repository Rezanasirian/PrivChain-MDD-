package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/golang/protobuf/proto"
	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/msp"
)

// Ledger key under which the federation coordinator's MSP ID is stored. It is
// written once at Init from the instantiation arguments rather than hardcoded,
// so the same chaincode serves any network (CLAUDE.md §3).
const coordinatorMSPKey = "config/coordinatorMSP"

// callerIdentity is the authenticated submitter of a transaction.
type callerIdentity struct {
	// MSPID is the submitter's membership service provider (its organization).
	MSPID string
	// Fingerprint is a stable hash of the submitter's serialized certificate,
	// used to bind a clientID to whoever registered it without storing the cert.
	Fingerprint string
}

// getCallerIdentity extracts the submitting identity from the transaction
// creator. Returns an error rather than panicking on a missing or malformed
// creator (CLAUDE.md §4).
func getCallerIdentity(stub shim.ChaincodeStubInterface) (callerIdentity, error) {
	creator, err := stub.GetCreator()
	if err != nil {
		return callerIdentity{}, fmt.Errorf("cannot read transaction creator: %w", err)
	}
	if len(creator) == 0 {
		return callerIdentity{}, fmt.Errorf("transaction has no creator identity")
	}

	var serialized msp.SerializedIdentity
	if err := proto.Unmarshal(creator, &serialized); err != nil {
		return callerIdentity{}, fmt.Errorf("creator is not a SerializedIdentity: %w", err)
	}
	if serialized.GetMspid() == "" {
		return callerIdentity{}, fmt.Errorf("creator identity carries no MSP ID")
	}

	digest := sha256.Sum256(serialized.GetIdBytes())
	return callerIdentity{
		MSPID:       serialized.GetMspid(),
		Fingerprint: hex.EncodeToString(digest[:]),
	}, nil
}

// setCoordinatorMSP stores the MSP ID allowed to perform coordinator-only
// operations (reputation updates and subgraph publication).
func setCoordinatorMSP(stub shim.ChaincodeStubInterface, mspID string) error {
	if mspID == "" {
		return fmt.Errorf("coordinator MSP ID must not be empty")
	}
	return stub.PutState(coordinatorMSPKey, []byte(mspID))
}

// getCoordinatorMSP returns the configured coordinator MSP ID.
func getCoordinatorMSP(stub shim.ChaincodeStubInterface) (string, error) {
	blob, err := stub.GetState(coordinatorMSPKey)
	if err != nil {
		return "", fmt.Errorf("cannot read coordinator MSP: %w", err)
	}
	if len(blob) == 0 {
		return "", fmt.Errorf("no coordinator MSP configured; pass it as an Init argument")
	}
	return string(blob), nil
}

// requireCoordinator rejects the transaction unless the submitter belongs to the
// coordinator MSP. Used for records a client must not be able to write about
// itself — above all its own reputation.
func requireCoordinator(stub shim.ChaincodeStubInterface) (callerIdentity, error) {
	caller, err := getCallerIdentity(stub)
	if err != nil {
		return callerIdentity{}, err
	}
	coordinator, err := getCoordinatorMSP(stub)
	if err != nil {
		return callerIdentity{}, err
	}
	if caller.MSPID != coordinator {
		return callerIdentity{}, fmt.Errorf(
			"only the coordinator MSP %q may perform this operation (caller is %q)",
			coordinator, caller.MSPID)
	}
	return caller, nil
}

// requireOwnerOrCoordinator rejects the transaction unless the submitter either
// registered clientID or belongs to the coordinator MSP. Used for the consumed
// privacy budget: a client reports its own spend, and the coordinator may
// backfill, but no client may write another's budget.
func requireOwnerOrCoordinator(stub shim.ChaincodeStubInterface, clientID string) (callerIdentity, error) {
	caller, err := getCallerIdentity(stub)
	if err != nil {
		return callerIdentity{}, err
	}
	coordinator, err := getCoordinatorMSP(stub)
	if err != nil {
		return callerIdentity{}, err
	}
	if caller.MSPID == coordinator {
		return caller, nil
	}

	record, err := readClient(stub, clientID)
	if err != nil {
		return callerIdentity{}, err
	}
	if record.OwnerMSPID != caller.MSPID || record.OwnerFingerprint != caller.Fingerprint {
		return callerIdentity{}, fmt.Errorf(
			"identity from MSP %q does not own client %q", caller.MSPID, clientID)
	}
	return caller, nil
}
