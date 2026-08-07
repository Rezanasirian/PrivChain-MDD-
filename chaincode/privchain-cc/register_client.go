package main

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-protos-go/peer"
)

// RegisterClient records a federated client and its declared modality
// capability vector, binding it to the submitting identity. That binding is what
// later lets LogPrivacyBudget reject a client writing another client's consumed
// epsilon (ADR-0006).
//
// Args: clientID, audio, video, text — where the last three are 0/1 flags.
// A client must declare at least one modality, and may not be registered twice.
func (s *SmartContract) RegisterClient(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 4 {
		return shim.Error("RegisterClient expects 4 args: clientID, audio, video, text")
	}
	clientID := args[0]
	if clientID == "" {
		return shim.Error("clientID must not be empty")
	}

	caller, err := getCallerIdentity(stub)
	if err != nil {
		return shim.Error(err.Error())
	}

	var capability CapabilityVector
	for i := 0; i < 3; i++ {
		flag, err := strconv.Atoi(args[i+1])
		if err != nil || (flag != 0 && flag != 1) {
			return shim.Error(fmt.Sprintf("capability flag for %s must be 0 or 1", modalities[i]))
		}
		capability[i] = flag
	}
	if capability[0]+capability[1]+capability[2] == 0 {
		return shim.Error("a client must declare at least one modality")
	}

	key, err := stub.CreateCompositeKey(clientObjectType, []string{clientID})
	if err != nil {
		return shim.Error(err.Error())
	}
	existing, err := stub.GetState(key)
	if err != nil {
		return shim.Error(err.Error())
	}
	if existing != nil {
		return shim.Error(fmt.Sprintf("client %q is already registered", clientID))
	}

	blob, err := json.Marshal(ClientRecord{
		ClientID:         clientID,
		Capability:       capability,
		OwnerMSPID:       caller.MSPID,
		OwnerFingerprint: caller.Fingerprint,
	})
	if err != nil {
		return shim.Error(err.Error())
	}
	if err := stub.PutState(key, blob); err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(blob)
}

// GetClient returns a registered client's record. Args: clientID.
func (s *SmartContract) GetClient(stub shim.ChaincodeStubInterface, args []string) peer.Response {
	if len(args) != 1 {
		return shim.Error("GetClient expects 1 arg: clientID")
	}
	record, err := readClient(stub, args[0])
	if err != nil {
		return shim.Error(err.Error())
	}
	blob, err := json.Marshal(record)
	if err != nil {
		return shim.Error(err.Error())
	}
	return shim.Success(blob)
}

// readClient loads a registered client's record, or reports that it is absent.
func readClient(stub shim.ChaincodeStubInterface, clientID string) (ClientRecord, error) {
	key, err := stub.CreateCompositeKey(clientObjectType, []string{clientID})
	if err != nil {
		return ClientRecord{}, err
	}
	blob, err := stub.GetState(key)
	if err != nil {
		return ClientRecord{}, err
	}
	if blob == nil {
		return ClientRecord{}, fmt.Errorf("client %q is not registered", clientID)
	}
	var record ClientRecord
	if err := json.Unmarshal(blob, &record); err != nil {
		return ClientRecord{}, fmt.Errorf("client %q has a corrupt record: %w", clientID, err)
	}
	return record, nil
}
