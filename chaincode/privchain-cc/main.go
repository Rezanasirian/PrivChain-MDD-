// Command privchain-cc is the Hyperledger Fabric chaincode for the PrivChain-MDD
// auditability layer (Phase 5, objective H3).
//
// It records, on an immutable ledger, the federated-learning artifacts that make
// the per-modality differential privacy (H1) and capability-aware aggregation
// (H2) auditable: client registration + capability declaration, the per-modality
// privacy budget consumed each round (append-only — never overwritten), each
// client's per-modality reputation, and the aggregation subgraph published per
// round. See docs/architecture/ADR-0006 for the ledger schema and the rationale
// for which records are immutable.
package main

import (
	"fmt"

	"github.com/hyperledger/fabric-chaincode-go/shim"
)

func main() {
	if err := shim.Start(new(SmartContract)); err != nil {
		fmt.Printf("error starting privchain-cc chaincode: %s\n", err)
	}
}
