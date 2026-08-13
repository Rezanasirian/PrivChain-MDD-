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
	"os"

	"github.com/hyperledger/fabric-chaincode-go/shim"
)

// chaincodeID resolves the package ID the peer knows this chaincode by, under
// either of the two environment variable names Fabric has used for it.
func chaincodeID() string {
	if id := os.Getenv("CHAINCODE_ID"); id != "" {
		return id
	}
	return os.Getenv("CORE_CHAINCODE_ID_NAME")
}

func main() {
	// Chaincode-as-a-Service: the chaincode listens and the peer dials it,
	// instead of the peer building and launching a Docker container that dials
	// back. This is what makes the chaincode runnable where there is no Docker
	// daemon — the deployment environment here, and the normal arrangement for
	// Kubernetes deployments generally (ADR-0022). Selected by environment, so
	// the classic Docker-launched path is untouched when the variable is absent.
	if address := os.Getenv("CHAINCODE_SERVER_ADDRESS"); address != "" {
		server := &shim.ChaincodeServer{
			CCID:     chaincodeID(),
			Address:  address,
			CC:       new(SmartContract),
			TLSProps: shim.TLSProperties{Disabled: true},
		}
		if err := server.Start(); err != nil {
			fmt.Printf("error starting privchain-cc chaincode server: %s\n", err)
			os.Exit(1)
		}
		return
	}

	if err := shim.Start(new(SmartContract)); err != nil {
		fmt.Printf("error starting privchain-cc chaincode: %s\n", err)
		os.Exit(1)
	}
}
