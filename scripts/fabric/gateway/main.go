// Command privchain-gateway fronts the privchain-cc chaincode with the small JSON
// API that privchain.chain_client.fabric_gateway.FabricRestLedger already speaks
// (Phase 5, ADR-0022).
//
// The Python bridge was written against a documented gateway contract and has
// never been executed, because the gateway did not exist. Rather than replace the
// bridge with something easier to run, this supplies the missing half, so the
// client that Chapter 4 describes is the client that actually gets exercised.
//
// Contract (see the Python module's docstring):
//
//	POST /invoke  {"channel","chaincode","function","args":[...]}  -> {"payload": "..."}
//	POST /query   same shape                                        -> {"payload": "..."}
//	failure                                                         -> non-2xx {"error": "..."}
package main

import (
	"crypto/x509"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/hyperledger/fabric-gateway/pkg/client"
	"github.com/hyperledger/fabric-gateway/pkg/identity"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type request struct {
	Channel   string   `json:"channel"`
	Chaincode string   `json:"chaincode"`
	Function  string   `json:"function"`
	Args      []string `json:"args"`
}

// firstFile returns the single file inside dir, which is how Fabric's MSP layout
// stores the signing certificate and private key.
func firstFile(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			return filepath.Join(dir, entry.Name()), nil
		}
	}
	return "", fmt.Errorf("no file found in %s", dir)
}

func newIdentity(mspDir, mspID string) (*identity.X509Identity, identity.Sign, error) {
	certPath, err := firstFile(filepath.Join(mspDir, "signcerts"))
	if err != nil {
		return nil, nil, err
	}
	certPEM, err := os.ReadFile(certPath)
	if err != nil {
		return nil, nil, err
	}
	cert, err := identity.CertificateFromPEM(certPEM)
	if err != nil {
		return nil, nil, err
	}
	id, err := identity.NewX509Identity(mspID, cert)
	if err != nil {
		return nil, nil, err
	}

	keyPath, err := firstFile(filepath.Join(mspDir, "keystore"))
	if err != nil {
		return nil, nil, err
	}
	keyPEM, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, nil, err
	}
	privateKey, err := identity.PrivateKeyFromPEM(keyPEM)
	if err != nil {
		return nil, nil, err
	}
	sign, err := identity.NewPrivateKeySign(privateKey)
	if err != nil {
		return nil, nil, err
	}
	return id, sign, nil
}

func main() {
	var (
		listen     = envOr("GATEWAY_LISTEN", "127.0.0.1:8801")
		peerAddr   = envOr("PEER_ENDPOINT", "localhost:7051")
		peerHost   = envOr("PEER_HOST_ALIAS", "peer0.org1.privchain.local")
		tlsCert    = os.Getenv("PEER_TLS_CA")
		mspDir     = os.Getenv("MSP_DIR")
		mspID      = envOr("MSP_ID", "Org1MSP")
		defChannel = envOr("CHANNEL", "privchain-channel")
		defCC      = envOr("CHAINCODE", "privchain-cc")
	)
	if tlsCert == "" || mspDir == "" {
		log.Fatal("PEER_TLS_CA and MSP_DIR are required")
	}

	pem, err := os.ReadFile(tlsCert)
	if err != nil {
		log.Fatalf("cannot read peer TLS CA: %v", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		log.Fatal("peer TLS CA is not a valid certificate")
	}
	conn, err := grpc.NewClient(peerAddr,
		grpc.WithTransportCredentials(credentials.NewClientTLSFromCert(pool, peerHost)))
	if err != nil {
		log.Fatalf("cannot dial the peer: %v", err)
	}
	defer func() { _ = conn.Close() }()

	id, sign, err := newIdentity(mspDir, mspID)
	if err != nil {
		log.Fatalf("cannot load the gateway identity: %v", err)
	}
	gw, err := client.Connect(id,
		client.WithSign(sign),
		client.WithClientConnection(conn),
		client.WithEvaluateTimeout(30*time.Second),
		client.WithEndorseTimeout(30*time.Second),
		client.WithSubmitTimeout(30*time.Second),
		client.WithCommitStatusTimeout(60*time.Second),
	)
	if err != nil {
		log.Fatalf("cannot connect the gateway: %v", err)
	}
	defer gw.Close()

	handle := func(submit bool) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			var req request
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeError(w, http.StatusBadRequest, err)
				return
			}
			if req.Channel == "" {
				req.Channel = defChannel
			}
			if req.Chaincode == "" {
				req.Chaincode = defCC
			}
			contract := gw.GetNetwork(req.Channel).GetContract(req.Chaincode)

			// Endorsement is pinned to this organisation rather than left to
			// discovery. A single-org network with no anchor peers configured has
			// nothing to discover, and the gateway otherwise fails with "no peer
			// combination can satisfy the endorsement policy" even though the one
			// peer it is connected to is a perfectly good endorser.
			options := []client.ProposalOption{
				client.WithArguments(req.Args...),
				client.WithEndorsingOrganizations(mspID),
			}

			var payload []byte
			var callErr error
			if submit {
				payload, callErr = contract.Submit(req.Function, options...)
			} else {
				payload, callErr = contract.Evaluate(req.Function, options...)
			}
			if callErr != nil {
				// The chaincode's own rejection messages (append-only, immutability,
				// authorization) arrive here; pass them through so the Python side
				// can surface the reason rather than a bare 500.
				writeError(w, http.StatusBadGateway, callErr)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"payload": string(payload)})
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/invoke", handle(true))
	mux.HandleFunc("/query", handle(false))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	log.Printf("privchain-gateway listening on %s -> peer %s (%s/%s)",
		listen, peerAddr, defChannel, defCC)
	server := &http.Server{Addr: listen, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(server.ListenAndServe())
}

func writeError(w http.ResponseWriter, code int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
