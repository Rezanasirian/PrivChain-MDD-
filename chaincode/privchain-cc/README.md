# `privchain-cc` — Hyperledger Fabric chaincode (Go)

**Status: implemented in Phase 5** (objective H3). Ledger schema and immutability
decisions are recorded in [ADR-0006](../../docs/architecture/ADR-0006-blockchain-audit-layer.md).

The chaincode records the auditability layer for per-modality DP (H1) and
capability-aware aggregation (H2). It uses the low-level `shim.Chaincode`
(Init/Invoke) interface so it can be unit-tested with `shimtest.MockStub`
(CLAUDE.md §4).

## Functions

| Function | Args | Semantics |
|---|---|---|
| `RegisterClient` | `clientID, audio, video, text` | Register a client + capability vector; rejects duplicates and all-zero capability. |
| `LogPrivacyBudget` | `clientID, group, round, epsilonIncremental, epsilonCumulative` | **Append-only** executed-accountant entry; rejects overwriting `(clientID, group, round)` (CLAUDE.md §7). |
| `UpdateReputation` | `clientID, modality, score, round` | Set latest per-modality reputation (score ∈ [0,1]); updatable by design. |
| `PublishSubgraph` | `round, clientID…` | **Immutable** per-round aggregation subgraph; rejects re-publish. |
| `GetClient` / `GetReputation` / `GetSubgraph` / `GetBudgetHistory` | (reads) | Query helpers used by the Python bridge. |

Every function validates its inputs and returns an explicit error (never panics).

## Files

- `main.go` — chaincode entry point (`shim.Start`).
- `contract.go` — `SmartContract` + `Init`/`Invoke` dispatch.
- `models.go` — record structs, composite-key object types, modality helpers.
- `register_client.go`, `log_privacy_budget.go`, `update_reputation.go`, `publish_subgraph.go` — the functions.
- `smartcontract_test.go` — `shimtest.MockStub` unit tests for all four (happy path + validation + append-only/immutability).

## Build / test / lint

```bash
go mod tidy
gofmt -l .           # must print nothing
golangci-lint run    # must pass
go test ./...        # runs the MockStub tests
```

> Go and the Fabric `fabric-samples` test network are **not** installed in the
> current (offline) environment, so the above have not been run here — the
> Python side is validated end-to-end against the in-memory `MockLedger`
> (`src/privchain/chain_client/ledger.py`), which enforces the same invariants.
> See ADR-0006.
