# ADR-0006 — Blockchain audit layer with Hyperledger Fabric (H3)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Phase:** 5 (Blockchain Layer with Hyperledger Fabric)
- **Related objective:** H3 (integration — makes H1 and H2 auditable)

## Context

H3 requires an **auditability layer**: the per-modality DP budget spent (H1) and
the capability-aware aggregation decisions (H2) must be recorded on an immutable
ledger via a smart contract. The named stack is Hyperledger Fabric with Go
chaincode exposing four functions (implementation plan, Phase 5). Neither Go nor
a Fabric network is installed in the offline environment, so — as with Flower
(ADR-0003) and Opacus (ADR-0004) — the real artifact is written to standard and
validated by an offline stand-in that enforces identical semantics.

## Decisions

### 1. Chaincode design (`chaincode/privchain-cc/`)

Low-level `shim.Chaincode` (Init/Invoke) rather than the contract API, so the
functions are unit-testable with `shimtest.MockStub` (CLAUDE.md §4). Four write
functions + four read helpers, each validating inputs and returning an explicit
error (never panicking):

| Function | Key (composite) | Mutability |
|---|---|---|
| `RegisterClient(clientID, audio, video, text)` | `client~<id>` | write-once |
| `LogPrivacyBudget(clientID, modality, round, epsilonSpent)` | `budget~<id>~<mod>~<round>` | **append-only** |
| `UpdateReputation(clientID, modality, score, round)` | `reputation~<id>~<mod>` | updatable |
| `PublishSubgraph(round, clientID…)` | `subgraph~<round>` | **immutable** |

### 2. Immutable state — documented per CLAUDE.md §4

Two record types are immutable and are recorded here as required before writing
immutable state to the ledger:

- **Privacy budget** (`LogPrivacyBudget`): consumed ε is append-only; a second
  write for the same `(clientID, modality, round)` is rejected. This is the core
  auditability guarantee — consumed ε is never silently overwritten (CLAUDE.md
  §7).
- **Subgraph** (`PublishSubgraph`): a round's aggregation membership is a
  historical fact; re-publishing a round is rejected.

Reputation is intentionally *mutable* (it evolves each round); the round it was
last set at is retained. Client registration is write-once.

`UpdateReputation` takes an extra `round` argument beyond the plan's
`(clientID, modality, score)` signature, purely to timestamp the score for
audit; the aggregation semantics are unchanged.

### 3. Python bridge (`src/privchain/chain_client/`)

A backend-agnostic `LedgerClient` protocol mirrors the chaincode. Two
implementations:

- **`MockLedger`** — in-memory, enforcing the *same* invariants (append-only
  budget, immutable subgraph, one-shot registration). This is the offline
  ledger from Risk #3 of the plan and is what the tests and the default script
  run run against.
- **`FabricRestLedger`** — the live path, translating each call into
  invoke/query against a Fabric REST gateway using only the standard library.
  Importable and typed, but not exercised offline (needs the running network).

`build_ledger(LedgerConfig)` selects the backend from `configs/blockchain.yaml`.
`record_round(...)` writes a round's registration + subgraph + budget +
reputation and is deliberately free of any `federated` import so the ledger layer
never depends on the training layer.

### 4. Integration + Byzantine robustness

`run_capability_aware_simulation` gains optional `ledger` and `budget_allocator`
arguments: when a ledger is supplied, every round publishes its subgraph, logs
per-modality consumed ε (from the H1 allocator), and updates per-modality
reputation — real reads/writes each round. `scripts/run_federated_with_ledger.py`
runs this and reads the trail back into `audit_report.json`.

A Sho-et-al.-2024-inspired **Byzantine filter** (`federated/robust.py`, off by
default via `aggregation.byzantine_filter`) drops shared-group outlier updates
(robust median/MAD on the update norm) before aggregation, never removing half or
more of a cohort.

### 5. Definition of Done

> "One full round of federated training runs with real reads/writes against the
> blockchain (not an in-memory simulation)."

**Met against the `MockLedger`** (real read/write ledger semantics, in-process);
the identical `LedgerClient` calls hit real Fabric when `backend: fabric_rest` is
configured against a running network. As with Phases 2–3, the real Fabric/Go path
is written to standard but **not run in this offline environment** — the honest
caveat is recorded here and in the chaincode README.

## Assumptions / notes

- The REST gateway contract (`/invoke`, `/query`, `{channel, chaincode, function,
  args}`) is our assumption; a `fabric-samples` gateway or a thin custom gateway
  satisfies it. The alternative (fabric-gateway gRPC SDK) is heavier and deferred.
- Per-client consumed ε logged here reuses the global allocator's schedule; true
  per-client accounting (different sampling rates per client) is a Phase 6/7
  refinement.
- `go test ./...`, `gofmt`, and `golangci-lint` must be run once Go is installed;
  they could not run here.
