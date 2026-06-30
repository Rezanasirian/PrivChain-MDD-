# `privchain-cc` — Hyperledger Fabric chaincode (Go)

**Status: not yet implemented — scheduled for Phase 5** (objective H3).

This package will hold the Go chaincode with the four core functions named in
the implementation plan and [CLAUDE.md](../../CLAUDE.md) §4:

- `RegisterClient(clientID, capabilityVector)`
- `LogPrivacyBudget(clientID, modality, round, epsilonSpent)`
- `UpdateReputation(clientID, modality, score)`
- `PublishSubgraph(round, []clientID)`

Each function must validate inputs, return explicit errors (never panic), and
ship with a `shimtest` (MockStub) unit test. `gofmt` + `golangci-lint` are
mandatory before commit.

> Go and the Fabric `fabric-samples` test network are Phase 0 install items but
> are **not** installed in the current environment — set them up before
> starting Phase 5.
