# ADR-0022 — A real Fabric ledger without Docker, and what standing it up exposed

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 5
- **Related objectives:** H3 (auditable federation on a blockchain)

## Context

Phase 5's Definition of Done is *"one full round of federated training runs with
real reads/writes against the blockchain (not an in-memory simulation)"*. It had
only ever been met against `MockLedger`. ADR-0006 left the real network open.

That mattered more than it looked, because the pattern of this project has been
that code which has never run is broken. The Flower adapter was in that state
last phase and had four defects. `FabricRestLedger` and the chaincode were in the
same state here.

## Docker is not available, and that was established rather than assumed

Both routes were tested on the deployment container:

- **Normal Docker** needs `CAP_SYS_ADMIN` to mount overlayfs and to create each
  container's mount namespace. The effective capability set is
  `dac_override, fowner, fsetid, kill, setgid, setuid, setpcap,
  net_bind_service, sys_chroot, mknod, audit_write, setfcap` — no
  `CAP_SYS_ADMIN`, and no daemon or socket present.
- **Rootless Docker/Podman** would work around that with a user namespace, and
  the kernel permits them (`max_user_namespaces = 255787`,
  `unprivileged_userns_clone = 1`) — but `unshare -Urm` returns *Operation not
  permitted*, so the container's seccomp profile blocks the syscall. `/dev/fuse`
  is absent too, ruling out `fuse-overlayfs`.

This is deliberate hardening by the host, not a misconfiguration: nested
containers would let a tenant escape the resource controls.

## Decision: Chaincode-as-a-Service

Fabric needs Docker for exactly one thing — building and running the chaincode
container. `peer`, `orderer`, `configtxgen`, `cryptogen` and `osnadmin` are static
binaries. Since Fabric 2.4, an **external builder** lets chaincode run as an
ordinary process that the peer dials, so:

| component | how it runs here |
|---|---|
| orderer (etcdraft) | native process |
| peer | native process |
| channel | genesis via `configtxgen`, joined through the participation API |
| chaincode | native process; the package only carries the address to dial |
| lifecycle | unchanged: install → approve → commit → init |

Only the chaincode *launcher* differs from the Docker test network. Consensus,
endorsement, validation, the ledger and the lifecycle are identical — and CCaaS
is what Kubernetes deployments use in production, so this is closer to a real
deployment than the Docker test network, not further from it.

## What standing it up exposed

### 1. The chaincode had never compiled, because .gitignore silently ate two files

`.gitignore` carried:

```
chaincode/**/privchain-cc
```

intended for the compiled binary. As a gitignore pattern it also matches the
**directory** `chaincode/privchain-cc/`. Files added *before* the rule stayed
tracked — which is exactly what hid it — but every file added afterwards was
silently skipped by `git add`. Two were missing from the repository:

- `identity.go`, providing `getCallerIdentity`, `setCoordinatorMSP`,
  `getCoordinatorMSP`, `requireCoordinator` and `requireOwnerOrCoordinator`;
- `go.sum`.

So a clean checkout produced a chaincode that could not resolve its dependencies
and, once they were resolved, failed with seven undefined symbols. Every claim
that the chaincode "passes 12 shimtest unit tests" was untested: there was no Go
toolchain on the box and the code did not build.

The rule is now `chaincode/privchain-cc/privchain-cc`, which matches the binary
and not the directory.

With that fixed and the files committed: `gofmt -l` clean, `go vet` clean, and
**all 12 shimtest tests pass** — for the first time.

### 2. The external builder must live in `bin/`

Fabric requires `detect`/`build`/`release` under a `bin/` subdirectory of the
configured builder path. With them one level up, the peer finds no `detect`,
**silently falls back to its Docker builder**, and fails with
`dial unix /var/run/docker.sock: no such file or directory` — an error that
points at Docker rather than at the actual mistake.

### 3. My own verification script reported the opposite of the truth

The first run of `verify_ledger.sh` claimed *"duplicate budget write was accepted
— append-only is NOT enforced"*. That was a bug in the check, not the chaincode.
`peer` exits non-zero when the chaincode rejects a transaction — which is the
outcome being asserted — and piping it into `grep` under `set -o pipefail`
returns the peer's failure even when grep matched. A correct rejection therefore
read as a failed check.

Worth recording because it is the same failure mode as the Flower defects, seen
from the other side: a harness that can report success when nothing ran, or
failure when everything worked, is worse than no harness. The fix captures the
output before testing it.

## Result

Network: single-org, etcdraft orderer, one peer, channel `privchain-channel`,
chaincode `privchain-cc` deployed through the full lifecycle
(`install → approveformyorg → commit → Init`), all transactions committing VALID.

`scripts/fabric/verify_ledger.sh` establishes, in increasing order of how hard
each is to fake:

1. **A write reads back.** `RegisterClient` then `GetClient` returns the record,
   including the owner MSP and certificate fingerprint the chaincode derived.
2. **It survives a peer restart.** The peer is killed and restarted, and the
   record is still there. This is the step that distinguishes committed ledger
   state from a process that merely remembered.
3. **The real chaincode enforces the invariants**, not just the Python mock:
   - a second `LogPrivacyBudget` for the same (client, modality, round) is
     rejected — *"consumed epsilon is append-only and must not be overwritten"*;
   - republishing a subgraph for an existing round is rejected — *"subgraph for
     round 1 already published; it is immutable"*.

   These are the invariants CLAUDE.md §7 requires of the privacy accounting, and
   until now only `MockLedger` had ever demonstrated them.
4. Chain height 11 with real block hashes.

## Consequences

- Everything generated — MSP material, TLS keys, ledger data — lives in
  `.fabric/`, git-ignored. No key or certificate is committed (CLAUDE.md §7); the
  setup script regenerates the lot.
- `configs/blockchain.yaml` keeps `backend: mock` by default, so CI and offline
  runs are unchanged; the real network is opt-in.
- The chaincode's Go module is now part of the repository and must stay that way.
  A `.gitignore` pattern that can match a directory is a hazard worth avoiding
  generally — this one hid missing source for the entire life of Phase 5.

---

## Addendum — the Python bridge, and four more defects that only running found

### The gateway that was missing

`FabricRestLedger` was written against a documented JSON contract and had never
been executed, because nothing implemented that contract. Rather than replace it
with something easier to run, `scripts/fabric/gateway/` supplies the missing half
in Go using the official `fabric-gateway` SDK — so the client Chapter 4 describes
is the client that actually gets exercised.

Running it end to end found four more defects, none visible by inspection:

1. **Endorsement could not be satisfied.** The gateway failed with *"no peer
   combination can satisfy the endorsement policy"*. A single-org network with no
   anchor peers has nothing for discovery to find; the fix is to name the
   endorsing organisation explicitly (`WithEndorsingOrganizations`).
2. **Registration was not idempotent.** `record_round` tracked registrations in a
   per-process set, but a ledger outlives the process — and the chaincode rightly
   rejects a duplicate `RegisterClient`. The first audited run succeeded and every
   later one aborted. It now checks the ledger, which is what makes the audit
   trail re-runnable. Covered by a regression test.
3. **Background nodes died on script exit.** The orderer, peer and chaincode were
   started as background jobs and killed by SIGHUP when their script returned, so
   a network that reported itself up refused connections at the next step. They
   are `setsid`-detached now.
4. **`localhost` resolved to `::1`**, where nothing listens; the orderer binds
   IPv4. Addressed explicitly as `127.0.0.1`.

### The raft WAL outlived the purge

The most instructive one. A second `setup_network.sh` always died with
*"Could not append block: unexpected Previous block hash"*, which reads like
ledger corruption. The orderer log gave it away: `newRaft [term: 2, commit: 21,
lastindex: 21]` on a supposedly empty ledger.

The block ledger lived under `.fabric/` and was purged; the **raft write-ahead log
and snapshots** defaulted to `/var/hyperledger/production/orderer/etcdraft/`, well
outside it. So each rerun replayed a previous network's raft entries against a
freshly generated genesis block. That is exactly why the first run always worked
and every rerun failed — the failure mode looked like corruption and was really
incomplete cleanup. `ORDERER_CONSENSUS_WALDIR` and `ORDERER_CONSENSUS_SNAPDIR`
now live inside `.fabric/` too.

Two smaller ones fixed alongside: the package ID was read from
`lifecycle chaincode queryinstalled | head -1`, which returns *any* previously
installed package and silently deploys a stale build — it now comes from the
install response; and the shell scripts had picked up CRLF endings, where a `\r`
after `set -euo pipefail` makes bash reject the option (`.gitattributes` now pins
LF).

## End-to-end result

`teardown --purge → setup_network → deploy_chaincode → gateway → federated round`
runs clean from nothing:

- chaincode deployed through the full lifecycle, all transactions VALID;
- `scripts/fabric/verify_ledger.sh` passes: writes read back, **survive a peer
  restart**, and the real chaincode rejects both a duplicate privacy-budget entry
  and a republished subgraph;
- `scripts/run_federated_with_ledger.py` with `backend: fabric_rest` completes
  three rounds and reads its audit trail **back from the ledger**. The consumed
  per-modality ε converges on exactly the configured caps — audio 1.10 → 1.60 →
  2.00, video 2.16 → 3.18 → 4.00, text 4.22 → 6.30 → 8.00 against ε = 2 / 4 / 8 —
  which is the H1 accounting and the H2 subgraph, made auditable by H3;
- final channel height 68, with real block hashes.

**Phase 5's Definition of Done is met**: a full federated round with real reads
and writes against a real blockchain, not an in-memory simulation.

The accuracy numbers in that run come from mock data and remain meaningless; what
is demonstrated is the audit trail, which is what H3 claims.
