# ADR-0025 — Separate privacy models for training and embedding release

- **Status:** Accepted; training claim scope corrected 2026-09-12
- **Date:** 2026-08-21
- **Phases:** 3, 5, 6
- **Related objectives:** H1 (per-modality DP), H4 (auditable accounting)

## Context

Federated training and publishing one embedding per record are different
mechanisms with different adjacency relations. Treating them as one threat
model produces privacy claims that neither implementation supports.

## Decision 1 — federated training

The protected unit is one fused training record under add/remove adjacency,
matching the Poisson-subsampled RDP accountant. A record affects every active
encoder parameter group and the shared fusion/head group; its end-to-end cost is
therefore the RDP composition of those mechanisms.

- Every client owns its accountant state.
- Noise is calibrated before training against the maximum permitted steps:
  `rounds * local_epochs * ceil(client_records / batch_size)`.
- Participation is charged under the conservative assumption that a client can
  participate in every round.
- A client has mechanisms only for modalities in its capability plus the shared
  parameter group. An absent modality consumes neither noise nor budget.
- The configured ``epsilon_m`` values are calibration targets for modality
  encoder parameter groups, not standalone guarantees for modality-specific
  records released through the fused model. A record touching more active
  groups has the composition of those groups; the matched experiments report
  the conservative all-groups participant epsilon.
- Every ledger round records both incremental and cumulative epsilon obtained
  from the same client accountant state.
- A zero noise multiplier is a numerical test facility only. It represents
  infinite epsilon and must never be entered into an accountant or ledger.

Client subsampling may later reduce actual expenditure only if a compatible
odometer or stopping rule is designed. Until then, worst-case all-round
participation remains valid and conservative.

## Decision 2 — embedding publication

The release mechanism publishes one clipped vector per record under
record-replacement adjacency. If each vector is clipped to radius `C`, two
neighboring records can differ by `2C`; Gaussian noise therefore has standard
deviation `sigma * 2C`.

This guarantee applies to one released record embedding. It is not equivalent
to participant-level DP training and must not be described as such.

## Consequences

- The server-side budget allocator cannot stand in for client training
  accountants.
- Ledger values are evidence of mechanisms actually executed by each client,
  not planned allocations.
- Training and release results report their adjacency and protected unit
  separately.
