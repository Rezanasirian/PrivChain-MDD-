# ADR-0005 — Capability-aware aggregation, reputation & federated distillation (H2)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Phase:** 4 (Capability-Aware Aggregation + Reputation + Federated Distillation)
- **Related objective:** H2 (second core novelty — completes it)

## Context

Phase 2 established plain FedAvg over a heterogeneous client population where a
client lacking a modality **zero-imputes** it (a length-1 zero sequence). Under a
single FedAvg average this is actively harmful: a text-only client contributes a
zero-signal audio/video encoder update that drags the shared encoder toward the
degenerate solution, hurting exactly the missing-modality clients the system
should protect. H2 asks for capability-declared subgraph aggregation, reputation
weighting, and federated distillation, "without losing the clinical value of
cross-modal dependencies." This ADR records how each is realized.

## Decisions

### 1. Capability-declared subgraph aggregation (Chapter 3 math)

- Indices: client `i`, modality `m ∈ {audio, video, text}`, round `t`. Each
  client declares a capability vector `c_i ∈ {0,1}^3`.
- The **subgraph** for modality `m` is `S_m = { i : c_{i,m} = 1 }` — the clients
  that actually possess `m`.
- Model parameters are partitioned by name prefix (reusing the DP-SGD grouping,
  ADR-0004): `encoders.<m>.*` belong to modality `m`; fusion + heads form the
  `shared` group `G`.
- **Update rule** for round `t`, with per-client per-group weights `w_{i,g}`:
  - modality encoder `m`: `θ_m^{t+1} = Σ_{i∈S_m} w_{i,m} θ_{i,m} / Σ_{i∈S_m} w_{i,m}`
  - shared group: `θ_G^{t+1} = Σ_i w_{i,G} θ_{i,G} / Σ_i w_{i,G}`
  - if `S_m = ∅` this round, `θ_m` is **kept at its previous global value**
    (never overwritten with a zero-signal average).

Implemented in `federated/capability.py` (`param_group`, `modality_subgraphs`)
and `federated/aggregation.py` (`capability_aware_aggregate`).

### 2. Reputation weighting

Each round every participant earns a per-group reputation `ρ_{i,g} ∈ [0,1]`:

- **volume** `v_i` = local sample count; `ṽ_i = v_i / max_{j∈S} v_j`.
- **consistency** `κ_{i,g} = max(0, cos(Δ_{i,g}, Δ̄_g))`, where `Δ_{i,g}` is the
  client's update delta for group `g` and `Δ̄_g` is the volume-weighted consensus
  delta over the subgraph. A lone member (or a degenerate zero consensus) earns
  `κ = 1`. This down-weights updates that point against the consensus — a light
  Byzantine-robustness prior (Sho et al. 2024), sharpened further in Phase 5.
- blend `r_{i,g} = α ṽ_i + (1-α) κ_{i,g}`, then EMA-smoothed across rounds:
  `ρ_{i,g} ← d·ρ_{i,g} + (1-d)·r_{i,g}`.
- aggregation weight `w_{i,g} = max(ρ_{i,g}, ρ_min) · v_i`.

With `reputation_weighting: false` this collapses to `w_{i,g} = v_i` (FedAvg-style
volume weighting) while still respecting subgraph membership. The per-client,
per-modality `ρ` snapshot is written to `reputation.jsonl` each round and is
exactly what the Phase 5 `UpdateReputation` chaincode will persist to the ledger.

Implemented in `federated/reputation.py` (`ReputationTracker`).

### 3. Federated distillation for missing-modality clients

The frozen global model at the **start** of a round is the teacher; a
missing-modality client adds a response-based KD term to its local loss:

`L = L_sup + λ · T² · BCE( z_student / T , σ(z_teacher / T) )`

on its own (capability-masked) batches. Because capability-aware aggregation keeps
the teacher's encoders clean, its predictions carry the cross-modal signal the
client lacks. `apply_to` selects whether only missing-modality clients or all
clients distill; the teacher is detached so no gradient leaks into it.

Implemented in `federated/distillation.py` and wired through
`FederatedClient.fit` (optional `teacher` / `distill_weight` args, so Phase 2
behavior is unchanged when they are omitted).

### 4. Definition of Done

`scripts/run_capability_federated.py` runs plain FedAvg and the capability-aware
protocol on the **same** partition, seed, and initial model, then writes
`comparison.json` with F1 / ROC-AUC overall and per modality-access pattern, plus
full per-run logs under `experiments/phase4/<run-id>/`. The harness surfaces the
improvement — especially for the missing-modality patterns (`audio_only`,
`text_only`) — which is the Phase 4 acceptance criterion.

## Assumptions / notes

- On mock noise data the absolute accuracy numbers are meaningless (as in Phases
  1–3); the comparison becomes informative on real DAIC-WOZ. Tests therefore
  assert the *mechanism* (subgraph isolation, reputation ordering, distillation
  minimizer, logging) rather than an accuracy delta on noise.
- Parameter→group mapping is by name prefix, shared with ADR-0004; keeping the
  two in sync is a maintenance constraint (both live behind the `encoders.<m>`
  convention).
- Consistency uses cosine agreement rather than a robust median; a stronger
  Byzantine filter (coordinate-wise trimmed mean / Krum) is deferred to Phase 5
  where it is combined with the ledger-read reputation.
- The Flower backend (`flower_app.py`) still runs the Phase 2 path; porting the
  capability-aware strategy to a Flower `Strategy` is deferred until `flwr` is
  installed (see ADR-0003).
