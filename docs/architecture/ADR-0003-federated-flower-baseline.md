# ADR-0003 — Phase 2 heterogeneous federated baseline (Flower)

- **Status:** Accepted
- **Date:** 2026-07-01
- **Phase:** 2 (Simulating Heterogeneous Federated Clients with Flower)
- **Related objective:** H2 (federation basics, no privacy / no blockchain yet)

## Context

Phase 2 establishes federation *before* the per-modality DP (Phase 3) and the
capability-aware protocol (Phase 4). The thesis names **Flower** as the
orchestration framework, but `flwr` is an optional dependency and the
development/CI environment is offline (no PyPI), so a Flower-only implementation
could not be run or tested here.

## Decisions

### 1. Two backends over one shared core

The local-training and aggregation logic lives in framework-agnostic modules:

- `federated/partition.py` — heterogeneous client partitioning + modality masking
- `federated/client.py` — `FederatedClient` (local fit/evaluate)
- `federated/aggregation.py` — `fedavg` (sample-count-weighted parameter mean)

Two backends drive them:

- **`simulation.py`** — an in-house FedAvg server loop. Runs offline with no
  extra deps, so it is unit/integration tested and produces the Phase 2 results.
- **`flower_app.py`** — a Flower `NumPyClient` adapter + `start_simulation`
  FedAvg strategy that reuses the *same* `FederatedClient`. Lazily imports
  `flwr`; importing the module does not require it.

This keeps the thesis-mandated Flower path first-class while guaranteeing the
phase is verifiable offline.

### 2. Heterogeneous modality access

Clients are assigned capability vectors `[audio, video, text]` from the
population mix in `configs/federated.yaml` (default at N=10: 40% full, 30%
audio+text, 20% audio-only, 10% text-only). A client lacking a modality has that
modality replaced by a **length-1 zero sequence** (`ModalityMaskedDataset`).

### 3. Deliberately naive FedAvg

Plain FedAvg averages *all* client models, including the zero-imputed encoders of
missing-modality clients — there is intentionally **no** missing-modality
handling, reputation, or Byzantine robustness yet. Demonstrating the resulting
degradation is itself a Chapter 4 result and motivates the Phase 4 protocol.

### 4. Evaluation

Each round, the aggregated global model is evaluated on a held-out
**full-modality** validation split (F1/ROC-AUC/accuracy), logged per round to
`experiments/phase2/<run-id>/metrics.jsonl`.

## Assumptions / notes to validate

- The Flower adapter targets `flwr.simulation.start_simulation` with
  `NumPyClient` and `to_client()`. Flower's API has changed across versions;
  validate `flower_app.py` against your installed `flwr` (and consider the newer
  `ClientApp`/`ServerApp` API). It has **not** been executed in this offline
  environment.
- Mock data is random noise (ADR-0001), so Phase 2 metrics are meaningless in
  absolute terms; the degradation *comparison* becomes meaningful on real
  DAIC-WOZ.

## Consequences

- `scripts/run_federated.py` runs either backend (`--backend sim|flower`).
- The same `FederatedClient` carries into Phase 4, where aggregation/weighting is
  swapped out for the capability-aware protocol.

---

## Amendment, 2026-08-13 (ADR-0021) — Flower has now been executed

This ADR recorded that the Flower adapter was written but could not be run
offline, and that the in-house simulator would produce the Phase 2 results. That
gap is now closed: `scripts/run_flower_parity.py` runs both backends on identical
splits, partitions, initial parameters and seed, and the two **agree** (F1 and
accuracy identical, ROC-AUC within 0.022 against a 0.05 tolerance fixed in
advance).

Getting there required fixing three real defects in the adapter and one in the
harness — all invisible to inspection, all only reachable by running it:

- clients were constructed in the parent process, so Ray pickled CUDA tensors
  into every worker; they are now built inside `client_fn`;
- Ray workers were granted no GPU, so a CUDA client could not deserialize;
- Flower's `History` does not carry the final global parameters, so the caller
  was evaluating the untrained initial model.

Each of these failed **silently**: Flower logs `received 0 results and N
failures` and continues with the initial parameters, which reads as a converged
run. The lesson generalises beyond Flower — an integration that has never been
executed should be described as untested, never as working.

`flwr` also constrains the environment: 1.12 uses `np.float_` and cannot coexist
with the NumPy 2.x that torch and scipy require. Version 1.33 works, and is
installed into a separate directory placed on `PYTHONPATH` only for the parity
check, so it cannot perturb the main environment's dependency set again.
