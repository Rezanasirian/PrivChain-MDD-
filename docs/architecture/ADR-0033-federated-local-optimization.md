# ADR-0033 — The federated local optimizer is a config choice, measured on train only

- **Status:** Accepted
- **Date:** 2026-09-13
- **Phase:** 4 (federated aggregation) — objective H2
- **Related:** ADR-0015, ADR-0020, ADR-0021, ADR-0026, ADR-0029

## Context

Two facts had to be reconciled before Chapter 4 could quote a federated number.

1. The final evaluation of 2026-08-23 reported centralized ROC-AUC 0.676 against
   plain FedAvg 0.439 on the official dev split — a collapse, not a cost.
2. The CTD audit (`docs/evaluation/CTD-FEDERATED-AUDIT-2026-09-10-FA.md`) traced
   a similar collapse to optimization rather than to federation: the seed
   distribution was bimodal, and a single-client parity test reproduced
   centralized training to seven digits, so the aggregation loop was sound. Its
   candidate repair — class weights from securely summed counts, plain SGD, and a
   larger local step — was found while looking at the official **test** split,
   so adopting it on that evidence would repeat the defect ADR-0029 corrected.

Until 2026-09-13 the local optimizer, the local learning rate, and the client
class-weighting mode were implicit in the code: every federated arm ran Adam with
per-shard weights at the centralized learning rate. Those are hyperparameters,
and CLAUDE.md §3 does not allow them to live in code.

## Decision

1. `federation.class_weight_mode`, `federation.local_optimizer` and
   `federation.local_learning_rate` are added to `configs/federated.yaml` and
   validated by `FederationConfig`. Their defaults reproduce the previous
   behaviour exactly (`null`, `adam`, `null`), so no committed result moves.
2. The choice between them is decided by `scripts/run_federated_optimizer_cv.py`:
   five inner folds over the pooled official **train** split, three seeds, each
   arm's out-of-fold predictions pooled and compared paired. Official dev and
   test are never constructed, scored, or selected on.
3. A candidate replaces the default only if its paired interval against the
   **committed federated arm** excludes zero. Being closer to the centralized arm
   is not sufficient: that comparison has no power at this sample size.
4. Out-of-fold score vectors are persisted (`oof_scores.json`) so an arm-vs-arm
   question can be asked later without retraining.

## Measurement

`experiments/phase4/phase4_federated_optimizer_cv_20260913_082526` (full grid)
and `..._20260913_084212` (per-shard arms, with out-of-fold scores).

| arm | ROC-AUC | ±sd | Δ vs centralized | 95% CI | p |
|---|---|:--:|---|---|---|
| centralized | 0.719 | 0.092 | – | – | – |
| per_shard + SGD + lr 0.1 | 0.629 | 0.124 | -0.120 | [-0.249, +0.014] | 0.083 |
| aggregate_counts + SGD + lr 0.1 | 0.622 | 0.108 | -0.091 | [-0.223, +0.048] | 0.214 |
| **per_shard + Adam + lr 0.0003 (committed)** | 0.613 | 0.125 | -0.095 | [-0.220, +0.032] | 0.157 |
| aggregate_counts + Adam + lr 0.0003 | 0.603 | 0.121 | -0.179 | [-0.323, -0.037] | 0.016 |
| aggregate_counts + SGD + lr 0.0003 | 0.524 | 0.123 | -0.239 | [-0.415, -0.050] | 0.008 |
| per_shard + SGD + lr 0.0003 | 0.523 | 0.122 | -0.239 | [-0.422, -0.052] | 0.013 |
| per_shard + Adam + lr 0.1 | 0.450 | 0.155 | -0.258 | [-0.421, -0.093] | 0.001 |
| aggregate_counts + Adam + lr 0.1 | 0.423 | 0.146 | -0.276 | [-0.437, -0.117] | 0.000 |

Paired against the committed arm:

| candidate | Δ vs committed | 95% CI | p |
|---|---|---|---|
| per_shard + SGD + lr 0.1 | -0.025 | [-0.118, +0.078] | 0.659 |
| per_shard + SGD + lr 0.0003 | -0.144 | [-0.299, +0.001] | 0.053 |
| per_shard + Adam + lr 0.1 | -0.162 | [-0.346, +0.032] | 0.118 |

## Consequences

- **The defaults stand.** No candidate beats the committed setting; the closest,
  SGD at lr 0.1, is indistinguishable from it (Δ = -0.025, p = 0.659). The CTD
  audit's repair is therefore *not* adopted for the multimodal pipeline.
- **Federation costs a little here, it does not collapse.** On train-only inner
  folds the committed federated arm reaches 0.613 against centralized 0.719, and
  the paired interval includes zero. The 0.439 dev number of 2026-08-23 is a
  property of that 34-session split and its seeds, not evidence that FedAvg
  destroys the signal. Chapter 4 must report the dev/test figure it measures, and
  say this alongside it.
- Learning rate is the dominant factor, not the class-weight mode: Adam at lr 0.1
  loses 0.26 AUC, and the two weighting modes never separate.
- `aggregate_counts` remains available and documented; it is a secure-aggregation
  affordance (ADR-0026), not an accuracy fix.

## Boundaries

- Inner-fold folds hold ~21 participants, so each arm's own interval is wide;
  only the paired comparison is interpretable, and even it cannot resolve
  differences below roughly 0.1 AUC (see `daic-woz-underpowered`).
- A null result across this grid does not prove no local-optimization schedule
  helps. It closes the schedules the audit nominated.
