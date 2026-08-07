# ADR-0008 — Final comparative evaluation & prior-work baselines (H5)

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 7 (Comparative Baselines & Final Evaluation)
- **Related objective:** H5 (empirical evaluation — the Chapter-4 tables)

## Context

Phase 7 must produce the Chapter-4 tables: a comparison of the proposed
framework against a centralized baseline, plain FedAvg, and reproductions of
Xu et al. 2023 / De Chaudhury et al. 2024 / Fan et al. 2025; a 10-fold
cross-validation + held-out test for the main variants; an ablation of the
proposed framework's components; and inference latency under different compute
budgets. Most machinery already exists (Phases 1–4, plus DP-SGD from Phase 3), so
Phase 7 is an **evaluation harness**, not new modelling.

## Decisions

### 1. One shared CV protocol

`eval/benchmark.py` provides `held_out_split` (reserve a test fold), `k_fold_indices`
(k CV folds over the development set), `aggregate_metrics` (nan-aware mean/std —
ROC-AUC is undefined on a single-class fold and is skipped), and
`measure_inference_latency`. Every method is a fold closure
`(train_idx, test_idx, seed) -> metrics`, so all variants share the exact same
splits and seeds (`scripts/run_final_evaluation.py`). Reported per method: the CV
mean±std and a single held-out-fold score.

### 2. Method variants

Each variant is a configuration of components already in the repo:

| Variant | Configuration |
|---|---|
| `centralized` | Phase 1 centralized trainer (no FL, no DP) |
| `fedavg` | Phase 2 plain FedAvg over heterogeneous clients |
| `personalized` | capability-aware + reputation, no distillation |
| `proposed` | capability-aware + reputation + distillation (full) |
| `proposed_no_reputation` | proposed with reputation weighting off |

The **ablation** table reads directly off these: full (`proposed`), − reputation
(`proposed_no_reputation`), − distillation (`personalized`).

### 3. DP privacy–utility (adaptive vs uniform)

Run separately via centralized DP-SGD (Phase 3): the *adaptive* per-modality
allocation vs a *uniform* budget that splits the **same total ε** equally across
modalities. This isolates the H1 contribution (calibrating ε by modality risk)
with the privacy budget held constant. Federated per-client DP is deferred; the
DP dimension is evaluated centrally where the mechanism is already validated.

### 4. Prior-work reproductions are simplified stand-ins

The three cited papers are not available offline and target different exact
setups, so they are **not** faithfully reimplemented. Instead each is mapped to
the repo variant that captures its defining property, on the same data/model,
and labelled as such in the outputs:

- **Xu et al. 2023** → `centralized` (centralized multimodal fusion, no privacy).
- **De Chaudhury et al. 2024** → `dp_uniform` (privacy-preserving training with a
  single, non-adaptive budget).
- **Fan et al. 2025** → `personalized` (personalized/reputation aggregation).

This is an explicit, documented assumption (CLAUDE.md §9); a faithful
reproduction is a later task once the papers and real data are in hand.

### 5. Definition of Done

`scripts/run_final_evaluation.py` writes `cv_results.json`, `ablation.json`,
`dp_comparison.json`, `latency.json`, `inference_latency.png`, and a combined
`chapter4_summary.md` (markdown tables) under `experiments/phase7/<run-id>/`.
The `.png` requires the optional `viz` extra (`matplotlib`); when it is not
installed the harness logs a notice and writes `latency.json` only — the numeric
deliverable is never lost to a missing plotting library.

## Assumptions / notes

- On mock data the depression label is random, so all accuracy numbers are
  placeholders that demonstrate the table shapes; only the Phase 6 identity
  signal was ever real. DAIC-WOZ replaces these numbers without changing the
  harness.
- Federated variants report the **final-round** global model (not a
  val-selected checkpoint); the sim's internal validation loader is only used for
  its own logging and is discarded per fold (temp run dir).
- Latency is single-thread CPU forward-pass time; a GPU/quantized sweep can reuse
  `measure_inference_latency` unchanged.

---

## Revision (2026-08-06)

### R1. Cross-validation is stratified

Folds were drawn without stratification. DAIC-WOZ has ~190 sessions with a
minority depressed class, so 10-fold splitting routinely yields single-class test
folds on which ROC-AUC is undefined. `aggregate_metrics` then dropped those nan
values *silently*, so a reported "10-fold mean ROC-AUC" could in fact be a mean
over 7 folds with nothing in the output saying so.

`stratified_k_fold_indices` / `stratified_held_out_split` preserve the class
balance in every fold (and refuse a split that would leave a class with fewer
members than folds). `scripts/run_final_evaluation.py` derives the labels from
the dataset and uses them, printing the positive rate alongside the split sizes.

### R2. Fold counts are reported, not hidden

`aggregate_metrics` now emits `<metric>_num_valid_folds` next to every mean and
std. A count below `num_folds` is a visible signal that a metric was undefined
somewhere, instead of a silently shrinking denominator.

### R3. Splitting granularity

Splits are at session granularity, which on DAIC-WOZ is also subject granularity
(one interview per participant), so no subject spans a split. This is now stated
in `benchmark.py`: if the pipeline ever moves to utterance-level items, these
functions must be replaced by group-aware variants keyed on participant id, or
every reported number will be inflated by subject leakage.
