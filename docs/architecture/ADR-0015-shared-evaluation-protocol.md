# ADR-0015 — One evaluation protocol for every arm, and the selection bias it removed

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** cross-cutting (Phases 1, 3, 4, 7)
- **Related objectives:** H5 (evaluation), and every claim measured against the baseline

## Context

A critical review of the work to that point found that the numbers being
reported were not comparable to each other and were optimistically biased. Four
specific defects, all introduced by me:

1. **The arms were tuned differently.** `run_dp_sweep.py` selected the
   F1-maximizing decision threshold; `train_baseline.py` used a fixed 0.5. The
   DP curve was therefore compared against a handicapped baseline — biased *in
   favour of DP*, understating its cost.
2. **The threshold was chosen on the split it was reported on.** Picking the cut
   that maximizes dev F1 and then reporting dev F1 is circular.
3. **The dev split had been exhausted by selection.** 18 hyperparameter
   configurations x 3 seeds, then best-of-60-to-200 epochs, then the threshold —
   all scored on the same 34 sessions. Nothing read off dev was still an
   estimate of generalization.
4. **Two harnesses disagreed and the disagreement was excused.** The sweep
   harness reported F1 = 0.667 and the training script 0.640 for identical
   hyperparameters. The cause was concrete, not cosmetic: `build_train_val_loaders`
   passes an explicit seeded `torch.Generator` to its `DataLoader`, while the
   sweep harness relied on the global RNG. Hyperparameters were chosen with one
   code path and reported with the other.

A fifth issue compounded these: every headline figure was single-seed, on a
split where one flipped prediction moves F1 by ~0.03.

## Decision

### 1. Three splits with distinct, enforced roles

| Split | Role | Size (real data) |
|---|---|---|
| `train` | fits parameters | 86 |
| `selection` | chooses epoch **and** decision threshold | 21 |
| `report` | read once per run; never selected on | 34 (official dev) |

`selection` is carved out of the official train split, stratified by label, so
the official dev split stays clean. This costs ~20% of the training data; that
is the price of the dev number meaning something.

### 2. One module owns the protocol

`privchain.training.protocol` provides the splits, the loaders, the seed
repetition, and the aggregation. Every arm — the non-private baseline, the DP
sweep at each ε, the modality ablation, and later the federated variants —
imports `build_splits` from it rather than constructing its own. Defect 4 was
possible only because two scripts each built their own; making that impossible
is the fix, not fixing the two call sites.

`make_loader` refuses to shuffle without an explicit seed, so the global-RNG
path that caused the divergence cannot recur.

### 3. Threshold selected on one split, applied to another

`evaluate_with_selected_threshold(model, selection_loader, report_loader, ...)`
finds the F1-maximizing cut on `selection` and applies that fixed number to
`report`. Both arms call it. Tuning the threshold is legitimate — it is
post-processing of a released model and costs no privacy budget, and DAIC-WOZ's
~28% positive rate makes 0.5 arbitrary — but it must be tuned on data that is
not then reported on, and it must be tuned for *every* arm or not at all.

### 4. Every real-data figure is a mean ± std over seeds

`train.seeds: [42, 7, 2024]`. Single-seed numbers are not reported.

## Result: the bias was ~0.10 F1

The same model and hyperparameters, before and after the protocol fix:

| | dev F1 | dev ROC-AUC | dev accuracy |
|---|---|---|---|
| As previously reported (ADR-0014) | 0.640 | 0.767 | 0.735 |
| **Under this protocol** | **0.541 ± 0.021** | **0.740 ± 0.016** | **0.735 ± 0.000** |

Per seed: F1 = 0.571 / 0.526 / 0.526 (epoch ~36 of ~76 in each; thresholds
0.495, 0.500, 0.507).

Roughly 0.10 F1 of the previously reported figure was selection bias, not
signal. ROC-AUC moved much less (0.767 → 0.740), which is what one expects: it
is threshold-independent and so was never inflated by the threshold defect, only
by best-of-N epoch selection.

Part of the drop is also real capacity loss — training now sees 86 sessions
rather than 107, because 21 went to the selection split. That trade is
deliberate: a slightly weaker model whose number can be trusted beats a stronger
one whose number cannot.

**Every figure in ADR-0012 and ADR-0014 predates this protocol and is
optimistic.** They are kept as the record of how the work developed, with this
ADR as the correction; Chapter 4 must quote the protocol-corrected numbers.

## Consequences

- The DP privacy-utility curve in ADR-0013 must be regenerated: it compared a
  threshold-tuned DP arm against a fixed-threshold baseline, so **the true cost
  of DP is larger than that ADR reports**.
- The hyperparameters adopted in ADR-0014 were chosen on the old, biased harness.
  They are not re-swept here — the ranking of configurations is less sensitive to
  this bias than the absolute level is — but that is an assumption, not a
  verified claim, and a re-sweep under this protocol is owed before Chapter 4.
- Phase 7's k-fold harness (`eval/benchmark.py`, ADR-0008) already had the right
  shape. Nesting selection inside each fold, as this protocol does inside the
  single train/dev split, is what will let Phase 7 use all 141 non-test sessions
  instead of holding 21 back permanently.
- The held-out test split (`full_test_split.csv`, n=47) remains untouched and
  must be read exactly once, at the end.
