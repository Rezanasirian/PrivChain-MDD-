# ADR-0023 — The Chapter-4 evaluation protocol on real DAIC-WOZ, and switching to measured-risk allocation

- **Status:** Accepted
- **Date:** 2026-08-14
- **Phase:** 7
- **Related objectives:** H5 (comparative evaluation), H1 (per-modality DP)

## Context

Phase 7's Definition of Done — *"all tables and plots needed for Chapter 4 are
generated"* — was recorded as met on 2026-08-06. It was met against mock data,
where the depression label is random by construction, and the entry said so:
the accuracy numbers were placeholders demonstrating table shapes.

Phases 1–6 were all re-run on the real corpus on 2026-08-13. Phase 7 was not, and
could not have been: `scripts/run_final_evaluation.py` had no `--daic-config`
option and constructed `MockDaicWozDataset` directly. The one artifact Chapter 4
is assembled from was the only one that had never seen DAIC-WOZ.

## Decision 1 — folds come from pooled train+dev; the official test is reserved

DAIC-WOZ ships an official AVEC2017 three-way split, but the thesis protocol asks
for 10-fold cross-validation. These do not compose: 10-fold CV over the official
train split alone wastes dev, and CV over everything destroys the only number
comparable to published work.

**Folds are drawn from the pooled official train+dev split (141 sessions), and
`full_test_split` is held back for exactly one read per method at the end**,
written to `official_test.json`. The CV table carries the thesis protocol and its
fold-to-fold dispersion; the official-test column carries comparability.

Two properties make the pooling sound rather than convenient:

- **No normalization leakage.** `configs/daic_woz.yaml` sets
  `normalization: session` for audio and video, so feature statistics are
  computed per participant. Pooling train and dev therefore cannot let a fold's
  test participants influence its training features — the objection a reader
  would raise first. The alternative `corpus` mode (ADR-0019) fits its statistics
  on the train split only, so it would not leak either, but it would tie the
  fold statistics to the official split boundary rather than the fold's own.
- **No selection on the reported split.** Each fold carves its own selection
  split out of its own training indices (`_carve_fold`), and that split — never
  the fold's scored data — drives early stopping and the decision threshold.

The official test sessions are concatenated after the pool and addressed by
index, so the same fold runners score them with no second code path.

## Decision 2 — every arm trains under the Phase 1 protocol

The harness previously ran `for _ in range(epochs)` with `epochs: 3`, no
selection split, no early stopping, and no threshold selection. Meanwhile the
Phase 1 baseline on the same corpus trains up to 200 epochs with patience 40,
selects on F1, restores the best checkpoint, and fixes its threshold on selection
(ADR-0015) — reaching accuracy 0.735, F1 0.541, ROC-AUC 0.740.

Left alone, Chapter 4's headline "centralized" row would have been a model
trained to a small fraction of convergence, contradicting the baseline reported
two chapters earlier in the same thesis. **Every arm now follows ADR-0015**,
including the per-fold `pos_weight` that `class_weighting: true` asks for, since
each fold has its own class balance.

`dp_epochs` is deliberately separate from `epochs` and fixed at the Phase 3
sweep's 60. DP-SGD cannot early-stop: epochs are a *privacy* parameter, so
stopping early does not refund budget the accountant has already charged
(ADR-0012). Sharing one knob would have quietly coupled the privacy guarantee to
a convergence heuristic.

## Decision 3 — per-modality budgets derive from measured risk

`allocation.mode` moves from `explicit` to `inverse_risk`, which ADR-0017
explicitly left open as *"a deliberate change, not a side effect of measuring"*.

The hand-set budgets were audio 2.0 / video 4.0 / text 8.0 — a 2× separation
between audio and video. The Phase 6 measurement does not support it: normalized
re-identification risk came out 1.00 / 0.97 / 0.29, with audio and video
indistinguishable (22.2× vs 21.6× chance) and only text clearly separated. Both
arms are rescaled to the same composed participant ε afterwards (ADR-0018), so
what actually changes is the allocation *shape*: audio : video : text goes from
`1 : 2 : 4` to `1 : 1.03 : 3.45`.

The cost is that Chapter 1's asserted `audio > video > text` ordering is no
longer visible in the budgets — only `{audio, video} > text` is. That is what the
data showed, and asserting the finer ordering in the mechanism while the
measurement chapter reports it as unsupported would be the worse outcome. The
benefit is that the allocation becomes a *result*: budgets derived from measured
leakage rather than assumed leakage.

This also removes an inconsistency that was already in the thesis, in the
opposite direction from the one first assumed. `allocation.mode` is read only by
`allocate_target_epsilons`, which Phase 7's DP arm uses;
`scripts/run_allocation_comparison.py` never reads it and computes its arms
directly from the measured risks via `budget_shapes` (`risk ** -gamma`). So
Phase 3's "adaptive" arm was *already* risk-derived while Phase 7's "adaptive"
arm was the hand-set 2/4/8 — **two different allocations reported under one
name**. Changing Phase 7 is what reconciles them; Phase 3 needed no correction,
and was re-run only to produce a same-day artifact alongside the new tables.

## What pointing the harness at real data exposed

Four defects, none of which would have failed loudly:

1. **Input dims came from the mock config.** Every arm called
   `modality_input_dims(base.data)`, yielding the mock's 40/49/64 rather than the
   corpus's 74/20/768. The model would have been built at the wrong shape.
2. **CPU was hardcoded** in the DP and federated arms.
3. **Stratification decoded the whole corpus** — `full[i]["label"]` across 141
   sessions, materializing audio and video to obtain 141 integers. `labels_of`
   reads them from the split records.
4. **No `pos_weight`**, despite `class_weighting: true`, on a corpus with a 0.291
   positive rate.

## What the run measured, including the parts that do not support the thesis

Run `experiments/phase7/phase7_final_evaluation_20260814_083048`, 141 pooled
sessions, positive rate 0.291, 10 folds, RTX 4090.

| method | CV ROC-AUC | CV F1 | official test ROC-AUC |
|---|---|---|---|
| centralized | 0.680 ± 0.149 | 0.494 ± 0.138 | **0.709 ± 0.052** |
| fedavg | 0.584 ± 0.239 | 0.401 ± 0.166 | 0.422 ± 0.009 |
| personalized | 0.573 ± 0.256 | 0.382 ± 0.161 | 0.464 ± 0.012 |
| proposed | 0.580 ± 0.260 | 0.386 ± 0.164 | 0.462 ± 0.011 |
| proposed − reputation | 0.574 ± 0.255 | 0.386 ± 0.164 | 0.467 ± 0.014 |

The centralized arm reproduces the Phase 1 baseline (0.740), which is the check
that the protocol is what it claims to be. Three results do **not** support the
thesis and are recorded here rather than tuned away:

1. **The proposed framework is indistinguishable from plain FedAvg.** CV ROC-AUC
   spans 0.573–0.584 across all four federated arms while the fold-to-fold
   standard deviation is ±0.25. Removing reputation moves the mean by 0.006.
   Phase 4 reached the same conclusion on the dev split with a paired test
   (ADR-0021); Phase 7 does not overturn it. **H2 is not demonstrated on this
   corpus.**

2. **The DP allocation axis does nothing measurable.** The adaptive and uniform
   arms return *identical* F1, ROC-AUC and accuracy — only `loss_mean` differs
   (0.6901457 vs 0.6901366), the signature of two models whose weights differ but
   whose decisions and score ordering do not. The mechanism is genuinely
   different (σ = 9.43/9.17/3.27 adaptive vs 5.50 uniform), so this is collapse,
   not a wiring fault: at a composed participant ε of 8.0 over 60 epochs with
   C = 0.1, DP-SGD on 141 sessions learns essentially nothing either way.
   Phase 3's re-run agrees and is the stronger evidence, because it includes the
   control this table lacks: adaptive − uniform = +0.016 (95% CI [−0.058,
   +0.092], p = 0.746) and **anti_adaptive − uniform = 0.000 (p = 1.0)**. An
   anti-adaptive arm — deliberately giving the *riskiest* modalities the *most*
   budget — scoring identically to uniform is what
   `run_allocation_comparison.py` documents as the signal that the allocation
   axis is inert. **H1's utility claim is not demonstrated on this corpus**; its
   privacy claim stands on Phase 6 independently.

3. **Every federated arm falls to or below chance on the official test split**
   (0.422–0.467) while the centralized arm holds 0.709. The seed-to-seed spread
   is tiny (±0.01), so this is systematic rather than noise. The plausible cause
   is the known distribution shift between the AVEC2017 dev and test splits
   interacting with 10 clients of ~8 sessions each, but it is **not established**
   here and is flagged as open rather than explained.

Inference latency is 0.64 ms/batch, flat from batch 1 to 16, i.e. 0.64 → 0.041
ms/sample: the model is small enough that per-sample cost is dominated by fixed
overhead, which is the useful form of the compute-budget result.

## Consequences

- Chapter 4's tables are generated from the real corpus under one protocol shared
  with Chapter 3's baseline, so rows are comparable to each other and to Phase 1.
- The `official_test.json` column is the only figure comparable to published
  DAIC-WOZ work, and it is reported across three seeds because a single fit on
  47 sessions is not a result.
- CI still runs offline: with no `--daic-config` the mock path is unchanged, and
  `tests/unit/test_phase7_real_data_wiring.py` pins the pooling and the
  selection-split disjointness against a tiny on-disk fixture.
- Any Chapter-4 figure produced before 2026-08-14 is superseded.
