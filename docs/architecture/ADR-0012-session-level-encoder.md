# ADR-0012 — Session-level statistical encoder for the Phase 1 baseline

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 1 (Centralized Multimodal Baseline)
- **Related objectives:** H4 (prototype), H5 (evaluation)
- **Supersedes:** the encoder-type default in [ADR-0002](ADR-0002-daic-woz-integration.md) §4, for real-data runs

## Context

[ADR-0011](ADR-0011-baseline-training-protocol.md) fixed the training protocol
(dev-split validation, class weighting, full-session coverage) and closed with an
unresolved problem: **the model did not fit its own training data.** Training
loss was flat across every epoch, and the model oscillated between two degenerate
constant predictors — all-negative (`F1 = 0`) and all-positive (`recall = 1.0`,
`accuracy` = the positive rate).

The encoder inherited from ADR-0002 is a bidirectional DPGRU over the padded
sequence followed by masked mean-pooling. After ADR-0011 widened coverage to the
whole interview, that meant a recurrent model over ~3,000 timesteps per modality,
trained on **107 sessions**.

Two independent problems followed from that:

1. **It did not learn.** 107 samples cannot constrain a recurrent model of that
   capacity, and mean-pooling 3,000 recurrent states averages away the localized
   cues the recurrence exists to capture.
2. **It was too slow to iterate on.** A GRU is inherently sequential — step
   *t+1* waits on step *t* — so 3,000 steps cannot be parallelized by adding
   hardware. A 20-session, 4-configuration diagnostic ran >12 minutes at 99% GPU
   utilization without finishing a single configuration. On a rented GPU this
   made the experiment loop unaffordable, and Phases 3–7 need *dozens* of runs
   (the ε sweep, federated rounds, the final comparison table).

Note that GPU memory was never the constraint: peak usage was ~2.5 GB of 24 GB.
The bottleneck is sequential depth, which more VRAM cannot address.

## Decision

### 1. Add a `stats` encoder type and make it the real-data default

A third `EncoderConfig.type` alongside `mean` and `gru`. It summarizes each
session into fixed-size **statistical functionals** before any learned layer,
then applies an MLP:

```
(B, T, D) --masked_statistics--> (B, 5D) --Linear+ReLU--> (B, hidden) --Linear--> (B, out_dim)
```

Per feature channel, over valid timesteps only: **mean, standard deviation, min,
max, and mean absolute first difference**. The last term retains a coarse measure
of temporal variability that a plain mean discards.

This is the "functionals over low-level descriptors" representation used by the
**AVEC2017 DAIC-WOZ baseline itself**, so the baseline is now methodologically
comparable to the literature this thesis is measured against, rather than being
an idiosyncratic sequence model that happens not to work at this data size.

Sequence length stops mattering for compute: the session collapses before the
first parameter, so the model is small and fast regardless of interview length.

**Interface is unchanged.** `SequenceEncoder` still maps
`(B, T, input_dim) + lengths → (B, out_dim)`, so fusion, the prediction heads,
the federated clients, and the DP wrapper are untouched.

**DP compatibility is preserved.** Every statistic is computed from a single
sample's own valid prefix, so an embedding never depends on its batch mates —
the property Opacus needs for correct per-sample gradients, and the same property
the hand-unrolled bidirectional DPGRU was built to guarantee (ADR-0002). The
functionals themselves are parameter-free; only `nn.Linear` layers carry
gradients. A unit test asserts batch-independence directly.

The `gru` path is kept, not deleted: it remains the right choice if a future
phase adds segment-level chunking (splitting each interview into many short
segments), which is how the literature makes deep sequence models work on
DAIC-WOZ. That is recorded as possible future work, not adopted now.

### 2. Memoize parsed features on disk

Parsing dominated wall-clock: a COVAREP file is ~36 MB / ~90k rows, and the
subsampling loop must *read* every row to keep every `frame_stride`-th one. That
cost was paid again on every run — minutes per experiment, repeated across the
sweep.

`DaicWozDataset` now writes each subsampled matrix to `feature_cache_dir`
(default `data/daic_woz/_feature_cache`, git-ignored, 138 MB). The cache key is a
hash of **all** parsing options, so changing `frame_stride`, `max_frames`, or
`standardize` produces a new entry instead of silently reusing a stale one.
Entries are written via a temp file and renamed, so an interrupted run cannot
leave a truncated `.npy` that a later run would load as valid data. Set
`feature_cache_dir: null` to disable.

### 3. Select on F1, not ROC-AUC

ADR-0011 chose ROC-AUC for being threshold-independent. In practice it was
actively harmful: it selected epoch 1 — whose F1 was 0 — and triggered early
stopping at epoch 11 while training loss was still falling. F1 is also the
headline metric AVEC2017 reports for this task. `selection_metric: f1`, with
`early_stopping_patience: 40` and `epochs: 200`, since a 34-session dev split
gives a noisy per-epoch signal.

## Hyperparameters

Grid over learning rate x hidden dim x dropout, 3 seeds each, selecting on mean
dev F1 (ROC-AUC reported alongside so a degenerate high-AUC/zero-F1 configuration
could not win silently):

| lr | hidden | dropout | dev F1 (mean ± spread) | dev AUC |
|---|---|---|---|---|
| **1e-3** | **64** | **0.3** | **0.588 ± 0.037** | **0.675** |
| 3e-3 | 32 | 0.3 | 0.588 ± 0.037 | 0.643 |
| 3e-3 | 128 | 0.5 | 0.584 ± 0.021 | 0.636 |
| 3e-4 | 128 | 0.5 | 0.498 ± 0.029 | 0.524 |

`lr = 1e-3, hidden = 64, dropout = 0.3` is adopted (tied on F1, clearly better on
AUC). The whole sweep — 18 configurations x 3 seeds — completes in minutes with
the feature cache warm; before these changes a single configuration did not
finish in twelve.

## Result

Single run at the adopted configuration (seed 42, dev split, n=34):

```
Device=cuda  pos_weight=2.567  epochs_run=119/200
Best (epoch 79, by f1) — F1=0.5600  ROC-AUC=0.6640  acc=0.6765
```

Progression across the three ADRs:

| | dev F1 | dev ROC-AUC | Behaviour |
|---|---|---|---|
| ADR-0010 (first real run) | 0.0000 | 0.5432 | predicts one class |
| ADR-0011 (protocol fixed) | 0.0000 | 0.6838 | oscillates between constants |
| **ADR-0012 (this)** | **0.5600** | **0.6640** | learns; loss converges |

This is in the range the AVEC2017 DAIC-WOZ baseline reports for depression
classification, so it is a defensible reference point for the phases that are
measured against it.

## Consequences

- Phase 1 now has a baseline that actually learns, which is the precondition for
  every later comparison: the DP cost curve (H1), the capability-aware
  aggregation gain (H2), and the final comparison table (H5) are all differences
  from this number.
- Phases 3–7 must use `encoder.type: stats` when running on real data, or their
  results will not be comparable to this baseline.
- Experiment iteration is now cheap enough for the ε sweep and the federated
  simulations to be run repeatedly rather than once.
- **Not claimed:** that this is a competitive DAIC-WOZ result. It is a sound
  baseline, not a tuned state-of-the-art system; the thesis contribution is the
  privacy/federation/audit layer built on top of it.
- Test-set numbers are deliberately not reported here. `full_test_split.csv` is
  touched only once, for the final Chapter 4 table, to keep it uncontaminated by
  the model selection performed above.
