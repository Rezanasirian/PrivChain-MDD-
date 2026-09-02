# ADR-0027 — Segment-aligned, quality-gated multimodal fusion

- **Status:** Accepted
- **Date:** 2026-09-01
- **Phase:** 1 (centralized multimodal baseline), objective H4, with effects on Phases 2–4
- **Supersedes nothing.** Extends ADR-0012 (session-level encoder), ADR-0014 (transformer
  text encoder) and ADR-0019 (feature normalization).

## Context

The committed model reduces a ~15-minute clinical interview to one vector per modality.
`daic_woz.text.representation` defaults to `document`, so every participant turn is
concatenated into a single MPNet embedding and the text encoder pools over a length-1
sequence. Audio and video are collapsed by `masked_statistics` into five functionals per
channel over the whole session. Fusion then concatenates the three session vectors.

Whatever the interview's structure carried is gone before the model sees anything: which
answer was about sleep and which about hopelessness, in what order they came, and how the
participant sounded *while* saying each one.

Measured on real DAIC-WOZ under the locked protocol (ADR-0015):

| arm | ROC-AUC |
|---|---|
| text only | 0.710 |
| audio only (session normalization) | 0.503 |
| video only | 0.494 |
| all three | 0.740 |
| audio only (no normalization) | 0.619 |

Two things follow. The tri-modal model is not significantly better than text alone under a
paired bootstrap, so audio and video are contributing close to nothing as currently
represented. And audio gains 0.12 ROC-AUC when normalization is removed, so `session`
z-scoring is deleting signal rather than nuisance (as ADR-0019 warned it might).

## Decision

Keep the model small — 107 training sessions and DP-SGD noise rule out a tri-modal
transformer — and spend the change on structure instead of capacity.

```
interview → K turn-group envelopes over the participant's timestamped turns
  text[k]  = MPNet(turns in group k)
  audio[k] = functionals(COVAREP rows inside the UNION of participant intervals in k)
  video[k] = functionals(AU rows inside group k's envelope)
  quality[m][k] = [valid, ...]
        ↓ encoders.<m>: project each segment, and score how much to trust it
        ↓ masked softmax across modalities, per segment
        ↓ additive attention over valid segments (+ sinusoidal positions)
        ↓ binary head + PHQ-8 head
```

### D1 — Audio takes participant speech; video takes the whole envelope

A group's envelope runs from its first turn's `start_time` to its last turn's `stop_time`,
so it also spans Ellie's questions. Attributing the interviewer's voice to the participant
would corrupt the acoustic branch, so audio frames are selected from the **union of the
participant's own `[start_time, stop_time]` intervals**.

Video deliberately keeps the full envelope: facial behaviour while *listening* is
plausibly informative for depression, and unlike audio it is unambiguously the
participant's. The asymmetry is a decision, not an oversight.

These are **turn-group envelopes, not a partition of the interview**. Consecutive envelopes
can have a gap between them, and nothing in the code or the writing claims otherwise. The
alternative — extending boundaries to the midpoint of each gap — would assign silence to
whichever group happened to be adjacent, which is a stronger claim than the data supports.

### D2 — Timestamps never come from retained row positions

The parser records each kept row's index **in the source file**, before striding and before
any malformed row is skipped. Audio time is `source_row_index / sample_rate_hz`; video uses
OpenFace's own `timestamp` column. Deriving audio time from the position in the subsampled
matrix would shift every later frame whenever one row was dropped — and the result would be
invisible, since every segment would still be full, just of the wrong frames.

Features, timestamps and quality columns are produced by **one parsing pass**
(`ParsedFeatures`). A second parser reading the metadata columns separately would only have
to skip one malformed line differently to attach every quality value to the wrong frame.

A transcript timestamp that is missing, unparseable, or non-finite makes the whole pair
unusable: the turn keeps its text and gets a **zero-length** interval, contributing no
frames. A half-readable pair is never completed from its surviving endpoint — a turn with
an unreadable start and a stop of 120 would claim the interview's first two minutes of
audio, and the mis-attribution would be invisible because every segment would still look
full. `nan` and `inf` parse happily as floats and are rejected for the same reason: they
would select either nothing or everything.

### D3 — Validity is explicit at every level

`quality[m][k][0]` is `valid ∈ {0, 1}` by contract, for every modality. A group with text
but no usable audio frames gets `audio.valid = 0`. A segment where no modality is valid
fuses to zeros with zero gates and is masked out of the temporal attention; a sample with
no valid segment at all pools to zeros.

This is not defensive decoration. A softmax over three `-inf` scores is `NaN`, the state is
reachable (a silent participant, or modality dropout removing every branch), and a `NaN`
that appears in one sample propagates into every parameter's gradient for the whole batch.

### D4 — Quality vectors have fixed, published widths

The model sizes its gate layers from these, so they cannot drift with the loader. Unbounded
counts are `log1p`-compressed so the gate cannot simply learn "more frames = better" off a
raw scale.

| modality | vector | width |
|---|---|---|
| audio | `[valid, voiced_ratio, log1p(frame_count)]` | 3 |
| video | `[valid, mean_confidence, success_ratio, log1p(frame_count)]` | 4 |
| text | `[valid, log1p(token_count), log1p(turn_count)]` | 3 |

`voiced_ratio` reads the COVAREP VUV column, configured by index because that file has no
header; clearing `audio.quality_columns` reports 0 and the gate simply cannot use it.

### D5 — Quiet participants are padded, never dropped

`contiguous_spans` cannot split 3 turns into 8 groups, yet the model needs a fixed `(K, D)`.
So `K_eff = min(K, n_turns)` real segments, right-padded to `K` with zero features,
`valid = 0`. Requiring `K` turns would drop the quietest participants, who are not a random
subset in a depression corpus.

One `SegmentPlan` is computed per session and consumed by all three modalities. If text
grouped its turns while audio and video sliced their own frame counts, segment `k` would
mean a different stretch of interview in each branch and "aligned" would be a claim the data
does not support.

### D6 — Modality-specific parameters stay in a modality group

Per-modality DP budgets (`map_parameter_groups`) and capability-aware aggregation
(`param_group`) both classify parameters by name prefix; anything not matching falls into
`shared`. So in the new network each modality encoder owns **its projection and its gate
head**, under `encoders.<modality>.`, and the fusion module holds only the masked softmax,
the weighted sum, and a genuinely shared post-projection.

The existing `GatedFusion` (commit e15e951) put its per-modality gate heads at
`fusion.gates.<modality>`, which the rule classified as `shared`. A capability-restricted
client's audio-gate gradient is already zero — presence zeroes it — so the bug is not that
those parameters train on nothing. It is that they are **averaged in the shared pool**,
diluting the updates from clients that do hold the modality, and that their DP noise is
charged to the wrong ε group.

Rather than relocate them, which would rename `state_dict` keys and orphan every checkpoint
written before this ADR, the **grouping rule was extended** to recognize
`fusion.gates.<modality>` as that modality's group, in both consumers. The math is
unchanged and old checkpoints still load with `strict=True`.

### D7 — Huber for the PHQ-8 head, in normalized units

The regression target is `phq8_score / phq8_max`, so `huber_delta` lives in `[0, 1]`:
`0.1` is about 2.4 PHQ-8 points. A delta of `1.0` would keep every achievable error inside
the quadratic region and be indistinguishable from MSE. Default stays `mse`; the ladder's
segment arms are what test the change.

`DepressionObjective` is now built by `build_objective(model_config, ...)` at every
config-driven call site. There are 24 construction sites; hand-constructing them is what
would let one arm keep MSE while another, under the same config, got Huber — a difference
that would surface as a method effect in Chapter 4.

### D8 — Modality dropout during centralized training

The centralized model sees all three modalities in every step, then meets Phase 2 clients
holding one or two. The draw is **per sample** (a batch-level draw gives only a handful of
distinct capability mixes per epoch at batch size 32 on 107 sessions), seeded, applied to
training steps only, and it never mutates the batch.

## What was deliberately not done

No MTC-Former, no tri-modal transformer, no multi-layer cross-attention. With 107 training
sessions, a frozen MPNet already accounting for most of the parameter count, DP-SGD adding
noise per parameter, and federated averaging on top, the expected outcome is an
unreproducible fit. If the segment-gated architecture shows a real gain, a **single-layer,
low-dimensional cross-attention** is the natural next ablation.

## Consequences

- The parsed-feature cache moves from `.npy` to `.npz` (it now carries timestamps and
  quality columns); the extension change retires every entry written by the values-only
  parser. The text cache schema is bumped 2 → 3.
- `Sample`/`Batch` gain an optional `quality` key (`NotRequired`, so nothing that omits it
  breaks — the synthesized distillation anchors included). `move_batch_to_device` now moves
  any mapping-valued field rather than only `presence`.
- `ModalityMaskedDataset` preserves the segment count when blanking an absent modality on
  segment-aligned samples, and still collapses to one zero frame otherwise — keeping 3000
  zero frames per absent modality would make every capability-restricted client pay to
  encode padding.
- `build_depression_model` is the only way production paths construct a model.
  `run_attack_eval.py`, `run_reid_risk.py` and `run_text_representation.py` reach into the
  baseline model's structure, so they call `require_baseline_architecture` and refuse a
  `segment_gated` config instead of quietly measuring the wrong model. A test enforces that
  no other script constructs `MultimodalDepressionModel` directly.
- `run_reid_risk.py` refuses to run with `segments.enabled`: `session_views.segment_session`
  slices the frame matrix, and with segments on it would slice segment rows instead —
  changing what Phase 6 measures without changing what it reports.
- `average_precision` is added to the metrics module and surfaced as `pr_auc`. It is
  **average precision**, not a trapezoidal PR integral; the two are both called "PR-AUC" and
  they disagree, since interpolating between PR operating points is optimistic.

## Evaluation

`scripts/run_segment_architecture.py` runs the ladder, each rung changing one thing:

1. `document+concat` — the committed baseline
2. `document+gated` — the existing gated fusion
3. `segments+attn` — segment text with attention, audio/video genuinely ablated
4. `segments+av+gated` — segment text plus session-level audio/video through the existing gate
5. `aligned+quality_gated` — the full architecture, corpus audio normalization
6. `aligned+dropout` — arm 5 plus modality dropout
7. `aligned+session_norm` — arm 5 under session normalization, **sensitivity only**

Ablation in arm 3 zeroes the features *and* clears presence (`MaskedModalityModel`).
Shrinking the input to one frame per session — the earlier trick — is not an ablation: that
frame is real data and the presence flag still says the modality is there. The wrapper
multiplies the arm's mask into the batch's **incoming** presence rather than replacing it,
so an arm that keeps video cannot hand a partial client back a modality it never had.

Arm 6 changes the PHQ-8 loss and nothing else, deliberately as its own rung: folding Huber
into arm 5 would leave the architecture's effect and the loss's effect inseparable.

Ranking is on inner k-fold CV over the official **train** split only (107 participants; 141
is train + dev). Arms are compared by paired bootstrap on out-of-fold scores **averaged
across seeds and resampled per participant**. Concatenating the seeds would enter each
participant five times and break the resampling's independence assumption, narrowing every
interval for free. Per-fold win counts are reported alongside as a stability check, not as
the test.

```
inner CV over train   → model-selection evidence (all arms)
official dev          → unbiased report for the ONE selected arm
official test         → read once, after the architecture is frozen
```

The run manifest records `official_dev_scored: false` and `official_dev_constructed: true`
rather than a blanket "dev untouched". `build_splits` does construct the dev dataset, and
constructing one parses a single dev session to infer feature dims. Nothing dev-side reaches
training, selection or a reported number, but the manifest says what happened rather than
what would read better.

At this sample size, prior bootstrap intervals in this project are wide enough that
differences around 0.02 ROC-AUC are not resolvable. An arm is promoted to the dev report
only if its paired CI excludes zero **and** its win rate holds across folds and seeds.
