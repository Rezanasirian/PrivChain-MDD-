# ADR-0030 — Frame validity and aggregation before modality modelling

- **Status:** Accepted (implementation complete; defaults retained after inner CV)
- **Date:** 2026-09-04
- **Phase:** 1 (Centralized Multimodal Baseline), objective H4
- **Related:** ADR-0011 (coverage), ADR-0019 (normalization), ADR-0031

## Context

The committed loader keeps the first row of every `frame_stride` window. It also
drops OpenFace's `success` and `confidence` columns from model inputs without
using them to decide whether the accompanying action units are measurements.
COVAREP's VUV flag is retained as quality, but F0 and the glottal channels are
still summarized on unvoiced frames even though the DAIC-WOZ documentation says
they must not be used there.

These are separate questions and must not be hidden inside one preprocessing
change:

1. Does window aggregation improve on unfiltered decimation?
2. Does excluding failed OpenFace rows improve video features?
3. Does excluding unvoiced F0/glottal entries improve audio features?

## Decision

1. `downsample` is explicit per modality: `decimate`, `mean`, or
   `window_functionals`. The committed default remains `decimate` until an
   inner-CV comparison supports changing it. Window mean is a boxcar prefilter;
   it attenuates high-frequency content but is not described as an ideal or
   complete anti-alias filter.
2. `validity` is data-shaped. OpenFace validity masks whole rows using
   `success` and an optional, predeclared confidence threshold. COVAREP validity
   masks only F0 and the declared glottal channels when VUV is false.
3. A window with no valid value for one channel is imputed from that channel's
   non-empty window aggregates within the same session. It must never fall back
   to the measurements the validity rule rejected. If a channel is invalid for
   the whole session, its aggregate is zero.
4. The parser preserves averaged `success`, `confidence`, and VUV columns as
   quality ratios. Filtering model features therefore does not erase the audit
   signal used by quality-aware fusion.
5. A validity mask with `decimate` is rejected. The ladder includes an
   aggregation-only control before every aggregation-plus-validity arm, so the
   effects remain attributable.

## Measurement protocol

`scripts/run_preprocessing_ladder.py` ranks arms by inner CV on the official
train split. In particular, video is tested as `video+mean`, then
`video+mean+success`, then `video+mean+valid`; the success-only arm disables the
confidence threshold so the two policies are not conflated. No default changes
and no Chapter 4 result is adopted from implementation alone.

## Result (2026-09-12)

The predeclared five-arm comparison ran on 107 official-train participants with
five inner folds and three seeds. Neither window averaging nor either OpenFace
validity policy improved pooled out-of-fold ROC-AUC significantly:

| arm | fold-mean AUC | pooled Δ vs committed | 95% CI | p |
|---|---:|---:|---:|---:|
| `video+mean` | 0.732 | +0.007 | [-0.041, +0.051] | 0.743 |
| `video+mean+success` | 0.698 | -0.048 | [-0.111, +0.013] | 0.121 |
| `video+mean+valid` | 0.689 | -0.038 | [-0.123, +0.041] | 0.377 |
| combined audio/video repair | 0.734 | +0.000 | [-0.084, +0.082] | 0.989 |

Across all available OpenFace AU exports, 280,466 of 5,377,703 rows (5.22%) had
`success == 0`; the union of failed tracking and confidence below 0.90 was 7.05%.
The defect is therefore real and non-trivial in prevalence, but removing those
rows did not improve this model under the selected protocol. The committed
`decimate`/validity-off defaults remain unchanged. Results:
`experiments/phase1/phase1_preprocessing_ladder_20260912_190924/`.

The follow-up audio ladder reached the same decision. Its best nominal arm,
`speech+mean+valid+corpus`, had pooled Δ AUC +0.043 with 95% CI
[-0.049, +0.126], p=0.337. The validity arm without corpus normalization had
Δ=-0.016, p=0.668. No isolated audio preprocessing step justified a default
change. Results:
`experiments/phase1/phase1_preprocessing_ladder_20260912_193056/`.

## Consequences

- The feature-cache schema is versioned, so old decimated matrices cannot be
  mistaken for validity-aware aggregates.
- Any winning preprocessing arm requires re-running modality ablation, DP,
  federation, and privacy attacks because the inputs—and sometimes their
  widths—have changed.
- `window_functionals` changes model parameter count under the existing `stats`
  encoder and is therefore exploratory, not a capacity-matched comparison.
