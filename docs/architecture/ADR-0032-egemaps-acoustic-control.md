# ADR-0032 — Capacity-matched eGeMAPS acoustic control

- **Status:** Rejected (measured 2026-09-12)
- **Date:** 2026-09-12
- **Phase:** 1
- **Related:** ADR-0019, ADR-0030, ADR-0031

## Context

The COVAREP repair ladder found no significant benefit from speech masking,
window averaging, validity masking, or corpus normalization. A standard,
independently specified acoustic representation is needed as a materially
different control.

## Decision

1. Extract the 88 openSMILE `eGeMAPSv02` functionals from each DAIC-WOZ WAV
   after concatenating only transcript intervals attributed to the participant.
   Use the same 0.10-second boundary padding as ADR-0031. Extraction is fixed and
   label-free.
2. Fit feature normalization on the official train split only. Dev receives the
   frozen train statistics.
3. Use a `mean` audio encoder because each functional vector has one row. The
   existing `stats` encoder would duplicate it as mean/min/max and add constant
   standard-deviation and difference blocks.
4. Match encoder capacity. Committed COVAREP uses `74 * 5 -> 128 -> 128`, or
   64,000 parameters. eGeMAPS uses `88 -> 294 -> 128`, or 63,926 parameters, a
   0.12% difference.
5. Compare `speech+egemaps` with `committed` on the same five inner folds and
   three seeds. Official dev and test remain untouched. Adoption requires a
   paired pooled out-of-fold confidence interval that excludes zero.

## Result

Run `experiments/phase1/phase1_preprocessing_ladder_20260912_201339`, 107
participants, 5 inner folds x 3 seeds, official dev/test untouched.

| arm | ROC-AUC | ±sd | F1 | wins | pooled Δ | 95% CI | p |
|---|---|:--:|---|---|---|---|---|
| committed | 0.728 | 0.094 | 0.457 | – | – | – | – |
| speech+egemaps | 0.531 | 0.124 | 0.289 | 1/15 | -0.213 | [-0.328, -0.106] | <0.001 |

Preflight (`egemaps_preflight.json`): all 188 vectors present and finite,
both arms see the same 107-participant pool, audio encoder 64,000 vs 63,926
parameters (0.12%).

Diagnostics (`egemaps_diagnostics.md`) rule out an extraction or scaling
fault: per-participant eGeMAPS F0 correlates 0.966 with COVAREP F0, and the
normalized inputs have unit spread. An audio-only logistic probe gives
0.49–0.50 pooled out-of-fold AUC for eGeMAPS against 0.61–0.62 for the
COVAREP functionals. The multimodal drop below the committed model, rather
than to it, comes from the audio branch fitting noise (88 inputs, 86 training
sessions, validation loss up to 2.9 on three folds) and the fusion following
it.

Decision: eGeMAPS is rejected as the acoustic front-end. COVAREP/FORMANT with
the committed pipeline stays the default; no downstream result changes.

## Boundaries

- The `mean` encoder has no non-linearity between its two linear layers, so
  the eGeMAPS branch is a linear 88 -> 128 map with matched parameter count.
  The linear probe in the diagnostics shows the null is not an artefact of
  that choice: no linear separability exists to begin with.

- Concatenating participant turns removes interviewer speech and most
  between-turn silence. This tests voice and prosody, not conversational timing.
- Functionals discard within-session order. A null result does not rule out a
  learned waveform sequence representation.
- Downstream experiments are rerun only if this control supports changing the
  acoustic default.

## Implementation

- `scripts/extract_egemaps.py`
- `scripts/run_preprocessing_ladder.py --arms committed speech+egemaps`
- optional dependency: `uv sync --extra audio`
