# ADR-0031 — Feature extraction: three defects, and the ladder that settles them

- **Status:** Accepted as an experimental facility; no default change supported
- **Date:** 2026-09-04
- **Phase:** 1 (Centralized Multimodal Baseline), objective H4
- **Supersedes nothing.** Revisits ADR-0011 (coverage), ADR-0012 (session-level
  encoder), ADR-0019 (normalization), ADR-0027 (segment alignment), ADR-0030
  (frame aggregation and measurement validity).

## Context

The acoustic branch has scored at chance on real DAIC-WOZ (~0.53 ROC-AUC against
~0.71 for text alone) through every phase so far. That was read as a property of
the corpus and worked around in the model — a gate to suppress the branch
(ADR-0027), a mixture to route around it (ADR-0028). Reading the loader instead
turns up preprocessing defects that can plausibly contribute to it. Their causal
effect on AUC remains a measurement question.

### Defect 1 — the session-level view never restricted audio to the participant

With `segments.enabled: false`, which is what every committed result used,
`DaicWozDataset.__getitem__` loads the **whole** COVAREP matrix for a ~15-minute
recording. A DAIC-WOZ session is a conversation: the virtual interviewer speaks,
the participant answers, and there is silence between turns. So the branch was
handed the interviewer's voice and the room's silence, and its session
functionals describe the interview's structure at least as much as the
participant's speech.

The segment-aligned path already had the right rule — audio takes the union of
the participant's own turn intervals, and only video keeps the whole envelope
(ADR-0027) — but the session-level path, the default, did not.

### Defect 2 — two session functionals are effectively constant

`normalization: session` (ADR-0019) forces every channel to mean 0 and standard
deviation 1 *within* the session. `masked_statistics` then computes, over exactly
those rows, the mean and the standard deviation. So the mean block is
approximately 0 and the std block approximately 1, up to floating-point error,
epsilon handling, masks, and degenerate channels — verified numerically for the
ordinary case and pinned by
`tests/unit/test_feature_normalization.py::test_session_mode_makes_two_of_the_five_functionals_constant`.

That leaves up to 148 of the audio branch's 370 inputs with negligible direct
between-session variation, and the same fraction of the video branch. They are
not mathematically guaranteed to be constant in every edge case, and their
effect on optimization or DP utility must be measured rather than inferred from
their count alone.

`corpus` normalization (fitted on train only) restores the two blocks. It also
preserves the between-subject differences a speaker-identification attacker uses,
which is exactly the trade ADR-0019 declined to make on argument alone. It is
now a measurement, and a win obliges re-running `scripts/run_reid_risk.py`.

### Defect 3 — the ladder was choosing on the report split

`scripts/run_preprocessing_ladder.py` compared its arms by shelling out to
`run_modality_ablation.py`, which reports on the official **dev** split. Picking
a preprocessing default that way turns the report split into a validation set,
which ADR-0015 exists to prevent. `run_text_representation.py` had already
established the right protocol: inner k-fold CV over the official train split.

## Decision

1. **A modality may restrict its frames to the participant's own speech.** New
   per-modality `speech_mask` block (`enabled`, `source: transcript`,
   `pad_seconds`), built from the transcript turn timings the loader already
   parsed for ADR-0027, and applied **inside** the parser — before striding and
   before window aggregation, so a retained frame is always a speech frame and
   `frame_stride` counts speech frames rather than recording frames. Corpus
   statistics are fitted under the same mask; otherwise the statistics would
   describe silence while each session was normalized by its own speech.
   It remains off by default because the completed ladder did not establish a
   significant improvement.

2. **A modality may read several files.** New per-modality `sources` list,
   concatenated channel-wise and aligned on the **source row index** so a short
   or damaged export cannot shift a channel by a frame. This unlocks signals the
   corpus already ships and the pipeline never read: OpenFace `gaze` and `pose`
   beside the action units, `FORMANT` beside COVAREP. A source inherits its
   section's keys except the file name, so the single-file case is unchanged.

3. **Configuration faults are distinguished from damaged sessions.** New
   `FeatureConfigError`. The loader deliberately skips a session it cannot read
   while inferring feature dims and fitting corpus statistics (ADR-0010); a
   speech mask with no clock, or an unknown downsample mode, is wrong for every
   session and must not surface as "no readable participant found".

4. **The ladder ranks on inner CV over the train split only.** Rewritten as a
   self-contained script on the `run_text_representation.py` protocol, reporting
   both per-fold wins and a bootstrap over pooled out-of-fold predictions. Dev is
   constructed (one session is parsed for feature dims) and never trained on,
   selected on, or scored. The manifest records that.

## The arms

Each rung adds one thing to the rung before it, so a win can be attributed.
`committed` is the pipeline every result in the repo was produced under.

| arm | what it changes |
|---|---|
| `committed` | decimate, session normalization, no validity mask, no speech mask |
| `speech` | audio restricted to the participant's turns |
| `speech+mean` | + boxcar window mean instead of decimation (crude alias attenuation) |
| `speech+mean+valid` | + F0/glottal channels masked on unvoiced frames |
| `speech+functionals+valid` | + `[mean, std, min, max, mean\|diff\|]` per window (74 → 370) |
| `speech+corpus` | speech mask, frames otherwise untouched, corpus normalization — isolates defect 2 |
| `speech+mean+valid+corpus` | the combination |
| `video+mean` | video window mean only — aggregation control for F2 |
| `video+mean+success` | + exclude OpenFace rows with `success == 0` |
| `video+mean+valid` | + the predeclared confidence threshold |
| `speech+mean+valid+video+mean+success` | combined audio and success-only video repair |
| `speech+video_speech` | + video also restricted to speech, testing ADR-0027's listening claim |
| `speech+video_sources` | + OpenFace gaze and pose beside the action units |
| `speech+formant` | + FORMANT beside COVAREP |

## Consequences

- **Every Chapter 4 number is re-run under the winner.** Audio's input width
  changes (74 → 370 under window functionals, or wider with FORMANT), which
  changes the audio branch's parameter count and therefore the DP noise its
  group carries. Mixing pre- and post-repair numbers in one table would not be a
  comparison. Order: baseline → modality ablation → DP sweep → federated + MoE →
  re-identification and membership attacks → final evaluation.
- **ADR-0019's leakage argument is reopened** if `corpus` wins. The re-id attack
  is the measurement that settles what it costs.
- **The mechanism claims of ADR-0027 and ADR-0028 are re-examined, not
  retracted.** Both were tested against an audio branch that was fed the wrong
  frames. A gate that learned to suppress audio, and a mixture that routed
  around it, were solving a data defect; whether they still pay once the branch
  carries signal is an open question, and the honest reading is that their
  earlier null results may not transfer unchanged to the repaired inputs.
- **Caching.** `FEATURE_CACHE_SCHEMA_VERSION` is bumped to 3, and the speech
  window enters the per-participant cache key; corpus statistics key on the mask
  *settings* plus every source's parsing options.

## Measurement update (2026-09-12)

The focused ladder covering the committed pipeline, three video aggregation /
validity controls, and the combined audio/video repair completed on 107
official-train participants (five folds, three seeds). The combined arm had
fold-mean AUC 0.734 versus 0.728 for committed, but its paired pooled difference
was +0.0004, 95% CI [-0.0836, +0.0815], p=0.989. The success-only video arm was
nominally worse (Δ=-0.0485, p=0.121), not significantly different. These results
do not support changing the preprocessing default or rerunning downstream thesis
experiments under a new winner. They also do not make invalid measurements
valid; the validity-aware path remains available for sensitivity analyses.

Artifact: `experiments/phase1/phase1_preprocessing_ladder_20260912_190924/`.

### Audio follow-up

The six-arm audio ladder then isolated speech masking, window averaging,
COVAREP validity, and corpus normalization:

| arm | fold-mean AUC | pooled Δ vs committed | 95% CI | p |
|---|---:|---:|---:|---:|
| `speech` | 0.719 | +0.020 | [-0.034, +0.078] | 0.463 |
| `speech+mean` | 0.715 | +0.004 | [-0.058, +0.066] | 0.928 |
| `speech+mean+valid` | 0.721 | -0.016 | [-0.087, +0.049] | 0.668 |
| `speech+corpus` | 0.700 | -0.026 | [-0.104, +0.038] | 0.473 |
| `speech+mean+valid+corpus` | 0.760 | +0.043 | [-0.049, +0.126] | 0.337 |

No interval excludes zero. The nominal interaction in the combined corpus arm
is not evidence of a winner, and none of the individual repairs improves the
model measurably. The preprocessing investigation is therefore closed without
a default change. Further audio work should test a genuinely different
front-end, such as eGeMAPS or a pretrained waveform representation, rather than
continue tuning these COVAREP summaries.

Artifact: `experiments/phase1/phase1_preprocessing_ladder_20260912_193056/`.

Follow-up: the eGeMAPS control was run the same day and rejected (ADR-0032,
pooled Δ = -0.213, 95% CI [-0.328, -0.106]).

## Alternatives considered

- **Masking within fixed windows instead of filtering rows.** Keeps windows that
  are entirely silence, which then fall back to the unmasked aggregate — the
  defect, preserved behind a mask. Rejected.
- **Falling back to the whole session when a transcript has no usable timings.**
  Silently reintroduces defect 1 for the participants whose data is worst. The
  loader raises instead, and participant 440 is already excluded (ADR-0010).
- **Dropping the mean/std functionals under session normalization.** Fixes the
  symptom by hiding it, and makes the encoder's input width depend on a data
  setting. Corpus normalization addresses the cause and is measurable.
- **Replacing COVAREP with a raw-waveform front-end (eGeMAPS, wav2vec2).**
  `{pid}_AUDIO.wav` is on disk and this is the obvious next step, but it is a new
  extraction pass and a new loader path. Deferred until the ladder says how much
  of the gap the repair alone closes — and it would depend on the same speech
  mask, since the waveform also carries the interviewer.
