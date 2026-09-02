# Capability-conditioned logit MoE — one pre-registered comparison

Inner 5-fold CV over the official train split only. Both arms
train under the identical capability schedule and are scored
counterfactually under every capability, so each estimate uses the whole
fold.

## Primary test (fixed before the run)

`Δ = AUC_audio_only(MoE) − AUC_audio_only(baseline)`

| Δ | 95% CI | significant |
|---|---|:--:|
| +0.023 | [-0.010, +0.056] | no |

## Per-capability ROC-AUC (descriptive)

| capability | baseline | MoE | Δ |
|---|---|---|---|
| full | 0.701 | 0.655 | -0.046 |
| audio_text | 0.691 | 0.658 | -0.033 |
| audio_only | 0.639 | 0.662 | +0.023 |
| text_only | 0.764 | 0.550 | -0.214 |

## Secondary (no significance claimed)

- macro capability AUC: baseline 0.699, MoE 0.631
- min-capability Δ: -0.090 [-0.253, +0.036] — a minimum over four
  noisy estimates is biased downward, so this interval is descriptive
- argmin distribution (MoE): full 4%, audio_text 6%, audio_only 2%, text_only 87%

## Learned gate weights under full modality

| arm | audio | video | text |
|---|---|---|---|
| baseline_gated_fusion | 0.580 | 0.177 | 0.243 |
| capability_moe | 0.426 | 0.185 | 0.389 |

## Wall clock

- baseline_gated_fusion: 2.4s per fold
- capability_moe: 2.5s per fold
