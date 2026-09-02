# Capability-conditioned logit MoE — one pre-registered comparison

Inner 5-fold CV over the official train split only. Both arms
train under the identical capability schedule and are scored
counterfactually under every capability, so each estimate uses the whole
fold.

## Primary test (fixed before the run)

`Δ = AUC_audio_only(MoE) − AUC_audio_only(baseline)`

| Δ | 95% CI | significant |
|---|---|:--:|
| +0.028 | [-0.005, +0.062] | no |

## Per-capability ROC-AUC (descriptive)

| capability | baseline | MoE | Δ |
|---|---|---|---|
| full | 0.695 | 0.657 | -0.038 |
| audio_text | 0.684 | 0.659 | -0.025 |
| audio_only | 0.634 | 0.662 | +0.028 |
| text_only | 0.731 | 0.564 | -0.167 |

## Secondary (no significance claimed)

- macro capability AUC: baseline 0.686, MoE 0.635
- min-capability Δ: -0.070 [-0.223, +0.035] — a minimum over four
  noisy estimates is biased downward, so this interval is descriptive
- argmin distribution (MoE): full 7%, audio_text 5%, audio_only 5%, text_only 83%

## Learned gate weights under full modality

| arm | audio | video | text |
|---|---|---|---|
| baseline_gated_fusion | 0.578 | 0.177 | 0.245 |
| capability_moe | 0.454 | 0.181 | 0.365 |

## Wall clock

- baseline_gated_fusion: 2.3s per fold
- capability_moe: 2.6s per fold
