# Capability-conditioned logit MoE — one pre-registered comparison

Inner 5-fold CV over the official train split only. Both arms
train under the identical capability schedule and are scored
counterfactually under every capability, so each estimate uses the whole
fold.

## Primary test (fixed before the run)

`Δ = AUC_audio_only(MoE) − AUC_audio_only(baseline)`

| Δ | 95% CI | significant |
|---|---|:--:|
| +0.012 | [-0.029, +0.052] | no |

## Per-capability ROC-AUC (descriptive)

| capability | baseline | MoE | Δ |
|---|---|---|---|
| full | 0.703 | 0.674 | -0.029 |
| audio_text | 0.684 | 0.667 | -0.017 |
| audio_only | 0.653 | 0.665 | +0.012 |
| text_only | 0.622 | 0.562 | -0.060 |

## Secondary (no significance claimed)

- macro capability AUC: baseline 0.665, MoE 0.642
- min-capability Δ: -0.060 [-0.190, +0.078] — a minimum over four
  noisy estimates is biased downward, so this interval is descriptive
- argmin distribution (MoE): full 4%, audio_text 3%, audio_only 5%, text_only 88%

## Learned gate weights under full modality

| arm | audio | video | text |
|---|---|---|---|
| baseline_gated_fusion | 0.575 | 0.204 | 0.221 |
| capability_moe | 0.395 | 0.207 | 0.398 |

## Wall clock

- baseline_gated_fusion: 1.6s per fold
- capability_moe: 2.0s per fold
