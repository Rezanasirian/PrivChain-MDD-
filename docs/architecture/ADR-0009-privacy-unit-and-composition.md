# ADR-0009 — The privacy unit, and what composes with what

- **Status:** Accepted
- **Date:** 2026-08-06
- **Phase:** 3 (per-modality DP) with consequences for 5–7
- **Related objectives:** H1 (per-modality DP), H5 (empirical privacy evaluation)

## Context

H1's novelty is that each modality gets its own budget `ε_m` instead of one
budget over the whole gradient vector. Reporting `ε_audio = 2, ε_video = 4,
ε_text = 8` is only meaningful once two questions are answered explicitly, and
both are the sort of question a thesis committee asks first:

1. **What is one "unit" of privacy here?** ε bounds the effect of changing *one
   what*?
2. **A participant's data flows through all three encoders plus the shared
   fusion head. What does that participant actually spend?**

The Phase 3 implementation previously left both implicit. This ADR fixes them.

## Decision

### 1. The privacy unit is one participant's data *within one modality*

`ε_m` bounds the effect of changing one participant's **modality-`m` record**
(their audio, or their video, or their transcript) on the modality-`m` parameter
group. This is exactly what the mechanism enforces: per-sample gradients are
clipped and noised **per parameter group**, so the guarantee is per group.

This is the right unit for the thesis's claim — "audio carries more
re-identification risk than text, so it should be noised harder" is a statement
about modalities, not about whole participants — but it is *not* the unit a
reader assumes by default, so every reported table names it.

### 2. Participant-level budget is the RDP composition, and is reported alongside

A participant contributing all three modalities is exposed to four mechanisms:
the three encoders and the shared fusion/head group. Their true budget is the
composition of all four, computed by summing the RDP curves at each order
*before* converting to `(ε, δ)`:

```
ε_participant = min_α [ Σ_g RDP_g(α) + log(1/δ)/(α−1) ]
```

Implemented as `privchain.privacy.accountant.compose_epsilon` and surfaced as
`PerModalityBudgetAllocator.participant_epsilon`. `scripts/run_dp_sweep.py`
writes it to `allocation_report.json` as `participant_epsilon`.

Summing the per-modality `ε` values instead would also be sound but needlessly
loose: at the configured allocation (2 / 4 / 8) the naive sum is ≈ 12.6 while the
RDP composition is ≈ 7.7.

### 3. The shared group runs at `max_m σ_m`

The fusion layer and heads see all modalities, so they are noised at the largest
(most protective) multiplier among the modalities. Conservative and simple; a
dedicated shared budget remains possible later.

### 4. DP-SGD bounds membership inference, not re-identification

DP-SGD constrains how much the *trained model* depends on any one training
record. It says nothing about how distinctive an encoder's output is for a given
input — an encoder can map an unseen subject to a highly identifiable point no
matter how privately it was trained. Phase 6 confirmed this empirically:
re-identification stayed at 100% across every training ε.

Re-identification is therefore bounded by a **separate mechanism**: the released
embedding is clipped to `embedding_clip_norm` (bounding its sensitivity) and
perturbed by the Gaussian mechanism at the target ε
(`privchain.eval.attackers.release_embeddings_dp`). Chapter 4 reports the two
attacks against the two mechanisms and says which is which; conflating them
would be the single easiest result in this thesis to attack.

## Consequences

- Every ε table in Chapter 4 states the privacy unit and reports both the
  per-modality budgets and the composed participant budget.
- The composed budget is larger than any individual `ε_m`; that is the honest
  number and is presented as such rather than buried.
- Chapter 3's formalization carries the composition formula above, not just the
  per-modality allocation rule.

## Related

- [ADR-0004](ADR-0004-per-modality-dp.md) — the allocation mechanism and DP-SGD.
- [ADR-0007](ADR-0007-attacker-models.md) — the attacker models these budgets
  are evaluated against.
