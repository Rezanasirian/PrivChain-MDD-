# ADR-0016 — Measured modality contributions, and what they mean for the per-modality budget

- **Status:** Accepted (measurement); **Open question** (the allocation policy it challenges)
- **Date:** 2026-08-13
- **Phase:** 1 (measurement) / 3 (implication)
- **Related objectives:** H1 (per-modality privacy) — this is evidence for or against its central premise

## Context

H1 claims modalities deserve *different* privacy budgets. That presumes they
differ in two respects: what they **expose** (re-identification risk) and what
they **contribute** (utility). `configs/privacy.yaml` has asserted an ordering
since Phase 3 — audio riskiest, text least risky — with the note *"This ordering
is a hypothesis to be validated with attacker models in Phase 6 — do NOT treat
these numbers as final."*

Neither half had ever been measured on this corpus. Earlier ADRs also asserted
"text is the strongest DAIC-WOZ modality" from the literature, without testing it
on our own features. This ADR measures the utility half.

## Method

One arm per non-empty subset of {audio, video, text}. A modality is ablated by
**zeroing its input features and clearing its fusion presence flag**, leaving the
architecture, parameter count, and training schedule identical across arms — so
the difference between arms is information, not capacity. That also makes an
ablation arm here identical in construction to a capability-restricted client in
Phase 2.

All arms run under the shared evaluation protocol (ADR-0015): selection on a
held-back slice of train, reporting on the untouched dev split (n=34), three
seeds, mean ± std.

## Result

| Modalities | dev F1 | dev ROC-AUC | dev accuracy |
|---|---|---|---|
| audio | 0.464 ± 0.007 | **0.503 ± 0.004** | 0.412 |
| video | 0.400 ± 0.141 | **0.494 ± 0.026** | 0.412 |
| text | 0.346 ± 0.009 | **0.710 ± 0.010** | 0.667 |
| audio + video | 0.431 ± 0.025 | 0.528 ± 0.027 | 0.382 |
| audio + text | 0.492 ± 0.028 | 0.719 ± 0.010 | 0.676 |
| video + text | 0.362 ± 0.028 | 0.722 ± 0.023 | 0.657 |
| **all three** | **0.541 ± 0.021** | **0.740 ± 0.016** | **0.735** |

Read ROC-AUC, not F1: it is threshold-independent, and the F1 column is noisy
because the threshold is calibrated on only 21 selection sessions.

Three findings:

1. **Audio and video carry no usable signal on their own.** ROC-AUC 0.503 and
   0.494 — chance, to within their spread. Whatever depressive signal COVAREP
   and OpenFace AUs contain, the session-level functionals of ADR-0012 do not
   expose it to a linear-ish model at this sample size.
2. **Text alone reaches 0.710 of the full model's 0.740.** The literature claim
   asserted in ADR-0014 now holds *on our data, measured*, and much more sharply
   than "strongest": text is very nearly the whole model.
3. **The weak modalities still contribute in combination.** Adding audio and
   video to text moves ROC-AUC 0.710 → 0.740 and F1 0.346 → 0.541. Features that
   are useless alone are not useless jointly, which is a real multimodal result
   and not a foregone conclusion.

## The implication for H1 — and it is uncomfortable

Set the measured utility beside the configured budgets:

| Modality | configured risk | configured ε | protection | measured standalone AUC |
|---|---|---|---|---|
| audio | 0.9 | 2.0 | **strongest** | 0.503 (chance) |
| video | 0.6 | 4.0 | medium | 0.494 (chance) |
| text | 0.3 | 8.0 | **weakest** | **0.710** |

**The current allocation spends its tightest privacy budget protecting the
modality that contributes nothing, and its loosest budget on the modality that
carries essentially all of the signal.**

It is very likely wrong in the other direction too. The risk ordering
(audio > video > text) was set when text was a hashed bag-of-words. Text is now
a 768-dim contextual embedding of what a participant actually said (ADR-0014) —
free-text clinical disclosure, the most re-identifying and most sensitive content
in the corpus, not the least.

This is not a reason to abandon per-modality allocation; it is the strongest
argument yet *for* it, but with a different policy. Two candidate framings, to be
decided once Phase 6's attacker models supply measured risk:

* **Risk-proportional** (current intent): spend budget where exposure is
  greatest. Needs measured risk, not assumed risk.
* **Utility-aware**: a modality that contributes nothing should not be bought at
  any privacy price — the honest move for audio/video may be to drop them or
  give them a negligible budget, spending the participant's total ε on the
  modality that actually earns it.

The interesting thesis result is precisely that these two point in **opposite
directions** here, and that a naive risk-only allocation is dominated. That is a
Chapter 4 argument, not a footnote.

## Consequences

- `configs/privacy.yaml`'s `reidentification_risk` values are now explicitly
  **placeholders contradicted by evidence** on the utility axis, and untested on
  the risk axis. They must not appear in Chapter 4 as if derived.
- **Phase 6 becomes a blocker for Phase 3's headline claim**, not a follow-up:
  the attacker models must measure per-modality re-identification risk against
  the *current* features (including transformer text embeddings) before any
  allocation can be justified.
- Phase 3's DP results should be read knowing that ~all utility lives in text:
  the noise applied to the audio and video encoders is, on this evidence, almost
  pure cost.
- Worth revisiting the audio/video representation before concluding those
  modalities are useless — session-level functionals may simply be too lossy.
  Segment-level modelling (ADR-0012's deferred option) is the obvious test. The
  claim here is about *these features*, not about audio and video in principle.

---

## Amendment, 2026-08-13 (ADR-0019, ADR-0020) — two of three findings withdrawn

This ADR reported `mean ± std` over seeds and read it as though it bounded the
estimate. It does not: it is optimization variance on a fixed 34-session split,
roughly a tenth of the sampling uncertainty that actually applies (ADR-0020).
Re-running with paired bootstrap intervals, and under the three feature
normalization schemes of ADR-0019, changes what may be claimed.

### Finding 3 is withdrawn: "the weak modalities still contribute in combination"

It rested on text-alone 0.710 → all-three 0.740, reported as ±0.010 and ±0.016.
The paired bootstrap on the same sessions:

| comparison | Δ ROC-AUC | 95% CI | p |
|---|---|---|---|
| all three − text | +0.032 | [−0.102, +0.190] | 0.712 |
| all three − audio+text | +0.032 | [−0.027, +0.110] | 0.359 |
| all three − video+text | +0.020 | [−0.106, +0.161] | 0.793 |

**No measured difference**, and the same under `corpus` (p = 0.533) and `none`
(p = 0.491) normalization. Adding audio and video to text cannot be shown to help.

### Finding 1 is withdrawn as stated: "audio carries no usable signal"

Audio alone was 0.503 under `session` normalization. ADR-0019 showed that setting
z-scores each channel *within* a session, deleting absolute pitch, formant and
energy levels. Without it audio alone reaches **0.619**.

That does not make audio useful — its interval, [0.392, 0.808], still contains
chance. It makes the original claim unsupported: the experiment had normalized
away the information it then reported as absent. The defensible statement is that
**this corpus cannot resolve what audio carries**, not that audio carries nothing.

Video moved the other way (0.494 → 0.335 under `none`), and text was identical to
three decimals across all three schemes — the control confirming the manipulation
was specific.

### Finding 2 stands: text dominates

| comparison | Δ ROC-AUC | 95% CI | p |
|---|---|---|---|
| all three − audio | +0.249 | [+0.010, +0.479] | 0.043 |
| all three − audio+video | +0.221 | [+0.011, +0.439] | 0.041 |

Both separate. Text is where the diagnostic signal is, and dropping it costs
something measurable — which is the one thing here large enough for 34 sessions
to resolve.

### What survives for H1

The section "This resolves ADR-0016's discomfort" below must be read with
ADR-0017's own amendment: video's risk rank is not stable under normalization
either. The surviving claim is narrow — **text carries the most utility and the
least identity, and audio the reverse** — and it is directional, not quantitative.
