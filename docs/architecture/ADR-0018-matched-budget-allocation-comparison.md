# ADR-0018 — Comparing DP allocations at a matched participant budget

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 3 (H1's central claim)
- **Related objectives:** H1 (per-modality adaptive DP), H5 (empirical evaluation)

## Context

H1 claims that splitting the DP budget **per modality**, calibrated by
re-identification risk, buys more utility than one uniform budget. With ADR-0017
supplying measured risks and ADR-0015 supplying a shared evaluation protocol,
that claim is finally testable on real data.

Testing it requires the arms to cost the **same privacy**. Otherwise the
comparison measures how much budget each arm was handed, not how well it spent it.

## The defect this fixes

`scripts/run_final_evaluation.py` matched its adaptive and uniform arms like this:

```python
uniform_eps = sum(adaptive.values()) / len(MODALITIES)  # "same total budget"
```

That is the arithmetic mean of the per-modality ε, and it is **not** the same
privacy cost. What a participant contributing every modality actually spends is
the RDP *composition* of the three encoder mechanisms plus the shared head
(`PerModalityBudgetAllocator.participant_epsilon`, ADR-0009). Composition is
dominated by the **loosest** mechanism and is not linear in the individual
budgets, so two allocations with equal mean ε generally have different composed ε.

The adaptive allocation is deliberately uneven — that is the whole idea — and its
loosest budget goes to text. So at equal mean it spends **strictly more** real
privacy than the uniform arm. The bias runs in favour of the thesis's own
hypothesis, which is the worst possible direction for it to run: any "adaptive
wins" result read off that table would have been an artifact of the matching.

This is the same class of defect as the selection-bias problems found in
ADR-0015 — a comparison that quietly gives one arm an advantage — and it is
asserted as a regression test
(`test_matching_on_the_mean_would_not_have_been_equivalent`).

## Method

`scale_to_participant_epsilon(weights, target, …)` takes a budget **shape** (only
the ratios matter) and bisects on a scalar multiplier until the composed
participant ε equals the target. Composed ε is strictly increasing in the
multiplier, so the bisection is well-posed; the bracket is expanded from both
ends rather than assumed.

Every arm is built from a shape and scaled to the *same* target, so the arms are
matched by construction. `run_allocation_comparison.py` re-checks the achieved
budgets at runtime and **refuses to report** if they differ by more than 1% —
the match is the premise of the entire experiment, so it is enforced where it
matters rather than only in a test.

### Three arms

| arm | shape | role |
|---|---|---|
| `uniform` | ε equal for all three | the baseline H1 argues against |
| `adaptive` | ε ∝ 1/risk, measured risks (ADR-0017) | the H1 mechanism |
| `anti_adaptive` | ε ∝ risk | control |

The control earns its cost. With 34 dev sessions and three seeds, a two-arm gap
inside the seed spread reads as a result when it is not; a deliberately-wrong
third arm shows whether the allocation axis moves anything at all. If all three
land together, the honest finding is that per-modality allocation buys nothing
*at this corpus size* — which is reportable, and much better learned here than
in Chapter 4.

Each arm also reports its per-modality ε, giving the privacy half of the claim
for free: at equal participant cost, does the adaptive arm actually hand the
high-risk modalities a tighter budget?

## One DP arm, not four

The DP-SGD loop had been written out separately in `run_dp_sweep.py`,
`run_attack_eval.py` and `run_final_evaluation.py`. Adding a fourth copy for this
comparison was indefensible after ADR-0015, whose whole subject was two harnesses
silently disagreeing on identical hyperparameters.

It now lives once in `privchain/privacy/dp_training.py`
(`DpArmConfig` + `train_dp_arm`), carrying the details ADR-0013 paid for: Adam
rather than plain SGD, Poisson subsampling, epoch selection on the selection
split over trained epochs only, and early stopping on the baseline's patience.
`run_dp_sweep.py` was rewired onto it, and the sweep was re-run afterwards and
compared against its previously committed numbers — a behaviour-preserving
refactor has to prove it preserved behaviour.

## Result

86 training sessions, 34 dev sessions, three seeds, target participant ε = 8.
`experiments/phase3/phase3_allocation_comparison_20260813_160822/`.

Matched budgets — 8.004 / 8.003 / 8.001, agreeing to 0.04%:

| arm | ε audio | ε video | ε text | dev ROC-AUC | dev F1 |
|---|---|---|---|---|---|
| uniform | 3.617 | 3.617 | 3.617 | 0.549 ± 0.067 | 0.384 ± 0.082 |
| adaptive | **1.957** | **2.018** | 6.750 | 0.515 ± 0.090 | 0.351 ± 0.035 |
| anti_adaptive | 5.125 | 4.971 | 1.486 | 0.536 ± 0.076 | 0.390 ± 0.081 |

### The privacy half of the claim holds

At *identical* participant cost, the adaptive allocation gives the two
high-risk modalities materially tighter budgets than uniform does — audio
1.96 vs 3.62 and video 2.02 vs 3.62, both about **1.8× tighter** (σ 11.05 and
10.75 against 6.43). That is exactly what H1 promises, and it is now a
demonstrated property of the mechanism rather than an assertion.

### The utility half does not

The three arms are **statistically indistinguishable**: a spread of 0.034
ROC-AUC against a largest per-arm standard deviation of 0.090. The nominal
ordering is `uniform > anti_adaptive > adaptive` — the adaptive arm is nominally
*worst* — but none of that survives the seed spread.

**The control arm is what makes this readable.** `anti_adaptive` deliberately
inverts the allocation, spending the loosest budget where risk is highest, and it
scores like the others. If a sensible allocation and a deliberately wrong one are
indistinguishable, the allocation axis is not moving utility at all here. A
two-arm experiment would have shown "uniform 0.549 vs adaptive 0.515" and invited
exactly the wrong conclusion in either direction.

### Why there is nothing to reallocate

Put the arms next to the references from ADR-0013:

| | dev ROC-AUC |
|---|---|
| non-private baseline | 0.740 |
| DP path, no noise (clipping only) | 0.596 |
| all three arms at participant ε = 8 | 0.515 – 0.549 |

By the time the noise is on, the model retains almost none of the signal it
started with. Allocation decides *where* a fixed amount of damage lands — but
when every arm has already been reduced to near chance, there is no surviving
utility for a better allocation to protect. The allocation question is downstream
of the more basic finding in ADR-0013: on 86 training sessions, DP-SGD spends
most of the model before the budget is even split.

### What this does not establish

With 34 dev sessions and three seeds (σ ≈ 0.09), only differences larger than
roughly 0.18 ROC-AUC are detectable. "No measured difference" is a statement
about the power of this experiment, **not** evidence that the arms are equal. A
larger corpus, or a regime where DP leaves real utility standing, could separate
them.

## What Chapter 3 has to say now

The thesis claims per-modality allocation improves the privacy–utility trade-off.
On this corpus, the defensible version is narrower and should be stated as:

> At a fixed participant privacy budget, per-modality allocation concentrates
> protection on the modalities that measurably leak the most identity (≈1.8×
> tighter ε for audio and video) **at no measurable utility cost**. It does not
> improve accuracy, and no accuracy improvement should be claimed for it.

That is still a real contribution — protection where it is needed, for free — and
it is defensible because both halves were measured. The stronger accuracy claim
is not supported and must be removed from Chapter 1's framing.

## Consequences

- `configs/privacy.yaml` now carries the **measured** risks (1.00 / 0.97 / 0.29,
  ADR-0017) instead of the assumed 0.9 / 0.6 / 0.3, and a new
  `allocation.total_participant_epsilon` naming the matched budget.
- `allocation.mode` stays `explicit`. Deriving the ε values from the measured
  risks means switching to `inverse_risk`, which changes what every existing
  Phase 3 number refers to; it is a deliberate decision, not a side effect.
- The Phase 7 harness's DP comparison is corrected to use the same matching, so
  Chapter 4's privacy–utility table cannot inherit the old bias.
- **Chapter 1's framing of H1 needs revising** to the narrower claim above. The
  accuracy improvement is unmeasured and, on this evidence, absent.
- The obvious follow-up is not a better allocation but a cheaper mechanism: with
  clipping alone costing 0.093 ROC-AUC (ADR-0013), the leverage on this corpus is
  in reducing DP's baseline cost, not in redistributing what survives it.

---

## Amendment, 2026-08-13 (ADR-0020) — the null result is now properly powered

This ADR called the arms "not separable" using a two-standard-deviations rule on
the seed spread. That was the right conclusion reached by the wrong test. Redone
with the paired bootstrap over participants:

| comparison | Δ ROC-AUC | 95% CI | p |
|---|---|---|---|
| adaptive − uniform | +0.016 | [−0.058, +0.092] | 0.746 |
| anti_adaptive − uniform | +0.000 | [−0.054, +0.061] | 0.928 |

Two things are worth separating here.

**The conclusion is unchanged and now much better supported.** Pairing shrinks the
interval from roughly ±0.20 (each arm's own bootstrap CI, e.g. uniform
[0.346, 0.763]) to about ±0.07, because the arms are scored on the same 34
sessions and the shared luck of that draw cancels. So this is no longer "we
cannot tell": the test **excludes any allocation effect larger than about ±0.09
ROC-AUC**. That is a real bound, and a far stronger statement than the original.

**The point estimate moved and the sign flipped.** The table above reported
uniform 0.549 against adaptive 0.515, a nominal 0.034 in uniform's favour; the
paired figure is +0.016 in *adaptive's* favour. Both are noise, but they differ
because they are different statistics: the table averages each seed's AUC, while
the paired test computes one AUC from the seeds' averaged scores. Neither is
wrong; the paired one is what the significance claim rests on, and the nominal
ordering in the table above should not be read as a ranking.

**The `anti_adaptive` control is the informative part.** A deliberately inverted
allocation differs from uniform by +0.000, p = 0.928. When the sensible and the
deliberately-wrong allocations are equally indistinguishable from the baseline,
the allocation axis is doing nothing to utility here — exactly as argued above,
and now with an interval tight enough to mean it.

The privacy half of the result is untouched: at matched participant ε the
adaptive arm still gives audio and video ~1.8× tighter budgets, which is an exact
property of the mechanism and needs no statistics. See ADR-0017's amendment,
though, for why video's *risk rank* — and hence part of the motivation for that
allocation — is less settled than it looked.
