# ADR-0020 — What our ± actually measured, and which claims it retires

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** cross-cutting (affects every reported number from Phase 1 onward)
- **Related objectives:** H5 (empirical evaluation) — and the credibility of every table in Chapter 4

## Context

Every result in this project has been reported as `mean ± std` over three seeds.
That number was read, by us, as though it were a confidence interval. It is not.

Spread over seeds measures **optimization variance on a fixed evaluation set**:
how much the answer moves when the weights are initialised differently and the
batches arrive in a different order. It holds the *participants* constant. The
question a thesis result has to answer is the opposite one — how much would this
move on a different sample of participants — and seed spread is silent on it.

On this corpus the difference is not subtle. The dev split is **34 sessions, 11
positive and 23 negative**. By Hanley–McNeil, the standard error of ROC-AUC there
is:

| AUC | SE | 95% CI |
|---|---|---|
| 0.740 | 0.097 | ±0.190 |
| 0.596 | 0.107 | ±0.210 |
| 0.515 | 0.108 | ±0.211 |

Our reported spreads ranged from ±0.010 to ±0.090. **The uncertainty that matters
is roughly ten times the one we were quoting**, and it was quoted next to gaps of
0.03.

## Decision

Report both, because they answer different questions and both are worth knowing:

- **seed spread** — is the training procedure stable?
- **bootstrap CI over participants** — would this survive a different sample?

`privchain.eval.metrics` gains `bootstrap_auc_ci` and
`paired_bootstrap_auc_difference`; `RunResult` carries the per-sample report-split
scores so an interval can be computed at all;
`privchain.training.protocol.uncertainty_report` is the one place that turns runs
into an interval, so no script can invent its own convention.

### Comparisons are paired

Two arms are scored on the *same* 34 sessions, and they rise and fall together
with the luck of that draw. Comparing their individual intervals throws that
pairing away and asks a much weaker question. Resampling participants **once per
replicate** and scoring both arms on that resample cancels the shared component
and leaves only the gap, which is far more sensitive — a unit test
(`test_pairing_is_more_sensitive_than_comparing_separate_intervals`) pins the
difference down with two arms whose own intervals overlap but whose paired
difference is significant.

Every comparative claim is therefore made on the paired difference, not by
eyeballing two overlapping intervals.

## Claims this retires

**ADR-0016 — "the weak modalities still contribute in combination."** Rested on
text-alone 0.710 → all-three 0.740, a gap of 0.030 reported as ±0.010 vs ±0.016.
It reads as decisive and is far inside the noise. **Withdrawn** pending the
paired test.

**ADR-0013 — the three-way decomposition of the cost of DP.** The end-to-end
figure (0.740 non-private → 0.480 at ε = 0.5) is large and stands. Its
attribution into −0.051 (path structure), −0.093 (clipping) and −0.024 (noise at
ε = 8) splits that total into three pieces **all of which sit below the noise
floor**. The qualitative headline — that clipping, not noise, is where most of
the cost is paid — remains the most plausible reading of the evidence, but it is
a hypothesis, not a measurement, and ADR-0013 is amended to say so.

Per the decision taken before this work, the decomposition is **withdrawn rather
than re-run at higher seed count**: more seeds shrink optimization variance,
which was never the binding constraint. The 34-session sampling uncertainty is,
and no number of seeds touches it.

## Claims that survive

- **text 0.710 vs audio 0.503** (ADR-0016) — a 0.21 gap, paired, on the same
  sessions.
- **non-private 0.740 vs every DP arm ~0.52** (ADR-0013, ADR-0018) — ~0.22.
- **ADR-0018's null result.** A null is *consistent* with low power rather than
  undermined by it, and the ADR already said the arms were not separable. The
  paired test now gives that statement a proper basis instead of a
  two-standard-deviations rule of thumb.
- **ADR-0017's re-identification rates.** A different statistic on a different
  sample size: 141 subjects × 3 probes = 423 trials, SE ≈ 0.017. The audio-vs-text
  gap (0.158 vs 0.046) is ~6 SE. The audio-vs-video gap was already reported as
  not separable.

## Consequences

- No comparative claim is made on the dev split for a gap below roughly 0.15 AUC
  unless the **paired** bootstrap excludes zero.
- **The dev split has now been looked at dozens of times** across ADRs 0011–0018,
  and design decisions (encoder type, clipping norm, early stopping, allocation)
  were taken from what it showed. It is no longer an unbiased estimate for the
  project as a whole, whatever any individual run's protocol says. The mitigation
  is that the **test split (n = 47) has never been read** and must stay that way
  until one final run — after which no further tuning is legitimate.
- The honest framing for Chapter 4 is that this corpus supports **direction**, not
  **magnitude**: it can show that text dominates audio, and that DP is expensive.
  It cannot resolve 0.03.
- Per-sample scores are written keyed by split index, never by participant ID, so
  no committed artifact can be linked back to a person.
