# ADR-0013 — Per-modality DP-SGD on the real corpus: protocol fixes and an unresolved utility collapse

- **Status:** Proposed — the protocol changes are accepted; the utility result is **not** yet a Chapter 4 number
- **Date:** 2026-08-13
- **Phase:** 3 (Adaptive Per-Modality DP Mechanism, H1)
- **Related objectives:** H1 (per-modality privacy), H5 (evaluation)

## Context

Phase 3 was built and unit-tested against mock data (ADR-0004, ADR-0009). This
is its first execution on the real corpus, now that Phase 1 has a baseline that
actually learns (ADR-0012, ADR-0014). `scripts/run_dp_sweep.py` was wired to
real data for the first time; it previously hardcoded `MockDaicWozDataset` and
CPU.

## Decisions (accepted)

### 1. The sweep runs on real data, under the Phase 1 protocol

`--daic-config` builds the official train/dev splits, infers feature dims from
the corpus, resolves the device from config, and — importantly — uses the **same
class-weighted objective** as the non-private baseline. Without that, the
privacy-utility gap would have conflated the cost of DP with a change of
protocol.

### 2. An ε = ∞ reference point

The sweep now begins at ε = ∞: the same architecture, data, and step budget with
the noise switched off (σ = 0) but **clipping still on**. This decomposes the
curve into two separately attributable costs — clipping/subsampling versus
perturbation — instead of reporting only absolute numbers.

This point immediately earned its keep. The first real run showed `F1 = 0` at
every ε *including ε = ∞*, which proved the failure was not differential privacy
at all. Without the reference point the obvious (and wrong) conclusion would have
been "DP destroys utility on DAIC-WOZ."

### 3. Adam, not plain SGD

The cause of that first failure: the sweep built `torch.optim.SGD` at the
baseline's learning rate. The "SGD" in DP-SGD refers to the noised,
per-sample-clipped *gradient*, not to the optimizer consuming it. Plain SGD at
`lr = 3e-4` does not move this model at all. The sweep now uses Adam with the
baseline's learning rate and weight decay, matching Phase 1.

### 4. Model selection matches the baseline

The baseline reports its **best** epoch (dev-selected); the sweep reported its
**last**. Comparing the two charges DP for a protocol difference. The sweep now
evaluates after every epoch and keeps the best by the configured
`selection_metric`. Evaluating a released model is post-processing, so this costs
no additional privacy budget.

An initial version of this also evaluated *before* training and let that compete.
Since DP training made the model worse than its initialization, the untrained
weights won at every ε — producing byte-identical metrics across ε = 0.5…16, an
artifact that looked like a broken sweep. Selection now runs over trained epochs
only.

### 5. `sweep.epochs` raised from 3 to 60

Epochs is a *privacy* parameter, not only a compute one: more steps means more
mechanism applications, so the accountant calibrates more noise for the same
target ε. But 3 epochs (≈12 steps) would have starved DP-SGD for reasons
unrelated to privacy and overstated the cost of DP. The non-private baseline
needs ~29 epochs to reach its best.

### 6. The decision threshold is tuned, not fixed at 0.5

The single most important fix. Under DP-SGD, **per-sample clipping normalizes
away the class weighting**: `pos_weight` works by scaling the positive class's
loss and therefore its gradient magnitude, but clipping every per-sample gradient
to the same norm `C` erases exactly that magnitude. The model then ranks cases
correctly while placing every score below 0.5, and a fixed threshold reports
`F1 = 0`.

A lever diagnostic made this unmistakable. Sweeping sampling rate x clipping norm
x width at ε = 8 gave `F1 = 0.000` in all eight cells while **ROC-AUC ranged
0.54–0.60 — above chance in every one**. The models were learning; the metric
was not seeing it.

`binary_classification_metrics` now accepts `threshold=None` to select the
F1-maximizing cut from the scores, and reports which cut it used. This is
post-processing of a released model's outputs, so it adds nothing to the privacy
budget. It is standard practice for imbalanced classification and is applied to
the private and non-private arms alike.

The same diagnostic also disposed of one hypothesis: **shrinking the model did
not help.** `hidden = 128` beat `hidden = 32` on AUC in every pairing, so the
`√(#parameters)` noise-norm argument is not the binding constraint here.

## Result — and why it is not yet reportable

Accounting (unchanged, and sound — σ calibrated per modality, consumed ε matches
target, participant budget composed per ADR-0009):

```
audio  risk=0.90  target_eps=2.00  sigma=10.068  consumed_eps=1.999
video  risk=0.60  target_eps=4.00  sigma= 5.485  consumed_eps=3.999
text   risk=0.30  target_eps=8.00  sigma= 3.092  consumed_eps=8.000
composed participant epsilon: 10.043
```

Utility, after the threshold fix (single seed, dev split, n=34):

| ε | dev F1 | dev ROC-AUC | dev accuracy |
|---|---|---|---|
| 0.5 | 0.489 | 0.387 | 0.324 |
| 1 | 0.512 | 0.506 | 0.382 |
| 2 | 0.537 | 0.494 | 0.441 |
| 4 | 0.595 | 0.553 | 0.559 |
| 8 | 0.611 | 0.577 | 0.588 |
| 16 | 0.611 | 0.581 | 0.588 |
| ∞ (clip only, no noise) | 0.564 | 0.510 | 0.500 |
| *non-private baseline* | *0.640* | *0.767* | *0.735* |

**F1 now rises monotonically with ε and saturates around ε ≈ 8.** That is a
privacy-utility trade-off curve, where before there was a flat line at zero. The
progression from the earlier state is entirely attributable to the fixes above,
not to any change in the privacy mechanism.

**It is still not a Chapter 4 number,** for two reasons:

**(a) The ε = ∞ control is *below* the private runs** (F1 0.564 vs 0.611 at
ε = 8). Removing noise cannot genuinely help less than adding it; something is
wrong with the comparison, not with DP. The likely cause is selection bias:
taking the best of 60 epochs on a 34-session dev set is optimistic, and *noisier*
runs benefit more because they visit more diverse checkpoints and so get more
chances at a lucky one. Until the control behaves coherently, the curve's levels
cannot be trusted even though its shape is plausible.

**(b) Every point is single-seed.** With 34 dev sessions, one flipped prediction
moves F1 by ~0.03; the whole spread from ε = 4 to ε = 16 is about that size.

The privacy cost, read off the saturated end, is roughly **F1 0.640 → 0.611 and
ROC-AUC 0.767 → 0.581**: modest in F1, large in ranking quality. Note the AUC gap
is much wider than the F1 gap, which is consistent with the threshold tuning
propping up F1 on a model whose ranking has genuinely degraded. Reporting only F1
here would flatter DP considerably — Chapter 4 must report both.

## What to try next, in order

1. **Repeat every point over ≥3 seeds** and report mean ± spread. Nothing else
   is worth tuning until the error bars are visible; (a) and (b) above may both
   dissolve into noise.
2. **Fix the selection bias.** Either evaluate at a fixed step budget rather
   than best-of-N, or hold out a separate split for the epoch choice, so the
   private and non-private arms are selected identically.
3. **Tune the clipping norm `C`** against real gradient magnitudes; it is still
   1.0, inherited from the mock configuration. The lever sweep hinted `C = 0.1`
   is slightly better for AUC.
4. **Only then** report the curve.

Two hypotheses from the first draft of this ADR were tested and **rejected**:
shrinking the model did not help (wider was consistently better), and raising the
sampling rate to `q = 1` changed little once the threshold was tuned.

## Consequences

- **Phase 3's Definition of Done is not met on real data.** It remains met on
  mock data (the mechanism, accountant, and allocator are correct and tested);
  what is missing is a real accuracy-vs-ε curve that actually varies with ε.
  `docs/implementation-plan.md` must not be ticked for real data yet.
- The per-modality allocation and RDP composition are unaffected by this — they
  are accounting, and they are correct. Only the utility axis is blocked.
- If the mitigations above are insufficient, that is itself a defensible thesis
  finding: single-institution DAIC-WOZ is too small for meaningful DP-SGD, which
  is a direct argument for the federated setting (H2) where the effective
  dataset spans institutions. It must be stated as a finding, not hidden behind
  a flat curve.
