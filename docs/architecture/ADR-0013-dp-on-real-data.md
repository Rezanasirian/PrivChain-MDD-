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

## Result — and why it is not yet reportable

```
audio  risk=0.90  target_eps=2.00  sigma=10.068  consumed_eps=1.999
video  risk=0.60  target_eps=4.00  sigma= 5.485  consumed_eps=3.999
text   risk=0.30  target_eps=8.00  sigma= 3.092  consumed_eps=8.000
composed participant epsilon: 10.043

eps=  inf -> acc=0.735  F1=0.400  ROC-AUC=0.545
eps= 0.50 -> acc=0.676  F1=0.000  ROC-AUC=0.387
eps= 1.00 -> acc=0.676  F1=0.000  ROC-AUC=0.387
eps= 2.00 -> acc=0.676  F1=0.000  ROC-AUC=0.387
eps= 4.00 -> acc=0.676  F1=0.000  ROC-AUC=0.387
eps= 8.00 -> acc=0.676  F1=0.000  ROC-AUC=0.395
eps=16.00 -> acc=0.676  F1=0.000  ROC-AUC=0.395
```

The accounting side is sound: σ is calibrated per modality, consumed ε matches
its target, and the composed participant budget is reported (ADR-0009). **The
utility side is not usable**, for two distinct reasons:

**(a) Clipping and Poisson subsampling alone cost more than half the F1.** At
ε = ∞ — no noise whatsoever — the model reaches F1 = 0.400 / AUC = 0.545 against
the baseline's 0.640 / 0.767.

**(b) Any finite ε collapses the model completely.** F1 = 0 and ROC-AUC below
chance at every budget from 0.5 to 16, essentially flat. A curve that does not
move with ε is not a privacy-utility trade-off curve; it is a broken operating
point.

The likely mechanism for (b) is a signal-to-noise argument, not a bug. Gaussian
noise is added per coordinate with standard deviation `σ·C / E[batch]`, while the
useful signal is an average of per-sample gradients each clipped to `C`. The
noise **vector** therefore grows as `√(#parameters)` while the signal norm stays
bounded by `C`. With ~200k parameters, `C = 1`, and `E[batch] = 32`, the noise
norm exceeds the signal norm by more than an order of magnitude even at ε = 16.
DP-SGD needs many samples per parameter; 107 participants is far from that.

## What to try next, in order

1. **Raise the sampling rate.** With only 107 training samples, full-batch
   DP-GD (`q = 1`) maximises signal per step. Fewer, stronger steps is the
   standard regime for tiny datasets.
2. **Shrink the private model.** The text branch alone is ~98k parameters
   (768 × 128). Projecting the frozen embeddings down before the trainable layer
   would cut the noise norm roughly as `√(#parameters)` without touching the
   privacy analysis — the projection is data-independent post-processing of a
   frozen encoder.
3. **Tune the clipping norm `C`.** It is currently 1.0, inherited from the mock
   configuration and never tuned against real gradient magnitudes.
4. **Only then** report the curve.

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
