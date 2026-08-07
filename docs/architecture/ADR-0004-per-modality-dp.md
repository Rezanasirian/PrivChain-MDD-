# ADR-0004 — Adaptive per-modality differential privacy (H1)

- **Status:** Accepted; substantially revised 2026-08-06 (see "Revision" below)
- **Date:** 2026-07-01
- **Phase:** 3 (Adaptive Per-Modality DP Mechanism with Opacus)
- **Related objective:** H1 (first core novelty)

## Context

The thesis's first novelty is a **per-modality adaptive DP budget allocation** —
rather than one uniform budget over the whole gradient vector, each modality gets
a budget calibrated by its re-identification risk (audio > video > text). The
named tool is **Opacus**, but `opacus` is an optional dependency and the
environment is offline, so it could not be run/tested here.

## Decisions

### 1. The allocation mechanism (Chapter 3 math)

- Indices: client `i`, modality `m ∈ {audio, video, text}`, step `t`.
- Parameters: risk `r_m ∈ (0,1]`, target `δ`, sampling rate `q`, planned steps
  `T`; either explicit budgets `ε_m` or a total budget `ε_total` + sharpness `γ`.
- **Adaptive (inverse-risk) allocation:**
  `ε_m = ε_total · r_m^{-γ} / Σ_k r_k^{-γ}` → higher risk gets a smaller budget
  (γ=0 → uniform; larger γ → more risk-sensitive).
- **Decision variable (noise):** `σ_m = min{ σ : ε_RDP(σ, q, T, δ) ≤ ε_m }`,
  obtained from the RDP accountant by binary search.
- **Auditable consumption:** after `t` steps,
  `ε_m(t) = ε_RDP(σ_m, q, t, δ)` — the per-modality budget each client reports
  and (Phase 5) logs to the ledger; never silently overwritten (CLAUDE.md §7).

Implemented in `privacy/budget_allocator.py`; modes in `configs/privacy.yaml`.

### 2. RDP accountant implemented in-house

`privacy/accountant.py` computes the RDP of the Sampled Gaussian Mechanism
(Mironov 2019) over integer orders 2..64 in log-space (SciPy `logsumexp`), then
converts to `(ε, δ)` via `ε = min_α RDP_total(α) + log(1/δ)/(α-1)`. This is the
same accounting Opacus performs. `opacus_engine.cross_check_epsilon` validates it
against Opacus's `RDPAccountant` when `opacus` is installed.

Validated offline: q=1 reduces to `α/(2σ²)`; ε monotonic ↓ in σ and ↑ in steps;
`get_noise_multiplier` round-trips within tolerance.

### 3. Per-modality DP-SGD

`privacy/dp_sgd.py` treats **each modality as an independent DP mechanism**:
per-sample gradients (via microbatching) of each modality's parameter group are
clipped to `C` and perturbed with Gaussian noise scaled by that modality's `σ_m`.
Parameter→group mapping is by name prefix (`encoders.audio/video/text`); the
fusion + heads form a `shared` group that conservatively takes `max_m σ_m`
(it processes all modalities). This is mathematically what Opacus's
`PrivacyEngine` does; the Opacus production wiring (one engine per modality
optimizer) is documented in `opacus_engine.py`.

### 4. Definition of Done

`scripts/run_dp_sweep.py` (a) writes a per-modality allocation report
(`σ_m`, consumed `ε_m`) and (b) sweeps target ε values, training DP-SGD at each
and plotting accuracy/F1/ROC-AUC vs ε (`accuracy_vs_epsilon.png` +
`sweep_curve.jsonl`) under `experiments/phase3/<run-id>/`.

## Assumptions / notes

- The `shared`-group `max σ` rule is a conservative simplification; a dedicated
  shared budget could be introduced later.
- Per-modality DP-SGD here clips each modality group **separately** (per-modality
  composition), which is the point of H1 — not a single global clip.
- On mock noise data the accuracy-vs-ε curve is not meaningful in absolute terms;
  it becomes informative on real DAIC-WOZ (and feeds Phase 6 attacker analysis).

---

## Revision (2026-08-06)

Four decisions above were found to be wrong or unsound and have been replaced.

### R1. The in-house accountant is gone; Opacus is the accountant

Decision 2 described a hand-written integer-order RDP bound as "the same
accounting Opacus performs". Measured against Opacus it over-reports ε by
**15–22%** (e.g. σ=1.1, q=0.05, T=100: 3.94 vs 3.31). Over-reporting is safe in
direction, but calibrating σ from a target ε then injects more noise than
necessary and costs accuracy for nothing.

`privchain/privacy/accountant.py` is now a thin typed wrapper over
`opacus.accountants.analysis.rdp` and `opacus.accountants.utils`, so every ε in
Chapter 4 comes from the reference implementation. `opacus` moved from an
optional extra to a core dependency. A unit test asserts equality with Opacus's
own `RDPAccountant`.

### R2. Sampling is genuinely Poisson

The accountant assumes each sample enters a step independently with probability
`q`. The training loop drew *shuffled fixed-size batches*, which do not satisfy
that, so the reported ε was not strictly valid. `poisson_batches` now draws real
Bernoulli(`q`) batches, and the noisy gradient sum is normalised by the
**expected** batch size `q·N` — normalising by the realised size would leak it.
Empty draws still consume a mechanism application, since that is what the
accountant charges for.

Consequence: "epoch" is now a budgeting convention (`steps_for_epochs`), not a
traversal, and all Phase 3/7 numbers were regenerated.

### R3. Per-sample gradients come from Opacus — and are verified

Microbatching (one forward/backward per sample) was correct but ~3.3× slower than
necessary and unusable on the real corpus. The fast path now uses
`opacus.GradSampleModule`, keeping our own per-group clipping and noise on top of
`p.grad_sample`. Two constraints emerged, both non-obvious:

- **`nn.GRU` is unsupported.** Opacus raises `ShouldReplaceModuleError`; the
  encoders now use `opacus.layers.DPGRU`.
- **`pack_padded_sequence` silently breaks per-sample attribution.** Packing
  collapses the batch dimension, and Opacus then attributes gradients to the
  wrong samples with *no error raised* — the failure mode that would have
  invalidated every DP claim in the thesis while all tests still passed.

Packing was therefore removed. Because a bidirectional RNN over a padded tensor
would start its backward direction inside another sample's padding (making an
embedding depend on its batch mates), the bidirectional case is built from **two
unidirectional DPGRUs**, the second fed the per-sample reversed valid prefix. A
unit test asserts this is numerically identical to `nn.GRU` +
`pack_padded_sequence` (max difference 3e-8), so the model is unchanged; another
asserts the two per-sample-gradient backends agree, so the fast path is never
trusted blindly.

### R4. The privacy unit and composition are now explicit

What `ε_m` bounds, and what a participant contributing all modalities actually
spends, are specified in [ADR-0009](ADR-0009-privacy-unit-and-composition.md) and
reported by `compose_epsilon` / `participant_epsilon`.
