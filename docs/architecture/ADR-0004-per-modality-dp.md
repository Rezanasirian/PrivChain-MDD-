# ADR-0004 — Adaptive per-modality differential privacy (H1)

- **Status:** Accepted
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

- Integer-order RDP (2..64) is a valid, slightly looser bound than Opacus's
  fractional-order search; adequate for a prototype and documented as such.
- The `shared`-group `max σ` rule is a conservative simplification; a dedicated
  shared budget could be introduced later.
- Per-modality DP-SGD here clips each modality group **separately** (per-modality
  composition), which is the point of H1 — not a single global clip.
- On mock noise data the accuracy-vs-ε curve is not meaningful in absolute terms;
  it becomes informative on real DAIC-WOZ (and feeds Phase 6 attacker analysis).
