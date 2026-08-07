# ADR-0007 — Privacy-attacker models and re-identification risk (H5)

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 6 (Attacker Models for Privacy Evaluation)
- **Related objective:** H5 (empirical privacy evaluation)

## Context

H1 claims that per-modality DP protects subjects, calibrated by each modality's
re-identification risk (audio > video > text). Phase 6 must *empirically* back
that claim: the three attacker models named in the thesis — speaker
identification (audio), face recognition (video), named-entity de-anonymization
(text) — plus a membership-inference attack, producing a table of **attack
success rate per modality and per privacy budget** (Chapter 4). The real
attackers use pretrained speaker/face/NER models on DAIC-WOZ; offline we need a
faithful stand-in that runs in CI on mock data.

## Decisions

### 1. One re-identification engine, three named attackers

All three named attackers reduce to the same operation on a modality's *released*
(DP-noised) embedding: given a gallery of enrolled subjects, identify the subject
of a probe embedding. We implement it once as a **nearest-centroid cosine
attacker** (`eval/attackers.py::ReidentificationAttacker`): enrol an
L2-normalized template per subject, assign each probe to its nearest template,
report **top-1 accuracy** as the re-identification success rate. The chance
baseline is `1 / num_subjects`. Speaker-ID / face-recognition / NER differ only
in which modality encoder produced the embedding.

### 2. Views per subject on mock data

A re-identification attack needs several views per subject. On mock data each
session is one subject whose random features are a *genuine* identity signal
(unlike the random depression label). We synthesize intra-subject variability by
adding small feature-space jitter and encoding each jittered view
(`eval/embeddings.py`); on real DAIC-WOZ the natural multiple utterances/frames
replace the jitter. This makes the attack curve meaningful even offline: at high
ε (little noise) re-identification is easy; as ε shrinks it falls toward chance.

### 3. Mapping ε to embedding noise

A target ε is mapped to a noise multiplier σ via the same RDP accountant Phase 3
uses (`get_noise_multiplier(ε, q, T, δ)`, with nominal `(q, T, δ)` in
`configs/attack.yaml`); the released embedding carries Gaussian noise of std
`σ · noise_scale`. This ties the attacker's input directly to the privacy budget,
so the sweep and the adaptive-allocation headline share one mechanism. The
adaptive headline evaluates each modality at its H1 allocation ε — the highest-
risk modality (audio, smallest ε) should be the best protected, which is the
thesis's central privacy claim.

### 4. Membership inference

`MembershipInferenceAttacker` is a loss/score-threshold attack: members
(training split) tend to score higher (lower loss) than non-members. It reports
ROC-AUC, best-threshold balanced accuracy, and advantage `2·AUC − 1`, symmetric
about 0.5 (the attacker may flip its rule). The run script fits the model briefly
on the member split so there is a membership signal to attack, then adds σ-scaled
noise to the scores at each ε.

### 5. Definition of Done

`scripts/run_attack_eval.py` writes `attack_success.json` (the per-modality ×
per-ε table, the adaptive-allocation headline, and the membership-inference
curve), `attack_curve.jsonl`, and `attack_success_vs_epsilon.png` under
`experiments/phase6/<run-id>/`. This is the Chapter-4 input.

## Assumptions / notes

- On mock data the *absolute* numbers are not the thesis result — only the
  identity signal is real. The DAIC-WOZ numbers replace them once the data is
  available; the mechanism and table shape are what Phase 6 delivers.
- The nearest-centroid attacker is a strong, standard verification baseline;
  a learned attacker (e.g. a small MLP or a pretrained speaker/face model) can be
  slotted behind the same interface in Phase 7 without changing the harness.
- `noise_scale` is a mock calibration knob; on real embeddings the per-modality
  sensitivity would be estimated rather than assumed.

---

## Revision (2026-08-06)

The Phase 6 harness measured the wrong thing in two ways, and both are fixed.

### R1. The attacked model is now actually trained with DP

Decisions 3–4 attacked a model trained with **Adam and no privacy at all**, then
approximated the effect of DP by adding σ-scaled noise to its embeddings and
scores afterwards. The resulting "attack success vs ε" curve was therefore a
statement about a post-hoc perturbation, not about the mechanism the thesis
proposes.

`scripts/run_attack_eval.py` now trains a fresh model with per-modality DP-SGD at
each swept ε and attacks *that* model, plus one non-private model as the ε = ∞
reference.

### R2. Re-identification needs its own mechanism — DP-SGD does not bound it

Running the corrected pipeline produced a flat curve: re-identification stayed at
**100% at every ε**, including the tightest. That is not a bug, it is the point.
DP-SGD bounds how much the trained model depends on any one *training record*; it
says nothing about how distinctive an encoder's output is for a given *input*.
The old `noise_scale` knob had been hiding this by making the curve move for
reasons unrelated to any guarantee.

Re-identification is now evaluated against an explicit, well-defined mechanism:
the released embedding is clipped to `embedding_clip_norm` (bounding its
sensitivity) and perturbed by the Gaussian mechanism at the target ε, calibrated
through the same accountant (`release_embeddings_dp`). `noise_scale`,
`sample_rate` and `steps` are gone from `configs/attack.yaml`; the sweep range
was widened so the curve shows the full transition rather than saturating.

Measured on mock data (32 subjects, chance = 0.031):

| ε (release) | audio | video | text |
|---|---|---|---|
| 0.5 | 0.010 | 0.010 | 0.010 |
| 8 | 0.010 | 0.000 | 0.021 |
| 128 | 0.042 | 0.104 | 0.188 |
| 512 | 0.333 | 0.698 | 0.865 |
| ∞ (no DP) | 1.000 | 1.000 | 1.000 |

Membership inference against the DP-SGD-trained models behaves exactly as the
theory predicts, and is the honest headline for DP-SGD: advantage 0.458 on the
non-private model, ≈ 0 at every ε with DP.

### R3. The membership-inference attacker was biased upward

It chose its threshold on the same scores it then scored itself against, and
folded the AUC about 0.5 (`max(auc, 1−auc)`), so pure noise always looked like
leakage and `advantage` could never be negative. The attacker now calibrates its
threshold *and* its decision direction on a held-out calibration split and
reports signed AUC on the remainder. A unit test asserts that the mean advantage
over 40 independent no-leakage trials is ≈ 0 and that it sometimes goes negative
— which the previous implementation could not do.

### R4. Which mechanism bounds which attack

Stated explicitly in [ADR-0009](ADR-0009-privacy-unit-and-composition.md) and in
`attack_success.json`'s `mechanisms` field, so Chapter 4 cannot accidentally
present one as evidence for the other.
