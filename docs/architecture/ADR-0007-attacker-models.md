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
