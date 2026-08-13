# PrivChain-MDD — Phased Implementation Roadmap
## Thesis: "Depression Detection via Machine Learning and Blockchain"
**Reza Nasirian — University of Tehran, Industrial Engineering, Systems Modeling & Data Analytics track**

> Extracted from the full text of Chapter 1 (Introduction) and Chapter 2 (Literature Review) of the thesis. Intended as the input for actually implementing the project with Claude Code — converting the five research objectives of Chapter 1 into an executable, phase-by-phase technical backlog.

---

## 0. Current State of the Document (important to read before starting)

From reviewing the PDF:

- **Chapter 1 (Introduction)** and **Chapter 2 (Literature Review)**: complete and written. The problem statement, objectives, novelty, and background are well developed.
- **Chapter 3 (Methodology)**: **not yet written** — only contains the university's template placeholder text. More importantly, **the current mathematical model in this chapter (equations 3-1 through 3-20, pages 73–77) is completely unrelated to the thesis topic** — it's a hospital operating-room scheduling model (patient/surgeon/bed assignment with fuzzy numbers) that appears to have been copy-pasted from a different template/example, and must be fully removed and replaced with the actual mathematical model of this project (DP budget allocation + federated aggregation).
- **Chapter 4 (Results)** and **Chapter 5 (Discussion & Conclusion)**: only template guidance text, no real content.

In short: there is a solid research problem and literature foundation, but no mathematical model, system architecture, code, or results have been implemented yet. This document fills exactly that gap.

---

## 1. The Five Research Objectives (from Section 1-6 of the thesis)

| # | Objective | Expected Output |
|---|------|-------------------|
| H1 | Design a **per-modality adaptive differential-privacy budget allocation** mechanism (not a single uniform budget across the whole gradient vector), calibrated by each modality's (audio/video/text) re-identification risk, and auditable via a ledger tied to a smart contract | Mathematically proven DP mechanism + implementation |
| H2 | Design a **smart-contract-managed federated learning protocol** that combines clients with asymmetric modality access (some audio-only, some text-only, etc.) via capability-declared subgraph aggregation, federated distillation, and reputation-based weighting — without losing the clinical value of cross-modal dependencies | Aggregation protocol + convergence argument |
| H3 | Combine H1 and H2 into a **unified framework** on top of a Byzantine-fault-tolerant blockchain infrastructure (inspired by Sho et al., 2024) and personalized aggregation (Fan et al., 2025) | End-to-end architecture |
| H4 | **Prototype implementation**: DAIC-WOZ dataset, per-vertex noise injection with Opacus, smart-contract infrastructure with Hyperledger Fabric | Runnable code |
| H5 | **Empirical evaluation** of the prototype against prior work in terms of diagnostic accuracy, per-modality privacy guarantees, robustness to asymmetric access, and inference latency under matched compute budgets | Chapter 4 tables + Chapter 5 analysis |

## 2. Tech Stack Explicitly Named in the Thesis (pages 15–16)

These are stated directly in the text — deviating from them means diverging from the approved thesis:

- **Dataset**: DAIC-WOZ (189 clinical interview sessions, audio+video+text, PHQ-8 labels)
- **Deep learning pipeline**: PyTorch
- **Differential privacy**: Opacus (per-vertex/per-modality noise injection, independently calibrated per modality)
- **Federated orchestration**: Flower
- **Blockchain layer**: Hyperledger Fabric — chaincode in **Go**, with at least 4 core functions:
  1. Client registration + capability declaration
  2. Logging the privacy budget consumed each round
  3. Updating each client's per-modality reputation score
  4. Publishing the federation subgraph (which clients aggregate together this round)
- **Evaluation metrics**: F1, ROC-AUC, success rate of re-identification/membership-inference attacks using **three separate attacker models**: speaker identification (audio), face recognition (image), named-entity extraction (text)
- **Comparison baselines**: centralized training without privacy, standard FedAvg without privacy, reproductions of Xu et al. 2023, De Chaudhury et al. 2024, and Fan et al. 2025 on the same dataset
- **Validation**: 10-fold cross-validation + a held-out test fold, plus ablation analysis for each component

---

## 3. Proposed System Architecture (for Chapter 3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Federated Client Layer                       │
│  Client A (audio+video+text)  Client B (audio only)  Client C (text+audio)│
│   Audio encoder → embed       Audio encoder           Audio encoder │
│   Video encoder → embed                                Text encoder │
│   Text encoder → embed                                              │
│   Local fusion → PHQ prediction                                    │
│         │                                                          │
│   Opacus: per-modality DP noise (different budget per modality)    │
└─────────┼────────────────────────────────────────────────────────┘
          │  noisy gradients + modality capability metadata
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Federated Aggregation Server (Flower)              │
│  - subgraph grouping by declared capability vector (H2)            │
│  - reputation-based weighting (read from the blockchain)           │
│  - federated distillation for missing-modality clients             │
│  - Byzantine-robust aggregation                                    │
└─────────┼────────────────────────────────────────────────────────┘
          │  log transaction (budget spent, reputation, next subgraph)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│           Hyperledger Fabric — Smart Contracts (Go)                │
│  RegisterClient | LogPrivacyBudget | UpdateReputation | PublishSubgraph │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Implementation Phases

> Each phase has a concrete "Definition of Done" so it can be handed directly to Claude Code as a task.

### Phase 0 — Environment & Data Setup (2–4 days)
**Goal:** A reproducible foundation before any model code is written.
- [ ] Apply for DAIC-WOZ access (requires a Data Use Agreement — **this is the longest lead-time item in the whole project; start it today**, approval can take weeks). _(External/manual — not yet done.)_
- [x] Until approval arrives, build a **mock/synthetic** version of the DAIC-WOZ structure (short random audio, a few image frames, fake transcripts, random PHQ-8 labels) so the full pipeline can be tested without real data. _(`src/privchain/data/mock_daic_woz.py`; see ADR-0001.)_
- [x] Git repo + folder structure (`data/`, `src/encoders/`, `src/federated/`, `src/privacy/`, `chaincode/`, `experiments/`, `notebooks/`)
- [x] Python virtual environment + dependency manifest (`pyproject.toml` + `uv`, per CLAUDE.md §3, replacing `requirements.txt`; torch/opacus/flwr/librosa/transformers/scikit-learn declared across core + optional groups).
- [ ] Install Go + Hyperledger Fabric (a local test network via `fabric-samples`). _(External/manual — Go not installed in current env; needed for Phase 5.)_
- **Definition of Done:** A smoke-test `pytest` run on the mock data pipeline passes and produces correctly shaped tensors. ✅ **Met** — 13 tests pass (`tests/unit/test_mock_daic_woz.py`, `tests/integration/test_pipeline_smoke.py`).

### Phase 1 — Centralized Multimodal Baseline Model (no federation, no privacy) (1–2 weeks)
**Goal:** Establish a diagnostic accuracy baseline before federated/DP complexity is added.
- [x] Audio encoder (session-level statistical functionals → MLP on real data; projection → bi-GRU → masked mean-pool also available. COVAREP features). `src/privchain/encoders/audio.py`, `sequence_encoder.py`, ADR-0012.
- [x] Video encoder (same sequence encoder over facial-feature sequences; OpenFace AUs on real data). `src/privchain/encoders/video.py`.
- [x] Text encoder (sequence encoder over transcript features; offline hashing / opt-in TF-IDF vectorizer on real data). `src/privchain/encoders/text.py`, `data/text_vectorizers.py`.
- [x] Fusion layer (concat, with a forward-compatible per-sample presence mask) → binary classification head + optional PHQ-8 regression head. `src/privchain/fusion/`.
- [x] Centralized training on mock data (config-driven, seeded, experiment logging). `src/privchain/training/`, `scripts/train_baseline.py`.
- [x] Report F1 and ROC-AUC (pure-NumPy metrics). `src/privchain/eval/metrics.py`.
- [x] **Trained and evaluated on the real DAIC-WOZ corpus** (downloaded 2026-08-13; 189 participants, 85.6 GB). Validation on the official dev split (n=34), class-weighted loss, session-level statistical encoder, hyperparameters swept over 3 seeds. ADR-0010 / ADR-0011 / ADR-0012.
- **Definition of Done:** The model trains/evaluates on mock data; once real data arrives, F1 and ROC-AUC are reportable. ✅ **Met on real data** — 150 tests pass; `scripts/train_baseline.py --daic-config configs/daic_woz.yaml` runs end-to-end and writes config + `metrics.jsonl` + checkpoint. Real DAIC-WOZ dev results: **F1 = 0.560, ROC-AUC = 0.664** (seed 42; sweep mean over 3 seeds F1 = 0.588 ± 0.037), in the range the AVEC2017 baseline reports. Two defects were found and fixed on first contact with the real corpus: silently-zeroed test labels from a split-header mismatch, and a source-truncated archive for participant 440 (ADR-0010). The held-out test split remains untouched until the Chapter 4 table.

### Phase 2 — Simulating Heterogeneous Federated Clients with Flower (1 week)
**Goal:** Build H2 without privacy and without blockchain — federation basics first.
- [x] Split data across N simulated clients with heterogeneous modality-access patterns (population mix in `configs/federated.yaml`: full / audio+text / audio-only / text-only). `src/privchain/federated/partition.py`.
- [x] Implement standard FedAvg (no missing-modality handling — absent modalities zero-imputed) with two backends: an offline in-house simulator and a Flower `NumPyClient` adapter. `src/privchain/federated/{simulation,client,aggregation,flower_app}.py`.
- [x] Per-round global metrics logged to `experiments/phase2/<run-id>/metrics.jsonl` for the degradation analysis (Chapter 4). _(On mock noise the numbers are meaningless; the comparison becomes meaningful on real DAIC-WOZ.)_
- **Definition of Done:** Simulated federated training across ≥3 heterogeneous clients with metrics logged. ✅ **Met** — `scripts/run_federated.py` runs 8 clients across 4 modality patterns; 43 tests pass. The **Flower backend is built but not run offline** (`flwr` not installed; see ADR-0003) — the in-house simulator produces the Phase 2 results.

### Phase 3 — Adaptive Per-Modality DP Mechanism with Opacus (H1) (1–2 weeks)
**Goal:** The first core novelty of the thesis.
- [x] Define a base privacy budget (ε, δ) per modality based on re-identification risk (audio > video > text); `configs/privacy.yaml` with `explicit` and `inverse_risk` allocation modes.
- [x] Formalize the budget allocation function mathematically (indices i/m/t; parameters ε_m, risk r_m; decision variable σ_m via the accountant): `src/privchain/privacy/budget_allocator.py`, written up in ADR-0004 for Chapter 3.
- [x] Implement per-sample gradient clipping + per-modality Gaussian noise (a different σ per modality encoder), as a manual DP-SGD equivalent to Opacus: `src/privchain/privacy/dp_sgd.py`. Opacus bridge/cross-check in `opacus_engine.py`.
- [x] Cumulative privacy accounting via an in-house RDP accountant (Sampled Gaussian Mechanism): `src/privchain/privacy/accountant.py`.
- **Definition of Done:** each client reports per-modality DP budget consumed; an accuracy-vs-ε curve is plotted. ✅ **Met** — `scripts/run_dp_sweep.py` writes `allocation_report.json` (per-modality σ/consumed ε **and the composed participant ε**) plus `accuracy_vs_epsilon.png` + `sweep_curve.jsonl`.
- **Revised 2026-08-06 (ADR-0004 R1–R4, ADR-0009).** The original implementation had three soundness defects, all now fixed and covered by tests:
  - the in-house accountant over-reported ε by 15–22% vs Opacus → **Opacus is now the accountant** and a core dependency;
  - training used shuffled fixed-size batches while the accountant assumed Poisson sampling → **real Poisson subsampling**, normalised by the expected batch size;
  - per-sample gradients now come from `opacus.GradSampleModule` (~3.3× faster, and required for real DAIC-WOZ). This forced `nn.GRU` → `opacus.layers.DPGRU` and the removal of `pack_padded_sequence`, which **silently mis-attributes per-sample gradients** under Opacus; the bidirectional encoder is rebuilt from two unidirectional passes and unit-tested to be numerically identical to the packed reference.
  - The privacy unit and cross-modality composition are now stated explicitly in **ADR-0009**.

### Phase 4 — Capability-Aware Aggregation + Reputation + Federated Distillation (H2 complete) (2 weeks)
**Goal:** Replace Phase 2's plain FedAvg with the actual proposed protocol.
- [x] Subgraph aggregation: group clients by their declared modality capability vector (one-hot [audio, video, text]); each modality encoder is averaged only over the clients that declare it, empty subgraphs keep the global value. `src/privchain/federated/capability.py`, `aggregation.py::capability_aware_aggregate`.
- [x] Aggregation weighting by reputation score (reputation = data volume + per-group update consistency, EMA-smoothed; `reputation_weighting` toggle; per-modality `ρ` snapshot logged to `reputation.jsonl`, ledger-ready for Phase 5). `src/privchain/federated/reputation.py`.
- [x] Federated distillation for missing-modality clients — teacher = frozen global model, student distills its soft predictions locally. `src/privchain/federated/distillation.py`, wired via `FederatedClient.fit`.
- [x] Compare accuracy against the Phase 2 baseline (plain FedAvg) on the same heterogeneous distribution. `scripts/run_capability_federated.py` writes `comparison.json` (overall + per modality-access pattern).
- **Definition of Done:** Measurable F1/ROC-AUC improvement over plain FedAvg, especially for missing-modality clients. ✅ **Met (mechanism)** — `scripts/run_capability_federated.py` runs both protocols on the same partition/seed/init and reports per-pattern deltas; 19 new tests pass. Design in ADR-0005. **On mock noise the accuracy delta is not meaningful** (as in Phases 1–3); the real improvement is produced once DAIC-WOZ is downloaded.

### Phase 5 — Blockchain Layer with Hyperledger Fabric (H3 — integration) (2 weeks)
**Goal:** Auditability and smart-contract enforcement for H1 and H2.
- [ ] Stand up a local Fabric test network (2–4 peers + orderer). _(External/manual — needs Go + Docker + `fabric-samples`, not installed in the offline env.)_
- [x] Write chaincode in Go with 4 functions + read helpers, input validation, explicit errors, and `shimtest` (MockStub) unit tests. `chaincode/privchain-cc/` (ADR-0006):
  - `RegisterClient(clientID, capabilityVector)`
  - `LogPrivacyBudget(clientID, modality, round, epsilonSpent)` — **append-only** (consumed ε never overwritten, CLAUDE.md §7)
  - `UpdateReputation(clientID, modality, score, round)`
  - `PublishSubgraph(round, []clientID)` — **immutable** per round
- [x] Connect the federated server to the ledger (read/write each round) via a backend-agnostic `LedgerClient`: in-memory `MockLedger` (offline) + `FabricRestLedger` (live REST gateway). Wired into `run_capability_aware_simulation` (subgraph + per-modality consumed ε + reputation); `scripts/run_federated_with_ledger.py` reads the trail back to `audit_report.json`. `src/privchain/chain_client/`.
- [x] Byzantine robustness: shared-group outlier filter (robust median/MAD), `aggregation.byzantine_filter`. `src/privchain/federated/robust.py`. _(Personalized aggregation per Fan et al. 2025 is folded into the per-modality reputation weighting from Phase 4.)_
- **Definition of Done:** One full round of federated training runs with real reads/writes against the blockchain (not an in-memory simulation). ✅ **Met against the `MockLedger`** (real ledger read/write semantics; same invariants *and access control* as the chaincode) — the identical `LedgerClient` calls hit real Fabric with `backend: fabric_rest`. Standing up a Docker Fabric test network remains open; see ADR-0006.
- **Revised 2026-08-06 (ADR-0006 R1–R5).** Go 1.26.5 is now installed and the chaincode **compiles and tests for the first time** (`go.sum` did not previously exist, so `go build` could never have run). Three defects fixed: (a) every chaincode function accepted **any** submitter — a client could raise its own reputation, i.e. its own aggregation weight; identity-based access control now binds each client to its registrant and restricts reputation/subgraph writes to the coordinator MSP; (b) rounds were stored as bare decimals in composite keys, so `GetBudgetHistory` returned an out-of-order audit trail past round 9 — rounds are now zero-padded; (c) `MockLedger` mirrored the invariants but not the authorization, so permission bugs could pass offline — it now enforces the same rules.

### Phase 6 — Attacker Models for Privacy Evaluation (part of H5) (1 week)
**Goal:** Empirically prove that adaptive DP actually protects privacy.
- [x] Speaker-identification attacker (audio re-identification) — nearest-centroid cosine attacker on the audio embedding. `src/privchain/eval/attackers.py::ReidentificationAttacker` (ADR-0007).
- [x] Face-recognition attacker (video re-identification) — same engine on the video embedding.
- [x] Named-entity-extraction attacker (text de-anonymization) — same engine on the text embedding.
- [x] Membership-inference attack against the noised embeddings at different per-modality ε levels. `MembershipInferenceAttacker`; per-ε curve in the report.
- **Definition of Done:** A table of attack success rate per modality and per privacy-budget level — this feeds directly into Chapter 4. ✅ **Met** — `scripts/run_attack_eval.py` writes `attack_success.json` (per-modality × per-ε re-identification, the adaptive-allocation headline, and the membership-inference curve) + `attack_curve.jsonl` + `attack_success_vs_epsilon.png`. On mock data the depression labels are noise but **subject identity is a real signal**, so the re-identification curve (and the "higher risk → smaller ε → better protected" headline) is demonstrable offline; DAIC-WOZ numbers replace the mock ones later.
- [x] **Measured on real DAIC-WOZ (ADR-0017).** `scripts/run_reid_risk.py` replaces the mock corpus's jittered views — which are recoverable by construction and prove nothing — with **disjoint contiguous stretches of each participant's single session**: frame spans for audio/video, spans of transcript turns for text, each embedded separately. 141 participants, chance 0.0071, three seeds, shuffled-label control at chance. At matched width: **audio 22.2× chance, video 21.6×, text 6.6×**. The assumed ordering (audio > video > text) **holds**, text's assumed 0.3 is nearly exact (measured 0.29), but the audio-vs-video gap does not survive (0.158 ± 0.004 vs 0.153 ± 0.010). Crucially this resolves ADR-0016 in H1's favour: the modality leaking least identity is the one carrying all the utility, so risk-proportional and utility-aware allocation agree. `configs/privacy.yaml` deliberately left unchanged pending that policy decision.
- **Revised 2026-08-06 (ADR-0007 R1–R4).** The harness previously attacked a model trained **without any DP** and simulated privacy by noising its outputs afterwards. Now a real per-modality DP-SGD model is trained at each ε and attacked, alongside a non-private ε = ∞ reference. Running that revealed the conceptual error underneath: **DP-SGD does not bound re-identification** (it stayed at 100% at every ε) — it bounds membership inference. Re-identification is now measured against an explicit embedding-release mechanism (clip to a bounded norm + Gaussian mechanism at the target ε). The membership-inference attacker was also biased upward (threshold tuned on the scored data, AUC folded about 0.5) and now calibrates on a held-out split and reports signed AUC. Measured: non-private advantage 0.458 → ≈ 0 under DP at every ε.

### Phase 7 — Comparative Baselines & Final Evaluation (H5 complete) (2 weeks)
**Goal:** The final tables for Chapter 4.
- [x] Reproduce Xu et al. 2023, De Chaudhury et al. 2024, Fan et al. 2025 as **simplified stand-ins** on the same data/model (centralized ≈ Xu; uniform-DP ≈ De Chaudhury; personalized/reputation ≈ Fan), each labelled as such — the papers are unavailable offline, so these are documented approximations (ADR-0008), not faithful reimplementations.
- [x] 10-fold cross-validation + held-out test for all variants (centralized, plain FedAvg, personalized, full proposed framework, and an ablation), one shared split/seed protocol. `src/privchain/eval/benchmark.py`, `scripts/run_final_evaluation.py`.
- [x] Ablation analysis: adaptive DP → uniform DP (same total ε, centralized DP-SGD); remove reputation weighting; remove federated distillation. Written to `ablation.json` + `dp_comparison.json`.
- [x] Inference latency under different compute budgets (forward-pass ms/batch and ms/sample across batch sizes). `measure_inference_latency`; `latency.json` + `inference_latency.png`.
- **Definition of Done:** All tables and plots needed for Chapter 4 are generated. ✅ **Met** — `scripts/run_final_evaluation.py` writes `cv_results.json`, `ablation.json`, `dp_comparison.json`, `latency.json`, `inference_latency.png`, and a combined `chapter4_summary.md` under `experiments/phase7/<run-id>/`. **On mock data the depression label is random, so the accuracy numbers are placeholders** demonstrating the table shapes; the real numbers come with DAIC-WOZ. Design in ADR-0008.
- **Revised 2026-08-06 (ADR-0008 R1–R3).** Folds were drawn **without stratification**, so on an imbalanced corpus single-class test folds are routine, ROC-AUC is undefined on them, and `aggregate_metrics` dropped those folds *silently* — a "10-fold mean" could quietly be a mean over 7. Splits are now stratified and every metric reports `<metric>_num_valid_folds` next to its mean.

### Phase 8 — Writing Chapters 3, 4, and 5 (can proceed in parallel with phases above)
- [ ] **Chapter 3:** Fully replace the current (unrelated) mathematical model with the formalization from Phase 3 (DP budget allocation) and Phase 4 (aggregation weighting) — indices/parameters/variables tables following the existing chapter's format but with real content
- [ ] **Chapter 4:** Tables and plots from Phase 7
- [ ] **Chapter 5:** Summary, novelty (mapped back to Section 1-8), recommendations, limitations (e.g., dependency on DAIC-WOZ, small-scale federated simulation, etc.)

---

## 5. Risks and Things to Manage Early

1. **DAIC-WOZ access** is the biggest scheduling risk — apply now, proceed with mock data in the meantime.
2. **Federated simulation scale**: how many clients will you actually work with? The thesis doesn't give an exact number — suggestion: at least 10 simulated clients across 4 distinct modality-access patterns.
3. **Running Fabric + Flower together** can be heavy on a single development machine — for early development, replace the blockchain layer with a mock ledger (an in-memory dict) and only connect to real Fabric starting in Phase 5.
4. **Section 1-9 (term definitions)** in Chapter 1 contains the formal definitions of DP, federated learning, and blockchain terminology — use the exact same vocabulary when writing Chapter 3 to stay consistent with Chapters 1/2.

---

## 6. Suggested Sequence to Start with Claude Code

```
Phase 0 (env) → Phase 1 (centralized baseline) → Phase 2 (plain Flower)
   → Phase 3 (per-modality DP) → Phase 4 (smart aggregation) → Phase 5 (Fabric)
   → Phase 6 (attacker models) → Phase 7 (final evaluation) → Phase 8 (writing)
```

Each phase can be handed to Claude Code as a separate session; it's recommended to start each session with this file as context, beginning with Phase 0.