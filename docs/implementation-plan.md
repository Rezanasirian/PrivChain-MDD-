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
- [x] Audio encoder (lightweight projection → bi-GRU → masked mean-pool over acoustic-feature sequences; COVAREP on real data). `src/privchain/encoders/audio.py`, `sequence_encoder.py`.
- [x] Video encoder (same sequence encoder over facial-feature sequences; OpenFace AUs on real data). `src/privchain/encoders/video.py`.
- [x] Text encoder (sequence encoder over transcript features; offline hashing / opt-in TF-IDF vectorizer on real data). `src/privchain/encoders/text.py`, `data/text_vectorizers.py`.
- [x] Fusion layer (concat, with a forward-compatible per-sample presence mask) → binary classification head + optional PHQ-8 regression head. `src/privchain/fusion/`.
- [x] Centralized training on mock data (config-driven, seeded, experiment logging). `src/privchain/training/`, `scripts/train_baseline.py`.
- [x] Report F1 and ROC-AUC (pure-NumPy metrics). `src/privchain/eval/metrics.py`. _(On mock data these are meaningless by design — random-noise features; real numbers come once DAIC-WOZ is downloaded.)_
- **Definition of Done:** The model trains/evaluates on mock data; once real data arrives, F1 and ROC-AUC are reportable. ✅ **Met** — 33 tests pass; `scripts/train_baseline.py` runs end-to-end and writes config + `metrics.jsonl` + checkpoint. Real DAIC-WOZ loader built (`src/privchain/data/daic_woz.py`, `configs/daic_woz.yaml`, ADR-0002) and unit-tested against a fabricated fixture; **not yet run against the real ~300 GB corpus** (pending download via the DUA link).

### Phase 2 — Simulating Heterogeneous Federated Clients with Flower (1 week)
**Goal:** Build H2 without privacy and without blockchain — federation basics first.
- [x] Split data across N simulated clients with heterogeneous modality-access patterns (population mix in `configs/federated.yaml`: full / audio+text / audio-only / text-only). `src/privchain/federated/partition.py`.
- [x] Implement standard FedAvg (no missing-modality handling — absent modalities zero-imputed) with two backends: an offline in-house simulator and a Flower `NumPyClient` adapter. `src/privchain/federated/{simulation,client,aggregation,flower_app}.py`.
- [x] Per-round global metrics logged to `experiments/phase2/<run-id>/metrics.jsonl` for the degradation analysis (Chapter 4). _(On mock noise the numbers are meaningless; the comparison becomes meaningful on real DAIC-WOZ.)_
- **Definition of Done:** Simulated federated training across ≥3 heterogeneous clients with metrics logged. ✅ **Met** — `scripts/run_federated.py` runs 8 clients across 4 modality patterns; 43 tests pass. The **Flower backend is built but not run offline** (`flwr` not installed; see ADR-0003) — the in-house simulator produces the Phase 2 results.

### Phase 3 — Adaptive Per-Modality DP Mechanism with Opacus (H1) (1–2 weeks)
**Goal:** The first core novelty of the thesis.
- [ ] Define a base privacy budget (ε, δ) separately for each modality based on re-identification risk (audio > video > text per the Chapter 1 claim — this hypothesis must be validated empirically with attacker models in Phase 6)
- [ ] Formalize the budget allocation function mathematically (this is exactly what should replace the wrong model in Chapter 3 — indices: client i, modality m, round t; parameters: ε_m (base budget per modality), identification risk r_m; decision variable: σ_{i,m,t} (noise level))
- [ ] Implement with Opacus: per-sample gradient clipping + Gaussian noise with a different σ per modality encoder
- [ ] Compute cumulative privacy budget consumption (privacy accountant) using RDP or GDP
- **Definition of Done:** Each client reports how much per-modality DP budget it has consumed; an accuracy-vs-ε curve is plotted.

### Phase 4 — Capability-Aware Aggregation + Reputation + Federated Distillation (H2 complete) (2 weeks)
**Goal:** Replace Phase 2's plain FedAvg with the actual proposed protocol.
- [ ] Subgraph aggregation: group clients by their declared modality capability vector (e.g., one-hot [audio, video, text])
- [ ] Aggregation weighting by reputation score (initially reputation = function of data volume and gradient consistency; later read from the blockchain)
- [ ] Federated distillation for missing-modality clients — teacher from full-modality clients, student locally
- [ ] Compare accuracy against the Phase 2 baseline (plain FedAvg) on the same heterogeneous distribution
- **Definition of Done:** Measurable F1/ROC-AUC improvement over plain FedAvg, especially for missing-modality clients.

### Phase 5 — Blockchain Layer with Hyperledger Fabric (H3 — integration) (2 weeks)
**Goal:** Auditability and smart-contract enforcement for H1 and H2.
- [ ] Stand up a local Fabric test network (2–4 peers + orderer)
- [ ] Write chaincode in Go with 4 functions:
  - `RegisterClient(clientID, capabilityVector)`
  - `LogPrivacyBudget(clientID, modality, round, epsilonSpent)`
  - `UpdateReputation(clientID, modality, score)`
  - `PublishSubgraph(round, []clientID)`
- [ ] Connect the Flower server to the Fabric network (Go SDK or REST gateway calls) to read/write these values each round
- [ ] Design Byzantine robustness (e.g., filter outlier gradients before aggregation — inspired by Sho et al. 2024) and personalized aggregation (Fan et al. 2025)
- **Definition of Done:** One full round of federated training runs with real reads/writes against the blockchain (not an in-memory simulation).

### Phase 6 — Attacker Models for Privacy Evaluation (part of H5) (1 week)
**Goal:** Empirically prove that adaptive DP actually protects privacy.
- [ ] Speaker-identification attacker model (audio re-identification)
- [ ] Face-recognition attacker model (face re-identification)
- [ ] Named-entity-extraction attacker model (text-based de-anonymization)
- [ ] Run a membership-inference attack against the noised embeddings and report success rate at different per-modality ε levels
- **Definition of Done:** A table of attack success rate per modality and per privacy-budget level — this feeds directly into Chapter 4.

### Phase 7 — Comparative Baselines & Final Evaluation (H5 complete) (2 weeks)
**Goal:** The final tables for Chapter 4.
- [ ] Reproduce Xu et al. 2023, De Chaudhury et al. 2024, Fan et al. 2025 (at least a simplified version) on the same data
- [ ] Run 10-fold cross-validation + held-out test for all variants (centralized baseline, plain FedAvg, full proposed framework)
- [ ] Ablation analysis: remove adaptive DP → uniform DP only; remove reputation weighting; remove federated distillation
- [ ] Measure inference latency under different compute budgets
- **Definition of Done:** All tables and plots needed for Chapter 4 are generated.

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