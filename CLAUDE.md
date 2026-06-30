# CLAUDE.md — PrivChain-MDD Project Standards

> This file is read automatically by Claude Code and must be followed for all code in this repo.
> **Thesis:** Depression Detection via Machine Learning and Blockchain — Reza Nasirian, University of Tehran

---

## 1. Project Summary (keep this in mind at all times)

A multimodal (audio/video/text) federated learning system for depression detection featuring:
- **Per-modality** differential privacy budget allocation (Opacus)
- Capability-aware aggregation + reputation weighting + federated distillation (Flower)
- An auditability layer on a blockchain ledger (Hyperledger Fabric, chaincode in Go)

The full phased implementation plan lives in `docs/implementation-plan.md` — read it before starting any phase.

---

## 2. Repo Structure (mandatory — don't change without approval)

```
privchain-mdd/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── implementation-plan.md
│   └── architecture/            # ADRs (Architecture Decision Records)
├── configs/                     # all hyperparameters live here, never in code
│   ├── baseline.yaml
│   ├── federated.yaml
│   └── privacy.yaml
├── src/
│   └── privchain/
│       ├── encoders/            # audio.py, video.py, text.py
│       ├── fusion/
│       ├── privacy/             # opacus wrappers, budget allocator
│       ├── federated/           # flower client/server
│       ├── chain_client/        # python<->Fabric bridge (gRPC/REST)
│       └── eval/                # attacker models, metrics
├── chaincode/
│   └── privchain-cc/            # Go chaincode (Hyperledger Fabric)
├── tests/
│   ├── unit/
│   └── integration/
├── experiments/                 # output of every run: configs + logs + checkpoints
├── notebooks/                   # exploration only, never a source of truth for code
├── data/                        # never committed — only .gitkeep
├── pyproject.toml
├── .pre-commit-config.yaml
└── .env.example
```

---

## 3. Python Standards

- **Formatting + lint:** `ruff` (acts as both formatter and linter) — config in `pyproject.toml`, max line length 100.
- **Type hints are mandatory** on every public function (parameters and return value). Run `mypy --strict` on `src/`.
- **Docstrings:** Google style, including `Args`, `Returns`, `Raises`. Every module gets a top-of-file docstring stating which phase/goal (H1–H5) of the implementation plan it relates to.
- **Dependency management:** `pyproject.toml` + `uv` or `poetry` — never `pip install` manually without adding it to the lockfile.
- **Configuration:** No hyperparameter is ever hardcoded; everything comes from `configs/*.yaml` (validated with `pydantic` or `omegaconf`).
- **Random seeding:** Every training script must take `seed` from config and seed `torch`, `numpy`, and `random` with it — reproducibility is critical for Chapter 4 results.
- **Experiment logging:** Every run writes to `experiments/<phase>/<run-id>/`: the config used, metrics (as jsonl), and checkpoints. If MLflow/Weights & Biases is available, use it; otherwise the folder structure above is the minimum standard.
- **Testing:** `pytest`. Every new module needs at least one unit test before merge. Tests for encoders/privacy/federated components must run against mock data (never real DAIC-WOZ) so CI works without sensitive data.

## 4. Go Standards (chaincode)

- `gofmt` + `golangci-lint` mandatory before commit.
- Each chaincode function (`RegisterClient`, `LogPrivacyBudget`, `UpdateReputation`, `PublishSubgraph`) must:
  - Validate its inputs and return an explicit error (never panic)
  - Have a unit test using `shimtest` (MockStub)
- Don't write immutable state to the ledger unless it's documented in a design doc (`docs/architecture/`).

## 5. Naming Conventions

| Item | Rule | Example |
|---|---|---|
| Python file/module | `snake_case` | `privacy_budget_allocator.py` |
| Class | `PascalCase` | `ModalityAwareAggregator` |
| Function/variable | `snake_case` | `compute_epsilon_per_modality` |
| Go file | `snake_case.go` | `register_client.go` |
| Git branch | `phase-N/short-description` | `phase-3/opacus-per-modality-dp` |
| Experiment/run name | `phaseN_<description>_<date>` | `phase3_dp_budget_sweep_20260630` |

## 6. Commit Convention (Conventional Commits)

```
feat(privacy): add per-modality epsilon allocator
fix(federated): correct reputation weighting normalization
test(chaincode): add unit tests for LogPrivacyBudget
docs(plan): update phase 4 definition of done
```
Every commit must reference which phase of the implementation plan it relates to (in the commit body, not the title).

## 7. Security and Data Privacy (the most important section for this project)

- **No real DAIC-WOZ data is ever committed or pushed.** `data/` is in `.gitignore`. Only mock data is allowed in the repo.
- Fabric keys, certificates, and any secrets go in `.env` (which is in `.gitignore`) — `.env.example` holds a template with no real values.
- Each client's consumed epsilon (ε) must be logged so privacy accounting stays auditable — never silently overwritten.
- Before every commit, `pre-commit` should check that no raw data file extensions (`.wav`, `.mp4`, `.csv` under `data/`) have been added.

## 8. Definition of "Done" for Every Pull Request

- [ ] `ruff check` and `mypy` pass with no errors (Python) / `golangci-lint` passes with no errors (Go)
- [ ] New tests are written and passing
- [ ] If a new phase was completed, the corresponding checkbox in `docs/implementation-plan.md` is checked off
- [ ] No hyperparameter is hardcoded
- [ ] No raw data or secrets are committed

## 9. When Claude Code Is Working

- Before starting any phase, read `docs/implementation-plan.md` and state which phase you're currently implementing.
- If you make a significant architectural decision (e.g., how per-modality re-identification risk is computed), write a short ADR in `docs/architecture/`.
- If something in the thesis is ambiguous (e.g., the exact number of simulated federated clients), make an explicit assumption and document it in the config and an ADR — don't guess silently.