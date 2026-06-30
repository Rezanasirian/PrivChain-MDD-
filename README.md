# PrivChain-MDD

**Depression Detection via Machine Learning and Blockchain** — a multimodal
(audio/video/text) federated learning system with per-modality differential
privacy and a Hyperledger Fabric auditability layer.

> M.Sc. thesis — Reza Nasirian, University of Tehran (Industrial Engineering,
> Systems Modeling & Data Analytics). Project standards: [`CLAUDE.md`](CLAUDE.md).
> Phased roadmap: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Status

| Phase | Description | State |
|------:|-------------|-------|
| 0 | Environment & data setup (scaffold + mock data pipeline) | ✅ done |
| 1 | Centralized multimodal baseline (encoders + fusion + trainer) | ✅ done (mock; real loader pending download) |
| 2 | Heterogeneous federated clients (FedAvg; sim + Flower backends) | ✅ done (sim verified; Flower backend pending `flwr`) |
| 3 | Per-modality DP with Opacus (H1) | ⬜ |
| 4 | Capability-aware aggregation + reputation + distillation (H2) | ⬜ |
| 5 | Hyperledger Fabric blockchain layer (H3) | ⬜ |
| 6 | Attacker models for privacy evaluation (H5) | ⬜ |
| 7 | Comparative baselines & final evaluation (H5) | ⬜ |

## Quick start (Phase 0)

Requires Python ≥ 3.11. This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
# Create the environment and install core + dev dependencies.
uv venv
uv pip install -e ".[dev]"

# Run the mock-data pipeline smoke test (Phase 0 Definition of Done).
uv run pytest

# (Optional) write a mock DAIC-WOZ dataset to data/mock/ for inspection.
uv run python scripts/generate_mock_data.py
```

### Train the Phase 1 baseline

```bash
# Centralized multimodal baseline on mock data (writes to experiments/phase1/).
uv run python scripts/train_baseline.py

# On real DAIC-WOZ once downloaded (verify configs/daic_woz.yaml first):
uv run python scripts/train_baseline.py --daic-config configs/daic_woz.yaml
```

Real DAIC-WOZ integration (loader + config + format assumptions) is documented
in [ADR-0002](docs/architecture/ADR-0002-daic-woz-integration.md).

### Run Phase 2 federated training

```bash
# Heterogeneous FedAvg across N clients (in-house simulator; writes experiments/phase2/).
uv run python scripts/run_federated.py --rounds 5 --num-clients 8

# Same run via the Flower backend (requires the optional dependency):
uv run pip install flwr
uv run python scripts/run_federated.py --backend flower
```

The federated design (two backends, modality heterogeneity, naive-FedAvg
degradation) is documented in
[ADR-0003](docs/architecture/ADR-0003-federated-flower-baseline.md).

Lint and type-check (CLAUDE.md §3):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Layout

```
configs/        all hyperparameters (baseline / federated / privacy YAML)
src/privchain/  encoders, fusion, privacy, federated, chain_client, eval, data
chaincode/      Go chaincode for Hyperledger Fabric (Phase 5)
tests/          unit + integration (run only against mock data)
data/           git-ignored; never commit real DAIC-WOZ
experiments/    per-run config + metrics + checkpoints (git-ignored)
docs/           implementation plan + architecture ADRs
```

## Data & privacy

Real DAIC-WOZ data is **never** committed (it requires a Data Use Agreement).
Development and CI run against the synthetic dataset in
[`src/privchain/data/mock_daic_woz.py`](src/privchain/data/mock_daic_woz.py)
(see [ADR-0001](docs/architecture/ADR-0001-mock-daic-woz-dataset.md)). Secrets
and Fabric identity material live in a local `.env` (template: `.env.example`).

## Outstanding Phase 0 setup (manual / external)

- **Apply for DAIC-WOZ access now** — longest lead time in the project.
- **Install Go + Hyperledger Fabric `fabric-samples`** before Phase 5 (not yet
  installed in this environment).
