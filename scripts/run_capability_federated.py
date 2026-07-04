"""CLI: Phase 4 capability-aware federation vs. the Phase 2 FedAvg baseline.

Runs both protocols on the *same* heterogeneous partition, seed, and initial
model, then reports the F1 / ROC-AUC difference — overall and, crucially, under
each modality-access pattern (the missing-modality clients the Phase 4 protocol
targets). Both runs write full logs under ``experiments/phase4/<run-id>/``.

Phase 4 (objective H2): capability-aware subgraph aggregation + reputation
weighting + federated distillation.

Usage:
    python scripts/run_capability_federated.py
    python scripts/run_capability_federated.py --rounds 5 --num-clients 6
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from privchain.config import (
    load_baseline_config,
    load_federated_config,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.federated.partition import ModalityMaskedDataset, build_client_partitions
from privchain.federated.simulation import (
    build_federated_clients,
    run_capability_aware_simulation,
    run_simulation,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.loaders import split_dataset
from privchain.training.objective import DepressionObjective, evaluate_model


def _val_indices(val_subset: Any) -> list[int]:
    """Recover the underlying dataset indices behind a (possibly nested) Subset."""
    if isinstance(val_subset, Subset):
        inner = _val_indices(val_subset.dataset) if isinstance(val_subset.dataset, Subset) else None
        if inner is None:
            return [int(i) for i in val_subset.indices]
        return [inner[int(i)] for i in val_subset.indices]
    raise TypeError("expected a torch Subset from split_dataset")


def _build_clients(
    train_subset: Any, partitions: Any, input_dims: dict[str, int], base: Any, federation: Any
) -> list[Any]:
    """Build a fresh client population (own loader RNG) for one run."""
    return build_federated_clients(
        train_subset,
        partitions,
        input_dims=input_dims,
        model_config=base.model,
        batch_size=base.train.batch_size,
        local_epochs=federation.local_epochs,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        seed=base.seed,
    )


def main() -> None:
    """Run and compare Phase 2 FedAvg against the Phase 4 capability-aware protocol."""
    parser = argparse.ArgumentParser(description="Compare FedAvg vs. capability-aware federation.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--rounds", type=int, default=None, help="Override num_rounds.")
    parser.add_argument("--num-clients", type=int, default=None, help="Override num_clients.")
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    seed_everything(base.seed)

    federation = fed.federation
    if args.rounds is not None:
        federation = federation.model_copy(update={"num_rounds": args.rounds})
    if args.num_clients is not None:
        federation = federation.model_copy(
            update={
                "num_clients": args.num_clients,
                "clients_per_round": min(federation.clients_per_round, args.num_clients),
            }
        )

    full_dataset = MockDaicWozDataset(base.data, seed=base.seed)
    train_subset, val_subset = split_dataset(full_dataset, base.train.val_fraction, base.seed)
    val_loader: DataLoader[Sample] = DataLoader(
        val_subset, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Per-pattern masked validation loaders: how well does the global model do when
    # only a given modality-access pattern's modalities are available at inference?
    val_indices = _val_indices(val_subset)
    capability_val_loaders: dict[str, DataLoader[Sample]] = {}
    for pattern in federation.modality_patterns:
        masked = ModalityMaskedDataset(
            full_dataset, val_indices, tuple(pattern.capability)  # type: ignore[arg-type]
        )
        capability_val_loaders[pattern.name] = DataLoader(
            masked, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
        )

    partitions = build_client_partitions(len(train_subset), federation, base.seed)
    input_dims = modality_input_dims(base.data)

    # Identical initial global parameters for both runs.
    init_model = MultimodalDepressionModel(input_dims, base.model)
    init_state: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in init_model.state_dict().items()
    )

    run_dir = create_run_dir(base.train.output_dir, "phase4", "phase4_capability_vs_fedavg")
    save_config(run_dir, {"baseline": base.model_dump(), "federated": fed.model_dump()})

    pattern_summary: dict[str, int] = {}
    for p in partitions:
        pattern_summary[p.pattern_name] = pattern_summary.get(p.pattern_name, 0) + 1
    print(f"Clients by pattern: {pattern_summary}")

    # ── Phase 2 baseline: plain FedAvg ───────────────────────────────────────
    baseline_dir = run_dir / "baseline_fedavg"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_model = MultimodalDepressionModel(input_dims, base.model)
    baseline_model.load_state_dict(copy.deepcopy(init_state))
    baseline_clients = _build_clients(train_subset, partitions, input_dims, base, federation)
    run_simulation(
        baseline_model,
        baseline_clients,
        val_loader,
        num_rounds=federation.num_rounds,
        clients_per_round=federation.clients_per_round,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        run_dir=baseline_dir,
        seed=base.seed,
    )

    # ── Phase 4 protocol: capability-aware aggregation ───────────────────────
    capability_dir = run_dir / "capability_aware"
    capability_dir.mkdir(parents=True, exist_ok=True)
    capability_model = MultimodalDepressionModel(input_dims, base.model)
    capability_model.load_state_dict(copy.deepcopy(init_state))
    capability_clients = _build_clients(train_subset, partitions, input_dims, base, federation)
    run_capability_aware_simulation(
        capability_model,
        capability_clients,
        val_loader,
        aggregation=fed.aggregation,
        num_rounds=federation.num_rounds,
        clients_per_round=federation.clients_per_round,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        run_dir=capability_dir,
        seed=base.seed,
        capability_val_loaders=capability_val_loaders,
    )

    # ── Comparison table (overall + per modality-access pattern) ─────────────
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight)
    device = torch.device("cpu")
    comparison: dict[str, dict[str, Any]] = {}
    loaders = {"overall": val_loader, **capability_val_loaders}
    for name, loader in loaders.items():
        fedavg_m = evaluate_model(baseline_model, loader, objective, device)
        cap_m = evaluate_model(capability_model, loader, objective, device)
        comparison[name] = {
            "fedavg": {k: fedavg_m[k] for k in ("f1", "roc_auc")},
            "capability_aware": {k: cap_m[k] for k in ("f1", "roc_auc")},
            "delta_f1": cap_m["f1"] - fedavg_m["f1"],
            "delta_roc_auc": cap_m["roc_auc"] - fedavg_m["roc_auc"],
        }
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"\nRun dir: {run_dir}")
    print(f"{'setting':<14}{'FedAvg F1':>12}{'CapAware F1':>14}{'ΔF1':>10}")
    for name, row in comparison.items():
        print(
            f"{name:<14}{row['fedavg']['f1']:>12.4f}"
            f"{row['capability_aware']['f1']:>14.4f}{row['delta_f1']:>10.4f}"
        )
    print("\n(On mock noise these numbers are not meaningful; the harness reports "
          "the real improvement once DAIC-WOZ is available.)")


if __name__ == "__main__":
    main()
