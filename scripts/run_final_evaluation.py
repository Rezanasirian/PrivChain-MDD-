"""CLI: final comparative evaluation for Chapter 4 (Phase 7, objective H5).

Runs one shared 10-fold cross-validation (+ a held-out test fold) over every
method variant and writes the Chapter-4 tables/plots:

* **Main comparison** — centralized (no FL/DP), plain FedAvg, a personalized
  variant, and the full proposed framework.
* **Ablations** — the proposed framework minus reputation, minus distillation.
* **DP privacy–utility** — per-modality *adaptive* allocation vs a *uniform*
  budget of the same total ε (centralized DP-SGD).
* **Inference latency** — forward-pass latency across batch sizes (compute
  budget).

Each named variant also maps to a representative baseline from the literature
(centralized ≈ Xu et al. 2023, uniform-DP ≈ De Chaudhury et al. 2024,
personalized ≈ Fan et al. 2025); these are **simplified stand-ins** on the same
data/model, not faithful reimplementations — see ADR-0008.

Outputs live under ``experiments/phase7/<run-id>/``: ``cv_results.json``,
``ablation.json``, ``dp_comparison.json``, ``latency.json``, and a combined
``chapter4_summary.md``. On mock data the depression label is random, so the
accuracy numbers are placeholders (as in earlier phases); the harness and table
shapes are what Phase 7 delivers, and DAIC-WOZ fills in the real numbers.

Usage:
    python scripts/run_final_evaluation.py
    python scripts/run_final_evaluation.py --k-folds 5 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from privchain.config import (
    AggregationConfig,
    load_baseline_config,
    load_evaluation_config,
    load_federated_config,
    load_privacy_config,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, collate_fn
from privchain.eval.benchmark import (
    aggregate_metrics,
    held_out_split,
    k_fold_indices,
    measure_inference_latency,
)
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import (
    build_federated_clients,
    run_capability_aware_simulation,
    run_simulation,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.budget_allocator import PerModalityBudgetAllocator, allocate_target_epsilons
from privchain.privacy.dp_sgd import dp_train_epoch, map_parameter_groups, resolve_group_sigmas
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import DepressionObjective, evaluate_model
from privchain.training.trainer import CentralizedTrainer

MODALITIES = ("audio", "video", "text")
FoldRunner = Callable[[list[int], list[int], int], dict[str, float]]


def _loader(
    dataset: Any, indices: list[int], batch_size: int, *, shuffle: bool = False
) -> DataLoader:
    """Build a padded DataLoader over a subset of ``dataset``."""
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def _make_batches(num_items: int, batch_size: int, seed: int) -> list[list[int]]:
    """Shuffle indices and chunk into logical batches (for DP-SGD)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items).tolist()
    return [order[i : i + batch_size] for i in range(0, num_items, batch_size)]


def _eval_centralized(
    full: Any, train_idx: list[int], test_idx: list[int], *, base: Any, epochs: int, seed: int
) -> dict[str, float]:
    """Train the centralized baseline on a fold and score the test split."""
    seed_everything(seed)
    input_dims = modality_input_dims(base.data)
    model = MultimodalDepressionModel(input_dims, base.model)
    trainer = CentralizedTrainer(
        model,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
    )
    train_loader = _loader(full, train_idx, base.train.batch_size, shuffle=True)
    for _ in range(epochs):
        trainer.train_epoch(train_loader)
    return trainer.evaluate(_loader(full, test_idx, base.train.batch_size))


def _eval_centralized_dp(
    full: Any,
    train_idx: list[int],
    test_idx: list[int],
    *,
    base: Any,
    priv: Any,
    epochs: int,
    seed: int,
    mode: str,
) -> dict[str, float]:
    """Train centralized per-modality DP-SGD (adaptive or uniform ε) on a fold."""
    seed_everything(seed)
    input_dims = modality_input_dims(base.data)
    device = torch.device("cpu")
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight)

    adaptive = allocate_target_epsilons(priv.allocation, priv.per_modality)
    if mode == "uniform":
        uniform_eps = sum(adaptive.values()) / len(MODALITIES)  # same total budget
        targets = {m: uniform_eps for m in MODALITIES}
    else:
        targets = adaptive
    risks = {m: priv.per_modality[m].reidentification_risk for m in MODALITIES}

    batch_size = base.train.batch_size
    n_train = len(train_idx)
    sample_rate = min(1.0, batch_size / n_train)
    steps = epochs * math.ceil(n_train / batch_size)
    allocator = PerModalityBudgetAllocator(
        targets, risks, delta=priv.delta, sample_rate=sample_rate, steps=steps
    )
    group_sigmas = resolve_group_sigmas(allocator.noise_multipliers())

    model = MultimodalDepressionModel(input_dims, base.model).to(device)
    groups = map_parameter_groups(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=base.train.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    train_subset = Subset(full, train_idx)
    for _ in range(epochs):
        dp_train_epoch(
            model,
            train_subset,
            _make_batches(n_train, batch_size, seed),
            objective,
            groups=groups,
            group_sigmas=group_sigmas,
            max_grad_norm=priv.max_grad_norm,
            optimizer=optimizer,
            device=device,
            generator=generator,
        )
    return evaluate_model(model, _loader(full, test_idx, batch_size), objective, device)


def _eval_federated(
    full: Any,
    train_idx: list[int],
    test_idx: list[int],
    *,
    base: Any,
    fed: Any,
    rounds: int,
    seed: int,
    strategy: str,
    aggregation: AggregationConfig,
) -> dict[str, float]:
    """Train a federated variant on a fold and score the held-out test split."""
    seed_everything(seed)
    input_dims = modality_input_dims(base.data)
    device = torch.device("cpu")
    train_subset = Subset(full, train_idx)
    federation = fed.federation.model_copy(
        update={"num_rounds": rounds, "clients_per_round": fed.federation.num_clients}
    )
    partitions = build_client_partitions(len(train_subset), federation, seed)
    clients = build_federated_clients(
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
        seed=seed,
    )
    global_model = MultimodalDepressionModel(input_dims, base.model)
    test_loader = _loader(full, test_idx, base.train.batch_size)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        if strategy == "fedavg":
            run_simulation(
                global_model, clients, test_loader,
                num_rounds=rounds, clients_per_round=federation.num_clients,
                phq8_max=base.data.phq8_max, phq_loss_weight=base.model.phq_loss_weight,
                run_dir=run_dir, seed=seed,
            )
        else:
            run_capability_aware_simulation(
                global_model, clients, test_loader, aggregation=aggregation,
                num_rounds=rounds, clients_per_round=federation.num_clients,
                phq8_max=base.data.phq8_max, phq_loss_weight=base.model.phq_loss_weight,
                run_dir=run_dir, seed=seed,
            )
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight)
    return evaluate_model(global_model, test_loader, objective, device)


def _cross_validate(
    runner: FoldRunner,
    dev_idx: list[int],
    splits: list[tuple[list[int], list[int]]],
    held_out_idx: list[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run a method over CV folds + one held-out fit, returning aggregate stats."""
    per_fold = [
        runner([dev_idx[i] for i in tr], [dev_idx[i] for i in te], seed + fold)
        for fold, (tr, te) in enumerate(splits)
    ]
    held_out = runner(dev_idx, held_out_idx, seed)
    return {
        "cv": aggregate_metrics(per_fold),
        "held_out": {k: held_out[k] for k in ("f1", "roc_auc", "accuracy")},
    }


def _fmt(agg: dict[str, float], key: str) -> str:
    """Format ``mean±std`` for a metric from an aggregate dict."""
    mean, std = agg.get(f"{key}_mean", float("nan")), agg.get(f"{key}_std", float("nan"))
    return f"{mean:.3f}±{std:.3f}"


def main() -> None:
    """Run the full Chapter-4 comparison and write its tables/plots."""
    parser = argparse.ArgumentParser(description="Final comparative evaluation (Phase 7).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--k-folds", type=int, default=None, help="Override k_folds.")
    parser.add_argument("--rounds", type=int, default=None, help="Override federated rounds.")
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    priv = load_privacy_config(args.privacy_config).privacy
    ev = load_evaluation_config(args.evaluation_config).evaluation
    seed_everything(base.seed)

    k_folds = args.k_folds if args.k_folds is not None else ev.k_folds
    rounds = args.rounds if args.rounds is not None else ev.rounds

    full = MockDaicWozDataset(base.data, seed=base.seed)
    n = len(full)
    dev_idx, held_out_idx = held_out_split(n, ev.held_out_fraction, base.seed)
    splits = k_fold_indices(len(dev_idx), min(k_folds, len(dev_idx)), base.seed)
    print(f"n={n}  dev={len(dev_idx)}  held_out={len(held_out_idx)}  folds={len(splits)}")

    def cap_config(*, reputation: bool, distillation: bool) -> AggregationConfig:
        return fed.aggregation.model_copy(
            update={
                "strategy": "capability_aware",
                "reputation_weighting": reputation,
                "federated_distillation": distillation,
            }
        )

    # Method runners (each a fold closure: train_idx, test_idx, seed -> metrics).
    methods: dict[str, FoldRunner] = {
        "centralized": lambda tr, te, s: _eval_centralized(
            full, tr, te, base=base, epochs=ev.epochs, seed=s
        ),
        "fedavg": lambda tr, te, s: _eval_federated(
            full, tr, te, base=base, fed=fed, rounds=rounds, seed=s,
            strategy="fedavg", aggregation=fed.aggregation,
        ),
        "personalized": lambda tr, te, s: _eval_federated(
            full, tr, te, base=base, fed=fed, rounds=rounds, seed=s,
            strategy="capability_aware",
            aggregation=cap_config(reputation=True, distillation=False),
        ),
        "proposed": lambda tr, te, s: _eval_federated(
            full, tr, te, base=base, fed=fed, rounds=rounds, seed=s,
            strategy="capability_aware", aggregation=cap_config(reputation=True, distillation=True),
        ),
        "proposed_no_reputation": lambda tr, te, s: _eval_federated(
            full, tr, te, base=base, fed=fed, rounds=rounds, seed=s,
            strategy="capability_aware",
            aggregation=cap_config(reputation=False, distillation=True),
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, runner in methods.items():
        print(f"  running {name} ...")
        results[name] = _cross_validate(runner, dev_idx, splits, held_out_idx, seed=base.seed)

    # DP privacy–utility: adaptive vs uniform allocation (centralized DP-SGD).
    dp_results: dict[str, dict[str, Any]] = {}
    for mode in ("adaptive", "uniform"):
        print(f"  running dp_{mode} ...")
        dp_results[mode] = _cross_validate(
            lambda tr, te, s, mode=mode: _eval_centralized_dp(
                full, tr, te, base=base, priv=priv, epochs=ev.epochs, seed=s, mode=mode
            ),
            dev_idx, splits, held_out_idx, seed=base.seed,
        )

    # Inference latency across batch sizes.
    seed_everything(base.seed)
    latency_model = MultimodalDepressionModel(modality_input_dims(base.data), base.model)
    latency = measure_inference_latency(
        latency_model, full, batch_sizes=ev.latency_batch_sizes,
        repeats=ev.latency_repeats, device=torch.device("cpu"),
    )

    run_dir = create_run_dir(base.train.output_dir, "phase7", "phase7_final_evaluation")
    save_config(
        run_dir,
        {"baseline": base.model_dump(), "federated": fed.model_dump(),
         "privacy": priv.model_dump(), "evaluation": ev.model_dump()},
    )
    (run_dir / "cv_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ablation = {k: results[k] for k in ("proposed", "proposed_no_reputation", "personalized")}
    (run_dir / "ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    (run_dir / "dp_comparison.json").write_text(json.dumps(dp_results, indent=2), encoding="utf-8")
    (run_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
    _write_summary(run_dir / "chapter4_summary.md", results, ablation, dp_results, latency)
    _plot_latency(latency, run_dir / "inference_latency.png")

    print(f"\n{'method':<24}{'CV F1':>16}{'CV ROC-AUC':>16}{'held-out F1':>14}")
    for name, res in results.items():
        print(
            f"{name:<24}{_fmt(res['cv'], 'f1'):>16}{_fmt(res['cv'], 'roc_auc'):>16}"
            f"{res['held_out']['f1']:>14.3f}"
        )
    print(f"\nRun dir: {run_dir}")
    print("(On mock data the labels are random, so accuracy is a placeholder; "
          "the tables are the deliverable.)")


def _write_summary(
    path: Path,
    results: dict[str, dict[str, Any]],
    ablation: dict[str, dict[str, Any]],
    dp_results: dict[str, dict[str, Any]],
    latency: list[dict[str, Any]],
) -> None:
    """Write the combined Chapter-4 markdown tables."""
    lines = ["# Chapter 4 — final evaluation (mock data)\n"]
    lines.append("> On mock data the depression label is random; these are "
                 "placeholder numbers demonstrating the tables. DAIC-WOZ fills them in.\n")

    lines.append("\n## Main comparison (10-fold CV, mean±std)\n")
    lines.append("| method | F1 | ROC-AUC | accuracy | held-out F1 |")
    lines.append("|---|---|---|---|---|")
    labels = {
        "centralized": "Centralized (no FL/DP) — cf. Xu et al. 2023",
        "fedavg": "Plain FedAvg",
        "personalized": "Personalized (reputation) — cf. Fan et al. 2025",
        "proposed": "Proposed (full framework)",
        "proposed_no_reputation": "Proposed − reputation",
    }
    for name, res in results.items():
        lines.append(
            f"| {labels.get(name, name)} | {_fmt(res['cv'], 'f1')} | "
            f"{_fmt(res['cv'], 'roc_auc')} | {_fmt(res['cv'], 'accuracy')} | "
            f"{res['held_out']['f1']:.3f} |"
        )

    lines.append("\n## Ablation (proposed vs. component removed)\n")
    lines.append("| variant | F1 | ROC-AUC |")
    lines.append("|---|---|---|")
    abl_labels = {
        "proposed": "Full framework",
        "proposed_no_reputation": "− reputation weighting",
        "personalized": "− federated distillation",
    }
    for name, res in ablation.items():
        lines.append(
            f"| {abl_labels.get(name, name)} | {_fmt(res['cv'], 'f1')} | "
            f"{_fmt(res['cv'], 'roc_auc')} |"
        )

    lines.append("\n## DP privacy–utility (centralized DP-SGD, same total ε)\n")
    lines.append("| allocation | F1 | ROC-AUC |")
    lines.append("|---|---|---|")
    for mode in ("adaptive", "uniform"):
        res = dp_results[mode]
        lines.append(
            f"| {mode} per-modality | {_fmt(res['cv'], 'f1')} | {_fmt(res['cv'], 'roc_auc')} |"
        )

    lines.append("\n## Inference latency (forward pass)\n")
    lines.append("| batch size | ms/batch | ms/sample |")
    lines.append("|---|---|---|")
    for row in latency:
        lines.append(
            f"| {row['batch_size']} | {row['ms_per_batch']:.3f} | {row['ms_per_sample']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_latency(latency: list[dict[str, Any]], path: Path) -> None:
    """Plot ms/sample vs batch size (no-op without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote latency.json only.")
        return

    sizes = [row["batch_size"] for row in latency]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, [row["ms_per_sample"] for row in latency], marker="o")
    ax.set_xlabel("Batch size (compute budget)")
    ax.set_ylabel("Latency (ms / sample)")
    ax.set_title("Inference latency vs. batch size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
