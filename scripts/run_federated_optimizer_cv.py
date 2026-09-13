"""CLI: is the FedAvg collapse an optimization defect? Train-only inner CV (Phase 4, H2).

The Chapter-4 comparison of 2026-08-23 reported centralized ROC-AUC 0.676 against
plain FedAvg 0.439 on real DAIC-WOZ. The CTD audit
(``docs/evaluation/CTD-FEDERATED-AUDIT-2026-09-10-FA.md``) traced a similar
collapse to optimizer instability rather than to federation itself: the per-round
seed spread was bimodal, and a single-client parity test reproduced centralized
training to seven digits, so the aggregation loop was sound. Its candidate repair
— class weights from securely summed counts plus plain SGD at a larger step — was
found while looking at the official test split, so it cannot be adopted on that
evidence (ADR-0029's lesson, restated in the audit's decision list).

This script re-asks the question where the answer is admissible: five inner folds
over the pooled official **train** split, repeated over seeds, with each arm's
out-of-fold predictions pooled and compared paired against the centralized arm.
Official dev and test are never constructed, scored, or selected on.

Each arm is one (class-weight mode, local optimizer, learning-rate) combination.
Nothing is hardcoded: the modes come from the CLI, the remaining federation and
training hyperparameters from ``configs/federated.yaml`` and
``configs/baseline.yaml``.

Usage:
    python scripts/run_federated_optimizer_cv.py --daic-config configs/daic_woz.yaml
    python scripts/run_federated_optimizer_cv.py --daic-config configs/daic_woz.yaml \
        --class-weight-modes per_shard aggregate_counts --optimizers adam sgd \
        --learning-rates 0.001 0.1
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import (
    BaselineConfig,
    load_baseline_config,
    load_federated_config,
    resolve_device,
)
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.eval.benchmark import stratified_k_fold_indices
from privchain.eval.metrics import paired_bootstrap_auc_difference
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import ClassWeightMode, build_federated_clients, run_simulation
from privchain.fusion.factory import build_depression_model
from privchain.seeding import derive_seed, seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    build_objective,
    collect_scores,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import build_splits, carve_selection_split, labels_of
from privchain.training.trainer import CentralizedTrainer

#: The arm every paired difference is measured against.
BASELINE_ARM = "centralized"


@dataclass(frozen=True)
class FederatedArm:
    """One federated configuration under test.

    Attributes:
        name: Arm label used in the report and the results file.
        class_weight_mode: How each client weights its BCE term.
        optimizer_name: Local optimizer, ``"adam"`` or ``"sgd"``.
        learning_rate: Local learning rate.
    """

    name: str
    class_weight_mode: ClassWeightMode
    optimizer_name: str
    learning_rate: float


def _loader(
    dataset: Dataset[Sample], batch_size: int, *, shuffle: bool = False, seed: int | None = None
) -> DataLoader[Sample]:
    """Wrap a dataset in a loader with the project's collate function."""
    generator = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
    )


def _run_centralized_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    base: BaselineConfig,
    input_dims: dict[str, int],
    quality_dims: dict[str, int] | None,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], NDArray[np.float64]]:
    """Train the non-federated reference on one fold and score its held-out part.

    Args:
        pool: The official train split as one dataset.
        train_idx: Fold training indices.
        val_idx: Fold scored indices.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        input_dims: Per-modality input dims.
        quality_dims: Per-modality quality widths, or ``None``.
        device: Torch device.
        seed: Seed for this repetition.

    Returns:
        ``(metrics, scores)``, the scores in ``val_idx`` order.
    """
    seed_everything(seed)
    batch_size = base.train.batch_size
    fold_train, selection = carve_selection_split(
        Subset(pool, train_idx),
        [pool_labels[i] for i in train_idx],
        selection_fraction=base.train.selection_fraction,
        seed=seed,
    )
    pos_weight = (
        positive_class_weight(_loader(fold_train, batch_size))
        if base.train.class_weighting
        else None
    )
    model = build_depression_model(input_dims, base.model, quality_dims)
    started = time.monotonic()
    selection_loader = _loader(selection, batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        CentralizedTrainer(
            model,
            learning_rate=base.train.learning_rate,
            weight_decay=base.train.weight_decay,
            phq8_max=base.data.phq8_max,
            phq_loss_weight=base.model.phq_loss_weight,
            device=str(device),
            pos_weight=pos_weight,
            objective=build_objective(base.model, base.data.phq8_max, pos_weight),
        ).fit(
            _loader(fold_train, batch_size, shuffle=True, seed=seed),
            selection_loader,
            epochs=base.train.epochs,
            run_dir=run_dir,
            selection_metric=base.train.selection_metric,
            early_stopping_patience=base.train.early_stopping_patience,
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    model = model.to(device)
    objective = build_objective(base.model, base.data.phq8_max, pos_weight).to(device)
    scored = _loader(Subset(pool, val_idx), batch_size)
    metrics = evaluate_with_selected_threshold(model, selection_loader, scored, objective, device)
    metrics["seconds"] = time.monotonic() - started
    scores, _ = collect_scores(model, scored, device)
    return metrics, scores


def _run_federated_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    arm: FederatedArm,
    base: BaselineConfig,
    federation: Any,
    input_dims: dict[str, int],
    quality_dims: dict[str, int] | None,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], NDArray[np.float64]]:
    """Run one federated arm on one fold and score its held-out part.

    The fold's training half is partitioned across clients; the selection split
    carved out of it drives per-round model selection and the decision
    threshold, exactly as in the centralized arm.

    Args:
        pool: The official train split as one dataset.
        train_idx: Fold training indices.
        val_idx: Fold scored indices.
        pool_labels: Binary label per pool item.
        arm: The federated configuration under test.
        base: Validated baseline config.
        federation: Validated federation config.
        input_dims: Per-modality input dims.
        quality_dims: Per-modality quality widths, or ``None``.
        device: Torch device.
        seed: Seed for this repetition.

    Returns:
        ``(metrics, scores)``, the scores in ``val_idx`` order.
    """
    seed_everything(seed)
    batch_size = base.train.batch_size
    fold_train, selection = carve_selection_split(
        Subset(pool, train_idx),
        [pool_labels[i] for i in train_idx],
        selection_fraction=base.train.selection_fraction,
        seed=seed,
    )
    fold_train_labels = labels_of(fold_train)
    # A fold's training half is smaller than the full train split, so the
    # configured client count can exceed it; one client per session is the floor.
    num_clients = min(federation.num_clients, len(fold_train_labels))
    scaled = federation.model_copy(
        update={
            "num_clients": num_clients,
            "clients_per_round": min(federation.clients_per_round, num_clients),
        }
    )
    partitions = build_client_partitions(
        len(fold_train_labels), scaled, seed, labels=fold_train_labels
    )
    clients = build_federated_clients(
        fold_train,
        partitions,
        input_dims=input_dims,
        model_config=base.model,
        batch_size=batch_size,
        local_epochs=scaled.local_epochs,
        learning_rate=arm.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        seed=seed,
        device=str(device),
        class_weight_mode=arm.class_weight_mode,
        quality_dims=quality_dims,
        optimizer_name=arm.optimizer_name,
    )
    model = build_depression_model(input_dims, base.model, quality_dims)
    started = time.monotonic()
    selection_loader = _loader(selection, batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        run_simulation(
            model,
            clients,
            selection_loader,
            num_rounds=scaled.num_rounds,
            clients_per_round=scaled.clients_per_round,
            phq8_max=base.data.phq8_max,
            phq_loss_weight=base.model.phq_loss_weight,
            run_dir=run_dir,
            seed=seed,
            device=str(device),
            early_stopping_patience=scaled.early_stopping_patience,
        )
        model.load_state_dict(torch.load(run_dir / "best_global_model.pt", map_location=device))

    model = model.to(device)
    # The reported threshold and metrics come from the selection split, never
    # from the scored fold.
    pos_weight = (
        positive_class_weight(_loader(fold_train, batch_size))
        if base.train.class_weighting
        else None
    )
    objective = build_objective(base.model, base.data.phq8_max, pos_weight).to(device)
    scored = _loader(Subset(pool, val_idx), batch_size)
    metrics = evaluate_with_selected_threshold(model, selection_loader, scored, objective, device)
    metrics["seconds"] = time.monotonic() - started
    scores, _ = collect_scores(model, scored, device)
    return metrics, scores


def _assemble_oof(
    folds: list[tuple[list[int], NDArray[np.float64]]], pool_size: int
) -> NDArray[np.float64]:
    """Stitch per-fold predictions into one out-of-fold score per pool item.

    Args:
        folds: ``(val_idx, scores)`` pairs covering the pool exactly once.
        pool_size: Number of items in the pool.

    Returns:
        One score per pool item.

    Raises:
        RuntimeError: If an item was scored twice or never.
    """
    out = np.full(pool_size, np.nan, dtype=np.float64)
    for val_idx, scores in folds:
        for position, index in enumerate(val_idx):
            if not np.isnan(out[index]):
                raise RuntimeError(f"pool item {index} scored twice in one repetition")
            out[index] = scores[position]
    if np.isnan(out).any():
        raise RuntimeError("some pool items were never scored")
    return out


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean and spread of the per-fold metrics that the report table uses."""
    summary: dict[str, float] = {}
    for key in ("roc_auc", "f1", "macro_f1", "accuracy", "pr_auc", "loss"):
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_sd"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return summary


def _build_arms(
    modes: list[str], optimizers: list[str], learning_rates: list[float]
) -> list[FederatedArm]:
    """Expand the CLI grid into named federated arms."""
    arms: list[FederatedArm] = []
    for mode in modes:
        for optimizer in optimizers:
            for rate in learning_rates:
                arms.append(
                    FederatedArm(
                        name=f"fedavg_{mode}_{optimizer}_lr{rate:g}",
                        class_weight_mode=mode,  # type: ignore[arg-type]
                        optimizer_name=optimizer,
                        learning_rate=rate,
                    )
                )
    return arms


def main() -> None:
    """Run the train-only federated inner CV and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=None, help="Override federated rounds.")
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument(
        "--class-weight-modes",
        nargs="+",
        default=["per_shard", "aggregate_counts"],
        choices=("off", "per_shard", "aggregate_counts", "pooled_oracle"),
    )
    parser.add_argument("--optimizers", nargs="+", default=["adam", "sgd"], choices=("adam", "sgd"))
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="Local learning rates to test. Defaults to the configured one.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    if args.daic_config is None:
        print(
            "warning: no --daic-config, so this run only exercises the harness on "
            "mock data, where the label is random.",
            flush=True,
        )

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
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
    seeds = args.seeds if args.seeds else list(base.train.seeds)
    rates = args.learning_rates if args.learning_rates else [base.train.learning_rate]
    arms = _build_arms(args.class_weight_modes, args.optimizers, rates)

    seed_everything(base.seed)
    device = torch.device(resolve_device(base.train.device))
    splits, input_dims = build_splits(base, args.daic_config)
    pool: Dataset[Sample] = ConcatDataset([splits.train, splits.selection])
    pool_labels = labels_of(splits.train) + labels_of(splits.selection)
    folds = stratified_k_fold_indices(pool_labels, args.inner_folds, base.seed)

    run_dir = create_run_dir(base.train.output_dir, "phase4", "phase4_federated_optimizer_cv")
    save_config(run_dir, {"baseline": base.model_dump(), "federated": fed.model_dump()})
    results_path = run_dir / "results.jsonl"
    print(
        f"device={device}  pool={len(pool_labels)} folds={args.inner_folds} seeds={seeds}\n"
        f"arms: {BASELINE_ARM} + {[arm.name for arm in arms]}\n",
        flush=True,
    )

    oof: dict[str, NDArray[np.float64]] = {}
    per_fold: dict[str, list[dict[str, float]]] = {}
    with results_path.open("w", encoding="utf-8") as sink:
        for arm_name in [BASELINE_ARM, *[arm.name for arm in arms]]:
            arm = next((a for a in arms if a.name == arm_name), None)
            print(f"=== {arm_name} ===", flush=True)
            collected: list[tuple[list[int], NDArray[np.float64]]] = []
            rows: list[dict[str, float]] = []
            for seed in seeds:
                for fold_i, (train_idx, val_idx) in enumerate(folds):
                    fold_seed = derive_seed(seed, fold_i)
                    if arm is None:
                        metrics, scores = _run_centralized_fold(
                            pool,
                            train_idx,
                            val_idx,
                            pool_labels,
                            base=base,
                            input_dims=input_dims,
                            quality_dims=splits.quality_dims,
                            device=device,
                            seed=fold_seed,
                        )
                    else:
                        metrics, scores = _run_federated_fold(
                            pool,
                            train_idx,
                            val_idx,
                            pool_labels,
                            arm=arm,
                            base=base,
                            federation=federation,
                            input_dims=input_dims,
                            quality_dims=splits.quality_dims,
                            device=device,
                            seed=fold_seed,
                        )
                    row = {"arm": arm_name, "seed": seed, "fold": fold_i, **metrics}
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    rows.append(metrics)
                    collected.append((list(val_idx), scores))
                    print(
                        f"  seed={seed} fold={fold_i} roc_auc={metrics['roc_auc']:.3f} "
                        f"f1={metrics['f1']:.3f} ({metrics['seconds']:.1f}s)",
                        flush=True,
                    )
            per_fold[arm_name] = rows
            # Average the per-seed out-of-fold vectors: every participant then
            # carries one score per arm, which is what the paired test needs.
            repetitions = [
                _assemble_oof(collected[i : i + len(folds)], len(pool_labels))
                for i in range(0, len(collected), len(folds))
            ]
            oof[arm_name] = np.mean(np.vstack(repetitions), axis=0)

    labels = np.asarray(pool_labels, dtype=np.int_)
    # Persisted so any arm-vs-arm paired question can be asked later without
    # retraining: the adoption decision is "does a candidate beat the committed
    # federated setting", not only "how far is it from centralized".
    (run_dir / "oof_scores.json").write_text(
        json.dumps({"labels": pool_labels, "scores": {k: v.tolist() for k, v in oof.items()}}),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "pool_size": len(pool_labels),
        "inner_folds": args.inner_folds,
        "seeds": seeds,
        "baseline_arm": BASELINE_ARM,
        "arms": {},
    }
    for arm_name, rows in per_fold.items():
        entry: dict[str, Any] = _summarize(rows)
        if arm_name != BASELINE_ARM:
            delta = paired_bootstrap_auc_difference(
                labels,
                oof[arm_name],
                oof[BASELINE_ARM],
                n_resamples=args.bootstrap_resamples,
                seed=base.seed,
            )
            entry["pooled_vs_baseline"] = delta
        report["arms"][arm_name] = entry
    (run_dir / "optimizer_cv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Federated optimization — train-only inner CV (Phase 4)",
        "",
        "Inner folds over the pooled official train split. Dev and test are",
        "neither constructed nor scored. `pooled` bootstraps the paired",
        "difference between an arm's out-of-fold predictions and the",
        f"`{BASELINE_ARM}` arm's.",
        "",
        "| arm | ROC-AUC | ±sd | macro-F1 | pooled Δ | 95% CI | p |",
        "|---|---|:--:|---|---|---|---|",
    ]
    ordered = sorted(
        report["arms"].items(), key=lambda kv: kv[1].get("roc_auc_mean", 0.0), reverse=True
    )
    for arm_name, entry in ordered:
        delta = entry.get("pooled_vs_baseline")
        if delta is None:
            cells = ["-", "-", "-"]
        else:
            cells = [
                f"{delta['difference']:+.3f}",
                f"[{delta['low']:+.3f}, {delta['high']:+.3f}]",
                f"{delta['p_two_sided']:.3f}",
            ]
        lines.append(
            f"| `{arm_name}` | {entry.get('roc_auc_mean', float('nan')):.3f} | "
            f"{entry.get('roc_auc_sd', float('nan')):.3f} | "
            f"{entry.get('macro_f1_mean', float('nan')):.3f} | " + " | ".join(cells) + " |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
