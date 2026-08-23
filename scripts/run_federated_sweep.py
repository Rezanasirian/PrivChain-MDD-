"""CLI: nested-CV hyperparameter sweep for the federated arms (Phase 4, H2).

Chapter-4 leaves federation far behind the centralized baseline while every arm
shares one model and one ``configs/baseline.yaml``, so the gap is not model
quality. This searches for what closes it, under a protocol that can say *why*.

**Nested, because the report split is the official dev split.**
``build_splits`` makes the official DAIC-WOZ dev split the report split
(``privchain.training.protocol``), so ranking dozens of configurations on it
would turn dev into a tuning set and re-running the winner would not undo that.
Instead:

* **Inner** — k-fold CV over the official *train* split only. Every ranking and
  every stage decision uses inner-fold mean ROC-AUC and nothing else.
* **Outer** — dev is never read here. The single winning configuration is
  evaluated on it once, separately, and that number is the reported estimate.
* **Baseline** — the centralized arm is re-measured under the identical inner CV
  by ``--include-centralized``. Neither the 0.676 of the phase-7 pooled CV nor
  the 0.740 of the dev harness is a valid reference for inner-CV numbers.

The official test split is never touched.

**Compute budget.** Varying ``batch_size`` and ``local_epochs`` together changes
optimizer steps, local drift, Adam warm-up and cost at once. ``--budget matched``
rescales ``num_rounds`` so every configuration is *allowed* the same number of
local optimizer steps, isolating how steps are grouped from how many there are.
``--budget fixed`` holds rounds constant instead, the uncontrolled comparison.

The match is a cap, not a guaranteed spend: early stopping can end a run well
short of it, and does — at one step per round the selection metric plateaus and
patience fires around round 41 of a planned 120. Every row therefore carries
``planned_rounds`` beside ``rounds_run`` and ``steps_per_client``, so a
configuration that stopped early is visible rather than silently compared as
though it had spent its budget.

Usage:
    python scripts/run_federated_sweep.py --daic-config configs/daic_woz.yaml \\
        --stage steps --budget matched --include-centralized
    python scripts/run_federated_sweep.py --daic-config configs/daic_woz.yaml \\
        --stage clients
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import (
    load_baseline_config,
    load_federated_config,
    resolve_device,
)
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.eval.benchmark import aggregate_metrics, stratified_k_fold_indices
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import (
    ClassWeightMode,
    build_federated_clients,
    run_capability_aware_simulation,
    run_simulation,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    DepressionObjective,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import build_splits, carve_selection_split, labels_of
from privchain.training.trainer import CentralizedTrainer

#: Local optimizer steps per client the matched budget targets. The committed
#: setup (batch_size 32, local_epochs 1, 120 rounds) spends exactly this, so
#: "matched" means "same cost as the configuration we are trying to beat".
STEP_BUDGET_PER_CLIENT = 120


def _loader(dataset: Dataset[Sample], batch_size: int, *, shuffle: bool = False) -> DataLoader:
    """Build a padded DataLoader over ``dataset``."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _steps_per_round(shard_size: int, batch_size: int, local_epochs: int) -> int:
    """Local optimizer steps one client takes in one round.

    A shard smaller than ``batch_size`` collapses to a single batch, which is why
    the committed setup spends one step per client per round however large the
    round budget looks.

    Args:
        shard_size: Training samples the client holds.
        batch_size: Local batch size.
        local_epochs: Local epochs per round.

    Returns:
        Steps per round, at least one.
    """
    return max(1, math.ceil(shard_size / max(1, batch_size))) * local_epochs


def _rounds_for_budget(shard_size: int, batch_size: int, local_epochs: int) -> int:
    """Rounds that spend :data:`STEP_BUDGET_PER_CLIENT` steps per client.

    Args:
        shard_size: Training samples the client holds.
        batch_size: Local batch size.
        local_epochs: Local epochs per round.

    Returns:
        A round count of at least one.
    """
    return max(
        1, round(STEP_BUDGET_PER_CLIENT / _steps_per_round(shard_size, batch_size, local_epochs))
    )


def _grid(stage: str) -> list[dict[str, Any]]:
    """Configurations for a named stage.

    Stage ``steps`` asks whether the gap is about how local optimizer steps are
    grouped: under a matched budget, ``(4, 1)`` and ``(4, 3)`` spend the same
    total steps but group them differently, so Adam warm-up within a round
    separates from step count. Stage ``clients`` asks whether data per client is
    the binding constraint, with class weighting varied alongside because its
    measured effect flipped sign between harnesses.

    Args:
        stage: ``"steps"`` or ``"clients"``.

    Returns:
        One dict of overrides per configuration.

    Raises:
        ValueError: If ``stage`` is unknown.
    """
    if stage == "steps":
        return [{"batch_size": bs, "local_epochs": le} for bs in (32, 8, 4) for le in (1, 3, 5)]
    if stage == "clients":
        return [
            {"num_clients": n, "class_weight_mode": mode}
            for n in (10, 5, 3)
            for mode in ("off", "per_shard", "pooled_oracle")
        ]
    raise ValueError(f"unknown stage: {stage}")


def _run_federated_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    base: Any,
    fed: Any,
    input_dims: dict[str, int],
    device: torch.device,
    seed: int,
    arm: str,
    overrides: dict[str, Any],
    budget: str,
) -> dict[str, float]:
    """Train one federated arm on one inner fold and score its validation part.

    Mirrors ``scripts/run_final_evaluation.py``: clients partition the fold's
    training data only, per-round selection happens on a split carved out of it,
    and the fold's scored split is read once at the end.

    Args:
        pool: The official train split, as one dataset.
        train_idx: Indices of the fold's training part.
        val_idx: Indices of the fold's scored part.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        fed: Validated federated config.
        input_dims: Per-modality input dims.
        device: Torch device.
        seed: Seed for this repetition; also reseeds the partition.
        arm: ``"fedavg"`` or ``"proposed"``.
        overrides: Stage overrides (batch size, local epochs, clients, weighting).
        budget: ``"matched"`` or ``"fixed"``.

    Returns:
        Metrics on the fold's scored part, plus measured cost.
    """
    seed_everything(seed)
    batch_size = int(overrides.get("batch_size", base.train.batch_size))
    local_epochs = int(overrides.get("local_epochs", fed.federation.local_epochs))
    weight_mode: ClassWeightMode = overrides.get(
        "class_weight_mode", "per_shard" if base.train.class_weighting else "off"
    )

    fold_train, selection = carve_selection_split(
        Subset(pool, train_idx),
        [pool_labels[i] for i in train_idx],
        selection_fraction=base.train.selection_fraction,
        seed=seed,
    )
    n_train = len(fold_train)  # type: ignore[arg-type]
    num_clients = min(int(overrides.get("num_clients", fed.federation.num_clients)), n_train)
    shard = max(1, n_train // num_clients)
    rounds = (
        _rounds_for_budget(shard, batch_size, local_epochs)
        if budget == "matched"
        else fed.federation.num_rounds
    )

    federation = fed.federation.model_copy(
        update={
            "num_rounds": rounds,
            "num_clients": num_clients,
            "clients_per_round": num_clients,
            "local_epochs": local_epochs,
        }
    )
    # Reseeded per repetition, so the reported spread covers partition draw and
    # not only initialization -- unlike run_federated_comparison.py, which builds
    # the partition once from base.seed outside its seed loop.
    partitions = build_client_partitions(
        n_train, federation, seed, labels=[pool_labels[i] for i in train_idx][:n_train]
    )
    clients = build_federated_clients(
        fold_train,
        partitions,
        input_dims=input_dims,
        model_config=base.model,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        seed=seed,
        device=str(device),
        class_weight_mode=weight_mode,
    )
    global_model = MultimodalDepressionModel(input_dims, base.model).to(device)
    selection_loader = _loader(selection, batch_size)
    common: dict[str, Any] = {
        "num_rounds": rounds,
        "clients_per_round": num_clients,
        "phq8_max": base.data.phq8_max,
        "phq_loss_weight": base.model.phq_loss_weight,
        "seed": seed,
        "device": str(device),
        "early_stopping_patience": federation.early_stopping_patience,
    }

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        if arm == "fedavg":
            history = run_simulation(
                global_model, clients, selection_loader, run_dir=run_dir, **common
            )
        else:
            history = run_capability_aware_simulation(
                global_model,
                clients,
                selection_loader,
                aggregation=fed.aggregation,
                run_dir=run_dir,
                **common,
            )
        global_model.load_state_dict(
            torch.load(run_dir / "best_global_model.pt", map_location=device)
        )

    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight).to(device)
    metrics = evaluate_with_selected_threshold(
        global_model.to(device),
        selection_loader,
        _loader(Subset(pool, val_idx), batch_size),
        objective,
        device,
    )
    last = history[-1] if history else {}
    steps = float(last.get("cumulative_optimizer_steps", 0))
    metrics["planned_rounds"] = float(rounds)
    metrics["rounds_run"] = float(len(history))
    metrics["optimizer_steps"] = steps
    metrics["steps_per_client"] = steps / max(1, len(clients))
    metrics["examples_seen"] = float(last.get("cumulative_examples_seen", 0))
    metrics["seconds"] = float(last.get("elapsed_seconds", 0.0))
    metrics["num_clients"] = float(len(clients))
    return metrics


def _run_centralized_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    base: Any,
    input_dims: dict[str, int],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Train the centralized baseline on one inner fold, for a matched reference.

    Args:
        pool: The official train split, as one dataset.
        train_idx: Indices of the fold's training part.
        val_idx: Indices of the fold's scored part.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        input_dims: Per-modality input dims.
        device: Torch device.
        seed: Seed for this repetition.

    Returns:
        Metrics on the fold's scored part.
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
    model = MultimodalDepressionModel(input_dims, base.model)
    trainer = CentralizedTrainer(
        model,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        device=str(device),
        pos_weight=pos_weight,
    )
    selection_loader = _loader(selection, batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trainer.fit(
            _loader(fold_train, batch_size, shuffle=True),
            selection_loader,
            epochs=base.train.epochs,
            run_dir=run_dir,
            selection_metric=base.train.selection_metric,
            early_stopping_patience=base.train.early_stopping_patience,
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight).to(device)
    return evaluate_with_selected_threshold(
        model.to(device),
        selection_loader,
        _loader(Subset(pool, val_idx), batch_size),
        objective,
        device,
    )


def _label(overrides: dict[str, Any]) -> str:
    """Render a configuration as a stable, sortable label."""
    return " ".join(f"{k}={v}" for k, v in sorted(overrides.items()))


def _summarize(rows: list[dict[str, Any]], centralized: dict[str, float] | None) -> str:
    """Render the ranking, with intervals and the selection caveat stated.

    Args:
        rows: One row per configuration x arm x seed x fold.
        centralized: Protocol-matched centralized aggregate, if measured.

    Returns:
        Markdown.
    """
    lines = ["# Federated sweep — inner-CV ranking", ""]
    if centralized is not None:
        lines += [
            f"Protocol-matched centralized reference: **ROC-AUC "
            f"{centralized.get('roc_auc_mean', float('nan')):.3f}"
            f"±{centralized.get('roc_auc_std', float('nan')):.3f}** on the same inner folds.",
            "",
        ]
    lines += [
        "Ranked on inner-fold mean ROC-AUC over the official train split. Dev is",
        "untouched here. Selecting the maximum over many configurations inflates",
        "the winner's inner score, so treat the top row as a *candidate*: its",
        "unbiased estimate is the single outer dev evaluation, run separately.",
        "",
        "``steps/client`` is what a run actually spent; ``rounds`` shows run vs",
        "planned, so a configuration truncated by early stopping is visible.",
        "",
        "| arm | configuration | ROC-AUC | ±sd | F1 | steps/client | rounds | seconds |",
        "|---|---|---|:--:|---|---|---|---|",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["arm"], row["config"]), []).append(row)
    ranked = sorted(
        grouped.items(),
        key=lambda kv: statistics.fmean(r["roc_auc"] for r in kv[1]),
        reverse=True,
    )
    for (arm, config), group in ranked:
        aucs = [r["roc_auc"] for r in group]
        f1s = [r["f1"] for r in group]
        sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        steps = statistics.fmean(r["steps_per_client"] for r in group)
        ran = statistics.fmean(r["rounds_run"] for r in group)
        planned = statistics.fmean(r["planned_rounds"] for r in group)
        seconds = statistics.fmean(r["seconds"] for r in group)
        lines.append(
            f"| {arm} | {config} | {statistics.fmean(aucs):.3f} | {sd:.3f} | "
            f"{statistics.fmean(f1s):.3f} | {steps:.0f} | {ran:.0f}/{planned:.0f} | {seconds:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _iter_jobs(
    configs: list[dict[str, Any]], arms: Sequence[str], seeds: Sequence[int]
) -> Iterator[tuple[dict[str, Any], str, int]]:
    """Yield every (configuration, arm, seed) triple."""
    for overrides in configs:
        for arm in arms:
            for seed in seeds:
                yield overrides, arm, seed


def main() -> None:
    """Run one sweep stage and write its ranking."""
    parser = argparse.ArgumentParser(description="Nested-CV federated sweep (Phase 4).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--stage", choices=("steps", "clients"), required=True)
    parser.add_argument("--budget", choices=("matched", "fixed"), default="matched")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=["fedavg", "proposed"])
    parser.add_argument(
        "--include-centralized",
        action="store_true",
        help="Measure the centralized baseline on the same inner folds.",
    )
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    seeds = args.seeds if args.seeds else list(base.train.seeds)
    seed_everything(base.seed)
    device = torch.device(resolve_device(base.train.device))

    # Tuning happens on the official train split only; splits.report is the
    # official dev split and is deliberately never constructed into a loader.
    splits, input_dims = build_splits(base, args.daic_config)
    pool: Dataset[Sample] = ConcatDataset([splits.train, splits.selection])
    pool_labels = labels_of(splits.train) + labels_of(splits.selection)
    folds = stratified_k_fold_indices(pool_labels, args.inner_folds, base.seed)

    run_dir = create_run_dir(
        base.train.output_dir, "phase4", f"phase4_federated_sweep_{args.stage}"
    )
    configs = _grid(args.stage)
    rows: list[dict[str, Any]] = []
    results_path = run_dir / "sweep_results.jsonl"

    with results_path.open("w", encoding="utf-8") as sink:
        for overrides, arm, seed in _iter_jobs(configs, args.arms, seeds):
            for fold_i, (train_idx, val_idx) in enumerate(folds):
                metrics = _run_federated_fold(
                    pool,
                    train_idx,
                    val_idx,
                    pool_labels,
                    base=base,
                    fed=fed,
                    input_dims=input_dims,
                    device=device,
                    seed=seed + fold_i,
                    arm=arm,
                    overrides=overrides,
                    budget=args.budget,
                )
                row = {
                    "arm": arm,
                    "config": _label(overrides),
                    "overrides": overrides,
                    "seed": seed,
                    "fold": fold_i,
                    **metrics,
                }
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            print(f"  done {arm} {_label(overrides)} seed={seed}", flush=True)

        centralized_aggregate: dict[str, float] | None = None
        if args.include_centralized:
            per_fold = []
            for seed in seeds:
                for fold_i, (train_idx, val_idx) in enumerate(folds):
                    metrics = _run_centralized_fold(
                        pool,
                        train_idx,
                        val_idx,
                        pool_labels,
                        base=base,
                        input_dims=input_dims,
                        device=device,
                        seed=seed + fold_i,
                    )
                    per_fold.append(metrics)
                    sink.write(
                        json.dumps(
                            {
                                "arm": "centralized",
                                "config": "reference",
                                "seed": seed,
                                "fold": fold_i,
                                **metrics,
                            }
                        )
                        + "\n"
                    )
            centralized_aggregate = aggregate_metrics(per_fold)
            print("  done centralized reference", flush=True)

    (run_dir / "sweep_summary.md").write_text(
        _summarize(rows, centralized_aggregate), encoding="utf-8"
    )
    # save_config already records git state, dependency versions and GPU; these
    # extras are what make this run's protocol and population checkable.
    save_config(
        run_dir,
        {"baseline": base.model_dump(), "federated": fed.model_dump()},
        manifest_extra={
            "dataset": "daic_woz" if args.daic_config else "mock",
            "pool_size": len(pool_labels),
            "protocol": "nested-inner-cv-train-split-only",
            "official_dev_read": False,
            "official_test_read": False,
            "inner_folds": args.inner_folds,
            "seeds": list(seeds),
            "budget": args.budget,
            "stage": args.stage,
            "capability_sample_share": _capability_shares(pool_labels, fed, base.seed),
        },
    )
    print(f"\nwrote {run_dir}")


def _capability_shares(pool_labels: list[int], fed: Any, seed: int) -> dict[str, float]:
    """Share of training samples behind each capability pattern.

    Recorded because ``assign_capabilities`` rounds ``fraction * num_clients``,
    so changing the client count silently changes the population — at three
    clients ``text_only`` disappears entirely. Writing the realized shares makes
    that confound checkable rather than assumed away.

    Args:
        pool_labels: Labels of the tuning pool (its length is the sample count).
        fed: Validated federated config.
        seed: Partition seed.

    Returns:
        ``{pattern_name: share_of_samples}``.
    """
    partitions = build_client_partitions(len(pool_labels), fed.federation, seed)
    total = sum(len(p.indices) for p in partitions) or 1
    shares: dict[str, float] = {}
    for partition in partitions:
        shares[partition.pattern_name] = (
            shares.get(partition.pattern_name, 0.0) + len(partition.indices) / total
        )
    return shares


if __name__ == "__main__":
    main()
