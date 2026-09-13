"""Compare centralized, FedAvg and participant-level DP on 24-D CTD features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from privchain.data.ctd_reference import CtdLinearClassifier, CtdReferenceDataset
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.federated.client import ClientDPConfig, FederatedClient
from privchain.federated.partition import ClientPartition, partition_indices
from privchain.federated.simulation import run_simulation
from privchain.seeding import seed_everything
from privchain.training.objective import (
    DepressionObjective,
    collect_scores,
    evaluate_model,
    positive_class_weight,
)
from privchain.training.trainer import CentralizedTrainer


def loader(
    dataset: CtdReferenceDataset, batch_size: int, *, shuffle: bool = False
) -> DataLoader[Sample]:
    """Create a loader using the project-wide batch contract."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def build_datasets(
    artifact_dir: Path, *, include_test: bool = True
) -> dict[str, CtdReferenceDataset]:
    """Load requested splits and apply train-only standardization to each."""
    raw_train = CtdReferenceDataset(artifact_dir, "train")
    mean = raw_train.features.mean(axis=0)
    std = raw_train.features.std(axis=0)
    splits = ("train", "dev", "test") if include_test else ("train", "dev")
    return {
        split: CtdReferenceDataset(
            artifact_dir, split, feature_mean=mean, feature_std=std
        )
        for split in splits
    }


def evaluate_best(
    model: CtdLinearClassifier,
    checkpoint: Path,
    datasets: dict[str, CtdReferenceDataset],
    batch_size: int,
    device: str,
    report_test: bool = True,
) -> dict[str, float]:
    """Load the dev-selected checkpoint and score dev or the untouched test."""
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    objective = DepressionObjective(24, 0.0)
    torch_device = torch.device(device)
    dev_scores, dev_labels = collect_scores(
        model, loader(datasets["dev"], batch_size), torch_device
    )
    candidates = np.linspace(0.05, 0.95, 19)

    def macro_f1(threshold: float) -> float:
        predicted = dev_scores >= threshold
        positive_tp = int(((predicted == 1) & (dev_labels == 1)).sum())
        positive_fp = int(((predicted == 1) & (dev_labels == 0)).sum())
        positive_fn = int(((predicted == 0) & (dev_labels == 1)).sum())
        negative_tp = int(((predicted == 0) & (dev_labels == 0)).sum())
        negative_fp = positive_fn
        negative_fn = positive_fp

        def class_f1(tp: int, fp: int, fn: int) -> float:
            denominator = 2 * tp + fp + fn
            return 2 * tp / denominator if denominator else 0.0

        return (
            class_f1(positive_tp, positive_fp, positive_fn)
            + class_f1(negative_tp, negative_fp, negative_fn)
        ) / 2.0

    threshold = float(max(candidates, key=macro_f1))
    report_split = "test" if report_test else "dev"
    return evaluate_model(
        model,
        loader(datasets[report_split], batch_size),
        objective,
        torch_device,
        threshold=threshold,
    )


def run_central(
    datasets: dict[str, CtdReferenceDataset],
    run_dir: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    optimizer_name: str,
    selection_metric: str,
    report_test: bool,
) -> dict[str, float]:
    """Train and evaluate the centralized arm."""
    seed_everything(seed)
    # Construct the model before iterating any DataLoader. Creating a
    # DataLoader iterator consumes Torch RNG state even with ``shuffle=False``;
    # doing the class-count pass first therefore gave Central and FedAvg
    # different initial weights despite receiving the same seed.
    model = CtdLinearClassifier()
    train_loader = loader(datasets["train"], batch_size, shuffle=True)
    pos_weight = positive_class_weight(loader(datasets["train"], batch_size))
    trainer = CentralizedTrainer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        phq8_max=24,
        phq_loss_weight=0.0,
        pos_weight=pos_weight,
        device=device,
        optimizer_name=optimizer_name,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.fit(
        train_loader,
        loader(datasets["dev"], batch_size),
        epochs=epochs,
        run_dir=run_dir,
        selection_metric=selection_metric,
    )
    return evaluate_best(
        model,
        run_dir / "best_model.pt",
        datasets,
        batch_size,
        device,
        report_test=report_test,
    )


def build_clients(
    dataset: CtdReferenceDataset,
    *,
    num_clients: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    rounds: int,
    seed: int,
    device: str,
    epsilon: float | None,
    max_grad_norm: float,
    optimizer_name: str,
    class_weight_mode: str,
) -> list[FederatedClient]:
    """Construct IID CTD-only clients, optionally with participant-level DP."""
    shards = partition_indices(len(dataset), num_clients, seed + 1)
    pooled_weight = positive_class_weight(loader(dataset, batch_size))
    # ``aggregate_counts`` emulates a secure-sum round: every client contributes
    # only its positive and negative counts and all clients receive the weight
    # derived from the two aggregate totals. The simulator can see the backing
    # dataset, but no per-client count is retained or logged. Production must
    # transport these two counters through secure aggregation.
    aggregate_count_weight: float | None = None
    if class_weight_mode == "aggregate_counts":
        positive_total = sum(int(dataset.labels[list(shard)].sum()) for shard in shards)
        example_total = sum(len(shard) for shard in shards)
        negative_total = example_total - positive_total
        if positive_total > 0 and negative_total > 0:
            aggregate_count_weight = negative_total / positive_total
    clients: list[FederatedClient] = []
    for client_id, indices in enumerate(shards):
        partition = ClientPartition(client_id, "ctd", (1, 0, 0), indices)
        subset = torch.utils.data.Subset(dataset, partition.indices)
        train_loader: DataLoader[Sample] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed + client_id),
        )
        if class_weight_mode == "per_shard":
            pos_weight = positive_class_weight(
                DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn)
            )
        elif class_weight_mode == "pooled_oracle":
            pos_weight = pooled_weight
        elif class_weight_mode == "aggregate_counts":
            pos_weight = aggregate_count_weight
        elif class_weight_mode == "off":
            pos_weight = None
        else:
            raise ValueError(f"unknown class-weight mode: {class_weight_mode}")
        dp = None
        if epsilon is not None:
            dp = ClientDPConfig(
                target_epsilons={"audio": epsilon},
                delta=1e-5,
                max_grad_norm=max_grad_norm,
                batch_size=batch_size,
                num_rounds=rounds,
                seed=seed + client_id,
            )
        clients.append(
            FederatedClient(
                client_id,
                (1, 0, 0),
                CtdLinearClassifier(),
                train_loader,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                phq8_max=24,
                phq_loss_weight=0.0,
                device=device,
                dp=dp,
                pos_weight=pos_weight,
                optimizer_name=optimizer_name,
            )
        )
    return clients


def run_federated(
    datasets: dict[str, CtdReferenceDataset],
    run_dir: Path,
    *,
    seed: int,
    rounds: int,
    num_clients: int,
    clients_per_round: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    epsilon: float | None,
    max_grad_norm: float,
    optimizer_name: str,
    class_weight_mode: str,
    selection_metric: str,
    report_test: bool,
    tail_average_fraction: float | None,
    on_round_end: Any = None,
) -> dict[str, float]:
    """Train one FedAvg or DP-FedAvg arm and evaluate its best round."""
    seed_everything(seed)
    model = CtdLinearClassifier()
    clients = build_clients(
        datasets["train"],
        num_clients=num_clients,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        rounds=rounds,
        seed=seed,
        device=device,
        epsilon=epsilon,
        max_grad_norm=max_grad_norm,
        optimizer_name=optimizer_name,
        class_weight_mode=class_weight_mode,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_simulation(
        model,
        clients,
        loader(datasets["dev"], batch_size),
        num_rounds=rounds,
        clients_per_round=clients_per_round,
        phq8_max=24,
        phq_loss_weight=0.0,
        run_dir=run_dir,
        seed=seed,
        device=device,
        selection_metric=selection_metric,
        tail_average_fraction=tail_average_fraction,
        on_round_end=on_round_end,
    )
    return evaluate_best(
        model,
        run_dir / "best_global_model.pt",
        datasets,
        batch_size,
        device,
        report_test=report_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ctd_reference"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dp-local-epochs", type=int)
    parser.add_argument("--dp-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument(
        "--tail-average-fraction",
        type=float,
        help="Average this final fraction of FL rounds instead of dev checkpointing.",
    )
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="sgd")
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "f1", "loss"),
        default="roc_auc",
        help="One dev-only metric used to checkpoint every arm.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=("off", "per_shard", "aggregate_counts", "pooled_oracle"),
        default="per_shard",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Evaluate on dev only and never load/report the test split.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("central", "fedavg", "fedavg_dp"),
        default=("central", "fedavg", "fedavg_dp"),
    )
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2
        args.rounds = 2
        args.num_clients = 3
        args.clients_per_round = 3

    datasets = build_datasets(args.artifact_dir, include_test=not args.selection_only)
    dp_batch_size = args.dp_batch_size or args.batch_size
    dp_local_epochs = args.dp_local_epochs or args.local_epochs
    common: dict[str, Any] = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "optimizer_name": args.optimizer,
        "selection_metric": args.selection_metric,
        "report_test": not args.selection_only,
    }
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            config={
                "seed": args.seed,
                "rounds": args.rounds,
                "num_clients": args.num_clients,
                "clients_per_round": args.clients_per_round,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "class_weight_mode": args.class_weight_mode,
                "epsilon": args.epsilon,
                "max_grad_norm": args.max_grad_norm,
                "tail_average_fraction": args.tail_average_fraction,
                "dp_batch_size": dp_batch_size,
                "dp_local_epochs": dp_local_epochs,
                "selection_split": "dev",
                "selection_metric": args.selection_metric,
                "report_split": "selection_dev" if args.selection_only else "test",
            },
        )

    def log_round(record: dict[str, Any]) -> None:
        if wandb_run is not None:
            wandb_run.log(record, step=int(record["round"]))
    results: dict[str, Any] = {
        "protocol": {
            "train": len(datasets["train"]),
            "selection_dev": len(datasets["dev"]),
            "report_test": "not_accessed" if args.selection_only else len(datasets["test"]),
            "reported_split": "selection_dev" if args.selection_only else "test",
            "standardization": "train-only",
            "threshold_selection": "dev-only",
            "checkpoint_selection": args.selection_metric,
            "optimizer": args.optimizer,
            "class_weight_mode": args.class_weight_mode,
            "dp_epsilon": args.epsilon,
            "dp_batch_size": dp_batch_size,
            "dp_local_epochs": dp_local_epochs,
            "tail_average_fraction": args.tail_average_fraction,
        },
    }
    if "central" in args.arms:
        results["central"] = run_central(
            datasets, args.output_dir / "central", epochs=args.epochs, **common
        )
    if "fedavg" in args.arms:
        results["fedavg"] = run_federated(
            datasets,
            args.output_dir / "fedavg",
            rounds=args.rounds,
            num_clients=args.num_clients,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            epsilon=None,
            max_grad_norm=args.max_grad_norm,
            tail_average_fraction=args.tail_average_fraction,
            class_weight_mode=args.class_weight_mode,
            on_round_end=log_round,
            **common,
        )
    if "fedavg_dp" in args.arms:
        dp_common = dict(common)
        dp_common["batch_size"] = dp_batch_size
        results["fedavg_dp"] = run_federated(
            datasets,
            args.output_dir / "fedavg_dp",
            rounds=args.rounds,
            num_clients=args.num_clients,
            clients_per_round=args.clients_per_round,
            local_epochs=dp_local_epochs,
            epsilon=args.epsilon,
            max_grad_norm=args.max_grad_norm,
            tail_average_fraction=args.tail_average_fraction,
            class_weight_mode=args.class_weight_mode,
            on_round_end=log_round,
            **dp_common,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    if wandb_run is not None:
        metric_prefix = "selection_dev" if args.selection_only else "test"
        for arm in args.arms:
            if arm in results:
                for key, value in results[arm].items():
                    wandb_run.summary[f"{metric_prefix}/{arm}/{key}"] = value
        wandb_run.summary["result_file"] = str(output)
        wandb_run.finish()
    print(json.dumps(results, indent=2))
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
