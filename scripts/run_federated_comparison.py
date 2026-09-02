"""CLI: federated arms on real DAIC-WOZ under the shared protocol (Phases 2 & 4, H2).

H2 claims that when clients hold *different subsets of the modalities*, plain
FedAvg is damaged by zero-imputing what a client lacks, and that capability-aware
subgraph aggregation — plus reputation weighting and federated distillation —
repairs it. Every federated number in this project so far came from the mock
corpus, where the depression label is random noise.

The previous harness (``run_capability_federated.py``) also predates ADR-0015: it
selected the best global model and reported it on the *same* loader, ran one
seed, chose no decision threshold, and quoted no interval. This script replaces
it for anything Chapter 4 cites:

* splits from :func:`~privchain.training.protocol.build_splits`; clients
  partition **train only**;
* per-round model selection on **selection**, never on the reported split;
* the final number read once on the untouched dev split, at a threshold chosen on
  selection, with a bootstrap CI and a **paired** difference against FedAvG
  (ADR-0020) — each arm's own interval is ±0.19 wide on 34 sessions, so only the
  paired comparison has any power;
* all arms start each seed from the **same initial global state**, so they differ
  in aggregation and nothing else.

Arms: ``centralized`` (non-federated reference), ``fedavg`` (Phase 2),
``capability``, ``capability+reputation``, and the full proposed protocol. The
stepwise structure is what makes it an ablation rather than a single comparison.

Also reports the per-modality-pattern breakdown: a protocol can help the
audio-only clients without moving the pooled number, and on a 34-session split
that is the more likely way for it to show up.

Usage:
    python scripts/run_federated_comparison.py --daic-config configs/daic_woz.yaml
    python scripts/run_federated_comparison.py --daic-config configs/daic_woz.yaml \
        --partition dirichlet
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from privchain.config import (
    AggregationConfig,
    load_baseline_config,
    load_federated_config,
    resolve_device,
)
from privchain.eval.metrics import paired_bootstrap_auc_difference
from privchain.federated.partition import ModalityMaskedDataset, build_client_partitions
from privchain.federated.simulation import (
    build_federated_clients,
    run_capability_aware_simulation,
    run_simulation,
)
from privchain.fusion.base import DepressionModelBase
from privchain.fusion.factory import build_depression_model
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    build_objective,
    collect_scores,
    evaluate_model,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import (
    RunResult,
    build_splits,
    format_aggregate,
    labels_of,
    make_loader,
    pooled_scores,
    repeat_over_seeds,
    uncertainty_report,
)
from privchain.training.trainer import CentralizedTrainer

REPORTED = ("f1", "roc_auc", "accuracy")
# Arm name -> (reputation_weighting, federated_distillation). `fedavg` is handled
# separately because it uses the Phase 2 aggregation path, not this one.
CAPABILITY_ARMS = {
    "capability": (False, False),
    "capability+reputation": (True, False),
    "capability+reputation+distillation": (True, True),
}


def main() -> None:
    """Run every federated arm under the shared protocol and compare them."""
    parser = argparse.ArgumentParser(description="Federated arms on real data (Phases 2 & 4).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--partition", choices=("iid", "dirichlet"), default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override train.batch_size. A shard smaller than this collapses to "
        "one batch, so it also sets how many optimizer steps a round can carry.",
    )
    parser.add_argument("--local-epochs", type=int, default=None, help="Override local_epochs.")
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    seeds = args.seeds if args.seeds else base.train.seeds
    seed_everything(base.seed)

    federation = fed.federation
    if args.partition is not None:
        federation = federation.model_copy(
            update={"partition": federation.partition.model_copy(update={"mode": args.partition})}
        )
    if args.num_clients is not None:
        federation = federation.model_copy(
            update={
                "num_clients": args.num_clients,
                "clients_per_round": min(federation.clients_per_round, args.num_clients),
            }
        )
    if args.rounds is not None:
        federation = federation.model_copy(update={"num_rounds": args.rounds})
    if args.local_epochs is not None:
        federation = federation.model_copy(update={"local_epochs": args.local_epochs})
    if args.batch_size is not None:
        base = base.model_copy(
            update={"train": base.train.model_copy(update={"batch_size": args.batch_size})}
        )

    splits, input_dims = build_splits(base, args.daic_config)
    device = resolve_device(base.train.device)
    torch_device = torch.device(device)
    train_cfg = base.train

    selection_loader = make_loader(splits.selection, batch_size=train_cfg.batch_size, shuffle=False)
    report_loader = make_loader(splits.report, batch_size=train_cfg.batch_size, shuffle=False)
    pos_weight = (
        positive_class_weight(
            make_loader(splits.train, batch_size=train_cfg.batch_size, shuffle=False)
        )
        if train_cfg.class_weighting
        else None
    )
    objective = build_objective(base.model, base.data.phq8_max, pos_weight).to(
        torch_device
    )

    train_labels = labels_of(splits.train)
    partitions = build_client_partitions(
        len(splits.train),  # type: ignore[arg-type]
        federation,
        base.seed,
        labels=train_labels,
    )

    # A protocol that helps only the modality-poor clients would not move the
    # pooled number; these masked views of the report split are where that shows.
    pattern_report_loaders = {
        pattern.name: make_loader(
            ModalityMaskedDataset(
                splits.report,
                list(range(len(splits.report))),  # type: ignore[arg-type]
                (pattern.capability[0], pattern.capability[1], pattern.capability[2]),
            ),
            batch_size=train_cfg.batch_size,
            shuffle=False,
        )
        for pattern in federation.modality_patterns
    }

    run_dir = create_run_dir(train_cfg.output_dir, "phase4", "phase4_federated_comparison")
    save_config(run_dir, {"baseline": base.model_dump(), "federated": fed.model_dump()})

    _print_population(partitions, train_labels, federation.partition.mode)
    print(
        f"\ndevice={device}  splits: train={len(splits.train)} "  # type: ignore[arg-type]
        f"selection={len(splits.selection)} report={len(splits.report)}  "  # type: ignore[arg-type]
        f"rounds<={federation.num_rounds} patience={federation.early_stopping_patience}  "
        f"seeds={list(seeds)}\n"
    )

    def finalize(model: DepressionModelBase) -> RunResult:
        """Read the untouched split once, at a threshold chosen on selection."""
        metrics = evaluate_with_selected_threshold(
            model, selection_loader, report_loader, objective, torch_device
        )
        per_pattern = {
            f"pattern_{name}_roc_auc": evaluate_model(
                model, loader, objective, torch_device, threshold=metrics["threshold"]
            )["roc_auc"]
            for name, loader in pattern_report_loaders.items()
        }
        scores, labels = collect_scores(model, report_loader, torch_device)
        return RunResult(
            metrics={**metrics, **per_pattern},
            threshold=metrics["threshold"],
            scores=scores,
            labels=labels,
        )

    def initial_state(seed: int) -> OrderedDict[str, torch.Tensor]:
        """The shared starting point every arm of this seed departs from."""
        seed_everything(seed)
        model = build_depression_model(input_dims, base.model, splits.quality_dims)
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())

    def run_centralized(seed: int) -> RunResult:
        """The non-federated reference: one model over the pooled training split."""
        seed_everything(seed)
        model = build_depression_model(input_dims, base.model, splits.quality_dims)
        arm_dir = run_dir / "centralized" / f"seed_{seed}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        CentralizedTrainer(
            model,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            phq8_max=base.data.phq8_max,
            phq_loss_weight=base.model.phq_loss_weight,
            device=device,
            pos_weight=pos_weight,
        ).fit(
            make_loader(splits.train, batch_size=train_cfg.batch_size, shuffle=True, seed=seed),
            selection_loader,
            epochs=train_cfg.epochs,
            run_dir=arm_dir,
            selection_metric=train_cfg.selection_metric,
            early_stopping_patience=train_cfg.early_stopping_patience,
        )
        model.load_state_dict(torch.load(arm_dir / "best_model.pt", map_location=device))
        return finalize(model.to(torch_device))

    def run_federated(arm: str, seed: int) -> RunResult:
        """Run one federated arm from the seed's shared initial state."""
        seed_everything(seed)
        arm_dir = run_dir / arm.replace("+", "_") / f"seed_{seed}"
        arm_dir.mkdir(parents=True, exist_ok=True)

        model = build_depression_model(input_dims, base.model, splits.quality_dims)
        model.load_state_dict(copy.deepcopy(initial_state(seed)))
        clients = build_federated_clients(
            splits.train,
            partitions,
            input_dims=input_dims,
            model_config=base.model,
            batch_size=train_cfg.batch_size,
            local_epochs=federation.local_epochs,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            phq8_max=base.data.phq8_max,
            phq_loss_weight=base.model.phq_loss_weight,
            seed=seed,
            device=device,
            class_weight_mode="per_shard" if train_cfg.class_weighting else "off",
        )

        common: dict[str, Any] = {
            "num_rounds": federation.num_rounds,
            "clients_per_round": federation.clients_per_round,
            "phq8_max": base.data.phq8_max,
            "phq_loss_weight": base.model.phq_loss_weight,
            "run_dir": arm_dir,
            "seed": seed,
            "device": device,
            "early_stopping_patience": federation.early_stopping_patience,
        }
        if arm == "fedavg":
            run_simulation(model, clients, selection_loader, **common)
        else:
            reputation, distillation = CAPABILITY_ARMS[arm]
            aggregation: AggregationConfig = fed.aggregation.model_copy(
                update={
                    "strategy": "capability_aware",
                    "reputation_weighting": reputation,
                    "federated_distillation": distillation,
                }
            )
            run_capability_aware_simulation(
                model, clients, selection_loader, aggregation=aggregation, **common
            )

        model.load_state_dict(torch.load(arm_dir / "best_global_model.pt", map_location=device))
        return finalize(model.to(torch_device))

    arms = ["centralized", "fedavg", *CAPABILITY_ARMS]
    results: dict[str, dict[str, float]] = {}
    runs: dict[str, list[RunResult]] = {}
    for arm in arms:
        runner = run_centralized if arm == "centralized" else (lambda s, a=arm: run_federated(a, s))
        aggregate, arm_runs = repeat_over_seeds(runner, seeds)  # type: ignore[arg-type]
        aggregate.update(uncertainty_report(arm_runs))
        results[arm], runs[arm] = aggregate, arm_runs
        print(
            f"{arm:36s} {format_aggregate(aggregate, REPORTED)}  "
            f"auc95%CI=[{aggregate['roc_auc_ci_low']:.3f}, {aggregate['roc_auc_ci_high']:.3f}]"
        )

    # H2's headline is `capability - fedavg`; every arm is compared against the
    # Phase 2 baseline it is supposed to improve on.
    labels = pooled_scores(runs["fedavg"])[1]
    fedavg_scores = pooled_scores(runs["fedavg"])[0]
    paired = {
        f"{arm}_minus_fedavg": paired_bootstrap_auc_difference(
            labels, pooled_scores(runs[arm])[0], fedavg_scores, seed=base.seed
        )
        for arm in arms
        if arm != "fedavg"
    }
    print("\npaired bootstrap against `fedavg` (the only comparison with power here):")
    for comparison, stats in paired.items():
        verdict = "SEPARATES" if stats["significant"] else "no measured difference"
        print(
            f"  {comparison:44s} d={stats['difference']:+.3f}  "
            f"95%CI=[{stats['low']:+.3f}, {stats['high']:+.3f}]  "
            f"p={stats['p_two_sided']:.3f}  -> {verdict}"
        )

    _print_patterns(results, federation.modality_patterns)

    payload = {
        "partition": federation.partition.model_dump(),
        "num_clients": federation.num_clients,
        "num_rounds": federation.num_rounds,
        "early_stopping_patience": federation.early_stopping_patience,
        "seeds": list(seeds),
        "n_train": len(splits.train),  # type: ignore[arg-type]
        "n_report": len(splits.report),  # type: ignore[arg-type]
        "clients": [
            {
                "client_id": p.client_id,
                "pattern": p.pattern_name,
                "capability": list(p.capability),
                "num_samples": len(p.indices),
                "positive_rate": (
                    sum(train_labels[i] for i in p.indices) / len(p.indices) if p.indices else 0.0
                ),
            }
            for p in partitions
        ],
        "arms": results,
        "paired_auc_differences": paired,
    }
    (run_dir / "federated_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {run_dir / 'federated_comparison.json'}")


def _print_population(partitions: list[Any], labels: list[int], mode: str) -> None:
    """Show the client population, so the partition is visible not assumed."""
    print(f"partition mode: {mode}")
    print(f"{'client':>6} {'pattern':14s} {'n':>4} {'pos_rate':>9}")
    for p in partitions:
        rate = sum(labels[i] for i in p.indices) / len(p.indices) if p.indices else 0.0
        print(f"{p.client_id:>6} {p.pattern_name:14s} {len(p.indices):>4} {rate:>9.2f}")
    corpus_rate = sum(labels) / len(labels)
    print(f"{'corpus':>6} {'':14s} {len(labels):>4} {corpus_rate:>9.2f}")


def _print_patterns(results: dict[str, dict[str, float]], patterns: list[Any]) -> None:
    """Per-modality-pattern ROC-AUC: where zero-imputation damage would show."""
    names = [p.name for p in patterns]
    header = f"\n{'arm':36s}" + "".join(f"{n:>14s}" for n in names)
    print(header)
    print("-" * len(header.strip()))
    for arm, aggregate in results.items():
        cells = "".join(
            f"{aggregate.get(f'pattern_{n}_roc_auc_mean', float('nan')):>14.3f}" for n in names
        )
        print(f"{arm:36s}{cells}")


if __name__ == "__main__":
    main()
