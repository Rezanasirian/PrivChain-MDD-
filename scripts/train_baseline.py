"""CLI: train the centralized multimodal baseline (Phase 1).

Loads ``configs/baseline.yaml``, seeds everything, builds loaders over the mock
dataset (or real DAIC-WOZ — see ``--daic-config``), trains the model, and writes
config + metrics + checkpoint to ``experiments/phase1/<run-id>/``.

On real data this follows the shared evaluation protocol
(:mod:`privchain.training.protocol`, ADR-0015): a selection split carved out of
train chooses the epoch and the decision threshold, the official dev split is
reported on and never selected against, and the whole thing repeats over several
seeds so the reported figure carries a spread. Every arm compared in Chapter 4
uses this same protocol.

Usage:
    python scripts/train_baseline.py [--config configs/baseline.yaml]
    python scripts/train_baseline.py --daic-config configs/daic_woz.yaml   # real data
    python scripts/train_baseline.py --daic-config configs/daic_woz.yaml --seeds 42 7 2024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from privchain.config import load_baseline_config, resolve_device
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    DepressionObjective,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import (
    RunResult,
    build_splits,
    format_aggregate,
    make_loader,
    repeat_over_seeds,
)
from privchain.training.trainer import CentralizedTrainer

REPORTED = ("f1", "roc_auc", "accuracy", "precision", "recall")


def main() -> None:
    """Parse args, build everything from config, train, and report metrics."""
    parser = argparse.ArgumentParser(description="Train the centralized multimodal baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Baseline config (model + training + mock-data dims).",
    )
    parser.add_argument(
        "--daic-config",
        type=Path,
        default=None,
        help="Optional real DAIC-WOZ config; when set, trains on real data.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override train.seeds from the config.",
    )
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    seeds = args.seeds if args.seeds else config.train.seeds
    seed_everything(config.seed)

    splits, input_dims = build_splits(config, args.daic_config)
    device = resolve_device(config.train.device)
    train_cfg = config.train

    selection_loader = make_loader(splits.selection, batch_size=train_cfg.batch_size, shuffle=False)
    report_loader = make_loader(splits.report, batch_size=train_cfg.batch_size, shuffle=False)
    train_order_loader = make_loader(splits.train, batch_size=train_cfg.batch_size, shuffle=False)
    pos_weight = positive_class_weight(train_order_loader) if train_cfg.class_weighting else None

    run_dir = create_run_dir(train_cfg.output_dir, "phase1", train_cfg.run_name)
    save_config(run_dir, config.model_dump())

    def run_once(seed: int) -> RunResult:
        """Train at one seed; select on `selection`, report on `report`."""
        seed_everything(seed)
        train_loader = make_loader(
            splits.train,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            seed=seed,
            num_workers=train_cfg.num_workers,
        )
        model = MultimodalDepressionModel(input_dims, config.model)
        trainer = CentralizedTrainer(
            model,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            phq8_max=config.data.phq8_max,
            phq_loss_weight=config.model.phq_loss_weight,
            device=device,
            pos_weight=pos_weight,
        )

        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        history = trainer.fit(
            train_loader,
            selection_loader,
            epochs=train_cfg.epochs,
            run_dir=seed_dir,
            selection_metric=train_cfg.selection_metric,
            early_stopping_patience=train_cfg.early_stopping_patience,
        )

        # Restore the epoch chosen on `selection`, then report on `report`.
        best = max(history, key=lambda record: record[f"val_{train_cfg.selection_metric}"])
        model.load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=device))
        metrics = evaluate_with_selected_threshold(
            model,
            selection_loader,
            report_loader,
            DepressionObjective(config.data.phq8_max, config.model.phq_loss_weight, pos_weight).to(
                torch.device(device)
            ),
            torch.device(device),
        )
        return RunResult(
            metrics=metrics,
            best_epoch=int(best["epoch"]),
            epochs_run=len(history),
            threshold=metrics["threshold"],
        )

    aggregate, results = repeat_over_seeds(run_once, seeds)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "seeds": list(seeds),
                "aggregate": aggregate,
                "per_seed": [
                    {"seed": s, "metrics": r.metrics, "best_epoch": r.best_epoch}
                    for s, r in zip(seeds, results, strict=True)
                ],
                "n_train": len(splits.train),  # type: ignore[arg-type]
                "n_selection": len(splits.selection),  # type: ignore[arg-type]
                "n_report": len(splits.report),  # type: ignore[arg-type]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Run dir: {run_dir}")
    print(
        f"Device={device}  pos_weight="
        f"{'none' if pos_weight is None else format(pos_weight, '.3f')}  "
        f"splits: train={len(splits.train)} selection={len(splits.selection)} "  # type: ignore[arg-type]
        f"report={len(splits.report)}"  # type: ignore[arg-type]
    )
    for seed, result in zip(seeds, results, strict=True):
        print(
            f"  seed {seed:>5}: F1={result.metrics['f1']:.4f}  "
            f"ROC-AUC={result.metrics['roc_auc']:.4f}  "
            f"acc={result.metrics['accuracy']:.4f}  "
            f"(epoch {result.best_epoch}/{result.epochs_run}, thr={result.threshold:.3f})"
        )
    print(
        f"Reported on the dev split over {len(seeds)} seeds — "
        f"{format_aggregate(aggregate, REPORTED)}"
    )


if __name__ == "__main__":
    main()
