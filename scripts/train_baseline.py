"""CLI: train the centralized multimodal baseline (Phase 1).

Loads ``configs/baseline.yaml``, seeds everything, builds train/val loaders over
the mock dataset (or real DAIC-WOZ — see ``--daic-config``), trains the model,
and writes config + metrics + checkpoint to ``experiments/phase1/<run-id>/``.

Usage:
    python scripts/train_baseline.py [--config configs/baseline.yaml]
    python scripts/train_baseline.py --daic-config configs/daic_woz.yaml   # real data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privchain.config import load_baseline_config, load_yaml, modality_input_dims, resolve_device
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.loaders import build_train_val_loaders
from privchain.training.objective import positive_class_weight
from privchain.training.trainer import CentralizedTrainer


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
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    seed_everything(config.seed)

    dataset = None
    val_dataset = None
    # Mock dims come from the data config; real dims are inferred from the
    # actual DAIC-WOZ features so the model is built to match.
    input_dims = modality_input_dims(config.data)
    if args.daic_config is not None:
        # Imported lazily so the mock path has no real-data dependencies.
        from privchain.data.daic_woz import build_daic_woz_dataset

        daic_cfg = load_yaml(args.daic_config)
        dataset = build_daic_woz_dataset(daic_cfg, split="train")
        # Real runs validate on the official dev split rather than carving a
        # random slice out of train — comparable to published DAIC-WOZ results
        # and a larger, fixed validation set (ADR-0011).
        val_dataset = build_daic_woz_dataset(daic_cfg, split="dev")
        input_dims = dataset.feature_dims

    train_loader, val_loader = build_train_val_loaders(
        config.data, config.train, seed=config.seed, dataset=dataset, val_dataset=val_dataset
    )

    pos_weight = positive_class_weight(train_loader) if config.train.class_weighting else None
    device = resolve_device(config.train.device)

    model = MultimodalDepressionModel(input_dims, config.model)
    trainer = CentralizedTrainer(
        model,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        phq8_max=config.data.phq8_max,
        phq_loss_weight=config.model.phq_loss_weight,
        device=device,
        pos_weight=pos_weight,
    )

    run_dir = create_run_dir(config.train.output_dir, "phase1", config.train.run_name)
    save_config(run_dir, config.model_dump())

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config.train.epochs,
        run_dir=run_dir,
        selection_metric=config.train.selection_metric,
        early_stopping_patience=config.train.early_stopping_patience,
    )

    best = max(history, key=lambda r: r[f"val_{config.train.selection_metric}"])
    final = history[-1]
    print(f"Run dir: {run_dir}")
    print(
        f"Device={device}  pos_weight="
        f"{'none' if pos_weight is None else format(pos_weight, '.3f')}  "
        f"epochs_run={len(history)}/{config.train.epochs}"
    )
    print(
        f"Best (epoch {best['epoch']}, by {config.train.selection_metric}) — "
        f"F1={best['val_f1']:.4f}  ROC-AUC={best['val_roc_auc']:.4f}  "
        f"acc={best['val_accuracy']:.4f}"
    )
    print(
        "Final epoch — "
        f"F1={final['val_f1']:.4f}  ROC-AUC={final['val_roc_auc']:.4f}  "
        f"acc={final['val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
