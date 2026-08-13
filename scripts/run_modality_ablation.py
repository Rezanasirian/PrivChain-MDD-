"""CLI: per-modality ablation of the centralized baseline (Phase 1, H1/H5).

The thesis claim (H1) is that modalities deserve *different* privacy budgets,
which presumes they differ in what they contribute and in what they expose.
``configs/privacy.yaml`` asserts an ordering (audio > video > text by
re-identification risk) that has never been measured on this corpus. This script
measures the **utility** half: what each modality is worth on its own, and what
each adds to the others.

A modality is ablated by zeroing its input features and clearing its presence
flag, so the architecture, parameter count, and training schedule are identical
across arms — only the information differs. Ablating by rebuilding a smaller
model would confound the modality's contribution with a change in capacity.

Runs under the shared evaluation protocol (:mod:`privchain.training.protocol`,
ADR-0015): selection on a held-back slice of train, reporting on the untouched
dev split, aggregated over seeds.

Usage:
    python scripts/run_modality_ablation.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch import nn

from privchain.config import CAPABILITY_MODALITIES, load_baseline_config, resolve_device
from privchain.data.mock_daic_woz import Batch
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


class MaskedModalityModel(nn.Module):
    """Wraps the baseline model, hiding all modalities outside ``present``.

    Zeroing the features *and* the fusion presence flag is what "the modality is
    absent" means elsewhere in the project (Phase 2 clients), so an ablation arm
    here matches a capability-restricted client there.

    Args:
        model: The full multimodal model.
        present: Modalities the model is allowed to see.
    """

    def __init__(self, model: MultimodalDepressionModel, present: frozenset[str]) -> None:
        super().__init__()
        self.model = model
        self.present = present

    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Blank the hidden modalities, then delegate to the wrapped model.

        Args:
            batch: A collated batch.
            presence: Ignored; this wrapper derives presence from ``present``.

        Returns:
            The wrapped model's outputs.
        """
        masked = dict(batch)
        batch_size = batch["label"].shape[0]
        flags: dict[str, torch.Tensor] = {}
        for modality in CAPABILITY_MODALITIES:
            visible = modality in self.present
            if not visible:
                masked[modality] = torch.zeros_like(batch[modality])
            flags[modality] = torch.full(
                (batch_size,),
                float(visible),
                device=batch["label"].device,
            )
        return self.model(masked, flags)  # type: ignore[arg-type]


def main() -> None:
    """Train one arm per modality subset and report each on the dev split."""
    parser = argparse.ArgumentParser(description="Per-modality ablation (Phase 1).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--daic-config",
        type=Path,
        default=None,
        help="Optional real DAIC-WOZ config; when set, ablates on real data.",
    )
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    seed_everything(config.seed)
    train_cfg = config.train

    splits, input_dims = build_splits(config, args.daic_config)
    device = torch.device(resolve_device(train_cfg.device))
    selection_loader = make_loader(splits.selection, batch_size=train_cfg.batch_size, shuffle=False)
    report_loader = make_loader(splits.report, batch_size=train_cfg.batch_size, shuffle=False)
    pos_weight = (
        positive_class_weight(
            make_loader(splits.train, batch_size=train_cfg.batch_size, shuffle=False)
        )
        if train_cfg.class_weighting
        else None
    )
    objective = DepressionObjective(
        config.data.phq8_max, config.model.phq_loss_weight, pos_weight
    ).to(device)

    run_dir = create_run_dir(train_cfg.output_dir, "phase1", "phase1_modality_ablation")
    save_config(run_dir, config.model_dump())

    def train_arm(present: frozenset[str], seed: int) -> dict[str, float]:
        """Train one ablation arm at one seed."""
        seed_everything(seed)
        train_loader = make_loader(
            splits.train,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            seed=seed,
            num_workers=train_cfg.num_workers,
        )
        model = MaskedModalityModel(MultimodalDepressionModel(input_dims, config.model), present)
        trainer = CentralizedTrainer(
            model,  # type: ignore[arg-type]
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            phq8_max=config.data.phq8_max,
            phq_loss_weight=config.model.phq_loss_weight,
            device=str(device),
            pos_weight=pos_weight,
        )
        arm_dir = run_dir / "_".join(sorted(present)) / f"seed_{seed}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        history = trainer.fit(
            train_loader,
            selection_loader,
            epochs=train_cfg.epochs,
            run_dir=arm_dir,
            selection_metric=train_cfg.selection_metric,
            early_stopping_patience=train_cfg.early_stopping_patience,
        )
        if not history:
            raise RuntimeError(f"no epochs ran for arm {sorted(present)}")
        model.load_state_dict(torch.load(arm_dir / "best_model.pt", map_location=device))
        return evaluate_with_selected_threshold(
            model, selection_loader, report_loader, objective, device
        )

    # Every non-empty subset: singles show standalone value, pairs and the full
    # set show what each modality *adds* on top of the others.
    subsets = [
        frozenset(combo)
        for size in (1, 2, 3)
        for combo in combinations(CAPABILITY_MODALITIES, size)
    ]

    rows: list[dict[str, Any]] = []
    print(
        f"splits: train={len(splits.train)} selection={len(splits.selection)} "  # type: ignore[arg-type]
        f"report={len(splits.report)}  seeds={list(train_cfg.seeds)}"
    )  # type: ignore[arg-type]
    for present in subsets:
        aggregate, _ = repeat_over_seeds(
            lambda seed, subset=present: RunResult(metrics=train_arm(subset, seed)),  # type: ignore[misc]
            train_cfg.seeds,
        )
        name = "+".join(m for m in CAPABILITY_MODALITIES if m in present)
        rows.append({"modalities": sorted(present), "name": name, **aggregate})
        print(f"  {name:18s} {format_aggregate(aggregate, ('f1', 'roc_auc', 'accuracy'))}")

    (run_dir / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
