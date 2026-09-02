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

from privchain.config import CAPABILITY_MODALITIES, load_baseline_config, resolve_device
from privchain.eval.metrics import paired_bootstrap_auc_difference
from privchain.eval.modality_masking import MaskedModalityModel
from privchain.fusion.factory import build_depression_model
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    build_objective,
    collect_scores,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import (
    RunResult,
    build_splits,
    format_aggregate,
    make_loader,
    pooled_scores,
    repeat_over_seeds,
    uncertainty_report,
)
from privchain.training.trainer import CentralizedTrainer


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
    parser.add_argument(
        "--normalization",
        choices=("session", "corpus", "none"),
        default=None,
        help=(
            "Override the audio/video feature normalization (ADR-0019). `session` "
            "z-scores within each session, deleting the between-subject differences "
            "this ablation is trying to measure."
        ),
    )
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    seed_everything(config.seed)
    train_cfg = config.train

    overrides = (
        {m: {"normalization": args.normalization} for m in ("audio", "video")}
        if args.normalization
        else None
    )
    splits, input_dims = build_splits(config, args.daic_config, daic_overrides=overrides)
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
    objective = build_objective(config.model, config.data.phq8_max, pos_weight).to(device)

    run_dir = create_run_dir(train_cfg.output_dir, "phase1", "phase1_modality_ablation")
    save_config(run_dir, {"baseline": config.model_dump(), "daic_overrides": overrides})
    print(f"normalization override: {args.normalization or '(config default)'}")

    def train_arm(present: frozenset[str], seed: int) -> RunResult:
        """Train one ablation arm at one seed."""
        seed_everything(seed)
        train_loader = make_loader(
            splits.train,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            seed=seed,
            num_workers=train_cfg.num_workers,
        )
        model = MaskedModalityModel(
            build_depression_model(input_dims, config.model, splits.quality_dims), present
        )
        trainer = CentralizedTrainer(
            model,  # type: ignore[arg-type]
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            phq8_max=config.data.phq8_max,
            phq_loss_weight=config.model.phq_loss_weight,
            device=str(device),
            pos_weight=pos_weight,
            objective=build_objective(config.model, config.data.phq8_max, pos_weight),
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
        metrics = evaluate_with_selected_threshold(
            model, selection_loader, report_loader, objective, device
        )
        scores, labels = collect_scores(model, report_loader, device)
        return RunResult(metrics=metrics, epochs_run=len(history), scores=scores, labels=labels)

    # Every non-empty subset: singles show standalone value, pairs and the full
    # set show what each modality *adds* on top of the others.
    subsets = [
        frozenset(combo)
        for size in (1, 2, 3)
        for combo in combinations(CAPABILITY_MODALITIES, size)
    ]

    rows: list[dict[str, Any]] = []
    runs: dict[str, list[RunResult]] = {}
    print(
        f"splits: train={len(splits.train)} selection={len(splits.selection)} "  # type: ignore[arg-type]
        f"report={len(splits.report)}  seeds={list(train_cfg.seeds)}"
    )  # type: ignore[arg-type]
    for present in subsets:
        aggregate, arm_runs = repeat_over_seeds(
            lambda seed, subset=present: train_arm(subset, seed),  # type: ignore[misc]
            train_cfg.seeds,
        )
        aggregate.update(uncertainty_report(arm_runs))
        name = "+".join(m for m in CAPABILITY_MODALITIES if m in present)
        rows.append({"modalities": sorted(present), "name": name, **aggregate})
        runs[name] = arm_runs
        print(
            f"  {name:18s} {format_aggregate(aggregate, ('f1', 'roc_auc', 'accuracy'))}"
            f"  auc95%CI=[{aggregate['roc_auc_ci_low']:.3f}, {aggregate['roc_auc_ci_high']:.3f}]"
        )

    # The claim this ablation is really asked to support is "modality X adds
    # something on top of the others", which is a *difference* between two arms
    # scored on the same 34 sessions. Comparing their individual intervals would
    # throw away the pairing and answer a much weaker question (ADR-0020).
    full = "+".join(CAPABILITY_MODALITIES)
    labels = pooled_scores(runs[full])[1]
    paired = {
        f"{full}_minus_{name}": paired_bootstrap_auc_difference(
            labels, pooled_scores(runs[full])[0], pooled_scores(runs[name])[0], seed=config.seed
        )
        for name in runs
        if name != full
    }
    print(f"\npaired bootstrap: what each reduced arm gives up against `{full}`")
    for comparison, stats in sorted(paired.items(), key=lambda kv: -kv[1]["difference"]):
        verdict = "SEPARATES" if stats["significant"] else "no measured difference"
        print(
            f"  {comparison:34s} d={stats['difference']:+.3f}  "
            f"95%CI=[{stats['low']:+.3f}, {stats['high']:+.3f}]  "
            f"p={stats['p_two_sided']:.3f}  -> {verdict}"
        )

    (run_dir / "ablation.json").write_text(
        json.dumps({"arms": rows, "paired_auc_differences": paired}, indent=2), encoding="utf-8"
    )
    print(f"\nRun dir: {run_dir}")


if __name__ == "__main__":
    main()
