"""CLI: does a sequential text representation beat the document baseline? (Phase 1).

Text is the strongest DAIC-WOZ modality and had the weakest representation: the
whole interview collapsed to one ``(1, 768)`` row, so the encoder's pooling ran
over a length-1 sequence and was an identity. This measures, under one locked
protocol, whether keeping the interview's shape actually buys accuracy.

**Arms**, each a step that isolates one thing:

1. ``document+mean`` — the committed baseline.
2. ``segments+mean`` — the effect of merely keeping several stretches of text.
3. ``segments+attn`` — the effect of *selecting* among them.
4. ``segments+attn+pos`` — selection that also knows where in the session a
   segment sat.

**Protocol**, matching ``scripts/run_federated_sweep.py`` so the numbers are
comparable: ranking is on inner k-fold CV over the official *train* split only.
``build_splits`` makes the official dev split the report split, so tuning on it
would turn dev into a validation set. Dev and test are not read here.

Arms are paired: every arm sees identical folds and seeds, and the comparison is
a paired difference per fold, which removes fold difficulty from the contrast. An
arm is only worth keeping if it wins on most folds, not on the mean alone.

Usage:
    python scripts/run_text_representation.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import load_baseline_config, resolve_device
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.eval.benchmark import stratified_k_fold_indices
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.fusion.factory import require_baseline_architecture
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    build_objective,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import build_splits, carve_selection_split, labels_of
from privchain.training.trainer import CentralizedTrainer

#: Arm name -> (daic text overrides, text encoder overrides).
ARMS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "document+mean": ({"representation": "document"}, {"type": "mean"}),
    "segments+mean": ({"representation": "segments"}, {"type": "mean"}),
    "segments+attn": ({"representation": "segments"}, {"type": "attn"}),
    "segments+attn+pos": (
        {"representation": "segments"},
        {"type": "attn", "positional": True},
    ),
}


def _loader(dataset: Dataset[Sample], batch_size: int, *, shuffle: bool = False) -> DataLoader:
    """Build a padded DataLoader over ``dataset``."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _run_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    base: Any,
    input_dims: dict[str, int],
    device: torch.device,
    seed: int,
    text_only: bool,
    encoder_overrides: dict[str, Any],
) -> dict[str, float]:
    """Train one arm on one inner fold and score its held-out part.

    Args:
        pool: The official train split as one dataset.
        train_idx: Fold training indices.
        val_idx: Fold scored indices.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        input_dims: Per-modality input dims.
        device: Torch device.
        seed: Seed for this repetition.
        text_only: Zero the audio/video branches, isolating the text change.
        encoder_overrides: Text-encoder settings for this arm.

    Returns:
        Metrics on the fold's scored part, plus wall-clock.
    """
    seed_everything(seed)
    batch_size = base.train.batch_size
    fold_train, selection = carve_selection_split(
        Subset(pool, train_idx),
        [pool_labels[i] for i in train_idx],
        selection_fraction=base.train.selection_fraction,
        seed=seed,
    )
    overrides = dict(base.model.encoder_overrides or {})
    overrides["text"] = {**overrides.get("text", {}), **encoder_overrides}
    model_config = base.model.model_copy(update={"encoder_overrides": overrides})

    pos_weight = (
        positive_class_weight(_loader(fold_train, batch_size))
        if base.train.class_weighting
        else None
    )
    model = MultimodalDepressionModel(input_dims, model_config)
    trainer = CentralizedTrainer(
        model,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=model_config.phq_loss_weight,
        device=str(device),
        pos_weight=pos_weight,
    )
    started = time.monotonic()
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

    objective = build_objective(model_config, base.data.phq8_max).to(device)
    metrics = evaluate_with_selected_threshold(
        model.to(device),
        selection_loader,
        _loader(Subset(pool, val_idx), batch_size),
        objective,
        device,
    )
    metrics["seconds"] = time.monotonic() - started
    metrics["text_only"] = float(text_only)
    return metrics


def _paired_report(rows: list[dict[str, Any]], baseline: str) -> str:
    """Render per-arm means and paired per-fold differences against ``baseline``.

    A mean can move because one fold swung. The paired columns say on how many
    folds an arm actually won, which is the check that separates a real effect
    from a lucky fold.

    Args:
        rows: One row per arm x seed x fold.
        baseline: Arm every other arm is differenced against.

    Returns:
        Markdown.
    """
    by_arm: dict[str, dict[tuple[int, int], float]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[(row["seed"], row["fold"])] = row["roc_auc"]

    lines = [
        "# Text representation — inner-CV comparison",
        "",
        "Ranked on inner-fold mean ROC-AUC over the official train split; dev and",
        "test are untouched. `wins` counts folds where the arm beat the baseline,",
        "so an effect carried by a single fold is visible as such.",
        "",
        f"Baseline arm: `{baseline}`",
        "",
        "| arm | ROC-AUC | ±sd | F1 | paired Δ | ±sd | wins | seconds |",
        "|---|---|:--:|---|---|:--:|---|---|",
    ]
    base_scores = by_arm.get(baseline, {})
    f1_by_arm: dict[str, list[float]] = {}
    secs_by_arm: dict[str, list[float]] = {}
    for row in rows:
        f1_by_arm.setdefault(row["arm"], []).append(row["f1"])
        secs_by_arm.setdefault(row["arm"], []).append(row["seconds"])

    ordered = sorted(by_arm, key=lambda a: statistics.fmean(by_arm[a].values()), reverse=True)
    for arm in ordered:
        scores = by_arm[arm]
        aucs = list(scores.values())
        shared = sorted(set(scores) & set(base_scores))
        deltas = [scores[k] - base_scores[k] for k in shared]
        delta_mean = statistics.fmean(deltas) if deltas else 0.0
        delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        wins = sum(1 for d in deltas if d > 0)
        lines.append(
            f"| {arm} | {statistics.fmean(aucs):.3f} "
            f"| {statistics.stdev(aucs) if len(aucs) > 1 else 0.0:.3f} "
            f"| {statistics.fmean(f1_by_arm[arm]):.3f} "
            f"| {delta_mean:+.3f} | {delta_sd:.3f} | {wins}/{len(deltas)} "
            f"| {statistics.fmean(secs_by_arm[arm]):.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the text-representation comparison and write its report."""
    parser = argparse.ArgumentParser(description="Text representation comparison (Phase 1).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument(
        "--all-modalities",
        action="store_true",
        help="Keep audio/video on; the default isolates the text change.",
    )
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    # The ladder sweeps this model's text encoder; a segment_gated config here
    # would report the wrong architecture under the arm names below.
    require_baseline_architecture(base.model, "the text-representation ladder")
    seeds = args.seeds if args.seeds else list(base.train.seeds)
    seed_everything(base.seed)
    device = torch.device(resolve_device(base.train.device))
    text_only = not args.all_modalities

    run_dir = create_run_dir(base.train.output_dir, "phase1", "phase1_text_representation")
    results_path = run_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as sink:
        for arm, (text_overrides, encoder_overrides) in ARMS.items():
            overrides = {"text": {**text_overrides, "num_segments": args.num_segments}}
            if text_only:
                # Zeroing the other branches keeps the fusion width fixed while
                # making sure a text-side win is not audio noise moving.
                overrides["audio"] = {"max_frames": 1, "frame_stride": 1}
                overrides["video"] = {"max_frames": 1, "frame_stride": 1}
            splits, input_dims = build_splits(base, args.daic_config, daic_overrides=overrides)
            pool: Dataset[Sample] = ConcatDataset([splits.train, splits.selection])
            pool_labels = labels_of(splits.train) + labels_of(splits.selection)
            folds = stratified_k_fold_indices(pool_labels, args.inner_folds, base.seed)

            for seed in seeds:
                for fold_i, (train_idx, val_idx) in enumerate(folds):
                    metrics = _run_fold(
                        pool,
                        train_idx,
                        val_idx,
                        pool_labels,
                        base=base,
                        input_dims=input_dims,
                        device=device,
                        seed=seed + fold_i,
                        text_only=text_only,
                        encoder_overrides=encoder_overrides,
                    )
                    row = {"arm": arm, "seed": seed, "fold": fold_i, **metrics}
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                print(f"  done {arm} seed={seed}", flush=True)

    (run_dir / "summary.md").write_text(_paired_report(rows, "document+mean"), encoding="utf-8")
    save_config(
        run_dir,
        {"baseline": base.model_dump()},
        manifest_extra={
            "dataset": "daic_woz" if args.daic_config else "mock",
            "protocol": "nested-inner-cv-train-split-only",
            # `build_splits` constructs the dev dataset (and parses one dev
            # session for feature dims); it is never trained on, selected on, or
            # scored here.
            "official_dev_scored": False,
            "official_dev_selected_on": False,
            "official_dev_constructed": True,
            "official_test_read": False,
            "pool_size": len(pool_labels),
            "inner_folds": args.inner_folds,
            "seeds": list(seeds),
            "num_segments": args.num_segments,
            "text_only": text_only,
            "arms": list(ARMS),
        },
    )
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
