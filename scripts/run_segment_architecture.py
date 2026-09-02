"""CLI: does segment-aligned, quality-gated fusion beat the session-level model? (Phase 1).

The committed baseline pools every modality to one session vector before fusing,
so it can never prefer the voice in one stretch of the interview and the words in
another. ADR-0027 proposes cutting the session into aligned segments, fusing per
segment under a quality-aware gate, and pooling the segments by attention. This
script measures whether that is worth anything on real DAIC-WOZ — and, because
each arm changes exactly one thing, where any gain comes from.

**Arms** (``--arms`` selects a subset):

1. ``document+concat`` — the committed baseline.
2. ``document+gated`` — the existing gated fusion, session-level.
3. ``segments+attn`` — segment-level text with attention, audio/video genuinely
   ablated (features zeroed *and* presence cleared).
4. ``segments+av+gated`` — segment text pooled to a session embedding, fused with
   session-level audio/video through the existing gate. Isolates "better text +
   existing gate" from the architecture change.
5. ``aligned+quality_gated`` — the full segment-aligned network, corpus audio
   normalization.
6. ``aligned+huber`` — arm 5 with the Huber PHQ-8 head. A separate rung on
   purpose: folding the loss into arm 5 would leave the architecture's effect and
   the loss's effect inseparable.
7. ``aligned+dropout`` — arm 5 plus per-sample modality dropout.
5b. ``aligned+session_norm`` — arm 5 under session normalization, reported as a
   sensitivity analysis rather than as a selectable arm.

**Protocol**, matching ``scripts/run_text_representation.py`` so the numbers are
comparable: ranking is on inner k-fold CV over the official **train** split only
(107 participants on the real corpus). ``build_splits`` makes the official dev
split the report split, so tuning on it would turn dev into a validation set. Dev
and test are not read here.

Arms are paired: identical folds and seeds, and the comparison is a paired
bootstrap over **participants**, on out-of-fold scores averaged across seeds.
Concatenating the seeds instead would enter each participant five times and break
the resampling's independence assumption, narrowing every interval for free.

Usage:
    python scripts/run_segment_architecture.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import (
    CAPABILITY_MODALITIES,
    BaselineConfig,
    load_baseline_config,
    resolve_device,
)
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.eval.benchmark import stratified_k_fold_indices
from privchain.eval.metrics import paired_bootstrap_auc_difference
from privchain.eval.modality_masking import MaskedModalityModel
from privchain.fusion.factory import build_depression_model
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.modality_dropout import ModalityDropout
from privchain.training.objective import (
    build_objective,
    collect_scores,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import build_splits, carve_selection_split, labels_of
from privchain.training.trainer import CentralizedTrainer

BASELINE_ARM = "document+concat"


@dataclass(frozen=True)
class ArmSpec:
    """One rung of the ladder: exactly one thing different from the rung below.

    Attributes:
        daic: Overrides layered onto the real-data config sections.
        model: Fields overridden on the validated model config.
        encoders: Per-modality encoder overrides.
        present: Modalities the model may see; the rest are ablated for real
            (zeroed features *and* cleared presence), not merely shrunk.
        dropout: Whether to apply per-sample modality dropout during training.
        selectable: ``False`` marks a sensitivity arm that is reported but never
            competes for selection.
    """

    daic: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    encoders: dict[str, dict[str, Any]] = field(default_factory=dict)
    present: frozenset[str] = frozenset(CAPABILITY_MODALITIES)
    dropout: bool = False
    selectable: bool = True


def _arms(segments: int) -> dict[str, ArmSpec]:
    """Build the arm table at a given segment count.

    Args:
        segments: Number of segments the segment-level arms cut the session into.

    Returns:
        Ordered ``{arm name: spec}``.
    """
    document = {"text": {"representation": "document"}}
    text_segments = {"text": {"representation": "segments", "num_segments": segments}}
    aligned = {"segments": {"enabled": True, "count": segments}}
    return {
        BASELINE_ARM: ArmSpec(daic=document, model={"fusion_type": "concat"}),
        "document+gated": ArmSpec(daic=document, model={"fusion_type": "gated"}),
        "segments+attn": ArmSpec(
            daic=text_segments,
            model={"fusion_type": "concat"},
            encoders={"text": {"type": "attn", "positional": True}},
            present=frozenset({"text"}),
        ),
        "segments+av+gated": ArmSpec(
            daic=text_segments,
            model={"fusion_type": "gated"},
            encoders={"text": {"type": "attn", "positional": True}},
        ),
        "aligned+quality_gated": ArmSpec(
            daic={**aligned, "audio": {"normalization": "corpus"}},
            model={"architecture": "segment_gated", "fusion_type": "quality_gated"},
        ),
        "aligned+huber": ArmSpec(
            daic={**aligned, "audio": {"normalization": "corpus"}},
            model={
                "architecture": "segment_gated",
                "fusion_type": "quality_gated",
                "phq_loss": "huber",
            },
        ),
        "aligned+dropout": ArmSpec(
            daic={**aligned, "audio": {"normalization": "corpus"}},
            model={"architecture": "segment_gated", "fusion_type": "quality_gated"},
            dropout=True,
        ),
        "aligned+session_norm": ArmSpec(
            daic={**aligned, "audio": {"normalization": "session"}},
            model={"architecture": "segment_gated", "fusion_type": "quality_gated"},
            selectable=False,
        ),
    }


def _model_config(base: BaselineConfig, spec: ArmSpec) -> Any:
    """Apply an arm's model overrides to the validated baseline config."""
    updates = dict(spec.model)
    fusion_type = updates.pop("fusion_type", None)
    if fusion_type is not None:
        updates["fusion"] = base.model.fusion.model_copy(update={"type": fusion_type})
    if spec.encoders:
        overrides = {k: dict(v) for k, v in (base.model.encoder_overrides or {}).items()}
        for modality, values in spec.encoders.items():
            overrides[modality] = {**overrides.get(modality, {}), **values}
        updates["encoder_overrides"] = overrides
    return base.model.model_copy(update=updates)


def _loader(dataset: Dataset[Sample], batch_size: int, *, shuffle: bool = False) -> DataLoader:
    """Build a padded DataLoader over ``dataset``."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _dropout_for(base: BaselineConfig, spec: ArmSpec, seed: int) -> ModalityDropout | None:
    """Build the modality dropout an arm asks for.

    The *arm* decides whether dropout runs, not ``train.modality_dropout.enabled``
    — that flag governs ordinary training runs, and honouring it here would make
    the dropout arm silently identical to the arm below it whenever the committed
    config has dropout off (which it does by default).

    Args:
        base: Validated baseline config, for the pattern mix.
        spec: The arm being run.
        seed: Seed for the draw.

    Returns:
        The dropout, or ``None`` when this arm does not use it.

    Raises:
        SystemExit: If the arm needs dropout but no patterns are configured.
    """
    if not spec.dropout:
        return None
    if not base.train.modality_dropout.patterns:
        raise SystemExit(
            "the dropout arm needs train.modality_dropout.patterns in the baseline config"
        )
    forced = base.train.modality_dropout.model_copy(update={"enabled": True})
    return ModalityDropout(forced, seed)


def _run_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    base: BaselineConfig,
    spec: ArmSpec,
    input_dims: dict[str, int],
    quality_dims: dict[str, int] | None,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Train one arm on one inner fold and score its held-out part.

    Args:
        pool: The official train split as one dataset.
        train_idx: Fold training indices.
        val_idx: Fold scored indices.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        spec: This arm's specification.
        input_dims: Per-modality input dims.
        quality_dims: Per-modality quality widths, or ``None``.
        device: Torch device.
        seed: Seed for this repetition.

    Returns:
        ``(metrics, scores)`` — the fold's metrics and its per-sample scores, in
        ``val_idx`` order, so out-of-fold predictions can be reassembled.
    """
    seed_everything(seed)
    batch_size = base.train.batch_size
    fold_train, selection = carve_selection_split(
        Subset(pool, train_idx),
        [pool_labels[i] for i in train_idx],
        selection_fraction=base.train.selection_fraction,
        seed=seed,
    )
    model_config = _model_config(base, spec)
    pos_weight = (
        positive_class_weight(_loader(fold_train, batch_size))
        if base.train.class_weighting
        else None
    )
    model = build_depression_model(input_dims, model_config, quality_dims)
    if spec.present != frozenset(CAPABILITY_MODALITIES):
        model = MaskedModalityModel(model, spec.present)  # type: ignore[assignment]

    trainer = CentralizedTrainer(
        model,  # type: ignore[arg-type]
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=model_config.phq_loss_weight,
        device=str(device),
        pos_weight=pos_weight,
        objective=build_objective(model_config, base.data.phq8_max, pos_weight),
        modality_dropout=_dropout_for(base, spec, seed),
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

    objective = build_objective(model_config, base.data.phq8_max)
    report_loader = _loader(Subset(pool, val_idx), batch_size)
    metrics = evaluate_with_selected_threshold(
        model.to(device), selection_loader, report_loader, objective.to(device), device
    )
    metrics["seconds"] = time.monotonic() - started
    scores, _ = collect_scores(model, report_loader, device)
    return metrics, scores


def _paired_report(
    rows: list[dict[str, Any]],
    oof: dict[str, dict[int, np.ndarray]],
    labels: np.ndarray,
    seed: int,
) -> str:
    """Render per-arm means and the paired bootstrap against the baseline arm.

    A mean can move because one fold swung, and a bootstrap over rows that repeat
    each participant once per seed would understate the interval. Both are
    reported: the participant-level paired bootstrap as the test, the per-fold
    win count as a stability check.

    Args:
        rows: One row per arm x seed x fold.
        oof: ``{arm: {seed: out-of-fold scores}}``.
        labels: Binary label per pool participant.
        seed: Seed for the bootstrap resampling.

    Returns:
        Markdown.
    """
    by_arm: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_arm[row["arm"]][(row["seed"], row["fold"])] = row["roc_auc"]
        for key in ("f1", "pr_auc", "seconds"):
            metrics[row["arm"]][key].append(row[key])

    averaged = {
        arm: np.stack(list(per_seed.values())).mean(axis=0) for arm, per_seed in oof.items()
    }
    lines = [
        "# Segment-aligned architecture — inner-CV comparison",
        "",
        "Ranked on inner-fold mean ROC-AUC over the official train split; dev and",
        "test are untouched. `Δ` is a paired bootstrap on out-of-fold scores",
        "averaged across seeds, resampled **per participant** — the unit that is",
        "actually independent. `wins` counts folds beating the baseline arm.",
        "",
        f"Baseline arm: `{BASELINE_ARM}`",
        "",
        "| arm | ROC-AUC | ±sd | PR-AUC | F1 | OOF Δ | 95% CI | sig | wins | seconds |",
        "|---|---|:--:|---|---|---|---|:--:|---|---|",
    ]
    base_scores = by_arm.get(BASELINE_ARM, {})
    ordered = sorted(by_arm, key=lambda a: statistics.fmean(by_arm[a].values()), reverse=True)
    for arm in ordered:
        aucs = list(by_arm[arm].values())
        shared = sorted(set(by_arm[arm]) & set(base_scores))
        wins = sum(1 for k in shared if by_arm[arm][k] > base_scores[k])
        if arm in averaged and BASELINE_ARM in averaged and arm != BASELINE_ARM:
            test = paired_bootstrap_auc_difference(
                labels, averaged[arm], averaged[BASELINE_ARM], seed=seed
            )
            delta = f"{test['difference']:+.3f}"
            interval = f"[{test['low']:+.3f}, {test['high']:+.3f}]"
            significant = "yes" if test["significant"] else "no"
        else:
            delta, interval, significant = "—", "—", "—"
        lines.append(
            f"| {arm} | {statistics.fmean(aucs):.3f} "
            f"| {statistics.stdev(aucs) if len(aucs) > 1 else 0.0:.3f} "
            f"| {statistics.fmean(metrics[arm]['pr_auc']):.3f} "
            f"| {statistics.fmean(metrics[arm]['f1']):.3f} "
            f"| {delta} | {interval} | {significant} | {wins}/{len(shared)} "
            f"| {statistics.fmean(metrics[arm]['seconds']):.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the segment-architecture ladder and write its report."""
    parser = argparse.ArgumentParser(description="Segment-aligned architecture ladder (Phase 1).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument(
        "--arms", nargs="+", default=None, help="Subset of arm names to run (default: all)."
    )
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    seeds = args.seeds if args.seeds else list(base.train.seeds)
    seed_everything(base.seed)
    device = torch.device(resolve_device(base.train.device))

    arms = _arms(args.num_segments)
    unknown = sorted(set(args.arms or []) - set(arms))
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; available: {list(arms)}")
    selected = {name: arms[name] for name in (args.arms or list(arms))}
    if BASELINE_ARM not in selected:
        # Every reported difference is against this arm; running without it would
        # leave the deltas undefined rather than merely absent.
        selected = {BASELINE_ARM: arms[BASELINE_ARM], **selected}

    run_dir = create_run_dir(base.train.output_dir, "phase1", "phase1_segment_architecture")
    results_path = run_dir / "results.jsonl"
    oof_path = run_dir / "oof_predictions.jsonl"
    rows: list[dict[str, Any]] = []
    oof: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    pool_labels: list[int] = []

    with (
        results_path.open("w", encoding="utf-8") as sink,
        oof_path.open("w", encoding="utf-8") as oof_sink,
    ):
        for arm, spec in selected.items():
            splits, input_dims = build_splits(base, args.daic_config, daic_overrides=spec.daic)
            pool: Dataset[Sample] = ConcatDataset([splits.train, splits.selection])
            pool_labels = labels_of(splits.train) + labels_of(splits.selection)
            folds = stratified_k_fold_indices(pool_labels, args.inner_folds, base.seed)

            for seed in seeds:
                fold_scores = np.zeros(len(pool_labels), dtype=np.float64)
                for fold_i, (train_idx, val_idx) in enumerate(folds):
                    metrics, scores = _run_fold(
                        pool,
                        train_idx,
                        val_idx,
                        pool_labels,
                        base=base,
                        spec=spec,
                        input_dims=input_dims,
                        quality_dims=splits.quality_dims,
                        device=device,
                        seed=seed + fold_i,
                    )
                    fold_scores[val_idx] = scores
                    row = {"arm": arm, "seed": seed, "fold": fold_i, **metrics}
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                oof[arm][seed] = fold_scores
                # Positional indices only: the file carries no participant id, so
                # it cannot be linked back to a person.
                oof_sink.write(
                    json.dumps(
                        {
                            "arm": arm,
                            "seed": seed,
                            "scores": [round(float(s), 6) for s in fold_scores],
                            "labels": pool_labels,
                        }
                    )
                    + "\n"
                )
                oof_sink.flush()
                print(f"  done {arm} seed={seed}", flush=True)

    labels = np.asarray(pool_labels, dtype=np.int_)
    (run_dir / "summary.md").write_text(
        _paired_report(rows, oof, labels, base.seed), encoding="utf-8"
    )
    save_config(
        run_dir,
        {"baseline": base.model_dump()},
        manifest_extra={
            "dataset": "daic_woz" if args.daic_config else "mock",
            "protocol": "nested-inner-cv-train-split-only",
            # Precise rather than flattering: `build_splits` constructs the dev
            # dataset (and parses one dev session to infer feature dims), so
            # "untouched" would be false. What matters for the protocol is that
            # dev is never trained on, selected on, or scored here.
            "official_dev_scored": False,
            "official_dev_selected_on": False,
            "official_dev_constructed": True,
            "official_test_read": False,
            "pool_size": len(pool_labels),
            "inner_folds": args.inner_folds,
            "seeds": list(seeds),
            "num_segments": args.num_segments,
            "arms": list(selected),
            "selectable_arms": [name for name, spec in selected.items() if spec.selectable],
            "bootstrap_unit": "participant (seed-averaged out-of-fold scores)",
        },
    )
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
