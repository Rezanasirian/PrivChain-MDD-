"""CLI: which feature-extraction pipeline should the thesis measure on? (ADR-0031)

Every arm trains the identical model on identical folds and seeds. Only the
**feature extraction** changes: which frames reach the encoder, how a stride
window collapses, whether unvoiced glottal channels count as measurements, and
whether normalization is fitted within the session or over the train split.

Two defects motivate the ladder, both in the loader rather than the model:

1. The session-level view handed the acoustic branch *every* frame of the
   recording — the interviewer's questions and the silence between turns
   included — so its functionals described the interview's structure more than
   the participant's voice. ``speech_mask`` restricts them to the participant's
   own turns.
2. ``normalization: session`` forces each channel to mean 0 / std 1 within the
   session, and the ``stats`` encoder then computes the mean and standard
   deviation over exactly those rows: two of its five functionals are the same
   constants for every participant. ``corpus`` (fitted on train only) makes them
   informative again, at the cost of preserving the between-subject differences a
   re-identification attacker uses — so a win here obliges a re-run of
   ``scripts/run_reid_risk.py`` (ADR-0019).

**Protocol** — the same one ``scripts/run_text_representation.py`` uses, and
deliberately *not* the dev-reporting path this script used to shell out to:
ranking is on inner k-fold CV over the official **train** split only. Choosing a
preprocessing default by comparing arms on the official dev split would turn the
report split into a validation set (ADR-0015). Dev is constructed (one session is
parsed for feature dims) but never trained on, selected on, or scored.

Arms are paired twice over: every arm sees identical folds and seeds, and the
comparison is both a per-fold difference (how many folds the arm actually won)
and a bootstrap over the pooled out-of-fold predictions.

Usage:
    python scripts/run_preprocessing_ladder.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import BaselineConfig, load_baseline_config, resolve_device
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.eval.benchmark import stratified_k_fold_indices
from privchain.eval.metrics import paired_bootstrap_auc_difference
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

#: The participant-speech mask, spelled once so no arm can differ in its padding.
_SPEECH = {"enabled": True, "source": "transcript", "pad_seconds": 0.10}

#: Arm name -> per-section ``daic_woz`` overrides. ``committed`` is the pipeline
#: every result in the repo was produced under and the baseline every other arm
#: is differenced against. Each later arm adds exactly one thing to the one
#: before it, so a win can be attributed.
ARMS: dict[str, dict[str, dict[str, Any]]] = {
    "committed": {},
    "speech": {"audio": {"speech_mask": _SPEECH}},
    "speech+mean": {"audio": {"speech_mask": _SPEECH, "downsample": "mean"}},
    "speech+mean+valid": {
        "audio": {
            "speech_mask": _SPEECH,
            "downsample": "mean",
            "validity": {"enabled": True},
        }
    },
    "speech+functionals+valid": {
        "audio": {
            "speech_mask": _SPEECH,
            "downsample": "window_functionals",
            "validity": {"enabled": True},
        }
    },
    # Isolates defect 2: the frames are unchanged, only the statistics the
    # `stats` encoder can still see.
    "speech+corpus": {"audio": {"speech_mask": _SPEECH, "normalization": "corpus"}},
    "speech+mean+valid+corpus": {
        "audio": {
            "speech_mask": _SPEECH,
            "downsample": "mean",
            "validity": {"enabled": True},
            "normalization": "corpus",
        }
    },
    # Video validity needs an aggregating control because the committed
    # decimation cannot apply a per-window mask. Keep success-only separate from
    # the pre-declared confidence threshold so F2 is not bundled with two changes.
    "video+mean": {"video": {"downsample": "mean"}},
    "video+mean+success": {
        "video": {
            "downsample": "mean",
            "validity": {"enabled": True, "min_confidence": None},
        }
    },
    "video+mean+valid": {"video": {"downsample": "mean", "validity": {"enabled": True}}},
    "speech+mean+valid+video+mean+success": {
        "audio": {
            "speech_mask": _SPEECH,
            "downsample": "mean",
            "validity": {"enabled": True},
        },
        "video": {
            "downsample": "mean",
            "validity": {"enabled": True, "min_confidence": None},
        },
    },
    # Tests ADR-0027's reading that facial behaviour while *listening* is
    # informative, rather than assuming it.
    "speech+video_speech": {
        "audio": {"speech_mask": _SPEECH},
        "video": {"speech_mask": _SPEECH},
    },
    # Signals the corpus ships and the pipeline never read: OpenFace gaze and
    # head pose beside the action units, and the FORMANT export beside COVAREP.
    # The FORMANT source states its own column layout because it inherits the
    # audio section, whose validity indices describe COVAREP's 74 channels.
    "speech+video_sources": {
        "audio": {"speech_mask": _SPEECH},
        "video": {
            "sources": [
                {"file_template": "{pid}_CLNF_AUs.txt"},
                {"file_template": "{pid}_CLNF_gaze.txt"},
                {"file_template": "{pid}_CLNF_pose.txt"},
            ]
        },
    },
    "speech+formant": {
        "audio": {
            "speech_mask": _SPEECH,
            "sources": [
                {"file_template": "{pid}_COVAREP.csv"},
                {
                    "file_template": "{pid}_FORMANT.csv",
                    "quality_columns": {},
                    "validity": None,
                },
            ],
        }
    },
    # Standard 88-dimensional eGeMAPSv02 functionals extracted by
    # `scripts/extract_egemaps.py` from the concatenated participant turns. A
    # length-1 vector must use a mean encoder: applying the stats encoder would
    # duplicate the vector as mean/min/max and append two constant blocks.
    "speech+egemaps": {
        "audio": {
            "file_template": "{pid}_eGeMAPSv02.csv",
            "has_header": False,
            "delimiter": ",",
            "drop_columns": [],
            "max_frames": 1,
            "frame_stride": 1,
            "quality_columns": None,
            "voiced_column": "",
            "validity": None,
            "speech_mask": {"enabled": False},
            "downsample": "decimate",
            "normalization": "corpus",
        }
    },
    # A1/A2 ladder. All four arms use the same segment-aware architecture;
    # only the CTD block and acoustic filtering change.
    "segment_baseline": {"segments": {"enabled": True}},
    "segment_a2": {
        "segments": {"enabled": True},
        "audio": {"speech_filter": {"remove_unvoiced_rows": True, "min_turn_seconds": 1.0}},
    },
    "segment_ctd": {
        "segments": {"enabled": True},
        "audio": {"ctd": {"enabled": True}},
    },
    "segment_ctd_a2": {
        "segments": {"enabled": True},
        "audio": {
            "ctd": {"enabled": True},
            "speech_filter": {"remove_unvoiced_rows": True, "min_turn_seconds": 1.0},
        },
    },
}

SEGMENT_ARMS = frozenset({"segment_baseline", "segment_a2", "segment_ctd", "segment_ctd_a2"})

# Baseline audio encoder: stats(74 channels -> 370 functionals), hidden=128,
# out=128 = 64,000 parameters. eGeMAPS uses mean(88), hidden=294, out=128 =
# 63,926 parameters: a 0.12% difference, so the comparison is capacity-matched.
ARM_ENCODER_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "speech+egemaps": {"audio": {"type": "mean", "hidden_dim": 294}}
}

#: The arm every difference is measured against.
BASELINE_ARM = "committed"


def _loader(dataset: Dataset[Sample], batch_size: int, *, shuffle: bool = False) -> DataLoader:
    """Build a padded DataLoader over ``dataset``."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _run_fold(
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
    """Train one arm on one inner fold and score its held-out part.

    Args:
        pool: The official train split as one dataset.
        train_idx: Fold training indices.
        val_idx: Fold scored indices.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        input_dims: Per-modality input dims for this arm's features.
        quality_dims: Per-modality quality widths (segment mode only).
        device: Torch device.
        seed: Seed for this repetition.

    Returns:
        ``(metrics, scores)`` — the fold's metrics and its per-sample scores, in
        ``val_idx`` order, so the arm can be pooled out-of-fold and compared
        paired against another arm.
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
    trainer = CentralizedTrainer(
        model,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        device=str(device),
        pos_weight=pos_weight,
        objective=build_objective(base.model, base.data.phq8_max, pos_weight),
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

    model = model.to(device)
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

    Every item is scored exactly once per seed by the fold that held it out, so
    the assembled vector is a prediction for the whole train split under this
    arm — with no item scored by a model that trained on it.

    Args:
        folds: ``(indices, scores)`` per fold, repeated over seeds.
        pool_size: Number of items in the pool.

    Returns:
        Mean out-of-fold score per item, shape ``(pool_size,)``.
    """
    total = np.zeros(pool_size, dtype=np.float64)
    counts = np.zeros(pool_size, dtype=np.float64)
    for indices, scores in folds:
        total[indices] += scores
        counts[indices] += 1.0
    return total / np.maximum(counts, 1.0)


def _report(rows: list[dict[str, Any]], paired: dict[str, dict[str, float]], baseline: str) -> str:
    """Render per-arm means, per-fold wins, and the pooled paired difference.

    Args:
        rows: One row per arm x seed x fold.
        paired: Bootstrap of the pooled out-of-fold difference, per arm.
        baseline: The arm every difference is measured against.

    Returns:
        Markdown.
    """
    by_arm: dict[str, dict[tuple[int, int], float]] = {}
    f1_by_arm: dict[str, list[float]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[(row["seed"], row["fold"])] = row["roc_auc"]
        f1_by_arm.setdefault(row["arm"], []).append(row["f1"])

    lines = [
        "# Feature extraction — inner-CV ladder (ADR-0031)",
        "",
        "Ranked on inner-fold mean ROC-AUC over the official train split; dev and",
        "test are neither scored nor selected on. `wins` counts folds where the arm",
        "beat the baseline, so an effect carried by one lucky fold is visible as",
        "such. `pooled` bootstraps the difference between the two arms' pooled",
        "out-of-fold predictions, which is the paired question worth asking.",
        "",
        f"Baseline arm: `{baseline}` — the pipeline every committed result used.",
        "",
        "| arm | ROC-AUC | ±sd | F1 | per-fold | wins | pooled | 95% CI | p |",
        "|---|---|:--:|---|---|---|---|---|---|",
    ]
    base_scores = by_arm.get(baseline, {})
    ordered = sorted(by_arm, key=lambda a: statistics.fmean(by_arm[a].values()), reverse=True)
    for arm in ordered:
        scores = by_arm[arm]
        aucs = list(scores.values())
        shared = sorted(set(scores) & set(base_scores))
        deltas = [scores[key] - base_scores[key] for key in shared]
        delta_mean = statistics.fmean(deltas) if deltas else 0.0
        wins = sum(1 for delta in deltas if delta > 0)
        stats = paired.get(arm)
        pooled = f"{stats['difference']:+.3f}" if stats else "-"
        interval = f"[{stats['low']:+.3f}, {stats['high']:+.3f}]" if stats else "-"
        p_value = f"{stats['p_two_sided']:.3f}" if stats else "-"
        lines.append(
            f"| {arm} | {statistics.fmean(aucs):.3f} "
            f"| {statistics.stdev(aucs) if len(aucs) > 1 else 0.0:.3f} "
            f"| {statistics.fmean(f1_by_arm[arm]):.3f} "
            f"| {delta_mean:+.3f} | {wins}/{len(deltas)} "
            f"| {pooled} | {interval} | {p_value} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the feature-extraction ladder and write its report."""
    parser = argparse.ArgumentParser(description="DAIC-WOZ feature-extraction ladder (ADR-0031).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=f"Subset of arms to run. Defaults to all: {' '.join(ARMS)}",
    )
    parser.add_argument(
        "--baseline-arm",
        choices=tuple(ARMS),
        default=BASELINE_ARM,
        help="Arm used for every paired difference.",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.arms or []) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
    selected = list(args.arms) if args.arms else list(ARMS)
    baseline_arm = args.baseline_arm
    if baseline_arm not in selected:
        # Every reported difference is against it, so it is not optional.
        selected.insert(0, baseline_arm)

    if args.daic_config is None:
        # Every arm here is a *data-side* override, and `build_splits` applies
        # those only on the real corpus. Without this warning a mock run looks
        # like a ladder whose rungs all tie.
        print(
            "warning: no --daic-config, so the arms are identical (mock data ignores "
            "daic_woz overrides). This run only exercises the harness.",
            flush=True,
        )

    base = load_baseline_config(args.config)
    seeds = args.seeds if args.seeds else list(base.train.seeds)
    seed_everything(base.seed)
    device = torch.device(resolve_device(base.train.device))

    run_dir = create_run_dir(base.train.output_dir, "phase1", "phase1_preprocessing_ladder")
    results_path = run_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []
    oof: dict[str, NDArray[np.float64]] = {}
    pool_labels: list[int] = []
    widths: dict[str, dict[str, int]] = {}

    with results_path.open("w", encoding="utf-8") as sink:
        for arm in selected:
            arm_base = base
            if arm in SEGMENT_ARMS:
                arm_base = base.model_copy(
                    update={
                        "model": base.model.model_copy(update={"architecture": "segment_gated"})
                    }
                )
            if arm in ARM_ENCODER_OVERRIDES:
                encoder_overrides = dict(arm_base.model.encoder_overrides)
                encoder_overrides.update(ARM_ENCODER_OVERRIDES[arm])
                arm_base = arm_base.model_copy(
                    update={
                        "model": arm_base.model.model_copy(
                            update={"encoder_overrides": encoder_overrides}
                        )
                    }
                )
            splits, input_dims = build_splits(arm_base, args.daic_config, daic_overrides=ARMS[arm])
            pool: Dataset[Sample] = ConcatDataset([splits.train, splits.selection])
            arm_labels = labels_of(splits.train) + labels_of(splits.selection)
            if pool_labels and arm_labels != pool_labels:
                raise RuntimeError(
                    f"arm {arm!r} sees a different pool than {baseline_arm!r}; the arms "
                    "would not be paired"
                )
            pool_labels = arm_labels
            widths[arm] = dict(input_dims)
            folds = stratified_k_fold_indices(pool_labels, args.inner_folds, base.seed)
            print(f"\n=== {arm}: dims={input_dims} pool={len(pool_labels)} ===", flush=True)

            collected: list[tuple[list[int], NDArray[np.float64]]] = []
            for seed in seeds:
                for fold_i, (train_idx, val_idx) in enumerate(folds):
                    metrics, scores = _run_fold(
                        pool,
                        train_idx,
                        val_idx,
                        pool_labels,
                        base=arm_base,
                        input_dims=input_dims,
                        quality_dims=splits.quality_dims,
                        device=device,
                        seed=derive_seed(seed, fold_i),
                    )
                    collected.append((list(val_idx), scores))
                    row = {"arm": arm, "seed": seed, "fold": fold_i, **metrics}
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                print(f"  done {arm} seed={seed}", flush=True)
            oof[arm] = _assemble_oof(collected, len(pool_labels))

    labels = np.asarray(pool_labels, dtype=np.int_)
    paired = {
        arm: paired_bootstrap_auc_difference(labels, oof[arm], oof[baseline_arm], seed=base.seed)
        for arm in oof
        if arm != baseline_arm
    }

    (run_dir / "summary.md").write_text(_report(rows, paired, baseline_arm), encoding="utf-8")
    (run_dir / "ladder.json").write_text(
        json.dumps(
            {
                "arms": {arm: ARMS[arm] for arm in selected},
                "encoder_overrides": {
                    arm: ARM_ENCODER_OVERRIDES[arm]
                    for arm in selected
                    if arm in ARM_ENCODER_OVERRIDES
                },
                "input_dims": widths,
                "paired_auc_differences": paired,
                "out_of_fold_scores": {arm: scores.tolist() for arm, scores in oof.items()},
                "labels": pool_labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
            "arms": selected,
            "baseline_arm": baseline_arm,
        },
    )

    print("\n" + _report(rows, paired, baseline_arm))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
