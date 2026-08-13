"""CLI: measure per-modality re-identification risk on real data (Phase 6, H5).

``configs/privacy.yaml`` gives audio the tightest privacy budget and text the
loosest, on the strength of three numbers — ``reidentification_risk`` 0.9 / 0.6 /
0.3 — that were **assumed, never measured**. The whole H1 claim ("the highest-risk
modality ends up best protected") rests on them, and ADR-0016 flagged the gap as
a blocker once the ablation showed text carrying nearly all the diagnostic
signal while audio sat at chance.

This script measures those risks. Each participant appears in exactly one
DAIC-WOZ session, so the attacker's views are disjoint contiguous stretches of
that session: frame spans for audio and video, spans of transcript turns for
text, each embedded on its own. The attacker enrols a template from some
stretches and re-identifies the rest against the whole candidate pool.

Three representations are measured per modality, because nearest-centroid
accuracy grows with feature width and the modalities are not the same width:

* ``raw`` — the summary the configured encoder consumes before its projection.
  Widths differ (audio 74x5, video 20x5, text 768).
* ``raw_pca`` — the same, projected to a common width, fitted on the enrollment
  rows only.
* ``encoded`` — the trained encoder's output, width-matched by construction.

The ordering claim rests on the width-matched rows. Nothing here is protected by
DP: this is the *unprotected* risk that the allocation is supposed to be
calibrated against. The attack-success-vs-ε curve is a different question and
belongs to ``scripts/run_attack_eval.py``.

Writes ``reidentification_risk.json`` and ``reidentification_risk.png`` under
``experiments/phase6/<run-id>/``. See ADR-0017.

Usage:
    python scripts/run_reid_risk.py --daic-config configs/daic_woz.yaml
    python scripts/run_reid_risk.py --skip-encoded          # raw rows only, no training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from privchain.config import load_attack_config, load_baseline_config, resolve_device
from privchain.data.mock_daic_woz import Sample
from privchain.eval.session_views import (
    ModalityViews,
    build_views,
    concat_views,
    run_reidentification,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import positive_class_weight
from privchain.training.protocol import RunResult, build_splits, make_loader, repeat_over_seeds
from privchain.training.trainer import CentralizedTrainer

MODALITIES = ("audio", "video", "text")
# The model is fitted on `train` and its epoch/threshold chosen on `selection`;
# `report` (the official dev split) never touches it. Re-identifying a subject
# the encoder has seen is an easier problem, so the groups are reported apart.
GROUP_NAMES = ("train", "selection", "report")
REPORTED = ("accuracy", "ratio_to_chance")


def main() -> None:
    """Measure and report per-modality re-identification risk."""
    parser = argparse.ArgumentParser(description="Per-modality re-identification risk (Phase 6).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--attack-config", type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument(
        "--daic-config",
        type=Path,
        default=None,
        help="Real DAIC-WOZ config; without it the mock corpus is used (smoke test).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--skip-encoded",
        action="store_true",
        help="Measure the raw representations only, skipping baseline training.",
    )
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    seg = load_attack_config(args.attack_config).attack.segments
    seeds = args.seeds if args.seeds else config.train.seeds
    seed_everything(config.seed)

    splits, input_dims = build_splits(config, args.daic_config)
    device = torch.device(resolve_device(config.train.device))
    grouped: dict[str, Dataset[Sample]] = {
        "train": splits.train,
        "selection": splits.selection,
        "report": splits.report,
    }

    # One subject id space across the three splits: the attacker picks from every
    # participant we are allowed to touch, so chance is 1/(all of them). The test
    # split stays sealed for the Chapter 4 table and is not loaded here.
    offsets: dict[str, int] = {}
    subject_group: dict[int, str] = {}
    cursor = 0
    for name in GROUP_NAMES:
        offsets[name] = cursor
        size = len(grouped[name])  # type: ignore[arg-type]
        subject_group.update({cursor + i: name for i in range(size)})
        cursor += size

    run_dir = create_run_dir(config.train.output_dir, "phase6", "phase6_reid_risk")
    save_config(run_dir, {"baseline": config.model_dump(), "segments": seg.model_dump()})

    def views_for(modality: str, encoder: torch.nn.Module | None) -> ModalityViews:
        """Build one modality's views across all three splits as a single pool."""
        return concat_views(
            *(
                build_views(
                    grouped[name],
                    modality,
                    num_segments=seg.num_segments,
                    encoder_type=config.model.encoder_for(modality).type,
                    encoder=encoder,  # type: ignore[arg-type]
                    device=device,
                    subject_offset=offsets[name],
                )
                for name in GROUP_NAMES
            )
        )

    trained: dict[int, MultimodalDepressionModel] = {}

    def train_model(seed: int) -> MultimodalDepressionModel:
        """Fit the Phase 1 baseline under the shared protocol, for the encoded rows.

        Memoized per seed: the three modalities are attacked through the *same*
        model, so training once per modality would only burn GPU time.
        """
        if seed in trained:
            return trained[seed]
        seed_everything(seed)
        pos_weight = (
            positive_class_weight(
                make_loader(splits.train, batch_size=config.train.batch_size, shuffle=False)
            )
            if config.train.class_weighting
            else None
        )
        model = MultimodalDepressionModel(input_dims, config.model)
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        CentralizedTrainer(
            model,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            phq8_max=config.data.phq8_max,
            phq_loss_weight=config.model.phq_loss_weight,
            device=str(device),
            pos_weight=pos_weight,
        ).fit(
            make_loader(splits.train, batch_size=config.train.batch_size, shuffle=True, seed=seed),
            make_loader(splits.selection, batch_size=config.train.batch_size, shuffle=False),
            epochs=config.train.epochs,
            run_dir=seed_dir,
            selection_metric=config.train.selection_metric,
            early_stopping_patience=config.train.early_stopping_patience,
        )
        model.load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=device))
        trained[seed] = model.to(device)
        return trained[seed]

    print(f"Run dir: {run_dir}")
    print(
        f"device={device}  segments={seg.num_segments} "
        f"(enrol {seg.enroll_segments})  seeds={list(seeds)}"
    )

    report: dict[str, dict[str, dict[str, float]]] = {}
    widths: dict[str, dict[str, int]] = {}

    # ── raw and raw_pca: no model involved, so the views are built once ───────
    raw_views = {modality: views_for(modality, None) for modality in MODALITIES}
    for modality, views in raw_views.items():
        widths.setdefault(modality, {})["raw"] = int(views.features.shape[1])
        widths[modality]["raw_pca"] = min(seg.pca_dim, int(views.features.shape[1]))
        for label, pca_dim in (("raw", None), ("raw_pca", seg.pca_dim)):
            aggregate, _ = repeat_over_seeds(
                lambda s, v=views, d=pca_dim: RunResult(  # type: ignore[misc]
                    metrics=run_reidentification(
                        v,
                        enroll_segments=seg.enroll_segments,
                        seed=s,
                        pca_dim=d,
                        groups=subject_group,
                    )
                ),
                seeds,
            )
            report.setdefault(modality, {})[label] = aggregate

    # ── encoded: a freshly trained model per seed, then its encoder's output ──
    if not args.skip_encoded:
        for modality in MODALITIES:

            def attack_encoded(seed: int, m: str = modality) -> RunResult:
                """Train at this seed, then attack that model's encoder output."""
                encoder = train_model(seed).encoders[m]
                return RunResult(
                    metrics=run_reidentification(
                        views_for(m, encoder),
                        enroll_segments=seg.enroll_segments,
                        seed=seed,
                        groups=subject_group,
                    )
                )

            aggregate, _ = repeat_over_seeds(attack_encoded, seeds)
            report.setdefault(modality, {})["encoded"] = aggregate
            widths[modality]["encoded"] = config.model.encoder_for(modality).out_dim

    # ── negative control: shuffled probe labels must land on chance ───────────
    control = {
        modality: run_reidentification(
            raw_views[modality],
            enroll_segments=seg.enroll_segments,
            seed=seeds[0],
            shuffle_subjects=True,
        )["accuracy"]
        for modality in MODALITIES
    }

    chance = float(next(iter(report.values()))["raw"]["chance_mean"])
    skipped = {m: list(v.skipped) for m, v in raw_views.items() if v.skipped}
    payload = {
        "protocol": {
            "views": "disjoint contiguous stretches of the single session per participant",
            "num_segments": seg.num_segments,
            "enroll_segments": seg.enroll_segments,
            "pca_dim": seg.pca_dim,
            "seeds": list(seeds),
            "splits_pooled": list(GROUP_NAMES),
            "note": "test split untouched; no DP applied — this is the unprotected risk",
        },
        "chance_accuracy": chance,
        "feature_widths": widths,
        "assumed_reidentification_risk": {"audio": 0.9, "video": 0.6, "text": 0.3},
        "measured": report,
        "negative_control_shuffled_labels": control,
        "skipped_subjects": skipped,
    }
    (run_dir / "reidentification_risk.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _plot(report, chance, run_dir / "reidentification_risk.png")

    _print_table(report, widths, chance, control, skipped)
    print(f"\nWrote {run_dir / 'reidentification_risk.json'}")


def _print_table(
    report: dict[str, dict[str, dict[str, float]]],
    widths: dict[str, dict[str, int]],
    chance: float,
    control: dict[str, float],
    skipped: dict[str, list[int]],
) -> None:
    """Print the measured table and the ordering it implies."""
    print(f"\nchance accuracy = {chance:.4f}  (1 / candidate pool)")
    if skipped:
        print(f"skipped (too few segments): {skipped}")
    header = (
        f"\n{'modality':8s} {'representation':15s} {'width':>6s} {'top-1':>14s} {'x chance':>9s}"
    )
    print(header)
    print("-" * len(header.strip()))
    for modality, rows in report.items():
        for label, aggregate in rows.items():
            mean, std = aggregate["accuracy_mean"], aggregate["accuracy_std"]
            print(
                f"{modality:8s} {label:15s} {widths[modality].get(label, 0):>6d} "
                f"{mean:>7.3f}±{std:<6.3f} {aggregate['ratio_to_chance_mean']:>8.1f}x"
            )
    for label in ("raw_pca", "encoded"):
        ranked = sorted(
            (m for m in report if label in report[m]),
            key=lambda m: report[m][label]["accuracy_mean"],
            reverse=True,
        )
        if ranked:
            print(f"measured ordering ({label}): " + " > ".join(ranked))
    print("assumed ordering (configs/privacy.yaml): audio > video > text")
    print(
        "negative control (shuffled labels): "
        + "  ".join(f"{m}={a:.4f}" for m, a in control.items())
    )


def _plot(report: dict[str, dict[str, dict[str, float]]], chance: float, path: Path) -> None:
    """Bar chart of re-identification success per modality (no-op without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote the JSON only.")
        return

    labels = sorted({label for rows in report.values() for label in rows})
    modalities = list(report)
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, label in enumerate(labels):
        positions = [i + offset * width for i in range(len(modalities))]
        ax.bar(
            positions,
            [report[m].get(label, {}).get("accuracy_mean", 0.0) for m in modalities],
            width,
            yerr=[report[m].get(label, {}).get("accuracy_std", 0.0) for m in modalities],
            capsize=3,
            label=label,
        )
    ax.axhline(chance, linestyle="--", color="grey", label="chance")
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(modalities))])
    ax.set_xticklabels(modalities)
    ax.set_ylabel("Top-1 re-identification accuracy")
    ax.set_title("Unprotected re-identification risk per modality (DAIC-WOZ)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
