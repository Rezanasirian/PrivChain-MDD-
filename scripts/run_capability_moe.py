"""CLI: does capability-conditioned logit fusion help the worst client? (ADR-0028).

**One inferential comparison, fixed before the run.** The segment ladder produced
eight arms and one significant result, which is what happens when a corpus of 107
participants is asked eight questions. This asks one:

    Δ = AUC_audio_only(MoE) − AUC_audio_only(baseline)

``audio_only`` is the primary capability because it is the only committed pattern
without text, the strongest modality — the hardest direct test of the thesis's
missing-modality claim. It was chosen from the *existing* ladder results, before
this model was implemented, and is not re-selected from this run's output.

**Both arms train under one schedule.** Each participant is seen under exactly one
capability per epoch, cycling through the deployment mix from
``configs/federated.yaml`` (4 full : 3 audio+text : 2 audio-only : 1 text-only).
One mask per participant-step keeps DP sensitivity at the participant, which is
what the accountant assumes.

**Evaluation is counterfactual.** Every held-out participant is scored once per
capability, so each capability's estimate rests on the full fold, not a quarter of
it. The four estimates are correlated, and the shared participant-level bootstrap
keeps that correlation instead of pretending they are independent.

Secondary metrics — macro AUC, the minimum over capabilities, the argmin
distribution, per-capability ROC/PR, the full-modality guardrail and the learned
gate weights — are reported without significance claims.

Usage:
    python scripts/run_capability_moe.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import BaselineConfig, ModalityPattern, load_baseline_config, resolve_device
from privchain.data.mock_daic_woz import MODALITIES, Sample, collate_fn
from privchain.eval.benchmark import stratified_k_fold_indices
from privchain.eval.metrics import average_precision, roc_auc_score
from privchain.federated.capability_patterns import capability_of, load_modality_patterns
from privchain.federated.partition import apply_capability_mask
from privchain.fusion.factory import build_depression_model
from privchain.seeding import derive_seed, seed_everything
from privchain.training.capability_schedule import (
    CapabilityScheduler,
    ScheduledCapabilityDataset,
)
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import DepressionObjective, positive_class_weight
from privchain.training.protocol import build_splits, carve_selection_split, labels_of
from privchain.training.trainer import CentralizedTrainer

#: The capability the single inferential test is about. Fixed here, in code, so
#: it cannot become whichever capability the run happened to favour.
PRIMARY_CAPABILITY = "audio_only"

#: Checkpoint/early-stopping criterion: mean BCE over the fixed selection set,
#: which holds every selection participant under every capability. Recorded in
#: the manifest because it is a protocol decision, not a tuned knob.
SELECTION_METRIC = "loss"

#: The two arms. Both are segment-aligned and both train under the same schedule;
#: the only difference is where the modalities meet.
ARMS: dict[str, str] = {
    "baseline_gated_fusion": "segment_gated",
    "capability_moe": "capability_moe",
}


@dataclass
class FoldScores:
    """One fold's held-out predictions under every capability.

    Attributes:
        indices: Pool positions of the scored participants.
        by_capability: ``{capability_name: scores}``, aligned with ``indices``.
        gate_weights: Mean gate weight per modality under the full capability.
    """

    indices: list[int]
    by_capability: dict[str, NDArray[np.float64]]
    gate_weights: dict[str, float]


class _MaskedView(Dataset[Sample]):
    """A fixed-capability view used only for scoring, never for training."""

    def __init__(self, base: Dataset[Sample], capability: dict[str, int]) -> None:
        self._base = base
        self._capability = capability

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Sample:
        return apply_capability_mask(self._base[index], self._capability)


def _loader(dataset: Dataset[Sample], batch_size: int, *, shuffle: bool = False) -> DataLoader:
    """Build a padded DataLoader over ``dataset``."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


@torch.no_grad()
def _score(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Return per-sample probabilities and the mean gate weight per modality."""
    model.eval()
    scores: list[NDArray[Any]] = []
    weights: dict[str, list[float]] = {modality: [] for modality in MODALITIES}
    for raw in loader:
        batch = {
            key: (
                {k: v.to(device) for k, v in value.items()}
                if isinstance(value, dict)
                else value.to(device)
            )
            for key, value in raw.items()
        }
        scores.append(torch.sigmoid(model(batch)["logit"]).cpu().numpy())
        # The MoE mixes one weight per modality per sample; the baseline gates
        # per *segment*, so its weights are averaged over the segment axis to put
        # both arms on the same "how much did this modality count" scale.
        learned = getattr(model, "last_weights", None) or getattr(
            getattr(model, "fusion", None), "last_gates", None
        )
        if learned:
            for modality in MODALITIES:
                value = learned[modality].cpu()
                weights[modality].extend(
                    value.mean(dim=-1).tolist() if value.dim() > 1 else value.tolist()
                )
    mean_weights = {
        modality: float(np.mean(values)) if values else float("nan")
        for modality, values in weights.items()
    }
    return np.concatenate(scores).astype(np.float64), mean_weights


def _run_fold(
    pool: Dataset[Sample],
    train_idx: list[int],
    val_idx: list[int],
    pool_labels: list[int],
    *,
    base: BaselineConfig,
    architecture: str,
    patterns: list[ModalityPattern],
    input_dims: dict[str, int],
    quality_dims: dict[str, int] | None,
    device: torch.device,
    seed: int,
) -> tuple[FoldScores, float]:
    """Train one arm on one fold, then score its held-out part under every capability.

    Args:
        pool: The official train split as one dataset.
        train_idx: Fold training positions.
        val_idx: Fold scored positions.
        pool_labels: Binary label per pool item.
        base: Validated baseline config.
        architecture: ``segment_gated`` or ``capability_moe``.
        patterns: The capability patterns, shared by schedule and evaluation.
        input_dims: Per-modality segment feature widths.
        quality_dims: Per-modality quality widths.
        device: Torch device.
        seed: Seed for this repetition.

    Returns:
        ``(scores, seconds)``.
    """
    seed_everything(seed)
    batch_size = base.train.batch_size
    fold_train, selection = carve_selection_split(
        Subset(pool, train_idx),
        [pool_labels[i] for i in train_idx],
        selection_fraction=base.train.selection_fraction,
        seed=seed,
    )
    # Only *training* rotates capabilities. A scheduled selection split changes
    # composition every epoch — simulating this schedule on ~17 participants, the
    # audio_only count swings between 1 and 6 and some epochs contain no
    # text_only at all — so two checkpoints would be compared on different data
    # and early stopping could pick whichever epoch drew the easier mix.
    scheduler = CapabilityScheduler(patterns, seed=seed)
    scheduled_train = ScheduledCapabilityDataset(fold_train, scheduler)
    # The fixed selection set: every participant under every capability, so it is
    # identical at every epoch and covers the mix the model must serve.
    fixed_selection: Dataset[Sample] = ConcatDataset(
        [_MaskedView(selection, capability_of(pattern)) for pattern in patterns]
    )

    # ADR-0028 D5 specifies one loss. Left to the committed config this would run
    # with phq_loss_weight 0.1, and the auxiliary head takes a *different route*
    # in each arm — one shared regressor after fusion in the baseline, three
    # per-modality regressors mixed by the gate in the MoE — so the arms would
    # differ in more than the fusion under test. Overridden here rather than in
    # the config so the committed defaults stay what other phases use.
    model_config = base.model.model_copy(
        update={
            "architecture": architecture,
            "phq_loss_weight": 0.0,
            "use_phq_regression": False,
        }
    )
    model = build_depression_model(input_dims, model_config, quality_dims)
    pos_weight = (
        positive_class_weight(_loader(scheduled_train, batch_size))
        if base.train.class_weighting
        else None
    )
    trainer = CentralizedTrainer(
        model,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=model_config.phq_loss_weight,
        device=str(device),
        objective=DepressionObjective(
            base.data.phq8_max,
            model_config.phq_loss_weight,
            pos_weight,
            phq_loss=model_config.phq_loss,
            huber_delta=model_config.huber_delta,
        ),
    )

    def _advance(epoch: int) -> None:
        scheduled_train.set_epoch(epoch)

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trainer.fit(
            _loader(scheduled_train, batch_size, shuffle=True),
            _loader(fixed_selection, batch_size),
            epochs=base.train.epochs,
            run_dir=run_dir,
            # Mean BCE across the four capabilities, not F1. On ~17 selection
            # participants F1 moves ~0.06 on a single flipped prediction, and it
            # would reward whichever epoch happened to sit near a threshold;
            # the loss is continuous and is what training minimizes. It is a mean
            # over samples, so each capability weighs exactly 25% whatever the
            # batch size — averaging per-batch means gave the trailing block
            # (one whole capability, by ConcatDataset's ordering) ~47%.
            selection_metric=SELECTION_METRIC,
            early_stopping_patience=base.train.early_stopping_patience,
            on_epoch_start=_advance,
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model = model.to(device)

    held_out = Subset(pool, val_idx)
    by_capability: dict[str, NDArray[np.float64]] = {}
    gate_weights: dict[str, float] = {}
    for pattern in patterns:
        view = _MaskedView(held_out, capability_of(pattern))
        scores, weights = _score(model, _loader(view, batch_size), device)
        by_capability[pattern.name] = scores
        if pattern.name == "full":
            gate_weights = weights
    return (
        FoldScores(indices=list(val_idx), by_capability=by_capability, gate_weights=gate_weights),
        time.monotonic() - started,
    )


def _assemble_oof(
    folds: list[FoldScores], pool_size: int, capabilities: list[str]
) -> dict[str, NDArray[np.float64]]:
    """Stitch per-fold predictions into one out-of-fold vector per capability."""
    out = {name: np.zeros(pool_size, dtype=np.float64) for name in capabilities}
    for fold in folds:
        for name, scores in fold.by_capability.items():
            out[name][fold.indices] = scores
    return out


def _paired_bootstrap(
    labels: NDArray[np.int_],
    treatment: NDArray[np.float64],
    control: NDArray[np.float64],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """Participant-level paired bootstrap of an AUC difference.

    Both arms are re-scored on the *same* resampled participants, so the fold
    difficulty and the class mix cancel out of the contrast.

    Args:
        labels: Binary labels.
        treatment: The proposed arm's scores.
        control: The baseline arm's scores.
        n_resamples: Bootstrap replicates.
        seed: Seed for the resampling.

    Returns:
        The observed difference and its percentile interval.
    """
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(labels, treatment) - roc_auc_score(labels, control)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(labels), len(labels))
        drawn = labels[idx]
        if drawn.min() == drawn.max():
            continue  # a one-class resample has no AUC
        deltas.append(roc_auc_score(drawn, treatment[idx]) - roc_auc_score(drawn, control[idx]))
    low, high = (
        (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
        if deltas
        else (float("nan"), float("nan"))
    )
    return {
        "delta": float(observed),
        "ci_low": low,
        "ci_high": high,
        "significant": float(low > 0.0 or high < 0.0),
    }


def _worst_capability_bootstrap(
    labels: NDArray[np.int_],
    treatment: dict[str, NDArray[np.float64]],
    control: dict[str, NDArray[np.float64]],
    capabilities: list[str],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Secondary: the min-over-capabilities contrast, and which capability wins it.

    Reported without a significance claim. A minimum over four noisy estimates is
    biased downward by an amount that depends on each arm's spread, so the
    interval below is descriptive, not a test — which is exactly why the primary
    estimand is a single named capability instead.

    Args:
        labels: Binary labels.
        treatment: The proposed arm's per-capability scores.
        control: The baseline arm's per-capability scores.
        capabilities: Capability names to minimize over.
        n_resamples: Bootstrap replicates.
        seed: Seed for the resampling.

    Returns:
        The observed min-AUC difference, a descriptive interval, and how often
        each capability was the argmin.
    """
    rng = np.random.default_rng(seed)

    def worst(scores: dict[str, NDArray[np.float64]], y: NDArray[np.int_], idx: Any) -> float:
        return min(roc_auc_score(y, scores[name][idx]) for name in capabilities)

    everyone = np.arange(len(labels))
    observed = worst(treatment, labels, everyone) - worst(control, labels, everyone)
    argmins: Counter[str] = Counter()
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(labels), len(labels))
        drawn = labels[idx]
        if drawn.min() == drawn.max():
            continue
        aucs = {name: roc_auc_score(drawn, treatment[name][idx]) for name in capabilities}
        argmins[min(aucs, key=lambda k: aucs[k])] += 1
        deltas.append(min(aucs.values()) - worst(control, drawn, idx))
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)) if deltas else float("nan"),
        "ci_high": float(np.percentile(deltas, 97.5)) if deltas else float("nan"),
        "argmin_distribution": {
            name: argmins[name] / max(sum(argmins.values()), 1) for name in capabilities
        },
    }


def _report(
    per_capability: dict[str, dict[str, dict[str, float]]],
    primary: dict[str, float],
    worst: dict[str, Any],
    gate_weights: dict[str, dict[str, float]],
    seconds: dict[str, float],
    inner_folds: int,
) -> str:
    """Render the pre-registered test first, then the descriptive metrics."""
    lines = [
        "# Capability-conditioned logit MoE — one pre-registered comparison",
        "",
        f"Inner {inner_folds}-fold CV over the official train split only. Both arms",
        "train under the identical capability schedule and are scored",
        "counterfactually under every capability, so each estimate uses the whole",
        "fold.",
        "",
        "## Primary test (fixed before the run)",
        "",
        f"`Δ = AUC_{PRIMARY_CAPABILITY}(MoE) − AUC_{PRIMARY_CAPABILITY}(baseline)`",
        "",
        "| Δ | 95% CI | significant |",
        "|---|---|:--:|",
        f"| {primary['delta']:+.3f} | [{primary['ci_low']:+.3f}, {primary['ci_high']:+.3f}] "
        f"| {'yes' if primary['significant'] else 'no'} |",
        "",
        "## Per-capability ROC-AUC (descriptive)",
        "",
        "| capability | baseline | MoE | Δ |",
        "|---|---|---|---|",
    ]
    arms = list(per_capability)
    control, treatment = arms[0], arms[-1]
    for name in per_capability[control]:
        base_auc = per_capability[control][name]["roc_auc"]
        moe_auc = per_capability[treatment][name]["roc_auc"]
        lines.append(f"| {name} | {base_auc:.3f} | {moe_auc:.3f} | {moe_auc - base_auc:+.3f} |")

    macro = {
        arm: statistics.fmean(m["roc_auc"] for m in caps.values())
        for arm, caps in per_capability.items()
    }
    lines += [
        "",
        "## Secondary (no significance claimed)",
        "",
        f"- macro capability AUC: baseline {macro[control]:.3f}, MoE {macro[treatment]:.3f}",
        f"- min-capability Δ: {worst['delta']:+.3f} "
        f"[{worst['ci_low']:+.3f}, {worst['ci_high']:+.3f}] — a minimum over four",
        "  noisy estimates is biased downward, so this interval is descriptive",
        "- argmin distribution (MoE): "
        + ", ".join(f"{k} {v:.0%}" for k, v in worst["argmin_distribution"].items()),
        "",
        "## Learned gate weights under full modality",
        "",
        "| arm | " + " | ".join(MODALITIES) + " |",
        "|---|" + "---|" * len(MODALITIES),
    ]
    for arm in arms:
        row = " | ".join(f"{gate_weights[arm].get(m, float('nan')):.3f}" for m in MODALITIES)
        lines.append(f"| {arm} | {row} |")
    lines += ["", "## Wall clock", ""]
    lines += [f"- {arm}: {value:.1f}s per fold" for arm, value in seconds.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the two-arm capability comparison and write its report."""
    parser = argparse.ArgumentParser(description="Capability-conditioned MoE comparison.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    seeds = args.seeds if args.seeds else list(base.train.seeds)
    patterns = load_modality_patterns(args.federated_config)
    capabilities = [pattern.name for pattern in patterns]
    if PRIMARY_CAPABILITY not in capabilities:
        raise ValueError(
            f"the primary capability {PRIMARY_CAPABILITY!r} is not declared in "
            f"{args.federated_config}; found {capabilities}"
        )

    seed_everything(base.seed)
    device = torch.device(resolve_device(base.train.device))
    overrides = {
        "segments": {"enabled": True, "count": args.num_segments},
        "text": {"representation": "segments", "num_segments": args.num_segments},
    }
    splits, input_dims = build_splits(base, args.daic_config, daic_overrides=overrides)
    pool: Dataset[Sample] = ConcatDataset([splits.train, splits.selection])
    pool_labels = labels_of(splits.train) + labels_of(splits.selection)
    labels = np.asarray(pool_labels, dtype=np.int_)
    folds = stratified_k_fold_indices(pool_labels, args.inner_folds, base.seed)

    run_dir = create_run_dir(base.train.output_dir, "phase1", "phase1_capability_moe")
    oof: dict[str, dict[str, NDArray[np.float64]]] = {}
    gate_weights: dict[str, dict[str, float]] = {}
    seconds: dict[str, float] = {}

    oof_path = run_dir / "oof_predictions.jsonl"
    with (
        (run_dir / "results.jsonl").open("w", encoding="utf-8") as sink,
        oof_path.open("w", encoding="utf-8") as oof_sink,
    ):
        for arm, architecture in ARMS.items():
            # Averaged over seeds, as in the segment ladder: the question is what
            # the procedure does, not what one lucky initialization did.
            per_seed: list[dict[str, NDArray[np.float64]]] = []
            weights_seen: list[dict[str, float]] = []
            timings: list[float] = []
            for seed in seeds:
                fold_results: list[FoldScores] = []
                for fold_i, (train_idx, val_idx) in enumerate(folds):
                    scored, elapsed = _run_fold(
                        pool,
                        train_idx,
                        val_idx,
                        pool_labels,
                        base=base,
                        architecture=architecture,
                        patterns=patterns,
                        input_dims=input_dims,
                        quality_dims=splits.quality_dims,
                        device=device,
                        seed=derive_seed(seed, fold_i),
                    )
                    fold_results.append(scored)
                    timings.append(elapsed)
                    if scored.gate_weights:
                        weights_seen.append(scored.gate_weights)
                    # The bootstrap's own input, written out so the reported CI
                    # can be recomputed without re-running the training. Only
                    # positional indices: the file carries no participant id.
                    for name, fold_scores in scored.by_capability.items():
                        for position, index in enumerate(scored.indices):
                            oof_sink.write(
                                json.dumps(
                                    {
                                        "arm": arm,
                                        "seed": seed,
                                        "fold": fold_i,
                                        "capability": name,
                                        "index": int(index),
                                        "label": int(pool_labels[index]),
                                        "score": round(float(fold_scores[position]), 6),
                                    }
                                )
                                + "\n"
                            )
                assembled = _assemble_oof(fold_results, len(pool_labels), capabilities)
                per_seed.append(assembled)
                for name, scores in assembled.items():
                    sink.write(
                        json.dumps(
                            {
                                "arm": arm,
                                "seed": seed,
                                "capability": name,
                                "roc_auc": roc_auc_score(labels, scores),
                                "pr_auc": average_precision(labels, scores),
                            }
                        )
                        + "\n"
                    )
                sink.flush()
                oof_sink.flush()
                print(f"  done {arm} seed={seed}", flush=True)
            oof[arm] = {
                name: np.stack([run[name] for run in per_seed]).mean(axis=0)
                for name in capabilities
            }
            gate_weights[arm] = {
                modality: statistics.fmean(w[modality] for w in weights_seen)
                if weights_seen
                else float("nan")
                for modality in MODALITIES
            }
            seconds[arm] = statistics.fmean(timings)

    control, treatment = list(ARMS)
    primary = _paired_bootstrap(
        labels,
        oof[treatment][PRIMARY_CAPABILITY],
        oof[control][PRIMARY_CAPABILITY],
        n_resamples=args.bootstrap_resamples,
        seed=base.seed,
    )
    worst = _worst_capability_bootstrap(
        labels,
        oof[treatment],
        oof[control],
        capabilities,
        n_resamples=args.bootstrap_resamples,
        seed=base.seed,
    )
    per_capability = {
        arm: {
            name: {
                "roc_auc": roc_auc_score(labels, scores),
                "pr_auc": average_precision(labels, scores),
            }
            for name, scores in arm_scores.items()
        }
        for arm, arm_scores in oof.items()
    }

    (run_dir / "summary.md").write_text(
        _report(per_capability, primary, worst, gate_weights, seconds, args.inner_folds),
        encoding="utf-8",
    )
    (run_dir / "primary_test.json").write_text(
        json.dumps(
            {
                "primary_capability": PRIMARY_CAPABILITY,
                "contrast": f"{treatment} - {control}",
                **primary,
                "secondary_min_capability": worst,
                "per_capability": per_capability,
                "gate_weights_full_modality": gate_weights,
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
            "official_dev_scored": False,
            "official_dev_selected_on": False,
            "official_dev_constructed": True,
            "official_test_read": False,
            "pool_size": len(pool_labels),
            "inner_folds": args.inner_folds,
            "seeds": list(seeds),
            "num_segments": args.num_segments,
            "arms": dict(ARMS),
            "capability_patterns": [
                {"name": p.name, "capability": p.capability, "fraction": p.fraction}
                for p in patterns
            ],
            "capability_pattern_source": str(args.federated_config),
            "primary_capability": PRIMARY_CAPABILITY,
            "primary_capability_fixed_before_run": True,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": base.seed,
            "bootstrap_method": "paired percentile bootstrap over participants",
            "bootstrap_unit": "participant",
            "confidence_level": 0.95,
            "seed_aggregation": "per-participant scores averaged across seeds before bootstrap",
            # ADR-0028 D5: one loss. Overridden in the script, so the value the
            # run actually used is recorded rather than the config default.
            "phq_loss_weight": 0.0,
            "use_phq_regression": False,
            "selection_metric": SELECTION_METRIC,
            "selection_set": "fixed: every selection participant under every capability",
            "selection_weighting": "equal macro (25% per capability); mean BCE over samples",
            "selection_schedule": "training only; the selection split never rotates",
            "gate_bias_initialization": dict(base.model.moe.gate_bias),
            "gate_bias_source": "pre-existing inner-CV result before MoE implementation",
            "gate_bias_swept": False,
        },
    )
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
