"""Shared evaluation protocol for real-data runs (ADR-0015).

Every arm compared in Chapter 4 — the non-private baseline, DP-SGD at each ε,
federated variants — must be *selected* and *reported* the same way, or the
differences between them measure the protocol rather than the method.

This module owns that protocol so no script can drift from it:

* **Three splits, distinct roles.** ``train`` fits parameters; ``selection``
  chooses the epoch and the decision threshold; ``report`` is read once per run
  and never selected on. The selection split is carved out of train, so the
  official dev split stays a clean estimate.
* **One threshold policy.** The F1-maximizing cut is found on ``selection`` and
  applied to ``report`` (:func:`evaluate_with_selected_threshold`).
* **One source of shuffling randomness**, so two harnesses given the same seed
  produce the same batches. Passing the generator explicitly is what the mock
  path already did and the sweep harness did not, which is why the two disagreed
  on identical hyperparameters.
* **Repeats over seeds**, aggregated to mean ± std, because a 34-session split
  moves ~0.03 F1 on a single flipped prediction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, Subset

from privchain.config import BaselineConfig, load_yaml, modality_input_dims
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.eval.benchmark import aggregate_metrics, stratified_held_out_split
from privchain.eval.metrics import bootstrap_auc_ci
from privchain.training.loaders import split_dataset


@dataclass(frozen=True)
class Splits:
    """The three roles a dataset plays in one run.

    Attributes:
        train: Fits model parameters.
        selection: Chooses epoch and decision threshold. Never reported.
        report: Read once at the end; never selected on.
        quality_dims: Per-modality quality-vector widths when the data is
            segment-aligned, else ``None``. Carried here rather than returned
            separately so adding it did not change ``build_splits``'s signature
            for the ten scripts that already unpack it.
    """

    train: Dataset[Sample]
    selection: Dataset[Sample]
    report: Dataset[Sample]
    quality_dims: dict[str, int] | None = None


@dataclass
class RunResult:
    """Outcome of one seeded run under the protocol.

    Only ``metrics`` is required: :func:`repeat_over_seeds` aggregates that, and
    arms that do not train epoch-by-epoch (the DP sweep, the ablation) have no
    meaningful epoch counts to report.

    ``scores``/``labels`` are the per-sample report-split predictions, kept so a
    run can quote a bootstrap confidence interval rather than only the spread
    over seeds (ADR-0020). They are positionally aligned with the split and carry
    no participant identifier.
    """

    metrics: dict[str, float]
    best_epoch: int = 0
    epochs_run: int = 0
    threshold: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)
    scores: NDArray[np.float64] | None = None
    labels: NDArray[np.int_] | None = None


def uncertainty_report(
    results: Sequence[RunResult], *, seed: int = 0, n_resamples: int = 2000
) -> dict[str, float]:
    """Bootstrap CI for an arm's ROC-AUC, from its pooled per-sample scores.

    Averaging the seeds' scores before bootstrapping asks the right question:
    how far would *this arm* — as a procedure, not as one lucky fit — move on a
    different sample of participants.

    Args:
        results: The per-seed results of one arm.
        seed: Seed for the resampling.
        n_resamples: Bootstrap replicates.

    Returns:
        ``roc_auc_ci_low`` / ``roc_auc_ci_high`` / ``roc_auc_ci_width``, or an
        empty mapping when the runs carry no scores.
    """
    try:
        mean_scores, labels = pooled_scores(results)
    except ValueError:
        return {}
    low, high = bootstrap_auc_ci(labels, mean_scores, n_resamples=n_resamples, seed=seed)
    return {
        "roc_auc_ci_low": low,
        "roc_auc_ci_high": high,
        "roc_auc_ci_width": high - low,
    }


def pooled_scores(results: Sequence[RunResult]) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Average an arm's per-seed scores, for paired comparison against another arm.

    Args:
        results: The per-seed results of one arm.

    Returns:
        ``(mean_scores, labels)``.

    Raises:
        ValueError: If none of the runs carried scores.
    """
    collected: list[NDArray[np.float64]] = []
    labels: NDArray[np.int_] | None = None
    for result in results:
        if result.scores is None or result.labels is None:
            continue
        collected.append(result.scores)
        labels = result.labels
    if not collected or labels is None:
        raise ValueError("no run carried per-sample scores")
    mean_scores: NDArray[np.float64] = np.stack(collected).mean(axis=0)
    return mean_scores, labels


def carve_selection_split(
    dataset: Dataset[Sample],
    labels: Sequence[int],
    *,
    selection_fraction: float,
    seed: int,
) -> tuple[Dataset[Sample], Dataset[Sample]]:
    """Split a training set into (train, selection), preserving class balance.

    Args:
        dataset: The full training dataset.
        labels: Binary label per sample, index-aligned with ``dataset``.
        selection_fraction: Share assigned to the selection split.
        seed: Seed for the split.

    Returns:
        ``(train_subset, selection_subset)``.

    Raises:
        ValueError: If the split would leave either side empty.
    """
    train_idx, selection_idx = stratified_held_out_split(
        list(labels), held_out_fraction=selection_fraction, seed=seed
    )
    if not train_idx or not selection_idx:
        raise ValueError(
            f"selection_fraction={selection_fraction} on {len(labels)} samples "
            "leaves an empty split"
        )
    return Subset(dataset, train_idx), Subset(dataset, selection_idx)


def make_loader(
    dataset: Dataset[Sample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    num_workers: int = 0,
) -> DataLoader[Sample]:
    """Build a DataLoader with an explicit, seeded generator when shuffling.

    Args:
        dataset: The dataset to wrap.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle each epoch.
        seed: Seed for the shuffle generator; required when ``shuffle``.
        num_workers: Worker processes.

    Returns:
        A configured DataLoader yielding padded batches.

    Raises:
        ValueError: If ``shuffle`` is set without a ``seed``.
    """
    if shuffle and seed is None:
        raise ValueError("a seed is required when shuffling, so runs are reproducible")
    generator = torch.Generator().manual_seed(seed) if seed is not None and shuffle else None
    loader: DataLoader[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        generator=generator,
    )
    return loader


def repeat_over_seeds(
    run: Callable[[int], RunResult],
    seeds: Sequence[int],
) -> tuple[dict[str, float], list[RunResult]]:
    """Run once per seed and aggregate the reported metrics.

    Args:
        run: Callable taking a seed and returning that run's result.
        seeds: Seeds to repeat over.

    Returns:
        ``(aggregate, results)`` where ``aggregate`` carries ``<metric>_mean``
        and ``<metric>_std`` (nan-aware) over the runs.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    results = [run(seed) for seed in seeds]
    return aggregate_metrics([result.metrics for result in results]), results


def format_aggregate(aggregate: dict[str, float], keys: Sequence[str]) -> str:
    """Render selected metrics as ``key=mean±std`` for logs.

    Args:
        aggregate: Output of :func:`repeat_over_seeds`.
        keys: Metric names (without the ``_mean``/``_std`` suffix).

    Returns:
        A single-line summary string.
    """
    parts: list[str] = []
    for key in keys:
        mean = aggregate.get(f"{key}_mean", float("nan"))
        std = aggregate.get(f"{key}_std", float("nan"))
        parts.append(f"{key}={mean:.3f}±{std:.3f}")
    return "  ".join(parts)


def build_splits(
    config: BaselineConfig,
    daic_config: Path | None,
    *,
    daic_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Splits, dict[str, int]]:
    """Build the three protocol splits and the model's input dims.

    Shared by every arm — the non-private baseline, the DP sweep, the ablations
    — so no two of them can end up comparing different data.

    On real data the official dev split is the report split and the selection
    split is carved out of train. On mock data the same three-way structure is
    applied to the synthetic sessions so the smoke path exercises the same code.

    Args:
        config: Validated baseline config.
        daic_config: Real-data config path, or ``None`` for mock data.
        daic_overrides: Per-section keys layered over the loaded real-data
            config, e.g. ``{"audio": {"normalization": "corpus"}}``. Lets one
            experiment sweep a data-side setting without a config file per arm,
            while the committed YAML stays the single source of defaults.

    Returns:
        ``(splits, input_dims)``.
    """
    if daic_config is not None:
        # Imported lazily so the mock path has no real-data dependencies.
        from privchain.data.daic_woz import build_daic_woz_dataset

        daic_cfg = load_yaml(daic_config)
        if daic_overrides:
            # The overrides name modality sections (`audio`, `video`), which live
            # under the file's top-level `daic_woz` key.
            inner = dict(daic_cfg["daic_woz"])
            for section, values in daic_overrides.items():
                if section not in inner:
                    raise KeyError(
                        f"cannot override unknown daic_woz section {section!r}; "
                        f"available: {sorted(inner)}"
                    )
                inner[section] = {**inner[section], **values}
            daic_cfg = {**daic_cfg, "daic_woz": inner}
        train_dataset = build_daic_woz_dataset(daic_cfg, split="train")
        full_train: Dataset[Sample] = train_dataset
        report: Dataset[Sample] = build_daic_woz_dataset(daic_cfg, split="dev")
        input_dims = train_dataset.feature_dims
        quality_dims = train_dataset.quality_dims
    else:
        full = MockDaicWozDataset(config.data, seed=config.seed)
        full_train, report = split_dataset(full, config.train.val_fraction, config.seed)
        input_dims = modality_input_dims(config.data)
        quality_dims = None

    train, selection = carve_selection_split(
        full_train,
        labels_of(full_train),
        selection_fraction=config.train.selection_fraction,
        seed=config.seed,
    )
    splits = Splits(train=train, selection=selection, report=report, quality_dims=quality_dims)
    return splits, input_dims


def labels_of(dataset: Any) -> list[int]:
    """Read per-sample binary labels without materializing feature tensors.

    ``DaicWozDataset`` keeps its split records in memory, so the labels are
    available without decoding audio/video. Falls back to indexing the dataset
    when that fast path is unavailable (e.g., the mock dataset).

    Args:
        dataset: A dataset following the :class:`Sample` contract.

    Returns:
        One binary label per sample, in dataset order.
    """
    records = getattr(dataset, "_records", None)
    if records is not None:
        return [int(record["label"]) for record in records]
    return [int(dataset[i]["label"]) for i in range(len(dataset))]
