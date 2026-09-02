"""Cross-validation, metric aggregation, and latency benchmarking (Phase 7, H5).

The reusable, framework-agnostic core of the final evaluation harness that
produces the Chapter-4 tables: k-fold index generation and a held-out split, a
nan-aware mean/std aggregator over per-fold metrics, and an inference-latency
benchmark across batch sizes (the "compute budget" axis). The method-specific
training (centralized / FedAvg / the proposed framework and its ablations) lives
in ``scripts/run_final_evaluation.py``; this module stays training-agnostic so it
can be unit-tested cheaply.

**Splits are stratified by default.** DAIC-WOZ has ~190 sessions with a minority
depressed class, so unstratified 10-fold splitting routinely produces
single-class test folds where ROC-AUC is undefined — silently shrinking the
number of folds the reported mean is taken over. Use
:func:`stratified_k_fold_indices` / :func:`stratified_held_out_split` whenever
labels are available, and read ``<metric>_num_valid_folds`` from
:func:`aggregate_metrics` to see how many folds actually contributed.

Splitting is at *session* granularity, which on DAIC-WOZ is also *subject*
granularity (one interview per participant), so no subject appears on both sides
of a split. If the pipeline ever moves to utterance-level items, these functions
must be replaced by group-aware variants keyed on participant id.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.fusion.base import DepressionModelBase
from privchain.training.objective import move_batch_to_device


def _splits_from_folds(n: int, folds: list[list[int]]) -> list[tuple[list[int], list[int]]]:
    """Turn a list of test folds into ``(train, test)`` index pairs."""
    splits: list[tuple[list[int], list[int]]] = []
    for test in folds:
        test_set = set(test)
        splits.append(([i for i in range(n) if i not in test_set], test))
    return splits


def k_fold_indices(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Split ``range(n)`` into ``k`` stratification-free CV folds.

    Prefer :func:`stratified_k_fold_indices` whenever labels are available; on an
    imbalanced corpus this variant can produce single-class test folds.

    Args:
        n: Number of items.
        k: Number of folds (``2 <= k <= n``).
        seed: Shuffle seed.

    Returns:
        A list of ``k`` ``(train_indices, test_indices)`` pairs; each item is a
        test index in exactly one fold, and the folds cover ``range(n)``.

    Raises:
        ValueError: If ``k`` is out of range.
    """
    if not 2 <= k <= n:
        raise ValueError(f"k must satisfy 2 <= k <= n (got k={k}, n={n})")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    folds = [sorted(int(i) for i in fold) for fold in np.array_split(shuffled, k)]
    return _splits_from_folds(n, folds)


def stratified_k_fold_indices(
    labels: Sequence[int], k: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Split into ``k`` folds that preserve the class balance of ``labels``.

    Each class is shuffled and dealt round-robin across the folds, so every fold
    holds (as close as integer arithmetic allows) the same positive rate as the
    full set — which is what keeps ROC-AUC defined on every fold.

    Args:
        labels: Binary label per item, indexed like the dataset.
        k: Number of folds (``2 <= k <= len(labels)``).
        seed: Shuffle seed.

    Returns:
        A list of ``k`` ``(train_indices, test_indices)`` pairs.

    Raises:
        ValueError: If ``k`` is out of range, or a class has fewer than ``k``
            members (which makes a stratified split impossible).
    """
    n = len(labels)
    if not 2 <= k <= n:
        raise ValueError(f"k must satisfy 2 <= k <= n (got k={k}, n={n})")

    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels)
    folds: list[list[int]] = [[] for _ in range(k)]
    for value in np.unique(label_array):
        members = np.flatnonzero(label_array == value)
        if len(members) < k:
            raise ValueError(
                f"class {value!r} has {len(members)} members but {k} folds were requested; "
                "a stratified split would leave folds without it"
            )
        for position, index in enumerate(rng.permutation(members)):
            folds[position % k].append(int(index))
    return _splits_from_folds(n, [sorted(fold) for fold in folds])


def stratified_held_out_split(
    labels: Sequence[int], held_out_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Development / held-out split that preserves the class balance.

    Args:
        labels: Binary label per item.
        held_out_fraction: Fraction reserved for the held-out test set, in ``(0, 1)``.
        seed: Shuffle seed.

    Returns:
        ``(dev_indices, held_out_indices)``.

    Raises:
        ValueError: If the split would leave either side (or a class) empty.
    """
    n = len(labels)
    held_out_size = int(round(n * held_out_fraction))
    if not 0 < held_out_size < n:
        raise ValueError(f"held_out_fraction={held_out_fraction} on n={n} yields an empty split")

    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels)
    held_out: list[int] = []
    for value in np.unique(label_array):
        members = rng.permutation(np.flatnonzero(label_array == value))
        take = int(round(len(members) * held_out_fraction))
        if not 0 < take < len(members):
            raise ValueError(
                f"held_out_fraction={held_out_fraction} leaves class {value!r} "
                "entirely on one side of the split"
            )
        held_out.extend(int(index) for index in members[:take])

    held_out_sorted = sorted(held_out)
    held_out_set = set(held_out_sorted)
    return [i for i in range(n) if i not in held_out_set], held_out_sorted


def held_out_split(n: int, held_out_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split ``range(n)`` into a development set and a held-out test set.

    Prefer :func:`stratified_held_out_split` when labels are available.

    Args:
        n: Number of items.
        held_out_fraction: Fraction reserved for the held-out test set, in ``(0, 1)``.
        seed: Shuffle seed.

    Returns:
        ``(dev_indices, held_out_indices)``.

    Raises:
        ValueError: If the split would leave either side empty.
    """
    held_out_size = int(round(n * held_out_fraction))
    if not 0 < held_out_size < n:
        raise ValueError(f"held_out_fraction={held_out_fraction} on n={n} yields an empty split")
    rng = np.random.default_rng(seed)
    shuffled = [int(i) for i in rng.permutation(n)]
    held_out = sorted(shuffled[:held_out_size])
    dev = sorted(shuffled[held_out_size:])
    return dev, held_out


def aggregate_metrics(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate a list of per-fold metric dicts into nan-aware mean/std.

    Every metric also reports ``<key>_num_valid_folds``: nan values (an undefined
    ROC-AUC on a single-class fold, say) are excluded from the mean, and a table
    that hides *how many* folds a mean was taken over is not reportable. A count
    below ``num_folds`` means the split should have been stratified.

    Args:
        per_fold: One metric mapping per fold (keys may differ).

    Returns:
        A mapping with ``{key}_mean``, ``{key}_std`` and ``{key}_num_valid_folds``
        for every key seen, plus ``num_folds``.

    Raises:
        ValueError: If ``per_fold`` is empty.
    """
    if not per_fold:
        raise ValueError("cannot aggregate an empty list of folds")
    keys = sorted({key for fold in per_fold for key in fold})
    aggregated: dict[str, float] = {"num_folds": float(len(per_fold))}
    for key in keys:
        values = np.array([fold[key] for fold in per_fold if key in fold], dtype=np.float64)
        finite = values[~np.isnan(values)]
        aggregated[f"{key}_num_valid_folds"] = float(len(finite))
        if len(finite) == 0:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")
        else:
            aggregated[f"{key}_mean"] = float(np.mean(finite))
            aggregated[f"{key}_std"] = float(np.std(finite))
    return aggregated


@torch.no_grad()
def measure_inference_latency(
    model: DepressionModelBase,
    dataset: Dataset[Sample],
    *,
    batch_sizes: list[int],
    repeats: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Time forward-pass latency at several batch sizes (the compute-budget axis).

    Args:
        model: The model to benchmark.
        dataset: Dataset to draw batches from (must have ``>= max(batch_sizes)``).
        batch_sizes: Batch sizes to time.
        repeats: Timed forward passes per batch size (plus one warm-up).
        device: Torch device.

    Returns:
        One record per batch size with ``ms_per_batch`` and ``ms_per_sample``.

    Raises:
        ValueError: If ``repeats`` is not positive or a batch size exceeds the
            dataset size.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    model = model.to(device)
    model.eval()
    n = len(dataset)  # type: ignore[arg-type]

    records: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size > n:
            raise ValueError(f"batch_size {batch_size} exceeds dataset size {n}")
        batch = move_batch_to_device(collate_fn([dataset[i] for i in range(batch_size)]), device)
        model(batch)  # warm-up (excluded from timing)

        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        elapsed = time.perf_counter() - start

        ms_per_batch = 1000.0 * elapsed / repeats
        records.append(
            {
                "batch_size": batch_size,
                "ms_per_batch": ms_per_batch,
                "ms_per_sample": ms_per_batch / batch_size,
            }
        )
    return records
