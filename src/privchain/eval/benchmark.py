"""Cross-validation, metric aggregation, and latency benchmarking (Phase 7, H5).

The reusable, framework-agnostic core of the final evaluation harness that
produces the Chapter-4 tables: k-fold index generation and a held-out split, a
nan-aware mean/std aggregator over per-fold metrics, and an inference-latency
benchmark across batch sizes (the "compute budget" axis). The method-specific
training (centralized / FedAvg / the proposed framework and its ablations) lives
in ``scripts/run_final_evaluation.py``; this module stays training-agnostic so it
can be unit-tested cheaply.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.training.objective import move_batch_to_device


def k_fold_indices(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Split ``range(n)`` into ``k`` stratification-free CV folds.

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
    splits: list[tuple[list[int], list[int]]] = []
    for test in folds:
        test_set = set(test)
        train = [i for i in range(n) if i not in test_set]
        splits.append((train, test))
    return splits


def held_out_split(n: int, held_out_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split ``range(n)`` into a development set and a held-out test set.

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

    Args:
        per_fold: One metric mapping per fold (keys may differ; nan values, e.g.
            an undefined ROC-AUC on a single-class fold, are ignored).

    Returns:
        A mapping ``{f"{key}_mean": ..., f"{key}_std": ...}`` over every key seen,
        plus ``num_folds``.

    Raises:
        ValueError: If ``per_fold`` is empty.
    """
    if not per_fold:
        raise ValueError("cannot aggregate an empty list of folds")
    keys = sorted({key for fold in per_fold for key in fold})
    aggregated: dict[str, float] = {"num_folds": float(len(per_fold))}
    for key in keys:
        values = np.array(
            [fold[key] for fold in per_fold if key in fold], dtype=np.float64
        )
        finite = values[~np.isnan(values)]
        if len(finite) == 0:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")
        else:
            aggregated[f"{key}_mean"] = float(np.mean(finite))
            aggregated[f"{key}_std"] = float(np.std(finite))
    return aggregated


@torch.no_grad()
def measure_inference_latency(
    model: MultimodalDepressionModel,
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
        batch = move_batch_to_device(
            collate_fn([dataset[i] for i in range(batch_size)]), device
        )
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
