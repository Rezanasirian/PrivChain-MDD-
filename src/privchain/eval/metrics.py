"""Binary-classification metrics (Phase 1, objective H5).

Pure-NumPy implementations of accuracy, precision/recall/F1, and ROC-AUC so the
evaluation path has no heavy third-party dependency and runs in CI against mock
data. F1 and ROC-AUC are the headline metrics for the centralized baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _rankdata(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assign average ranks to data, handling ties (1-based, like SciPy).

    Args:
        values: 1-D array of scores.

    Returns:
        Array of the same shape with average ranks.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    # Average ranks within tied groups.
    sorted_values = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc_score(labels: NDArray[np.int_], scores: NDArray[np.float64]) -> float:
    """Compute ROC-AUC via the rank-based (Mann–Whitney U) formula.

    Args:
        labels: Binary ground-truth labels in ``{0, 1}``, shape ``(N,)``.
        scores: Predicted scores/probabilities, shape ``(N,)``.

    Returns:
        The ROC-AUC, or ``nan`` if only one class is present (undefined).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    sum_pos_ranks = float(ranks[labels == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_classification_metrics(
    scores: NDArray[np.float64],
    labels: NDArray[np.int_],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and ROC-AUC.

    Args:
        scores: Predicted probabilities (e.g., ``sigmoid(logit)``), shape ``(N,)``.
        labels: Binary ground-truth labels in ``{0, 1}``, shape ``(N,)``.
        threshold: Decision threshold applied to ``scores`` for the hard metrics.

    Returns:
        Mapping with keys ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``.

    Raises:
        ValueError: If ``scores`` and ``labels`` differ in length or are empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(scores) == 0:
        raise ValueError("cannot compute metrics on empty inputs")

    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, scores),
    }
