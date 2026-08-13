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


def best_f1_threshold(scores: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """Find the decision threshold maximizing F1 on these scores.

    DAIC-WOZ is ~28% positive, so a fixed 0.5 cut is rarely where F1 peaks. Under
    DP-SGD it is badly wrong: per-sample gradient clipping normalizes every
    sample's gradient to the same norm, which erases the magnitude difference the
    class weighting introduces, so a model that *ranks* cases well can still put
    every score below 0.5 and score F1 = 0 (ADR-0013).

    Choosing the threshold is post-processing of the model's outputs, so it adds
    nothing to the privacy budget.

    Args:
        scores: Predicted probabilities, shape ``(N,)``.
        labels: Binary ground-truth labels in ``{0, 1}``, shape ``(N,)``.

    Returns:
        The threshold with the highest F1; ``0.5`` when no positive prediction
        ever scores above zero F1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    positives = int((labels == 1).sum())
    if positives == 0:
        return 0.5  # no threshold yields F1 > 0; keep the conventional cut

    # Sweep every distinct cut at once. Sorting descending, the prefix of length
    # i is exactly the set predicted positive at a threshold just below
    # scores[i-1], so cumulative sums give TP/FP for all cuts in one pass.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    true_positives = np.cumsum(labels[order] == 1)
    predicted = np.arange(1, len(scores) + 1)
    f1_at_cut = 2 * true_positives / (predicted + positives)

    # Only the last index of each run of equal scores is a realisable cut.
    realisable = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    f1_at_cut = np.where(realisable, f1_at_cut, -1.0)

    best = int(np.argmax(f1_at_cut))
    if f1_at_cut[best] <= 0.0:
        return 0.5

    # Sit midway between the last included score and the first excluded one.
    # The threshold is applied to a *different* split (ADR-0015), so landing
    # exactly on an observed score would make the cut needlessly brittle.
    if best + 1 < len(sorted_scores):
        return float((sorted_scores[best] + sorted_scores[best + 1]) / 2.0)
    return float(sorted_scores[best])


def binary_classification_metrics(
    scores: NDArray[np.float64],
    labels: NDArray[np.int_],
    threshold: float | None = 0.5,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and ROC-AUC.

    Args:
        scores: Predicted probabilities (e.g., ``sigmoid(logit)``), shape ``(N,)``.
        labels: Binary ground-truth labels in ``{0, 1}``, shape ``(N,)``.
        threshold: Decision threshold for the hard metrics. Pass ``None`` to
            select the F1-maximizing threshold from the scores themselves
            (:func:`best_f1_threshold`); the chosen value is returned under
            ``threshold``.

    Returns:
        Mapping with keys ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``, ``threshold``.

    Raises:
        ValueError: If ``scores`` and ``labels`` differ in length or are empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(scores) == 0:
        raise ValueError("cannot compute metrics on empty inputs")

    if threshold is None:
        threshold = best_f1_threshold(scores, labels)

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
        "threshold": float(threshold),
    }


def bootstrap_auc_ci(
    labels: NDArray[np.int_],
    scores: NDArray[np.float64],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ROC-AUC.

    Reported alongside the spread over seeds, because the two answer different
    questions and only one of them is about generalization (ADR-0020). Seed
    spread measures how stable optimization is on a *fixed* evaluation set;
    this measures how much the estimate would move on a *different* sample of
    participants. On 34 dev sessions the latter is roughly ten times the former,
    which is why several earlier claims did not survive re-examination.

    Args:
        labels: Binary labels, shape ``(N,)``.
        scores: Predicted scores, shape ``(N,)``.
        n_resamples: Bootstrap replicates.
        confidence: Coverage of the interval.
        seed: Seed for the resampling.

    Returns:
        ``(low, high)``. Replicates in which a class is absent are skipped; if
        none survive, ``(nan, nan)`` is returned rather than a fabricated bound.

    Raises:
        ValueError: If the inputs are empty or mismatched, or ``confidence`` is
            not in ``(0, 1)``.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    replicates: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(label_array), len(label_array))
        resampled = label_array[idx]
        # A replicate with only one class has no defined AUC; dropping it is
        # honest, inventing 0.5 for it would shrink the interval toward the null.
        if resampled.min() == resampled.max():
            continue
        replicates.append(roc_auc_score(resampled, score_array[idx]))

    if not replicates:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_bootstrap_auc_difference(
    labels: NDArray[np.int_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the AUC *difference* between two arms scored on the same sample.

    Resampling the participants **once per replicate** and scoring both arms on
    that same resample keeps the comparison paired, which is what makes a small
    difference detectable even when each arm's own interval is wide: the two arms
    rise and fall together with the luck of the draw, and only their gap is left.

    Args:
        labels: Binary labels shared by both arms, shape ``(N,)``.
        scores_a: First arm's scores.
        scores_b: Second arm's scores.
        n_resamples: Bootstrap replicates.
        confidence: Coverage of the interval.
        seed: Seed for the resampling.

    Returns:
        ``difference`` (a − b on the observed sample), ``low``, ``high``, and
        ``p_two_sided`` — the bootstrap proportion of replicates on the far side
        of zero, doubled. ``significant`` is 1.0 when the interval excludes zero.

    Raises:
        ValueError: If the inputs are empty or their lengths disagree.
    """
    if not len(labels) == len(scores_a) == len(scores_b):
        raise ValueError("labels and both score arrays must have the same length")
    if len(labels) == 0:
        raise ValueError("cannot bootstrap an empty sample")

    label_array = np.asarray(labels)
    array_a = np.asarray(scores_a, dtype=np.float64)
    array_b = np.asarray(scores_b, dtype=np.float64)
    observed = roc_auc_score(label_array, array_a) - roc_auc_score(label_array, array_b)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(label_array), len(label_array))
        resampled = label_array[idx]
        if resampled.min() == resampled.max():
            continue
        deltas.append(
            roc_auc_score(resampled, array_a[idx]) - roc_auc_score(resampled, array_b[idx])
        )

    if not deltas:
        return {
            "difference": observed,
            "low": float("nan"),
            "high": float("nan"),
            "p_two_sided": float("nan"),
            "significant": 0.0,
        }
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [tail, 1.0 - tail])
    below = float(np.mean(np.asarray(deltas) <= 0.0))
    p_two_sided = min(1.0, 2.0 * min(below, 1.0 - below))
    return {
        "difference": observed,
        "low": float(low),
        "high": float(high),
        "p_two_sided": p_two_sided,
        "significant": float(low > 0.0 or high < 0.0),
    }
