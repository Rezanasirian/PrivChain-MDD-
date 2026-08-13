"""Unit tests for bootstrap uncertainty on the report split (ADR-0020).

The project reported spread-over-seeds as if it were a confidence interval for
years of commits. It is not: it measures optimization stability on a *fixed*
evaluation set and says nothing about how the number would move on a different
sample of participants. These cover the helpers that answer the second question.
"""

from __future__ import annotations

import numpy as np
import pytest

from privchain.eval.metrics import (
    bootstrap_auc_ci,
    paired_bootstrap_auc_difference,
    roc_auc_score,
)


def _separable(n: int = 200, gap: float = 2.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores with a real, large signal."""
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = rng.standard_normal(n) + gap * labels
    return labels, scores


def test_interval_brackets_the_point_estimate() -> None:
    labels, scores = _separable()
    low, high = bootstrap_auc_ci(labels, scores, n_resamples=400, seed=1)
    assert low < roc_auc_score(labels, scores) < high


def test_random_scores_yield_an_interval_containing_chance() -> None:
    rng = np.random.default_rng(3)
    labels = np.array([0, 1] * 60)
    low, high = bootstrap_auc_ci(labels, rng.standard_normal(120), n_resamples=400, seed=2)
    assert low <= 0.5 <= high


def test_a_small_sample_gives_a_much_wider_interval() -> None:
    """The whole point: 34 sessions cannot support a tight claim."""
    labels_big, scores_big = _separable(n=400, gap=1.0, seed=5)
    labels_small, scores_small = _separable(n=34, gap=1.0, seed=5)
    wide = bootstrap_auc_ci(labels_small, scores_small, n_resamples=400, seed=0)
    narrow = bootstrap_auc_ci(labels_big, scores_big, n_resamples=400, seed=0)
    assert (wide[1] - wide[0]) > 2 * (narrow[1] - narrow[0])


def test_an_arm_compared_against_itself_shows_exactly_no_difference() -> None:
    labels, scores = _separable(n=60)
    result = paired_bootstrap_auc_difference(labels, scores, scores, n_resamples=200, seed=0)
    assert result["difference"] == 0.0
    assert result["low"] == 0.0 and result["high"] == 0.0
    assert not result["significant"]


def test_a_real_difference_is_detected() -> None:
    labels, good = _separable(n=200, gap=2.0, seed=7)
    rng = np.random.default_rng(8)
    result = paired_bootstrap_auc_difference(
        labels, good, rng.standard_normal(200), n_resamples=400, seed=0
    )
    assert result["difference"] > 0.2
    assert result["significant"]
    assert result["low"] > 0.0


def test_pairing_is_more_sensitive_than_comparing_separate_intervals() -> None:
    """Why the paired form exists rather than eyeballing two intervals.

    Two arms scored on the same participants rise and fall together with the
    luck of the draw. Cancelling that shared movement can resolve a difference
    the individually-wide intervals cannot.
    """
    rng = np.random.default_rng(11)
    labels = np.array([0] * 40 + [1] * 40)
    shared = rng.standard_normal(80) + 1.0 * labels
    better = shared + 0.35 * labels  # same ranking noise, slightly better signal

    own_a = bootstrap_auc_ci(labels, better, n_resamples=400, seed=0)
    own_b = bootstrap_auc_ci(labels, shared, n_resamples=400, seed=0)
    overlapping = own_a[0] < own_b[1] and own_b[0] < own_a[1]
    paired = paired_bootstrap_auc_difference(labels, better, shared, n_resamples=400, seed=0)

    assert overlapping  # the individual intervals do not separate the arms
    assert paired["significant"]  # the paired difference does


@pytest.mark.parametrize(
    "call",
    [
        lambda: bootstrap_auc_ci(np.array([], dtype=int), np.array([])),
        lambda: bootstrap_auc_ci(np.array([0, 1]), np.array([0.1])),
        lambda: bootstrap_auc_ci(np.array([0, 1]), np.array([0.1, 0.2]), confidence=1.5),
        lambda: paired_bootstrap_auc_difference(
            np.array([0, 1]), np.array([0.1, 0.2]), np.array([0.1])
        ),
    ],
)
def test_degenerate_input_is_rejected(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_single_class_sample_reports_nan_rather_than_a_fabricated_bound() -> None:
    labels = np.zeros(20, dtype=int)
    low, high = bootstrap_auc_ci(labels, np.random.default_rng(0).standard_normal(20))
    assert np.isnan(low) and np.isnan(high)
