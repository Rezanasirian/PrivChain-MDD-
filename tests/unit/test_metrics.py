"""Unit tests for the pure-NumPy evaluation metrics (Phase 1)."""

from __future__ import annotations

import math

import numpy as np

from privchain.eval.metrics import (
    _rankdata,
    best_f1_threshold,
    binary_classification_metrics,
    roc_auc_score,
)


def test_rankdata_handles_ties() -> None:
    ranks = _rankdata(np.array([10.0, 20.0, 20.0, 40.0]))
    # The two tied 20.0 values share the average of ranks 2 and 3 = 2.5.
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_perfect_classifier() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    m = binary_classification_metrics(scores, labels)
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_known_auc_value() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    # Concordant pairs: (0.35>0.1), (0.8>0.1), (0.8>0.4) = 3 of 4 -> 0.75.
    assert roc_auc_score(labels, scores) == 0.75


def test_tied_scores_give_half() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert roc_auc_score(labels, scores) == 0.5


def test_single_class_auc_is_nan() -> None:
    scores = np.array([0.2, 0.7, 0.9])
    labels = np.array([1, 1, 1])
    assert math.isnan(roc_auc_score(labels, scores))


def test_metrics_keys_present() -> None:
    m = binary_classification_metrics(np.array([0.3, 0.6]), np.array([0, 1]))
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc", "threshold"}


def test_best_f1_threshold_finds_a_cut_below_one_half() -> None:
    """A well-ranked but low-scoring model scores F1 = 0 at 0.5, not at its best cut.

    This is the DP-SGD case: clipping normalizes away the class weighting, so
    every score can land below 0.5 while the ranking is still correct.
    """
    scores = np.array([0.05, 0.10, 0.30, 0.35])
    labels = np.array([0, 0, 1, 1])

    at_half = binary_classification_metrics(scores, labels, threshold=0.5)
    assert at_half["f1"] == 0.0
    assert at_half["roc_auc"] == 1.0  # ranking is perfect

    tuned = binary_classification_metrics(scores, labels, threshold=None)
    assert tuned["f1"] == 1.0
    assert 0.10 < tuned["threshold"] < 0.30


def test_best_f1_threshold_is_reported_back() -> None:
    scores = np.array([0.2, 0.8])
    labels = np.array([0, 1])
    assert binary_classification_metrics(scores, labels)["threshold"] == 0.5


def test_best_f1_threshold_handles_all_negative_labels() -> None:
    """With no positives, no threshold yields F1 > 0; the default is kept."""
    threshold = best_f1_threshold(np.array([0.1, 0.9]), np.array([0, 0]))
    assert 0.0 <= threshold <= 1.0


def test_tuned_threshold_never_lowers_f1() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(50)
    labels = (rng.random(50) < 0.3).astype(int)

    fixed = binary_classification_metrics(scores, labels, threshold=0.5)["f1"]
    tuned = binary_classification_metrics(scores, labels, threshold=None)["f1"]
    assert tuned >= fixed
