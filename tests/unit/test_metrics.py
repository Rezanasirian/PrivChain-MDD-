"""Unit tests for the pure-NumPy evaluation metrics (Phase 1)."""

from __future__ import annotations

import math

import numpy as np

from privchain.eval.metrics import _rankdata, binary_classification_metrics, roc_auc_score


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
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
