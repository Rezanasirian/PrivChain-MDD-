"""Unit tests for the privacy-attacker models (Phase 6, H5)."""

from __future__ import annotations

import numpy as np
import pytest

from privchain.eval.attackers import (
    MembershipInferenceAttacker,
    ReidentificationAttacker,
    add_gaussian_noise,
)


def _subject_views(
    num_subjects: int, views: int, dim: int, spread: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((num_subjects, dim)) * 5.0
    embeddings, subjects, view_ids = [], [], []
    for subject in range(num_subjects):
        for view in range(views):
            embeddings.append(centers[subject] + rng.standard_normal(dim) * spread)
            subjects.append(subject)
            view_ids.append(view)
    return np.asarray(embeddings), np.asarray(subjects), np.asarray(view_ids)


def test_reidentification_is_perfect_without_noise() -> None:
    emb, subjects, views = _subject_views(8, 4, 16, spread=0.01, seed=0)
    attacker = ReidentificationAttacker()
    attacker.enroll(emb[views < 2], subjects[views < 2])
    accuracy = attacker.attack(emb[views >= 2], subjects[views >= 2])
    assert accuracy == 1.0


def test_reidentification_collapses_under_heavy_noise() -> None:
    emb, subjects, views = _subject_views(16, 4, 16, spread=0.01, seed=1)
    rng = np.random.default_rng(2)
    attacker = ReidentificationAttacker()
    attacker.enroll(emb[views < 2], subjects[views < 2])
    noisy = add_gaussian_noise(emb[views >= 2], std=50.0, rng=rng)
    accuracy = attacker.attack(noisy, subjects[views >= 2])
    # Far below perfect and near the 1/16 chance baseline.
    assert accuracy < 0.5
    assert ReidentificationAttacker.chance_accuracy(16) == pytest.approx(1 / 16)


def test_add_gaussian_noise_zero_std_is_identity() -> None:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((5, 4))
    out = add_gaussian_noise(emb, std=0.0, rng=rng)
    assert np.array_equal(out, emb)
    assert out is not emb  # returns a copy


def test_membership_inference_detects_separated_scores() -> None:
    rng = np.random.default_rng(3)
    members = rng.normal(1.0, 0.5, 200)
    nonmembers = rng.normal(-1.0, 0.5, 200)
    result = MembershipInferenceAttacker().attack(members, nonmembers)
    assert result["auc"] > 0.9
    assert result["advantage"] > 0.8
    assert 0.9 < result["accuracy"] <= 1.0


def test_membership_inference_near_chance_when_indistinguishable() -> None:
    rng = np.random.default_rng(4)
    members = rng.normal(0.0, 1.0, 300)
    nonmembers = rng.normal(0.0, 1.0, 300)
    result = MembershipInferenceAttacker().attack(members, nonmembers)
    assert result["auc"] < 0.65  # symmetric-max AUC stays close to 0.5
    assert result["advantage"] < 0.3


def test_reidentification_errors() -> None:
    attacker = ReidentificationAttacker()
    with pytest.raises(RuntimeError):
        attacker.attack(np.zeros((1, 2)), np.zeros(1, dtype=int))
    with pytest.raises(ValueError):
        attacker.enroll(np.zeros((0, 2)), np.zeros(0, dtype=int))
