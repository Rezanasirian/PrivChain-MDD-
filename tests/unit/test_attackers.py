"""Unit tests for the privacy-attacker models (Phase 6, H5)."""

from __future__ import annotations

import numpy as np
import pytest

from privchain.eval.attackers import (
    MembershipInferenceAttacker,
    ReidentificationAttacker,
    add_gaussian_noise,
    clip_rows,
    release_embeddings_dp,
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
    assert result["direction"] == 1.0


def test_membership_inference_finds_inverted_signal() -> None:
    """Calibration must be free to flip the rule when members score *lower*."""
    rng = np.random.default_rng(11)
    members = rng.normal(-1.0, 0.5, 200)
    nonmembers = rng.normal(1.0, 0.5, 200)
    result = MembershipInferenceAttacker().attack(members, nonmembers)
    assert result["direction"] == -1.0
    assert result["auc"] > 0.9


def test_membership_inference_is_unbiased_on_identical_distributions() -> None:
    """No leakage must report ~no advantage — averaged over many draws.

    The old implementation folded the AUC about 0.5 (``max(auc, 1-auc)``) and
    tuned its threshold on the scored data, so pure noise always looked like
    leakage. The mean advantage over independent trials pins that down.
    """
    attacker = MembershipInferenceAttacker()
    advantages = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        members = rng.normal(0.0, 1.0, 200)
        nonmembers = rng.normal(0.0, 1.0, 200)
        advantages.append(attacker.attack(members, nonmembers, rng=rng)["advantage"])

    assert abs(float(np.mean(advantages))) < 0.05
    assert min(advantages) < 0.0, "a truly unbiased estimator must sometimes go negative"


def test_membership_inference_rejects_tiny_groups() -> None:
    with pytest.raises(ValueError):
        MembershipInferenceAttacker().attack(np.array([1.0]), np.array([0.0, 1.0]))


# ── DP release of embeddings (what actually bounds re-identification) ────────


def test_clip_rows_bounds_the_norm() -> None:
    rng = np.random.default_rng(5)
    embeddings = rng.standard_normal((20, 8)) * 10.0
    clipped = clip_rows(embeddings, 1.5)
    assert np.all(np.linalg.norm(clipped, axis=1) <= 1.5 + 1e-9)
    # Rows already inside the ball are untouched.
    small = np.array([[0.1, 0.0]])
    assert np.allclose(clip_rows(small, 1.0), small)


def test_clip_rows_rejects_non_positive_norm() -> None:
    with pytest.raises(ValueError):
        clip_rows(np.ones((2, 2)), 0.0)


def test_dp_release_degrades_reidentification_as_epsilon_shrinks() -> None:
    """Tighter budget ⇒ more noise ⇒ the attacker does worse. The Phase 6 claim."""
    emb, subjects, views = _subject_views(16, 4, 16, spread=0.01, seed=6)

    def success(epsilon: float) -> float:
        rng = np.random.default_rng(7)
        released = release_embeddings_dp(
            emb, target_epsilon=epsilon, delta=1e-5, clip_norm=1.0, rng=rng
        )
        attacker = ReidentificationAttacker()
        attacker.enroll(released[views < 2], subjects[views < 2])
        return attacker.attack(released[views >= 2], subjects[views >= 2])

    tight, loose = success(0.5), success(512.0)
    assert tight < loose
    assert tight <= 2 * ReidentificationAttacker.chance_accuracy(16)
    assert loose > 0.5


def test_dp_release_bounds_the_released_norm() -> None:
    rng = np.random.default_rng(8)
    embeddings = rng.standard_normal((10, 6)) * 20.0
    released = release_embeddings_dp(
        embeddings, target_epsilon=512.0, delta=1e-5, clip_norm=1.0, rng=rng
    )
    # Tiny sigma at a huge epsilon: the release is essentially the clipped input.
    assert np.all(np.linalg.norm(released, axis=1) < 2.0)


def test_dp_release_uses_record_replacement_sensitivity() -> None:
    from privchain.privacy.accountant import get_noise_multiplier

    embeddings = np.array([[3.0, 4.0]], dtype=np.float64)
    epsilon, delta, clip_norm = 3.0, 1e-5, 2.0
    released = release_embeddings_dp(
        embeddings,
        target_epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        rng=np.random.default_rng(17),
    )
    clipped = clip_rows(embeddings, clip_norm)
    sigma = get_noise_multiplier(epsilon, sample_rate=1.0, steps=1, delta=delta)
    expected_noise = np.random.default_rng(17).normal(
        0.0, sigma * 2.0 * clip_norm, size=embeddings.shape
    )
    assert np.allclose(released, clipped + expected_noise)


def test_reidentification_errors() -> None:
    attacker = ReidentificationAttacker()
    with pytest.raises(RuntimeError):
        attacker.attack(np.zeros((1, 2)), np.zeros(1, dtype=int))
    with pytest.raises(ValueError):
        attacker.enroll(np.zeros((0, 2)), np.zeros(0, dtype=int))
