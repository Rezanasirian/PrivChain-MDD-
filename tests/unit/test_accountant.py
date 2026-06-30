"""Unit tests for the RDP accountant (Phase 3)."""

from __future__ import annotations

import math

import pytest

from privchain.privacy.accountant import (
    get_epsilon,
    get_noise_multiplier,
    rdp_sampled_gaussian,
)


def test_full_batch_matches_gaussian_rdp() -> None:
    # q=1 -> non-subsampled Gaussian: RDP(alpha) = alpha / (2 sigma^2).
    for alpha, sigma in ((2, 1.0), (5, 2.0), (10, 0.5)):
        assert rdp_sampled_gaussian(alpha, 1.0, sigma) == pytest.approx(alpha / (2 * sigma**2))


def test_epsilon_decreases_with_more_noise() -> None:
    e_low = get_epsilon(0.8, 0.01, 1000, 1e-5)
    e_high = get_epsilon(3.0, 0.01, 1000, 1e-5)
    assert e_high < e_low


def test_epsilon_increases_with_more_steps() -> None:
    assert get_epsilon(1.0, 0.01, 100, 1e-5) < get_epsilon(1.0, 0.01, 1000, 1e-5)


def test_zero_steps_is_zero_epsilon() -> None:
    assert get_epsilon(1.0, 0.01, 0, 1e-5) == 0.0


def test_noise_multiplier_round_trip() -> None:
    for target in (0.5, 1.0, 2.0, 8.0):
        sigma = get_noise_multiplier(target, 0.01, 1000, 1e-5)
        spent = get_epsilon(sigma, 0.01, 1000, 1e-5)
        assert spent <= target + 1e-3
        # And just below: a hair less noise should exceed the target.
        assert get_epsilon(sigma * 0.9, 0.01, 1000, 1e-5) > spent


def test_tighter_budget_needs_more_noise() -> None:
    sigma_tight = get_noise_multiplier(1.0, 0.01, 1000, 1e-5)
    sigma_loose = get_noise_multiplier(8.0, 0.01, 1000, 1e-5)
    assert sigma_tight > sigma_loose


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        get_epsilon(1.0, 0.01, 100, 0.0)  # delta out of range
    with pytest.raises(ValueError):
        get_noise_multiplier(0.0, 0.01, 100, 1e-5)  # non-positive target
    with pytest.raises(ValueError):
        rdp_sampled_gaussian(1, 0.5, 1.0)  # alpha < 2


def test_no_noise_is_infinite_rdp() -> None:
    assert math.isinf(rdp_sampled_gaussian(2, 0.5, 0.0))
