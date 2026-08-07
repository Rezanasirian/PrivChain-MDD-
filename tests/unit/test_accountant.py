"""Unit tests for the Opacus-backed RDP accountant (Phase 3)."""

from __future__ import annotations

import pytest

from privchain.privacy.accountant import (
    Mechanism,
    compose_epsilon,
    get_epsilon,
    get_noise_multiplier,
)


def test_matches_opacus_reference() -> None:
    """The wrapper must reproduce Opacus's own RDPAccountant number."""
    from opacus.accountants import RDPAccountant

    reference = RDPAccountant()
    for _ in range(100):
        reference.step(noise_multiplier=1.1, sample_rate=0.05)

    assert get_epsilon(1.1, 0.05, 100, 1e-5) == pytest.approx(
        reference.get_epsilon(delta=1e-5), rel=1e-6
    )


def test_epsilon_decreases_with_more_noise() -> None:
    assert get_epsilon(3.0, 0.01, 1000, 1e-5) < get_epsilon(0.8, 0.01, 1000, 1e-5)


def test_epsilon_increases_with_more_steps() -> None:
    assert get_epsilon(1.0, 0.01, 100, 1e-5) < get_epsilon(1.0, 0.01, 1000, 1e-5)


def test_zero_steps_is_zero_epsilon() -> None:
    assert get_epsilon(1.0, 0.01, 0, 1e-5) == 0.0


def test_noise_multiplier_round_trip() -> None:
    for target in (0.5, 1.0, 2.0, 8.0):
        sigma = get_noise_multiplier(target, 0.01, 1000, 1e-5)
        spent = get_epsilon(sigma, 0.01, 1000, 1e-5)
        assert spent <= target + 1e-2
        # And just below: a hair less noise should exceed the achieved epsilon.
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
        get_noise_multiplier(1.0, 0.01, 0, 1e-5)  # no steps to calibrate against
    with pytest.raises(ValueError):
        compose_epsilon([], 1.5)  # delta out of range


# ── composition (participant-level budget, ADR-0009) ────────────────────────


_AUDIO = Mechanism(noise_multiplier=1.5, sample_rate=0.05, steps=100, name="audio")
_VIDEO = Mechanism(noise_multiplier=1.0, sample_rate=0.05, steps=100, name="video")
_TEXT = Mechanism(noise_multiplier=0.8, sample_rate=0.05, steps=100, name="text")


def test_empty_composition_spends_nothing() -> None:
    assert compose_epsilon([], 1e-5) == 0.0


def test_single_mechanism_composition_matches_get_epsilon() -> None:
    assert compose_epsilon([_AUDIO], 1e-5) == pytest.approx(
        get_epsilon(_AUDIO.noise_multiplier, _AUDIO.sample_rate, _AUDIO.steps, 1e-5)
    )


def test_composition_exceeds_every_individual_mechanism() -> None:
    """A subject exposed to all three groups spends more than any single one."""
    composed = compose_epsilon([_AUDIO, _VIDEO, _TEXT], 1e-5)
    for mechanism in (_AUDIO, _VIDEO, _TEXT):
        individual = get_epsilon(
            mechanism.noise_multiplier, mechanism.sample_rate, mechanism.steps, 1e-5
        )
        assert composed > individual


def test_composition_is_tighter_than_naive_epsilon_sum() -> None:
    """Summing RDP curves beats summing epsilons — that is why this exists."""
    composed = compose_epsilon([_AUDIO, _VIDEO, _TEXT], 1e-5)
    naive = sum(
        get_epsilon(m.noise_multiplier, m.sample_rate, m.steps, 1e-5)
        for m in (_AUDIO, _VIDEO, _TEXT)
    )
    assert composed < naive


def test_composition_is_monotone_in_mechanism_count() -> None:
    two = compose_epsilon([_AUDIO, _VIDEO], 1e-5)
    three = compose_epsilon([_AUDIO, _VIDEO, _TEXT], 1e-5)
    assert three > two
