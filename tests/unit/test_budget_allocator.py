"""Unit tests for per-modality budget allocation (Phase 3)."""

from __future__ import annotations

import pytest

from privchain.config import AllocationConfig, ModalityPrivacy
from privchain.privacy.budget_allocator import (
    PerModalityBudgetAllocator,
    allocate_target_epsilons,
    scale_to_participant_epsilon,
)

_PER_MODALITY = {
    "audio": ModalityPrivacy(epsilon=2.0, reidentification_risk=0.9),
    "video": ModalityPrivacy(epsilon=4.0, reidentification_risk=0.6),
    "text": ModalityPrivacy(epsilon=8.0, reidentification_risk=0.3),
}


def test_explicit_allocation_uses_config_epsilons() -> None:
    targets = allocate_target_epsilons(AllocationConfig(mode="explicit"), _PER_MODALITY)
    assert targets == {"audio": 2.0, "video": 4.0, "text": 8.0}


def test_inverse_risk_gives_higher_risk_smaller_budget() -> None:
    alloc = AllocationConfig(mode="inverse_risk", total_epsilon=14.0, risk_sharpness=1.0)
    targets = allocate_target_epsilons(alloc, _PER_MODALITY)
    # Higher risk -> smaller epsilon.
    assert targets["audio"] < targets["video"] < targets["text"]
    # Budget is fully distributed.
    assert sum(targets.values()) == pytest.approx(14.0)


def test_uniform_when_gamma_zero() -> None:
    alloc = AllocationConfig(mode="inverse_risk", total_epsilon=9.0, risk_sharpness=0.0)
    targets = allocate_target_epsilons(alloc, _PER_MODALITY)
    for value in targets.values():
        assert value == pytest.approx(3.0)


def test_higher_risk_gets_more_noise() -> None:
    allocator = PerModalityBudgetAllocator.from_config(
        AllocationConfig(mode="explicit"),
        _PER_MODALITY,
        delta=1e-5,
        sample_rate=0.1,
        steps=200,
    )
    sigmas = allocator.noise_multipliers()
    # audio (eps=2, tight) needs more noise than text (eps=8, loose).
    assert sigmas["audio"] > sigmas["video"] > sigmas["text"]


def test_consumed_epsilon_grows_with_steps_and_respects_budget() -> None:
    allocator = PerModalityBudgetAllocator.from_config(
        AllocationConfig(mode="explicit"),
        _PER_MODALITY,
        delta=1e-5,
        sample_rate=0.1,
        steps=200,
    )
    early = allocator.consumed_epsilon(50)
    full = allocator.consumed_epsilon(200)
    assert early["audio"] < full["audio"]
    # At the planned horizon, consumption stays within the target budget.
    assert full["audio"] <= 2.0 + 1e-3
    assert full["text"] <= 8.0 + 1e-3


def test_inverse_risk_rejects_zero_risk() -> None:
    bad = {"audio": ModalityPrivacy(epsilon=2.0, reidentification_risk=0.0)}
    with pytest.raises(ValueError):
        allocate_target_epsilons(AllocationConfig(mode="inverse_risk"), bad)


# ── Matching arms on the composed participant budget (ADR-0018) ──────────────

# Each bisection step calibrates a sigma per modality through Opacus, so these
# stay deliberately small: the properties under test are structural, and a
# 200-step budget made the whole suite eight times slower for no extra coverage.
_MATCH_KWARGS = {"delta": 1e-5, "sample_rate": 0.1, "steps": 20}
_TARGET = 8.0
# The three shapes compared in scripts/run_allocation_comparison.py.
_SHAPES = {
    "uniform": {"audio": 1.0, "video": 1.0, "text": 1.0},
    "adaptive": {"audio": 1.0, "video": 1.0 / 0.97, "text": 1.0 / 0.29},
    "anti_adaptive": {"audio": 1.0, "video": 0.97, "text": 0.29},
}


def _participant_epsilon(epsilons: dict[str, float]) -> float:
    allocator = PerModalityBudgetAllocator(epsilons, dict.fromkeys(epsilons, 0.5), **_MATCH_KWARGS)
    return allocator.participant_epsilon(_MATCH_KWARGS["steps"])


def test_every_arm_lands_on_the_same_budget() -> None:
    """The premise of the whole comparison: matched arms, differing allocations."""
    scaled = {
        name: scale_to_participant_epsilon(shape, _TARGET, **_MATCH_KWARGS)
        for name, shape in _SHAPES.items()
    }
    achieved = {name: _participant_epsilon(eps) for name, eps in scaled.items()}
    for value in achieved.values():
        assert value == pytest.approx(_TARGET, rel=1e-2)
    assert max(achieved.values()) - min(achieved.values()) < 0.01 * _TARGET

    # Only the level changes; the ratios that define each allocation do not.
    for name, shape in _SHAPES.items():
        eps = scaled[name]
        assert eps["text"] / eps["audio"] == pytest.approx(shape["text"] / shape["audio"])
        assert eps["video"] / eps["audio"] == pytest.approx(shape["video"] / shape["audio"])


def test_matching_on_the_mean_would_not_have_been_equivalent() -> None:
    """Why the scaling exists: equal mean epsilon is not equal privacy.

    The uneven (adaptive) allocation is dominated by its loosest mechanism, so at
    equal *mean* budget it spends strictly more participant epsilon than the
    uniform arm — silently favouring the hypothesis under test.
    """
    uniform = dict.fromkeys(_SHAPES["adaptive"], 1.0)
    adaptive = _SHAPES["adaptive"]
    mean_adaptive = sum(adaptive.values()) / len(adaptive)

    matched_on_mean_uniform = dict.fromkeys(uniform, mean_adaptive)
    assert _participant_epsilon(adaptive) > _participant_epsilon(matched_on_mean_uniform)


def test_more_budget_means_a_larger_composed_epsilon() -> None:
    """Monotonicity, which is what makes the bisection valid."""
    small = scale_to_participant_epsilon(_SHAPES["uniform"], 4.0, **_MATCH_KWARGS)
    large = scale_to_participant_epsilon(_SHAPES["uniform"], 12.0, **_MATCH_KWARGS)
    assert small["audio"] < large["audio"]


@pytest.mark.parametrize(
    ("weights", "target"),
    [({}, 8.0), ({"audio": 0.0}, 8.0), ({"audio": 1.0}, 0.0)],
)
def test_scaling_rejects_degenerate_input(weights: dict[str, float], target: float) -> None:
    with pytest.raises(ValueError):
        scale_to_participant_epsilon(weights, target, **_MATCH_KWARGS)
