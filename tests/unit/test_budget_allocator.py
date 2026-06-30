"""Unit tests for per-modality budget allocation (Phase 3)."""

from __future__ import annotations

import pytest

from privchain.config import AllocationConfig, ModalityPrivacy
from privchain.privacy.budget_allocator import (
    PerModalityBudgetAllocator,
    allocate_target_epsilons,
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
