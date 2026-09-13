"""Unit tests for the balanced membership pool and seed aggregation (Phase 6, H5).

Covers the two helpers ADR-0029 adds to ``scripts/run_attack_eval.py``: the
member / non-member draw that removes the official-split confound from the
membership-inference attack, and the summary that turns per-seed attack metrics
into a mean, a spread and an interval.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from torch.utils.data import Subset

from privchain.config import (
    AudioConfig,
    DataConfig,
    TextConfig,
    VideoConfig,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset


def _load_attack_eval() -> ModuleType:
    """Import ``scripts/run_attack_eval.py`` as a module.

    The attacker evaluation is a CLI script rather than a package module, so it
    is loaded by path instead of imported by name.

    Returns:
        The imported module.
    """
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_attack_eval.py"
    spec = importlib.util.spec_from_file_location("run_attack_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_attack_eval"] = module
    spec.loader.exec_module(module)
    return module


ATTACK_EVAL = _load_attack_eval()


def _dataset(num_sessions: int, seed: int) -> MockDaicWozDataset:
    """Build a small mock corpus for the split helpers."""
    config = DataConfig(
        num_sessions=num_sessions,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=8, min_frames=6, max_frames=8),
        video=VideoConfig(n_features=6, min_frames=4, max_frames=6),
        text=TextConfig(embed_dim=8, min_tokens=4, max_tokens=6),
    )
    return MockDaicWozDataset(config, seed=seed)


def test_pool_labels_unwraps_subsets() -> None:
    dataset = _dataset(10, seed=3)
    direct = ATTACK_EVAL._pool_labels(dataset)
    indices = [7, 1, 4]
    through_subset = ATTACK_EVAL._pool_labels(Subset(dataset, indices))
    assert through_subset == [direct[i] for i in indices]


def test_balanced_pool_is_disjoint_and_covers_every_participant() -> None:
    first, second = _dataset(12, seed=5), _dataset(8, seed=6)
    members, nonmembers = ATTACK_EVAL._balanced_membership_pool([first, second], seed=42)

    assert len(members) + len(nonmembers) == len(first) + len(second)
    assert abs(len(members) - len(nonmembers)) <= 1
    assert set(members.indices).isdisjoint(nonmembers.indices)


def test_balanced_pool_redraws_per_seed() -> None:
    dataset = _dataset(20, seed=11)
    first, _ = ATTACK_EVAL._balanced_membership_pool([dataset], seed=1)
    again, _ = ATTACK_EVAL._balanced_membership_pool([dataset], seed=1)
    other, _ = ATTACK_EVAL._balanced_membership_pool([dataset], seed=2)

    assert list(first.indices) == list(again.indices)
    assert list(first.indices) != list(other.indices)


def test_aggregate_over_seeds_reports_mean_spread_and_interval() -> None:
    rows = [{"auc": 0.4}, {"auc": 0.6}, {"auc": 0.5}]
    aggregate = ATTACK_EVAL._aggregate_over_seeds(rows)

    assert aggregate["num_seeds"] == 3.0
    assert aggregate["auc_mean"] == pytest.approx(0.5)
    assert aggregate["auc_std"] == pytest.approx(0.1)
    assert aggregate["auc_ci95_low"] < 0.5 < aggregate["auc_ci95_high"]


def test_aggregate_over_seeds_gives_a_single_seed_no_spread() -> None:
    aggregate = ATTACK_EVAL._aggregate_over_seeds([{"auc": 0.55, "advantage": 0.1}])

    assert aggregate["auc_std"] == 0.0
    assert aggregate["auc_ci95_low"] == pytest.approx(aggregate["auc_ci95_high"])
    assert aggregate["advantage_mean"] == pytest.approx(0.1)
