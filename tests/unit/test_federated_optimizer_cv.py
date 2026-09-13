"""Tests for the train-only federated optimization CV harness (Phase 4, H2).

The harness exists to keep an optimizer choice out of the official dev and test
splits, so what matters is that its arm grid is faithful to the CLI and that its
out-of-fold assembly scores every participant exactly once.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_module() -> ModuleType:
    """Import ``scripts/run_federated_optimizer_cv.py`` as a module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_federated_optimizer_cv.py"
    spec = importlib.util.spec_from_file_location("run_federated_optimizer_cv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_arm_grid_covers_every_combination() -> None:
    module = _load_module()

    arms = module._build_arms(["per_shard", "aggregate_counts"], ["adam", "sgd"], [0.001, 0.1])

    assert len(arms) == 8
    assert {arm.name for arm in arms} == {
        "fedavg_per_shard_adam_lr0.001",
        "fedavg_per_shard_adam_lr0.1",
        "fedavg_per_shard_sgd_lr0.001",
        "fedavg_per_shard_sgd_lr0.1",
        "fedavg_aggregate_counts_adam_lr0.001",
        "fedavg_aggregate_counts_adam_lr0.1",
        "fedavg_aggregate_counts_sgd_lr0.001",
        "fedavg_aggregate_counts_sgd_lr0.1",
    }
    one = next(arm for arm in arms if arm.name == "fedavg_aggregate_counts_sgd_lr0.1")
    assert (one.class_weight_mode, one.optimizer_name, one.learning_rate) == (
        "aggregate_counts",
        "sgd",
        0.1,
    )


def test_out_of_fold_assembly_places_each_score_once() -> None:
    module = _load_module()

    folds = [([2, 0], np.asarray([0.9, 0.1])), ([1, 3], np.asarray([0.4, 0.6]))]

    out = module._assemble_oof(folds, 4)

    np.testing.assert_allclose(out, [0.1, 0.4, 0.9, 0.6])


def test_out_of_fold_assembly_rejects_an_unscored_participant() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="never scored"):
        module._assemble_oof([([0, 1], np.asarray([0.2, 0.3]))], 3)


def test_out_of_fold_assembly_rejects_a_double_scored_participant() -> None:
    module = _load_module()

    folds = [([0, 1], np.asarray([0.2, 0.3])), ([1], np.asarray([0.4]))]

    with pytest.raises(RuntimeError, match="scored twice"):
        module._assemble_oof(folds, 2)
