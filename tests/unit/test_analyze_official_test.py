"""Tests for the official-test paired analysis (Phase 7, H5).

Seeds are repetitions of one fit, so a participant's scores must be averaged
across them before the bootstrap sees that participant, and a seed that scored a
different cohort must not be silently mixed in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_module() -> ModuleType:
    """Import ``scripts/analyze_official_test.py`` as a module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_official_test.py"
    spec = importlib.util.spec_from_file_location("analyze_official_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scores_are_averaged_across_seeds_in_requested_order() -> None:
    module = _load_module()
    runs = [
        {"indices": [5, 2], "scores": [0.8, 0.2], "labels": [1, 0]},
        {"indices": [2, 5], "scores": [0.4, 0.6], "labels": [0, 1]},
    ]

    scores, labels = module.average_scores_over_seeds(runs, [2, 5])

    np.testing.assert_allclose(scores, [0.3, 0.7])
    np.testing.assert_array_equal(labels, [0, 1])


def test_a_seed_scoring_another_cohort_is_rejected() -> None:
    module = _load_module()
    runs = [
        {"indices": [1, 2], "scores": [0.5, 0.5], "labels": [0, 1]},
        {"indices": [1, 3], "scores": [0.5, 0.5], "labels": [0, 1]},
    ]

    with pytest.raises(ValueError, match="different participant set"):
        module.average_scores_over_seeds(runs, [1, 2])


def test_no_runs_is_an_error() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="no runs"):
        module.average_scores_over_seeds([], [1])
