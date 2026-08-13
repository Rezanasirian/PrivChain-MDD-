"""Unit tests for feature normalization modes (ADR-0019).

`session` normalization forces every participant's features to per-channel mean 0
and std 1, which deletes the between-subject differences both the model and the
attacker depend on. These tests pin that behaviour down so the choice stays a
visible, testable decision rather than a default nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from privchain.config import load_baseline_config
from privchain.data.daic_woz import apply_normalization
from privchain.training.protocol import build_splits


def _matrix(offset: float = 0.0, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (offset + scale * rng.standard_normal((50, 4))).astype(np.float32)


def test_session_mode_erases_between_subject_differences() -> None:
    """Two subjects on completely different scales become indistinguishable."""
    quiet = apply_normalization(_matrix(offset=0.0, scale=1.0), "session")
    loud = apply_normalization(_matrix(offset=100.0, scale=20.0), "session")

    for normalized in (quiet, loud):
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-3)


def test_corpus_mode_preserves_between_subject_differences() -> None:
    stats = (np.zeros((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32))
    quiet = apply_normalization(_matrix(offset=0.0), "corpus", stats)
    loud = apply_normalization(_matrix(offset=100.0), "corpus", stats)
    # The offset survives, which is exactly what `session` destroys.
    assert loud.mean() - quiet.mean() == pytest.approx(100.0, rel=1e-3)


def test_none_mode_is_the_identity() -> None:
    raw = _matrix(offset=7.0, scale=3.0)
    assert apply_normalization(raw, "none") is raw


def test_corpus_mode_applies_the_supplied_statistics() -> None:
    raw = _matrix(offset=5.0, scale=2.0)
    mean = np.full((1, 4), 5.0, dtype=np.float32)
    std = np.full((1, 4), 2.0, dtype=np.float32)
    normalized = apply_normalization(raw, "corpus", (mean, std))
    np.testing.assert_allclose(normalized, (raw - mean) / (std + 1e-6), rtol=1e-5)


def test_corpus_mode_requires_statistics() -> None:
    with pytest.raises(ValueError, match="fitted on the train split"):
        apply_normalization(_matrix(), "corpus")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown normalization mode"):
        apply_normalization(_matrix(), "zscore")


def test_constant_channel_does_not_divide_by_zero() -> None:
    constant = np.full((10, 3), 4.0, dtype=np.float32)
    assert np.isfinite(apply_normalization(constant, "session")).all()


def test_overrides_must_name_a_real_section(tmp_path: Path) -> None:
    """Overrides address sections under the file's top-level ``daic_woz`` key.

    Getting that nesting wrong silently produced a KeyError deep inside a sweep
    that had already been launched, so the mistake is now caught up front with a
    message naming the sections that do exist.
    """
    config_path = tmp_path / "daic.yaml"
    config_path.write_text("daic_woz:\n  audio: {}\n  video: {}\n", encoding="utf-8")
    base = load_baseline_config(Path("configs/baseline.yaml"))

    with pytest.raises(KeyError, match="unknown daic_woz section"):
        build_splits(
            base, config_path, daic_overrides={"audio_features": {"normalization": "none"}}
        )
