"""Modality dropout: per sample, reproducible, training-only, non-mutating."""

from __future__ import annotations

import pytest
import torch

from privchain.config import ModalityDropoutConfig, ModalityPattern
from privchain.data.mock_daic_woz import MODALITIES, Batch
from privchain.training.modality_dropout import ModalityDropout

PATTERNS = [
    ModalityPattern(name="text_only", capability=[0, 0, 1], fraction=0.5),
    ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.5),
]


def _batch(size: int) -> Batch:
    batch: Batch = {  # type: ignore[typeddict-item]
        "presence": {m: torch.ones(size, dtype=torch.long) for m in MODALITIES},
        "label": torch.zeros(size, dtype=torch.long),
    }
    return batch


def _dropout(seed: int = 0, patterns: list[ModalityPattern] | None = None) -> ModalityDropout:
    config = ModalityDropoutConfig(enabled=True, patterns=patterns or PATTERNS)
    return ModalityDropout(config, seed)


def test_draw_is_per_sample_not_per_batch() -> None:
    """One capability for a whole batch would give only a handful of draws per epoch."""
    flags = _dropout()(_batch(64))
    audio = flags["audio"]
    assert 0.0 < float(audio.mean()) < 1.0, "every sample got the same pattern"


def test_seeded_draws_are_reproducible() -> None:
    first = _dropout(seed=7)(_batch(32))
    second = _dropout(seed=7)(_batch(32))
    for modality in MODALITIES:
        assert torch.equal(first[modality], second[modality])


def test_observed_mix_matches_configured_fractions() -> None:
    patterns = [
        ModalityPattern(name="text_only", capability=[0, 0, 1], fraction=0.3),
        ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.7),
    ]
    flags = _dropout(seed=3, patterns=patterns)(_batch(4000))
    # Audio is present exactly for the `full` pattern.
    assert float(flags["audio"].mean()) == pytest.approx(0.7, abs=0.03)
    assert float(flags["text"].mean()) == 1.0  # text is in both patterns


def test_input_batch_is_not_mutated() -> None:
    batch = _batch(16)
    original = {m: batch["presence"][m].clone() for m in MODALITIES}
    _dropout()(batch)
    for modality in MODALITIES:
        assert torch.equal(batch["presence"][modality], original[modality])


def test_a_modality_the_sample_never_had_stays_absent() -> None:
    batch = _batch(16)
    batch["presence"]["video"] = torch.zeros(16, dtype=torch.long)
    flags = _dropout()(batch)
    assert float(flags["video"].abs().sum()) == 0.0


def test_disabled_dropout_passes_presence_through() -> None:
    dropout = ModalityDropout(ModalityDropoutConfig(enabled=False), seed=0)
    batch = _batch(8)
    flags = dropout(batch)
    for modality in MODALITIES:
        assert torch.equal(flags[modality], batch["presence"][modality])


def test_enabled_without_patterns_is_rejected() -> None:
    with pytest.raises(ValueError, match="no patterns"):
        ModalityDropout(ModalityDropoutConfig(enabled=True, patterns=[]), seed=0)


def test_fractions_must_sum_to_one() -> None:
    config = ModalityDropoutConfig(
        enabled=True,
        patterns=[ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.5)],
    )
    with pytest.raises(ValueError, match="sum to 1"):
        config.validated()
