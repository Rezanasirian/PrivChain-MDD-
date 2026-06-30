"""Unit tests for the per-modality sequence encoders (Phase 1)."""

from __future__ import annotations

import pytest
import torch

from privchain.config import EncoderConfig
from privchain.encoders.audio import AudioEncoder
from privchain.encoders.sequence_encoder import SequenceEncoder, masked_mean
from privchain.encoders.text import TextEncoder
from privchain.encoders.video import VideoEncoder


def test_masked_mean_ignores_padding() -> None:
    features = torch.tensor(
        [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]]  # last frame is padding
    )
    lengths = torch.tensor([2])
    pooled = masked_mean(features, lengths)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


@pytest.mark.parametrize("encoder_type", ["mean", "gru"])
def test_sequence_encoder_output_shape(encoder_type: str) -> None:
    config = EncoderConfig(type=encoder_type, hidden_dim=16, out_dim=8, dropout=0.0)
    encoder = SequenceEncoder(input_dim=5, config=config)
    x = torch.randn(4, 7, 5)
    lengths = torch.tensor([7, 5, 3, 1])
    out = encoder(x, lengths)
    assert out.shape == (4, 8)


def test_padding_does_not_change_output() -> None:
    # With the mean encoder, extra padding beyond `length` must not change output.
    config = EncoderConfig(type="mean", hidden_dim=16, out_dim=8, dropout=0.0)
    encoder = SequenceEncoder(input_dim=3, config=config).eval()

    base = torch.randn(1, 4, 3)
    lengths = torch.tensor([4])
    padded = torch.cat([base, torch.randn(1, 3, 3)], dim=1)  # 3 garbage frames

    with torch.no_grad():
        a = encoder(base, lengths)
        b = encoder(padded, lengths)
    assert torch.allclose(a, b, atol=1e-6)


def test_modality_encoders_are_sequence_encoders() -> None:
    config = EncoderConfig(type="gru", hidden_dim=8, out_dim=4)
    for cls in (AudioEncoder, VideoEncoder, TextEncoder):
        enc = cls(input_dim=6, config=config)
        assert isinstance(enc, SequenceEncoder)
        out = enc(torch.randn(2, 5, 6), torch.tensor([5, 2]))
        assert out.shape == (2, 4)
