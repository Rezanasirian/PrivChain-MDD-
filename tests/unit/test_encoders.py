"""Unit tests for the per-modality sequence encoders (Phase 1)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from privchain.config import EncoderConfig
from privchain.encoders.audio import AudioEncoder
from privchain.encoders.sequence_encoder import (
    SequenceEncoder,
    masked_mean,
    masked_statistics,
    reverse_padded,
)
from privchain.encoders.text import TextEncoder
from privchain.encoders.video import VideoEncoder


def test_masked_mean_ignores_padding() -> None:
    features = torch.tensor(
        [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]]  # last frame is padding
    )
    lengths = torch.tensor([2])
    pooled = masked_mean(features, lengths)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_masked_statistics_computes_functionals_over_the_valid_prefix() -> None:
    """mean/std/min/max/delta are taken over real frames only, never padding."""
    features = torch.tensor([[[1.0], [3.0], [5.0], [-99.0], [-99.0]]])  # 2 padded frames
    lengths = torch.tensor([3])

    stats = masked_statistics(features, lengths)

    # One channel x 5 functionals, in order: mean, std, min, max, delta-mean.
    assert stats.shape == (1, 5)
    mean, std, minimum, maximum, delta = stats[0].tolist()
    assert mean == pytest.approx(3.0)  # (1+3+5)/3
    assert std == pytest.approx(1.632993, abs=1e-5)  # population std of [1,3,5]
    assert minimum == pytest.approx(1.0)  # padding's -99 must not win
    assert maximum == pytest.approx(5.0)
    assert delta == pytest.approx(2.0)  # mean(|3-1|, |5-3|)


def test_masked_statistics_ignores_extra_padding() -> None:
    """Appending padding to a sample must not change its functionals."""
    base = torch.randn(1, 6, 4)
    lengths = torch.tensor([6])
    padded = torch.cat([base, torch.full((1, 5, 4), 7.5)], dim=1)

    assert torch.allclose(
        masked_statistics(base, lengths), masked_statistics(padded, lengths), atol=1e-6
    )


def test_masked_statistics_handles_a_single_frame() -> None:
    """A length-1 session has zero spread and no first differences."""
    stats = masked_statistics(torch.tensor([[[2.0], [9.0]]]), torch.tensor([1]))
    mean, std, minimum, maximum, delta = stats[0].tolist()
    assert (mean, minimum, maximum) == pytest.approx((2.0, 2.0, 2.0))
    assert std == pytest.approx(0.0, abs=1e-4)
    assert delta == pytest.approx(0.0)


def test_stats_encoder_is_independent_of_batch_mates() -> None:
    """A stats embedding must not depend on what else shares its batch.

    The DP path (Phase 3) relies on this for correct per-sample gradients.
    """
    config = EncoderConfig(type="stats", hidden_dim=16, out_dim=8, dropout=0.0)
    encoder = SequenceEncoder(input_dim=3, config=config).eval()

    sample = torch.randn(1, 5, 3)
    alone = encoder(sample, torch.tensor([5]))

    batched = torch.cat([sample, torch.randn(1, 5, 3)], dim=0)
    together = encoder(batched, torch.tensor([5, 2]))[:1]

    assert torch.allclose(alone, together, atol=1e-6)


@pytest.mark.parametrize("encoder_type", ["mean", "gru", "stats"])
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


def test_reverse_padded_reverses_only_the_valid_prefix() -> None:
    sequence = torch.tensor([[[1.0], [2.0], [3.0], [0.0], [0.0]]])
    lengths = torch.tensor([3])
    reversed_seq = reverse_padded(sequence, lengths)
    assert torch.allclose(reversed_seq, torch.tensor([[[3.0], [2.0], [1.0], [0.0], [0.0]]]))
    # Applying it twice is the identity.
    assert torch.allclose(reverse_padded(reversed_seq, lengths), sequence)


def test_gru_embedding_is_independent_of_batch_mates() -> None:
    """Padding from other samples must not leak into an embedding (DP-critical)."""
    config = EncoderConfig(type="gru", hidden_dim=8, out_dim=6, dropout=0.0, bidirectional=True)
    encoder = SequenceEncoder(input_dim=5, config=config).eval()

    lengths = torch.tensor([12, 4, 9, 2, 7])
    batch = torch.randn(5, 12, 5)
    for row, length in enumerate(lengths):
        batch[row, length:] = 0.0

    with torch.no_grad():
        batched = encoder(batch, lengths)
        for row, length in enumerate(lengths):
            alone = encoder(batch[row : row + 1, :length], lengths[row : row + 1])
            assert torch.allclose(batched[row], alone[0], atol=1e-5)


def test_bidirectional_gru_matches_packed_reference() -> None:
    """The hand-unrolled bidirection must equal `nn.GRU` + `pack_padded_sequence`."""
    hidden_dim = 8
    config = EncoderConfig(
        type="gru", hidden_dim=hidden_dim, out_dim=6, dropout=0.0, bidirectional=True
    )
    encoder = SequenceEncoder(input_dim=5, config=config).eval()
    assert encoder.rnn is not None and encoder.rnn_reverse is not None

    reference = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
    with torch.no_grad():
        for suffix, source in (("", encoder.rnn), ("_reverse", encoder.rnn_reverse)):
            for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(reference, f"{kind}_l0{suffix}").copy_(getattr(source, f"{kind}_l0"))

        lengths = torch.tensor([10, 3, 7])
        sequence = torch.randn(3, 10, 5)
        for row, length in enumerate(lengths):
            sequence[row, length:] = 0.0

        projected = encoder.proj(sequence)
        time_index = torch.arange(sequence.shape[1]).unsqueeze(0)
        projected = projected * (time_index < lengths.unsqueeze(1)).unsqueeze(-1).float()
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = reference(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        expected = encoder.out(masked_mean(unpacked, lengths))

        assert torch.allclose(encoder(sequence, lengths), expected, atol=1e-5)


def test_modality_encoders_are_sequence_encoders() -> None:
    config = EncoderConfig(type="gru", hidden_dim=8, out_dim=4)
    for cls in (AudioEncoder, VideoEncoder, TextEncoder):
        enc = cls(input_dim=6, config=config)
        assert isinstance(enc, SequenceEncoder)
        out = enc(torch.randn(2, 5, 6), torch.tensor([5, 2]))
        assert out.shape == (2, 4)
