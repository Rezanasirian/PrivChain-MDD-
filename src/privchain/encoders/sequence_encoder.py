"""Shared masked sequence encoder for all modalities.

Phase 1 (Centralized Multimodal Baseline). Each modality arrives as a padded
float sequence ``(B, T, input_dim)`` plus true per-sample ``lengths``. This
module projects, optionally runs a lightweight bidirectional GRU, and pools over
*valid* (unpadded) timesteps into a fixed-size embedding ``(B, out_dim)``.

The same building block backs the audio, video, and text encoders so they share
masking/pooling behaviour; modality-specific subclasses live in
:mod:`privchain.encoders.audio`, ``.video``, and ``.text``.
"""

from __future__ import annotations

import torch
from torch import nn

from privchain.config import EncoderConfig


def masked_mean(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool over valid timesteps, ignoring right-padding.

    Args:
        features: Tensor of shape ``(B, T, D)``.
        lengths: True per-sample lengths, shape ``(B,)``.

    Returns:
        Pooled tensor of shape ``(B, D)``.
    """
    batch, time_steps, _ = features.shape
    idx = torch.arange(time_steps, device=features.device).unsqueeze(0)  # (1, T)
    mask = (idx < lengths.unsqueeze(1)).unsqueeze(-1).to(features.dtype)  # (B, T, 1)
    summed = (features * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return summed / count


class SequenceEncoder(nn.Module):
    """Project → (optional GRU) → masked mean-pool → output projection.

    Args:
        input_dim: Feature dimension of the incoming sequence.
        config: Encoder hyperparameters (type, dims, dropout).
    """

    def __init__(self, input_dim: int, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(input_dim, config.hidden_dim)

        if config.type == "gru":
            self.rnn: nn.GRU | None = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                batch_first=True,
                bidirectional=config.bidirectional,
            )
            pooled_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        else:
            self.rnn = None
            pooled_dim = config.hidden_dim

        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(pooled_dim, config.out_dim)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a padded sequence into a fixed-size embedding.

        Args:
            sequence: Padded input, shape ``(B, T, input_dim)``.
            lengths: True per-sample lengths, shape ``(B,)`` (all ``>= 1``).

        Returns:
            Embedding tensor of shape ``(B, out_dim)``.
        """
        hidden = self.proj(sequence)  # (B, T, hidden_dim)

        if self.rnn is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                hidden, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        pooled = masked_mean(hidden, lengths)
        return self.out(self.dropout(pooled))
