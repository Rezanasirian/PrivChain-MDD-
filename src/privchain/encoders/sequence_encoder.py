"""Shared masked sequence encoder for all modalities.

Phase 1 (Centralized Multimodal Baseline). Each modality arrives as a padded
float sequence ``(B, T, input_dim)`` plus true per-sample ``lengths``, and leaves
as a fixed-size embedding ``(B, out_dim)``. Three encoder types share that
contract (``EncoderConfig.type``):

* ``stats`` — session-level statistical functionals (:func:`masked_statistics`)
  followed by an MLP. This is the AVEC2017 DAIC-WOZ baseline representation and
  the default for real-data runs: with 107 training sessions a recurrent model
  over thousands of timesteps does not fit (ADR-0012).
* ``mean`` — learned projection, then masked mean-pool.
* ``gru`` — bidirectional recurrence, then masked mean-pool.

The same building block backs the audio, video, and text encoders so they share
masking/pooling behaviour; modality-specific subclasses live in
:mod:`privchain.encoders.audio`, ``.video``, and ``.text``.

Two DP-driven implementation constraints shape this module (Phase 3, H1 — see
ADR-0004):

* the recurrent layer is :class:`opacus.layers.DPGRU`, because
  :class:`opacus.GradSampleModule` cannot produce per-sample gradients for
  ``nn.GRU``; and
* the sequence is **not** packed with ``pack_padded_sequence``. Packing collapses
  the batch dimension, which makes Opacus attribute per-sample gradients to the
  wrong samples *silently* — no error, wrong numbers, invalid DP guarantees.

Simply dropping the packing is not enough on its own: a bidirectional RNN run
over a padded tensor starts its backward direction inside another sample's
padding, so an embedding would depend on which samples share its batch. The
bidirectional case is therefore built from **two unidirectional** DPGRUs, the
second one fed the per-sample reversed *valid prefix* (:func:`reverse_padded`).
A unidirectional pass cannot see trailing padding, so each sample's embedding is
independent of its batch mates and numerically equal to the packed formulation —
verified against ``nn.GRU`` + ``pack_padded_sequence`` in the encoder tests.
"""

from __future__ import annotations

import torch
from opacus.layers import DPGRU
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


def masked_statistics(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Summarize a whole session into fixed-size statistical functionals.

    Computes, per feature channel and over valid timesteps only, the **mean**,
    **standard deviation**, **min**, **max**, and **mean absolute first
    difference** (a coarse measure of temporal variability). This is the
    "functionals over low-level descriptors" representation used by the AVEC2017
    DAIC-WOZ baseline, and it is what makes the model tractable at this dataset
    size — 107 training sessions cannot support a recurrent model over thousands
    of timesteps (ADR-0012).

    Every statistic is computed independently per sample from that sample's own
    valid prefix, so an embedding never depends on its batch mates — the same
    property the DP path requires of the recurrent encoders.

    Args:
        features: Padded tensor of shape ``(B, T, D)``.
        lengths: True per-sample lengths, shape ``(B,)`` (all ``>= 1``).

    Returns:
        Tensor of shape ``(B, 5 * D)``.
    """
    time_steps = features.shape[1]
    idx = torch.arange(time_steps, device=features.device).unsqueeze(0)  # (1, T)
    valid = idx < lengths.unsqueeze(1)  # (B, T)
    mask = valid.unsqueeze(-1).to(features.dtype)  # (B, T, 1)
    count = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

    mean = (features * mask).sum(dim=1) / count
    variance = (((features - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
    std = torch.sqrt(variance + 1e-8)

    # Padding must not win an extremum, so push it to the opposite infinity.
    minimum = features.masked_fill(~valid.unsqueeze(-1), float("inf")).min(dim=1).values
    maximum = features.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values

    # First differences are valid only where both endpoints are inside the prefix.
    deltas = (features[:, 1:] - features[:, :-1]).abs()
    delta_mask = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1).to(features.dtype)
    delta_count = delta_mask.sum(dim=1).clamp(min=1.0)
    delta_mean = (deltas * delta_mask).sum(dim=1) / delta_count

    return torch.cat([mean, std, minimum, maximum, delta_mean], dim=-1)


def reverse_padded(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sample's valid prefix in place, leaving right-padding put.

    ``[a, b, c, 0, 0]`` with ``length=3`` becomes ``[c, b, a, 0, 0]``. This is
    what lets a unidirectional RNN reproduce the backward direction of a packed
    bidirectional one.

    Args:
        sequence: Padded tensor of shape ``(B, T, D)``.
        lengths: True per-sample lengths, shape ``(B,)``.

    Returns:
        A tensor of the same shape with each valid prefix reversed.
    """
    time_steps = sequence.shape[1]
    idx = torch.arange(time_steps, device=sequence.device).unsqueeze(0)  # (1, T)
    mirrored = lengths.unsqueeze(1) - 1 - idx  # (B, T); negative inside padding
    gather_idx = torch.where(mirrored >= 0, mirrored, idx)
    return torch.gather(sequence, 1, gather_idx.unsqueeze(-1).expand_as(sequence))


class SequenceEncoder(nn.Module):
    """Project → (optional GRU) → masked mean-pool → output projection.

    Args:
        input_dim: Feature dimension of the incoming sequence.
        config: Encoder hyperparameters (type, dims, dropout).
    """

    # Number of functionals produced per feature channel by `masked_statistics`.
    NUM_STATISTICS = 5

    def __init__(self, input_dim: int, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        # The `stats` encoder summarizes the raw sequence *before* projecting, so
        # its projection consumes the concatenated functionals instead.
        proj_in = input_dim * self.NUM_STATISTICS if config.type == "stats" else input_dim
        self.proj = nn.Linear(proj_in, config.hidden_dim)

        if config.type == "gru":
            # Two unidirectional GRUs rather than one bidirectional one — see the
            # module docstring for why the direction is unrolled by hand.
            self.rnn: DPGRU | None = DPGRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                batch_first=True,
                bidirectional=False,
            )
            self.rnn_reverse: DPGRU | None = (
                DPGRU(
                    input_size=config.hidden_dim,
                    hidden_size=config.hidden_dim,
                    batch_first=True,
                    bidirectional=False,
                )
                if config.bidirectional
                else None
            )
            pooled_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        else:
            self.rnn = None
            self.rnn_reverse = None
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
        if self.config.type == "stats":
            # Pool first: the session collapses to functionals, then an MLP.
            summary = masked_statistics(sequence, lengths)  # (B, 5 * input_dim)
            pooled = torch.relu(self.proj(summary))  # (B, hidden_dim)
            stats_encoded: torch.Tensor = self.out(self.dropout(pooled))
            return stats_encoded

        hidden = self.proj(sequence)  # (B, T, hidden_dim)

        if self.rnn is not None:
            # Zero the padded steps so they cannot contribute state, then run the
            # directions separately (unpacked — see the module docstring).
            time_index = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            valid = (time_index < lengths.unsqueeze(1)).unsqueeze(-1).to(hidden.dtype)
            masked = hidden * valid

            forward_out, _ = self.rnn(masked)
            if self.rnn_reverse is not None:
                reversed_out, _ = self.rnn_reverse(reverse_padded(masked, lengths))
                # Undo the reversal so timesteps line up before pooling.
                backward_out = reverse_padded(reversed_out, lengths)
                hidden = torch.cat([forward_out, backward_out], dim=-1)
            else:
                hidden = forward_out

        pooled = masked_mean(hidden, lengths)
        encoded: torch.Tensor = self.out(self.dropout(pooled))
        return encoded
