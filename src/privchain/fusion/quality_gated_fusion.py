"""Quality-aware competitive fusion (Phase 1, ADR-0027).

:class:`~privchain.fusion.multimodal_fusion.GatedFusion` scores each modality
independently with a sigmoid, so three weak branches can all be trusted at once
and nothing forces them to compete. This module makes the weights a **softmax
across modalities**: attention has to be spent somewhere, and a branch only gains
weight by being more useful than the others *on this sample*.

Two properties matter beyond that:

* **The score sees data quality, not just the embedding.** An encoder given six
  usable audio frames still emits a confident-looking 128-dimensional vector; the
  quality vector (frames kept, voiced ratio, tracker confidence, tokens spoken)
  is what tells the gate the difference between a quiet segment and a broken one.
* **Missing means zero, exactly.** An absent modality is masked out of the
  softmax and the remaining weights renormalize, which is the behaviour a
  heterogeneous federated population needs — a client with no video should train
  the same fusion the full clients do, not a differently-scaled one.

This module holds **no per-modality parameters** on purpose. The gate score for
modality *m* is produced inside ``encoders.<m>``, so DP budget grouping and
capability-aware aggregation keep charging it to that modality (see
:class:`~privchain.fusion.base.DepressionModelBase`).
"""

from __future__ import annotations

import torch
from torch import nn


class QualityGatedFusion(nn.Module):
    """Masked softmax over per-modality gate scores, then a weighted sum.

    Args:
        modalities: Modality names, in a fixed order.
        embed_dim: Width every modality's embedding shares (they are summed, so
            they must match).
        hidden_dim: Output width of the post-fusion projection.
        dropout: Dropout applied after the projection.
    """

    def __init__(
        self,
        modalities: list[str],
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim
        self.last_gates: dict[str, torch.Tensor] = {}

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        scores: dict[str, torch.Tensor],
        valid: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-modality embeddings at every timestep.

        Args:
            embeddings: ``{modality: (B, T, embed_dim)}``.
            scores: ``{modality: (B, T)}`` raw gate scores (pre-softmax).
            valid: ``{modality: (B, T)}`` 0/1 flags combining the modality's
                presence with the timestep's own validity.

        Returns:
            ``(fused, any_valid)`` where ``fused`` is ``(B, T, hidden_dim)`` and
            ``any_valid`` is a ``(B, T)`` boolean marking the timesteps at least
            one modality could speak for. A timestep with nothing valid fuses to
            zeros with zero gates — never ``NaN``, which is what a softmax over
            three ``-inf`` scores would produce, and which is reachable whenever
            a participant fell silent or modality dropout removed every branch.
        """
        stacked_scores = torch.stack([scores[m] for m in self.modalities], dim=-1)  # (B, T, M)
        stacked_valid = torch.stack([valid[m] for m in self.modalities], dim=-1).bool()
        stacked_embeddings = torch.stack(
            [embeddings[m] for m in self.modalities], dim=-2
        )  # (B, T, M, D)

        any_valid = stacked_valid.any(dim=-1)  # (B, T)
        masked = stacked_scores.masked_fill(~stacked_valid, float("-inf"))
        # Neutralize the all-invalid rows before the softmax rather than after:
        # once a NaN exists it propagates through the sum and into every gradient.
        masked = torch.where(any_valid.unsqueeze(-1), masked, torch.zeros_like(masked))
        weights = torch.softmax(masked, dim=-1) * any_valid.unsqueeze(-1).to(masked.dtype)

        self.last_gates = {
            modality: weights[..., index].detach()
            for index, modality in enumerate(self.modalities)
        }
        pooled = (stacked_embeddings * weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, D)
        fused: torch.Tensor = self.project(pooled) * any_valid.unsqueeze(-1).to(pooled.dtype)
        return fused, any_valid
