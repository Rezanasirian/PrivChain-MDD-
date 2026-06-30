"""Multimodal fusion (Phase 1).

Combines per-modality embeddings into a single fused representation. Phase 1
uses concatenation (all modalities present). A per-sample ``presence`` mask is
supported now so the same module survives into Phase 2's heterogeneous,
missing-modality federated clients: absent modalities are zeroed rather than
dropped, keeping the concatenated width fixed.
"""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Concatenate per-modality embeddings, then project.

    Args:
        modality_dims: Ordered mapping ``{modality: embedding_dim}``.
        hidden_dim: Output dimension of the fusion projection.
        dropout: Dropout applied after the projection.
    """

    def __init__(self, modality_dims: dict[str, int], hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.modalities = list(modality_dims)
        total = sum(modality_dims.values())
        self.net = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        presence: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Fuse per-modality embeddings into ``(B, hidden_dim)``.

        Args:
            embeddings: Mapping ``{modality: (B, dim)}`` for every configured
                modality.
            presence: Optional mapping ``{modality: (B,)}`` of 0/1 flags marking
                which samples actually carry that modality; absent modalities are
                zeroed before concatenation.

        Returns:
            Fused tensor of shape ``(B, hidden_dim)``.
        """
        parts: list[torch.Tensor] = []
        for modality in self.modalities:
            emb = embeddings[modality]
            if presence is not None:
                emb = emb * presence[modality].unsqueeze(-1).to(emb.dtype)
            parts.append(emb)
        fused = torch.cat(parts, dim=-1)
        return self.net(fused)
