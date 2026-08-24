"""Multimodal fusion (Phase 1).

Combines per-modality embeddings into a single fused representation. A
per-sample ``presence`` mask is supported so the same module survives Phase 2's
heterogeneous, missing-modality federated clients: absent modalities are zeroed
rather than dropped, keeping the concatenated width fixed.

Two strategies:

* :class:`ConcatFusion` — concatenate, then project. Every modality reaches the
  classifier at full width whatever it contributes.
* :class:`GatedFusion` — score each modality and scale it before concatenating.
  On DAIC-WOZ audio alone scores near chance (~0.53 ROC-AUC against ~0.71 for
  text alone), so a fixed-width concatenation forces the classifier to spend
  capacity suppressing a branch that mostly carries noise. A gate lets the model
  learn that suppression once, in one scalar per modality per sample, instead of
  re-deriving it inside the projection.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Concatenate per-modality embeddings, then project.

    Args:
        modality_dims: Ordered mapping ``{modality: embedding_dim}``.
        hidden_dim: Output dimension of the fusion projection.
        dropout: Dropout applied after the projection.
    """

    def __init__(
        self, modality_dims: dict[str, int], hidden_dim: int, dropout: float = 0.0
    ) -> None:
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
        projected: torch.Tensor = self.net(fused)
        return projected


class GatedFusion(nn.Module):
    """Scale each modality by a learned gate, then concatenate and project.

    The gate is a per-modality scalar in ``(0, 1)`` predicted from that
    modality's own embedding, so a branch can be attenuated per sample rather
    than globally: a usable audio session and a noisy one need not share a
    weight. Gates are exposed on :attr:`last_gates` so a run can report what the
    model actually learned to trust instead of assuming.

    Absent modalities are gated to exactly zero, which keeps the ``presence``
    contract identical to :class:`ConcatFusion` — a missing branch must not be
    able to leak through a nonzero gate.

    Args:
        modality_dims: Ordered mapping ``{modality: embedding_dim}``.
        hidden_dim: Output dimension of the fusion projection.
        dropout: Dropout applied after the projection.
    """

    def __init__(
        self, modality_dims: dict[str, int], hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.modalities = list(modality_dims)
        self.gates = nn.ModuleDict(
            {modality: nn.Linear(dim, 1) for modality, dim in modality_dims.items()}
        )
        total = sum(modality_dims.values())
        self.net = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim
        self.last_gates: dict[str, torch.Tensor] = {}

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        presence: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Fuse per-modality embeddings into ``(B, hidden_dim)``.

        Args:
            embeddings: Mapping ``{modality: (B, dim)}`` for every modality.
            presence: Optional ``{modality: (B,)}`` 0/1 flags; absent modalities
                are gated to zero.

        Returns:
            Fused tensor of shape ``(B, hidden_dim)``.
        """
        parts: list[torch.Tensor] = []
        gates: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            emb = embeddings[modality]
            gate = torch.sigmoid(self.gates[modality](emb))  # (B, 1)
            if presence is not None:
                gate = gate * presence[modality].unsqueeze(-1).to(gate.dtype)
            gates[modality] = gate.detach().squeeze(-1)
            parts.append(emb * gate)
        self.last_gates = gates
        projected: torch.Tensor = self.net(torch.cat(parts, dim=-1))
        return projected


def build_fusion(modality_dims: dict[str, int], config: Any) -> ConcatFusion | GatedFusion:
    """Construct the fusion module named by ``config.type``.

    Args:
        modality_dims: Ordered mapping ``{modality: embedding_dim}``.
        config: Validated fusion configuration.

    Returns:
        The configured fusion module.

    Raises:
        ValueError: If ``config.type`` is unknown.
    """
    if config.type == "concat":
        return ConcatFusion(modality_dims, config.hidden_dim, config.dropout)
    if config.type == "gated":
        return GatedFusion(modality_dims, config.hidden_dim, config.dropout)
    raise ValueError(f"unknown fusion type {config.type!r}; expected concat or gated")
