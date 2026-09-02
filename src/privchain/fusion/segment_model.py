"""Segment-Aware Gated Multimodal Network (Phase 1, objective H4, ADR-0027).

The committed baseline encodes each modality into one session vector and only
then fuses, so the model can never say "trust the voice *here* and the words
*there*". This network inverts that order:

1. every modality arrives as ``K`` aligned segments (see
   :mod:`privchain.data.segment_alignment`);
2. a per-modality encoder projects each segment and scores how much that modality
   is worth listening to in it, given the embedding **and** the segment's
   measured quality;
3. the modalities compete through a masked softmax and are summed **per
   segment**;
4. additive attention over the fused segment sequence produces the session
   vector, which feeds the same two heads as the baseline.

Kept deliberately small: a projection and a gate head per modality, one attention
scorer, two heads. With 107 training sessions and DP-SGD noise on every
parameter, a tri-modal transformer or multi-layer cross-attention would be
fitting noise on a budget this corpus cannot pay.

Parameter naming follows the contract in
:class:`~privchain.fusion.base.DepressionModelBase`: everything modality-specific
lives under ``encoders.<modality>.``, so per-modality DP budgets and
capability-aware aggregation keep working unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from privchain.config import ModelConfig
from privchain.data.mock_daic_woz import MODALITIES, QUALITY_VALID_CHANNEL, Batch
from privchain.encoders.sequence_encoder import AdditiveAttentionPool, sinusoidal_positions
from privchain.fusion.base import DepressionModelBase
from privchain.fusion.quality_gated_fusion import QualityGatedFusion


class SegmentEncoder(nn.Module):
    """Project one modality's segments and score its per-segment usefulness.

    The gate head lives here rather than in the fusion module so that every
    parameter reading this modality's data stays inside its own DP/aggregation
    group (see the module docstring).

    Args:
        input_dim: Width of one segment's features for this modality.
        quality_dim: Width of this modality's per-segment quality vector.
        hidden_dim: Width of the projection's hidden layer.
        out_dim: Embedding width shared by all modalities (they are summed).
        dropout: Dropout applied inside the projection.
    """

    def __init__(
        self,
        input_dim: int,
        quality_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # Scores from the embedding *and* the quality vector: a thin segment and
        # a rich one look alike once projected, and only the quality channels say
        # which is which.
        self.gate = nn.Sequential(
            nn.Linear(out_dim + quality_dim, out_dim // 2 or 1),
            nn.Tanh(),
            nn.Linear(out_dim // 2 or 1, 1),
        )

    def forward(
        self, segments: torch.Tensor, quality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(B, K, input_dim)`` segments into embeddings and gate scores.

        Args:
            segments: This modality's per-segment features.
            quality: This modality's per-segment quality vectors ``(B, K, Q)``.

        Returns:
            ``(embeddings (B, K, out_dim), scores (B, K))``.
        """
        embedded = self.project(segments)
        scored: torch.Tensor = self.gate(torch.cat([embedded, quality], dim=-1)).squeeze(-1)
        return embedded, scored


class SegmentGatedNetwork(DepressionModelBase):
    """Per-segment quality-gated fusion followed by temporal attention.

    Args:
        input_dims: Per-modality segment feature widths.
        config: Validated model configuration.
        quality_dims: Per-modality quality-vector widths. Defaults to the
            project-wide layout in
            :data:`privchain.data.segment_alignment.QUALITY_DIMS`.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        config: ModelConfig,
        quality_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        # Imported here rather than at module scope so the model layer does not
        # depend on the data layer's parsing stack.
        from privchain.data.segment_alignment import QUALITY_DIMS

        self.config = config
        self.input_dims = dict(input_dims)
        self.quality_dims = dict(quality_dims or QUALITY_DIMS)

        encoder_configs = {modality: config.encoder_for(modality) for modality in MODALITIES}
        out_dims = {modality: encoder_configs[modality].out_dim for modality in MODALITIES}
        if len(set(out_dims.values())) != 1:
            raise ValueError(
                "segment_gated fusion sums the modalities, so every encoder must share "
                f"one out_dim; got {out_dims}"
            )
        embed_dim = next(iter(out_dims.values()))

        self.encoders = nn.ModuleDict(
            {
                modality: SegmentEncoder(
                    input_dims[modality],
                    self.quality_dims[modality],
                    encoder_configs[modality].hidden_dim,
                    encoder_configs[modality].out_dim,
                    encoder_configs[modality].dropout,
                )
                for modality in MODALITIES
            }
        )
        self.fusion = QualityGatedFusion(
            list(MODALITIES), embed_dim, config.fusion.hidden_dim, config.fusion.dropout
        )
        self.temporal = AdditiveAttentionPool(
            config.fusion.hidden_dim, config.temporal.attention_dim
        )
        self.temporal_dropout = nn.Dropout(config.temporal.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion.hidden_dim, config.head.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.head.dropout),
            nn.Linear(config.head.hidden_dim, 1),
        )
        self.regressor: nn.Linear | None = (
            nn.Linear(config.fusion.hidden_dim, 1) if config.use_phq_regression else None
        )

    def _segment_validity(
        self, batch: Batch, presence: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Combine sample-level presence with each segment's own ``valid`` flag.

        A modality is usable at segment ``k`` only if the client holds it at all
        *and* that segment actually contains data for it. A batch carrying no
        quality (the synthesized distillation anchors) is treated as valid
        everywhere, which is the only reading that leaves those paths unchanged.

        Args:
            batch: The collated batch.
            presence: Effective per-modality presence flags, shape ``(B,)``.

        Returns:
            ``{modality: (B, K)}`` float flags.
        """
        quality = batch.get("quality")
        length = batch["text"].shape[1]
        flags: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            held = presence[modality].unsqueeze(-1).to(batch["text"].dtype)  # (B, 1)
            if quality is None:
                flags[modality] = held.expand(-1, length)
            else:
                valid = quality[modality][..., QUALITY_VALID_CHANNEL]
                flags[modality] = valid * held
        return flags

    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass over a collated, segment-aligned batch.

        Args:
            batch: A collated batch whose three modalities share a segment count.
            presence: Optional per-modality 0/1 presence mask.

        Returns:
            Dict with ``logit`` ``(B,)`` and, when enabled, ``phq_pred`` ``(B,)``.

        Raises:
            ValueError: If the modalities disagree on the segment count — the
                alignment this architecture rests on would be silently untrue.
        """
        lengths = {modality: batch[modality].shape[1] for modality in MODALITIES}  # type: ignore[literal-required]
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"segment_gated needs one segment count across modalities, got {lengths}. "
                "Set daic_woz.segments.enabled to build aligned segments."
            )

        effective = batch["presence"] if presence is None else presence
        quality = batch.get("quality")
        valid = self._segment_validity(batch, effective)

        embeddings: dict[str, torch.Tensor] = {}
        scores: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            features = batch[modality]  # type: ignore[literal-required]
            if quality is None:
                # No measured quality: feed the gate a constant "present and
                # valid" vector so its input width still matches.
                shape = (*features.shape[:2], self.quality_dims[modality])
                modality_quality = torch.ones(shape, dtype=features.dtype, device=features.device)
            else:
                modality_quality = quality[modality]
            embedded, score = self.encoders[modality](features, modality_quality)
            # An absent modality must not leak through its embedding either.
            embeddings[modality] = embedded * valid[modality].unsqueeze(-1)
            scores[modality] = score

        fused, any_valid = self.fusion(embeddings, scores, valid)
        if self.config.temporal.positional:
            fused = fused + sinusoidal_positions(fused.shape[1], fused.shape[2], fused.device)
        session = self.temporal(
            fused,
            torch.full((fused.shape[0],), fused.shape[1], dtype=torch.long, device=fused.device),
            valid=any_valid,
        )
        session = self.temporal_dropout(session)

        outputs: dict[str, torch.Tensor] = {"logit": self.classifier(session).squeeze(-1)}
        if self.regressor is not None:
            outputs["phq_pred"] = self.regressor(session).squeeze(-1)
        return outputs
