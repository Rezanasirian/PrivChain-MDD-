"""Capability-Conditioned Logit Mixture of Experts (Phase 1, objective H4, ADR-0028).

Every fusion in this project so far combines modalities in *embedding* space: an
absent modality is zeroed and the zero still travels through a projection whose
weights were fitted with that slot occupied, so it shifts the pre-activation.
The model can be taught to tolerate that, but it cannot be made to ignore it.

Here each modality produces its own **logit**, and a masked softmax over the
present modalities mixes them:

```
audio → branch → logit_a, score_a ┐
video → branch → logit_v, score_v ├→ w = masked_softmax(score, presence)
text  → branch → logit_t, score_t ┘   logit = Σ w_m · logit_m
```

An absent modality gets weight exactly zero and the remaining weights
renormalize, so an audio-only client is scored by its audio expert alone rather
than by a fusion that is quietly missing two thirds of its input. Late fusion at
the logit is also the cheapest thing that can do this: three scalar heads, no
cross-modal parameters to fit on 107 sessions under DP noise.

Every modality-specific parameter — branch, expert head and gate scorer — lives
under ``encoders.<modality>.``, so per-modality DP budgets and capability-aware
subgraph aggregation keep working with no new rule (see
:class:`~privchain.fusion.base.DepressionModelBase`).
"""

from __future__ import annotations

import torch
from torch import nn

from privchain.config import ModelConfig
from privchain.data.mock_daic_woz import MODALITIES, QUALITY_VALID_CHANNEL, Batch
from privchain.encoders.sequence_encoder import AdditiveAttentionPool, sinusoidal_positions
from privchain.fusion.base import DepressionModelBase

#: Value used to mask an absent modality's gate score before the softmax. A true
#: ``-inf`` propagates NaN once *every* modality is masked, which is reachable
#: (a sample with no valid segment at all), so the sentinel is large and finite
#: and the all-absent case is handled explicitly instead.
_MASK_SCORE = -1.0e9


class ModalityBranch(nn.Module):
    """One modality's private path from segments to a logit and a gate score.

    The expert head is deliberately a single linear layer on the pooled session
    vector. A deeper head per modality would triple the parameters that only ever
    see one modality's data — precisely the parameters DP noise hits hardest,
    since a capability-restricted client trains them on a fraction of the corpus.

    Args:
        input_dim: Width of one segment's features for this modality.
        quality_dim: Width of this modality's per-segment quality vector.
        hidden_dim: Width of the projection's hidden layer.
        out_dim: Session-embedding width.
        attention_dim: Width of the temporal attention scorer.
        gate_hidden_dim: Width of the gate scorer's hidden layer.
        dropout: Dropout applied inside the projection and before the heads.
        gate_bias: Initial bias on this modality's gate score.
    """

    def __init__(
        self,
        input_dim: int,
        quality_dim: int,
        hidden_dim: int,
        out_dim: int,
        attention_dim: int,
        gate_hidden_dim: int,
        dropout: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.pool = AdditiveAttentionPool(out_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.expert = nn.Linear(out_dim, 1)
        # The gate sees the pooled quality too: two sessions with similar
        # embeddings can differ in how much data they were built from, and only
        # the quality channels carry that.
        self.gate = nn.Sequential(
            nn.Linear(out_dim + quality_dim, gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(gate_hidden_dim, 1),
        )
        # Start the mixture from what the ladder already measured rather than
        # from a uniform prior the optimizer would have to unlearn (ADR-0028).
        # This is a recorded prior, not a swept hyperparameter.
        final = self.gate[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, gate_bias)

    def forward(
        self, segments: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool this modality's valid segments into a logit, a score and a vector.

        The pooled session vector is returned rather than recomputed by the
        caller: the optional PHQ-8 head needs it, and pooling twice would double
        this branch's dropout draws, making the two heads disagree about which
        segments they saw.

        Args:
            segments: ``(B, K, input_dim)`` per-segment features.
            quality: ``(B, K, Q)`` per-segment quality vectors.
            valid: ``(B, K)`` float flags marking usable segments.

        Returns:
            ``(logit (B,), score (B,), session (B, out_dim))``.
        """
        embedded = self.project(segments) * valid.unsqueeze(-1)
        lengths = torch.full(
            (segments.shape[0],), segments.shape[1], dtype=torch.long, device=segments.device
        )
        # The pool masks with `~valid`, so it needs the boolean view; the float
        # view is what zeroes embeddings and weights the quality average.
        session = self.dropout(self.pool(embedded, lengths, valid=valid.bool()))

        # Quality is averaged over the segments that actually carry data, so a
        # padded segment cannot dilute the summary toward "poor".
        weight = valid.unsqueeze(-1)
        quality_summary = (quality * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

        logit: torch.Tensor = self.expert(session).squeeze(-1)
        score: torch.Tensor = self.gate(torch.cat([session, quality_summary], dim=-1)).squeeze(-1)
        return logit, score, session


class CapabilityLogitMoE(DepressionModelBase):
    """Mix per-modality logits with a presence-masked softmax.

    Args:
        input_dims: Per-modality segment feature widths.
        config: Validated model configuration.
        quality_dims: Per-modality quality-vector widths. Defaults to the
            project-wide layout.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        config: ModelConfig,
        quality_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        from privchain.data.segment_alignment import QUALITY_DIMS

        self.config = config
        self.input_dims = dict(input_dims)
        self.quality_dims = dict(quality_dims or QUALITY_DIMS)

        encoder_configs = {modality: config.encoder_for(modality) for modality in MODALITIES}
        self.encoders = nn.ModuleDict(
            {
                modality: ModalityBranch(
                    input_dims[modality],
                    self.quality_dims[modality],
                    encoder_configs[modality].hidden_dim,
                    encoder_configs[modality].out_dim,
                    config.temporal.attention_dim,
                    config.moe.gate_hidden_dim,
                    encoder_configs[modality].dropout,
                    config.moe.gate_bias.get(modality, 0.0),
                )
                for modality in MODALITIES
            }
        )
        # PHQ-8 regression, when enabled, is mixed by the same weights: giving it
        # its own fusion would let the two heads disagree about which modality
        # they trust, for a term this ADR's runs set to zero anyway.
        self.regressors: nn.ModuleDict | None = (
            nn.ModuleDict(
                {
                    modality: nn.Linear(encoder_configs[modality].out_dim, 1)
                    for modality in MODALITIES
                }
            )
            if config.use_phq_regression
            else None
        )
        #: Gate weights from the last forward pass, ``{modality: (B,)}``. Read by
        #: the comparison script: an MoE that silently learned to ignore audio
        #: should be visible as such, not inferred from its accuracy.
        self.last_weights: dict[str, torch.Tensor] = {}

    def _segment_validity(
        self, batch: Batch, presence: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Combine sample-level presence with each segment's own ``valid`` flag.

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
            held = presence[modality].unsqueeze(-1).to(batch["text"].dtype)
            if quality is None:
                flags[modality] = held.expand(-1, length)
            else:
                flags[modality] = quality[modality][..., QUALITY_VALID_CHANNEL] * held
        return flags

    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Score a batch by mixing the present modalities' logits.

        Args:
            batch: A collated batch whose modalities share a segment count.
            presence: Optional per-modality 0/1 presence mask.

        Returns:
            Dict with ``logit`` ``(B,)`` and, when enabled, ``phq_pred`` ``(B,)``.

        Raises:
            ValueError: If the modalities disagree on the segment count.
        """
        lengths = {modality: batch[modality].shape[1] for modality in MODALITIES}  # type: ignore[literal-required]
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"capability_moe needs one segment count across modalities, got {lengths}. "
                "Set daic_woz.segments.enabled to build aligned segments."
            )

        effective = batch["presence"] if presence is None else presence
        quality = batch.get("quality")
        valid = self._segment_validity(batch, effective)

        logits: dict[str, torch.Tensor] = {}
        scores: dict[str, torch.Tensor] = {}
        sessions: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            features = batch[modality]  # type: ignore[literal-required]
            if quality is None:
                shape = (*features.shape[:2], self.quality_dims[modality])
                modality_quality = torch.ones(shape, dtype=features.dtype, device=features.device)
            else:
                modality_quality = quality[modality]
            if self.config.temporal.positional:
                features = features + sinusoidal_positions(
                    features.shape[1], features.shape[2], features.device
                )
            logit, score, session = self.encoders[modality](
                features, modality_quality, valid[modality]
            )
            logits[modality] = logit
            scores[modality] = score
            sessions[modality] = session

        weights = self._mixture_weights(scores, valid)
        self.last_weights = {m: w.detach() for m, w in weights.items()}

        fused = torch.stack([logits[m] * weights[m] for m in MODALITIES], dim=0).sum(dim=0)
        outputs: dict[str, torch.Tensor] = {"logit": fused}
        if self.regressors is not None:
            outputs["phq_pred"] = self._mixed_phq(sessions, weights)
        return outputs

    def _mixture_weights(
        self, scores: dict[str, torch.Tensor], valid: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Masked softmax over the modalities this sample actually holds.

        A modality counts as usable when at least one of its segments is valid,
        which folds sample-level presence and segment-level emptiness into one
        test. When a sample holds nothing usable, every weight is zero: the fused
        logit is then 0.0, a genuine "no evidence" prediction, rather than the
        NaN a softmax over an all-masked row would produce.

        Args:
            scores: ``{modality: (B,)}`` raw gate scores.
            valid: ``{modality: (B, K)}`` per-segment validity flags.

        Returns:
            ``{modality: (B,)}`` weights, summing to 1 per sample or to 0 for a
            sample with no usable modality.
        """
        usable = torch.stack([(valid[m].sum(dim=1) > 0) for m in MODALITIES], dim=-1)  # (B, M)
        raw = torch.stack([scores[m] for m in MODALITIES], dim=-1)  # (B, M)
        masked = raw.masked_fill(~usable, _MASK_SCORE)
        weights = torch.softmax(masked, dim=-1)
        # Zero out the rows that had nothing usable, which softmax made uniform.
        weights = weights * usable.any(dim=-1, keepdim=True).to(weights.dtype)
        weights = weights * usable.to(weights.dtype)
        return {modality: weights[:, i] for i, modality in enumerate(MODALITIES)}

    def _mixed_phq(
        self, sessions: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Mix the per-modality PHQ-8 predictions with the same gate weights.

        Args:
            sessions: ``{modality: (B, out_dim)}`` pooled session vectors.
            weights: Mixture weights from :meth:`_mixture_weights`.

        Returns:
            ``(B,)`` PHQ-8 predictions in normalized units.
        """
        assert self.regressors is not None
        parts = [
            self.regressors[modality](sessions[modality]).squeeze(-1) * weights[modality]
            for modality in MODALITIES
        ]
        return torch.stack(parts, dim=0).sum(dim=0)
