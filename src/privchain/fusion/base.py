"""Shared contract for the depression-detection models (Phase 1, ADR-0027).

Two architectures now satisfy the same interface — the committed
encode-then-fuse baseline and the segment-aligned gated network — and everything
downstream (the trainer, the DP wrapper, the federated client, the attacker
evaluations) must accept either without caring which it holds.

An abstract base class rather than a ``Protocol``: the call sites also use
``state_dict``, ``to``, ``named_parameters`` and the rest of the ``nn.Module``
surface, and a Protocol would have to redeclare all of it to satisfy
``mypy --strict``. Subclassing ``nn.Module`` once gives that for free.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from privchain.data.mock_daic_woz import Batch


class DepressionModelBase(nn.Module):
    """A model mapping a collated batch to prediction heads.

    Subclasses must keep every modality-specific parameter under an
    ``encoders.<modality>.`` prefix. That naming is not cosmetic: per-modality DP
    budgets (:func:`privchain.privacy.dp_sgd.map_parameter_groups`) and
    capability-aware aggregation (:func:`privchain.federated.capability.param_group`)
    both classify parameters by that prefix, so a modality-specific tensor placed
    anywhere else is charged to the wrong privacy group and averaged over clients
    that never trained it.

    Attributes:
        input_dims: Per-modality input feature widths. Read by the distillation
            anchors, which have to synthesize inputs of the right shape.
        encoders: The per-modality encoders, keyed by modality name. The
            re-identification evaluation runs one of them on its own.
    """

    input_dims: dict[str, int]
    encoders: nn.ModuleDict

    @abstractmethod
    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass over a collated batch.

        Args:
            batch: A collated :class:`~privchain.data.mock_daic_woz.Batch`.
            presence: Optional per-modality 0/1 presence mask, overriding the
                batch's own flags (used by modality dropout and by ablations).

        Returns:
            Dict with ``logit`` ``(B,)`` always present, and ``phq_pred`` ``(B,)``
            when PHQ-8 regression is enabled.
        """
