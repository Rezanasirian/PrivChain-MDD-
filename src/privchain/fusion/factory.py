"""Model construction from config (Phase 1, ADR-0027).

There are two architectures now, and roughly fifty places that build a model.
Without a single factory, ``model.architecture: segment_gated`` would be honoured
by whichever script happened to be updated and silently ignored by the rest —
producing a comparison where some arms ran the new network and others ran the old
one under the new name. Every config-driven path calls this instead.
"""

from __future__ import annotations

from privchain.config import ModelConfig
from privchain.fusion.base import DepressionModelBase
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.fusion.segment_model import SegmentGatedNetwork


def build_depression_model(
    input_dims: dict[str, int],
    config: ModelConfig,
    quality_dims: dict[str, int] | None = None,
) -> DepressionModelBase:
    """Construct the model architecture named by ``config.architecture``.

    Args:
        input_dims: Per-modality input feature dimensions. In segment mode these
            are the per-segment widths reported by the dataset's
            ``feature_dims``, which already account for the functionals.
        config: Validated model configuration.
        quality_dims: Per-modality quality-vector widths (segment mode only).

    Returns:
        The configured model.

    Raises:
        ValueError: If ``config.architecture`` is unknown.
    """
    if config.architecture == "encode_then_fuse":
        return MultimodalDepressionModel(input_dims, config)
    if config.architecture == "segment_gated":
        return SegmentGatedNetwork(input_dims, config, quality_dims)
    raise ValueError(
        f"unknown model architecture {config.architecture!r}; "
        "expected encode_then_fuse or segment_gated"
    )


def require_baseline_architecture(config: ModelConfig, context: str) -> None:
    """Fail loudly where only the encode-then-fuse model is supported.

    Some evaluation paths (the re-identification attacker, the text-representation
    ladder) reach inside the baseline model's structure. Running them against a
    ``segment_gated`` config would not crash — it would quietly measure the wrong
    model, which is worse.

    Args:
        config: Validated model configuration.
        context: What the caller is, for the error message.

    Raises:
        ValueError: If the config asks for an unsupported architecture.
    """
    if config.architecture != "encode_then_fuse":
        raise ValueError(
            f"{context} supports only model.architecture=encode_then_fuse, "
            f"got {config.architecture!r}"
        )
