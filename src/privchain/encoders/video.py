"""Video encoder (Phase 1).

Consumes facial-feature sequences — random features on mock data, OpenFace
landmarks/AUs on real DAIC-WOZ — shaped ``(B, T, input_dim)`` and produces a
fixed-size embedding. See :class:`privchain.encoders.sequence_encoder.SequenceEncoder`.
"""

from __future__ import annotations

from privchain.config import EncoderConfig
from privchain.encoders.sequence_encoder import SequenceEncoder


class VideoEncoder(SequenceEncoder):
    """Sequence encoder specialized (by construction) for the video modality."""

    def __init__(self, input_dim: int, config: EncoderConfig) -> None:
        super().__init__(input_dim, config)
