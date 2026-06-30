"""Audio encoder (Phase 1).

Consumes acoustic feature sequences — log-mel features on mock data, COVAREP
features on real DAIC-WOZ — shaped ``(B, T, input_dim)`` and produces a
fixed-size embedding. See :class:`privchain.encoders.sequence_encoder.SequenceEncoder`.
"""

from __future__ import annotations

from privchain.config import EncoderConfig
from privchain.encoders.sequence_encoder import SequenceEncoder


class AudioEncoder(SequenceEncoder):
    """Sequence encoder specialized (by construction) for the audio modality."""

    def __init__(self, input_dim: int, config: EncoderConfig) -> None:
        super().__init__(input_dim, config)
