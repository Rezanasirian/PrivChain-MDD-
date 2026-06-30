"""Text encoder (Phase 1).

Consumes transcript feature sequences — random token embeddings on mock data, a
length-1 sequence holding a hashed/TF-IDF transcript vector on real DAIC-WOZ —
shaped ``(B, T, input_dim)`` and produces a fixed-size embedding. See
:class:`privchain.encoders.sequence_encoder.SequenceEncoder`.
"""

from __future__ import annotations

from privchain.config import EncoderConfig
from privchain.encoders.sequence_encoder import SequenceEncoder


class TextEncoder(SequenceEncoder):
    """Sequence encoder specialized (by construction) for the text modality."""

    def __init__(self, input_dim: int, config: EncoderConfig) -> None:
        super().__init__(input_dim, config)
