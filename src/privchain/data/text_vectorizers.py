"""Transcript text vectorizers (Phase 1).

Turns a participant's transcript into a fixed-size float vector so text fits the
same float-sequence contract as audio/video (as a length-1 sequence). The
default :class:`HashingTextVectorizer` is pure NumPy and needs no network or
pretrained model — important because the real-data path must run offline. A
TF-IDF option (sklearn) is provided for parity with the thesis plan.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens.

    Args:
        text: Raw transcript text.

    Returns:
        List of lowercase tokens.
    """
    return _TOKEN_RE.findall(text.lower())


class TextVectorizer(Protocol):
    """A callable mapping a transcript string to a fixed-size vector."""

    @property
    def dim(self) -> int:
        """The output vector dimension."""
        ...

    def transform(self, text: str) -> NDArray[np.float32]:
        """Vectorize one transcript into a ``(dim,)`` float32 vector."""
        ...


class HashingTextVectorizer:
    """Hashing-trick bag-of-words vectorizer (offline, deterministic).

    Tokens are hashed into ``dim`` buckets and L2-normalized. No vocabulary
    fitting is required, so the same instance works across train/dev/test.

    Args:
        dim: Number of hash buckets / output dimension.
        seed: Salt mixed into the token hash for reproducibility.
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        """The output vector dimension."""
        return self._dim

    def _bucket(self, token: str) -> int:
        """Map a token to a bucket index in ``[0, dim)`` deterministically."""
        salted = f"{self._seed}:{token}"
        return _stable_hash(salted) % self._dim

    def transform(self, text: str) -> NDArray[np.float32]:
        """Vectorize one transcript into a normalized ``(dim,)`` vector.

        Args:
            text: Raw transcript text.

        Returns:
            L2-normalized float32 vector of shape ``(dim,)``.
        """
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            vec[self._bucket(token)] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


def _stable_hash(text: str) -> int:
    """A small, process-stable string hash (FNV-1a, 64-bit).

    Python's built-in ``hash`` is salted per process, which would make features
    non-reproducible across runs; FNV-1a is deterministic.

    Args:
        text: Input string.

    Returns:
        A non-negative 64-bit hash.
    """
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
