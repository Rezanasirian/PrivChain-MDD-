"""Unit tests for transcript vectorizers and the vectorizer factory (Phase 1).

The transformer vectorizer is not exercised here: it needs pretrained weights
from the network, and CI must stay offline (CLAUDE.md §7). Its *selection* is
tested, and the embedding path itself is covered by the real-data runs recorded
in ADR-0014.
"""

from __future__ import annotations

import numpy as np
import pytest

from privchain.data.text_vectorizers import (
    HashingTextVectorizer,
    build_text_vectorizer,
    tokenize,
)


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("I don't Feel Great, really!") == ["i", "don't", "feel", "great", "really"]


def test_hashing_vectorizer_is_deterministic_and_normalized() -> None:
    vectorizer = HashingTextVectorizer(16, seed=3)
    first = vectorizer.transform("i feel tired and sad")
    second = vectorizer.transform("i feel tired and sad")

    assert first.shape == (16,)
    assert np.array_equal(first, second)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0)


def test_hashing_vectorizer_returns_zeros_for_empty_text() -> None:
    vector = HashingTextVectorizer(8).transform("   ")
    assert vector.shape == (8,)
    assert not vector.any()


def test_hashing_vectorizer_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        HashingTextVectorizer(0)


def test_factory_builds_the_hashing_vectorizer() -> None:
    vectorizer = build_text_vectorizer({"vectorizer": "hashing", "dim": 32}, seed=1)
    assert isinstance(vectorizer, HashingTextVectorizer)
    assert vectorizer.dim == 32


def test_factory_defaults_to_hashing_when_unspecified() -> None:
    assert isinstance(build_text_vectorizer({"dim": 4}), HashingTextVectorizer)


def test_factory_rejects_an_unknown_vectorizer() -> None:
    """`tfidf` was documented in config comments but never implemented.

    Falling back silently would put a different feature space behind the same
    config, so the factory refuses instead.
    """
    with pytest.raises(ValueError, match="tfidf"):
        build_text_vectorizer({"vectorizer": "tfidf", "dim": 8})
