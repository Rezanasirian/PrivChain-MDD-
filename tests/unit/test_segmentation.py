"""Unit tests for contiguous within-session segmentation (Phase 6, H5)."""

from __future__ import annotations

import pytest

from privchain.segmentation import contiguous_spans


@pytest.mark.parametrize(("total", "parts"), [(12, 4), (13, 4), (7, 7), (100, 6), (5, 1)])
def test_spans_partition_the_range_exactly(total: int, parts: int) -> None:
    spans = contiguous_spans(total, parts)
    assert len(spans) == parts
    assert spans[0][0] == 0
    assert spans[-1][1] == total
    # Contiguous and gapless: each span starts where the previous one stopped.
    assert all(prev[1] == nxt[0] for prev, nxt in zip(spans, spans[1:], strict=False))


@pytest.mark.parametrize(("total", "parts"), [(13, 4), (100, 6), (7, 3)])
def test_span_sizes_differ_by_at_most_one(total: int, parts: int) -> None:
    sizes = [stop - start for start, stop in contiguous_spans(total, parts)]
    assert min(sizes) >= 1
    assert max(sizes) - min(sizes) <= 1
    # The longer spans come first, so the split is deterministic.
    assert sizes == sorted(sizes, reverse=True)


def test_rejects_a_split_that_would_leave_an_empty_span() -> None:
    # A participant with 3 transcript turns cannot supply 6 attacker views.
    with pytest.raises(ValueError, match="cannot split 3 items into 6"):
        contiguous_spans(3, 6)


def test_rejects_non_positive_parts() -> None:
    with pytest.raises(ValueError, match="parts must be positive"):
        contiguous_spans(10, 0)
