"""Contiguous within-session segmentation (Phase 6, objective H5).

A re-identification attacker needs several *views* of the same subject: it
enrols a template from some of them and probes with the others. DAIC-WOZ gives
one recording per participant, so the views have to come from inside that
recording — disjoint, contiguous stretches of the same session.

This lives at package root because both the data layer (splitting a transcript
into chunks of turns) and the eval layer (splitting a feature matrix into chunks
of frames) need the identical rule, and the data layer must not import from
eval.
"""

from __future__ import annotations


def contiguous_spans(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(total)`` into ``parts`` contiguous, non-empty spans.

    Sizes differ by at most one, with the longer spans first. Splitting by
    *count* rather than by timestamp keeps every span non-empty, which matters
    for transcripts: a participant can fall silent for minutes, so equal-duration
    bins would leave some with no turns at all.

    Args:
        total: Number of items to split.
        parts: Number of spans to produce.

    Returns:
        ``parts`` ``(start, stop)`` half-open index pairs covering ``range(total)``.

    Raises:
        ValueError: If ``parts`` is not positive, or ``total < parts`` (which
            would force an empty span).
    """
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    if total < parts:
        raise ValueError(f"cannot split {total} items into {parts} non-empty spans")

    base, remainder = divmod(total, parts)
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        stop = start + base + (1 if index < remainder else 0)
        spans.append((start, stop))
        start = stop
    return spans
