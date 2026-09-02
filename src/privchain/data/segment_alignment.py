"""Segment-aligned multimodal views of one interview (Phase 1, ADR-0027).

The committed pipeline reduces a ~15-minute session to one vector per modality:
the transcript becomes a single document embedding, audio and video become
session-wide functionals. Whatever the interview's shape carried — which answer
was about sleep, which about hopelessness, how the participant sounded *while*
saying it — is gone before the model sees anything.

This module cuts the session into ``K`` segments and gives all three modalities
the *same* cut, so segment ``k`` means the same stretch of the interview in every
branch. The rules are deliberate and asymmetric:

* **Grouping.** The participant's timestamped turns are split into ``K_eff``
  contiguous groups of turns (:func:`~privchain.segmentation.contiguous_spans`).
  A group's **envelope** runs from its first ``start_time`` to its last
  ``stop_time``. Envelopes are *not* a partition of the interview — consecutive
  envelopes can have a gap between them, and nothing here pretends otherwise.
* **Audio takes the union of the participant's own turn intervals**, not the
  envelope. An envelope also spans Ellie's questions, and attributing the
  interviewer's voice to the participant would corrupt the acoustic branch.
* **Video takes the whole envelope.** Facial behaviour while *listening* is
  plausibly informative, so the video branch keeps it on purpose.
* **Quality travels with the features.** Each segment carries a small per-modality
  quality vector whose channel 0 is ``valid``; the fusion gate reads it, so the
  model can learn to discount a segment where a modality is thin rather than
  being fed a confidently-wrong zero.
* **Quiet participants are padded, not dropped.** A participant with fewer turns
  than ``K`` yields ``K_eff = min(K, n_turns)`` real segments, right-padded to
  ``K`` with zeros and ``valid = 0``. Demanding ``K`` turns would drop the
  quietest participants, who are not a random subset in a depression corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from privchain.segmentation import contiguous_spans

#: Width of each modality's per-segment quality vector. Fixed and public: the
#: model sizes its gate layers from these, so they cannot drift with the loader.
QUALITY_DIMS: dict[str, int] = {"audio": 3, "video": 4, "text": 3}

#: Number of functionals :func:`segment_functionals` emits per feature channel.
NUM_FUNCTIONALS = 5


@dataclass(frozen=True)
class TimedTurn:
    """One participant utterance with its transcript timestamps.

    Attributes:
        start: Utterance start, in seconds from the session start.
        stop: Utterance end, in seconds.
        text: The utterance text.
    """

    start: float
    stop: float
    text: str


@dataclass(frozen=True)
class SegmentPlan:
    """The one segmentation every modality of a session consumes.

    Computing this once and sharing it is the point: if text grouped its turns
    while audio and video sliced their own frame counts, segment ``k`` would mean
    a different stretch of the interview in each branch and "aligned" would be a
    claim the data does not support.

    Attributes:
        count: The padded segment count ``K`` every modality returns.
        groups: The ``K_eff <= K`` non-empty turn groups, in order.
    """

    count: int
    groups: tuple[tuple[TimedTurn, ...], ...]

    @property
    def effective(self) -> int:
        """Number of non-empty segments (``K_eff``)."""
        return len(self.groups)

    def envelope(self, index: int) -> tuple[float, float]:
        """Return group ``index``'s ``(start, stop)`` envelope in seconds."""
        group = self.groups[index]
        return group[0].start, group[-1].stop

    def intervals(self, index: int) -> tuple[tuple[float, float], ...]:
        """Return group ``index``'s participant-only speech intervals."""
        return tuple((turn.start, turn.stop) for turn in self.groups[index])


def plan_segments(turns: Sequence[TimedTurn], count: int) -> SegmentPlan:
    """Group a participant's turns into at most ``count`` contiguous segments.

    Args:
        turns: The participant's turns, in chronological order.
        count: The desired segment count ``K``.

    Returns:
        The shared :class:`SegmentPlan`. A participant with no turns yields a
        plan with zero groups, which every builder below renders as ``K`` invalid
        segments.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count < 1:
        raise ValueError(f"segment count must be positive, got {count}")
    effective = min(count, len(turns))
    if effective == 0:
        return SegmentPlan(count=count, groups=())
    spans = contiguous_spans(len(turns), effective)
    return SegmentPlan(
        count=count,
        groups=tuple(tuple(turns[start:stop]) for start, stop in spans),
    )


def segment_functionals(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Summarize one segment's frames into functionals over each channel.

    Mean, standard deviation, min, max, and mean absolute first difference —
    the same five statistics :func:`privchain.encoders.sequence_encoder.masked_statistics`
    computes over a whole session, applied per segment instead. Keeping the two
    definitions identical is what makes the segment arms comparable to the
    session-level baseline.

    Args:
        matrix: This segment's frames, shape ``(T, D)`` with ``T`` possibly 0.

    Returns:
        Concatenated functionals of shape ``(5 * D,)``; all zeros when the
        segment has no frames.
    """
    channels = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.zeros(NUM_FUNCTIONALS * channels, dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    if matrix.shape[0] > 1:
        delta = np.abs(np.diff(matrix, axis=0)).mean(axis=0)
    else:
        # One frame has no first difference; zero is the honest value, not a
        # fabricated spread.
        delta = np.zeros(channels, dtype=matrix.dtype)
    return np.concatenate([mean, std, minimum, maximum, delta]).astype(np.float32)


def rows_within(
    timestamps: NDArray[np.float64], intervals: Sequence[tuple[float, float]]
) -> NDArray[np.bool_]:
    """Mask the rows whose timestamp falls inside any of ``intervals``.

    Args:
        timestamps: Per-row timestamps in seconds, shape ``(T,)``.
        intervals: Half-open ``[start, stop)`` windows in seconds.

    Returns:
        Boolean mask of shape ``(T,)``.
    """
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, stop in intervals:
        mask |= (timestamps >= start) & (timestamps < stop)
    return mask


def _quality_scalar(values: NDArray[np.float32] | None, mask: NDArray[np.bool_]) -> float:
    """Mean of ``values`` over ``mask``, or 0.0 when unavailable/empty."""
    if values is None or not mask.any():
        return 0.0
    return float(values[mask].mean())


def build_frame_segments(
    values: NDArray[np.float32],
    timestamps: NDArray[np.float64],
    plan: SegmentPlan,
    *,
    modality: str,
    use_envelope: bool,
    voiced: NDArray[np.float32] | None = None,
    confidence: NDArray[np.float32] | None = None,
    success: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Build one frame modality's per-segment features and quality.

    Args:
        values: The session's frames, shape ``(T, D)``, already normalized.
        timestamps: Per-row timestamps in seconds, shape ``(T,)``.
        plan: The shared segmentation.
        modality: ``"audio"`` or ``"video"`` (selects the quality layout).
        use_envelope: ``True`` to take every frame in the group's envelope
            (video), ``False`` to take only the participant's own speech
            intervals (audio).
        voiced: Optional per-row voiced/unvoiced flags, for audio quality.
        confidence: Optional per-row tracker confidence, for video quality.
        success: Optional per-row tracking-success flags, for video quality.

    Returns:
        ``(features, quality)`` of shape ``(K, 5 * D)`` and ``(K, Q_m)``.

    Raises:
        ValueError: If ``modality`` has no defined quality layout.
    """
    if modality not in QUALITY_DIMS:
        raise ValueError(f"no quality layout for modality {modality!r}")

    width = NUM_FUNCTIONALS * values.shape[1]
    features = np.zeros((plan.count, width), dtype=np.float32)
    quality = np.zeros((plan.count, QUALITY_DIMS[modality]), dtype=np.float32)

    for index in range(plan.effective):
        windows = [plan.envelope(index)] if use_envelope else list(plan.intervals(index))
        mask = rows_within(timestamps, windows)
        count = int(mask.sum())
        if count == 0:
            # Text may exist here while the frames do not — a dropped tracker, a
            # window that fell between two retained audio frames. Say so rather
            # than feeding the gate a confident zero vector.
            continue
        features[index] = segment_functionals(values[mask])
        if modality == "audio":
            quality[index] = (1.0, _quality_scalar(voiced, mask), float(np.log1p(count)))
        else:
            quality[index] = (
                1.0,
                _quality_scalar(confidence, mask),
                _quality_scalar(success, mask),
                float(np.log1p(count)),
            )
    return features, quality


def build_text_quality(plan: SegmentPlan) -> NDArray[np.float32]:
    """Build the text branch's per-segment quality vectors.

    Counts are ``log1p``-compressed so the gate cannot simply learn "longer
    answer = more trustworthy" off an unbounded scale.

    Args:
        plan: The shared segmentation.

    Returns:
        Array of shape ``(K, 3)``: ``[valid, log1p(tokens), log1p(turns)]``.
    """
    quality = np.zeros((plan.count, QUALITY_DIMS["text"]), dtype=np.float32)
    for index in range(plan.effective):
        group = plan.groups[index]
        tokens = sum(len(turn.text.split()) for turn in group)
        if tokens == 0:
            continue
        quality[index] = (1.0, float(np.log1p(tokens)), float(np.log1p(len(group))))
    return quality


def segment_texts(plan: SegmentPlan) -> list[str]:
    """Return the concatenated transcript text of each non-empty segment.

    Args:
        plan: The shared segmentation.

    Returns:
        ``K_eff`` strings, in chronological order.
    """
    return [" ".join(turn.text for turn in group) for group in plan.groups]


def pad_to_count(matrix: NDArray[np.float32], count: int) -> NDArray[np.float32]:
    """Right-pad a ``(K_eff, D)`` matrix with zero rows up to ``count``.

    Args:
        matrix: The real segments' rows.
        count: The padded width ``K``.

    Returns:
        Array of shape ``(count, D)``.

    Raises:
        ValueError: If ``matrix`` already has more than ``count`` rows.
    """
    if matrix.shape[0] > count:
        raise ValueError(f"cannot pad {matrix.shape[0]} rows down to {count}")
    if matrix.shape[0] == count:
        return matrix
    padded = np.zeros((count, matrix.shape[1]), dtype=np.float32)
    padded[: matrix.shape[0]] = matrix
    return padded
