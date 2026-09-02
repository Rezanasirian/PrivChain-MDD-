"""Segment alignment: do the modalities really get cut at the same places?

The architecture's whole claim is that segment ``k`` means the same stretch of
interview in every branch. These tests check that against a fabricated corpus
whose timings are known exactly (:mod:`tests.fixtures.fake_corpus`), including
the cases that would otherwise fail silently: an empty window, a participant with
fewer turns than segments, and a malformed row in the middle of a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.fixtures.fake_corpus import (
    AUDIO_RATE_HZ,
    VIDEO_RATE_HZ,
    corpus_config,
    turn_window,
    write_corpus,
    write_file,
)

from privchain.data.daic_woz import DaicWozDataset, _load_feature_matrix
from privchain.data.segment_alignment import (
    QUALITY_DIMS,
    TimedTurn,
    plan_segments,
    rows_within,
    segment_functionals,
)
from privchain.data.text_vectorizers import HashingTextVectorizer

SPLITS = {"train": [(300, 0, 4), (301, 1, 15)]}


def _dataset(root: Path, **overrides: object) -> DaicWozDataset:
    config = corpus_config(root, **overrides)
    return DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "daic", splits=SPLITS, turns=12, seconds=30.0)


def test_plan_groups_turns_contiguously() -> None:
    turns = [TimedTurn(float(i), float(i) + 0.5, f"t{i}") for i in range(9)]
    plan = plan_segments(turns, 3)
    assert plan.effective == 3
    assert [len(group) for group in plan.groups] == [3, 3, 3]
    assert plan.envelope(0) == (0.0, 2.5)
    assert plan.intervals(1) == ((3.0, 3.5), (4.0, 4.5), (5.0, 5.5))


def test_quiet_participant_is_padded_not_dropped(tmp_path: Path) -> None:
    """Three turns, eight segments: three real rows, the rest marked invalid."""
    root = write_corpus(tmp_path / "daic", splits=SPLITS, turns=3, seconds=12.0)
    dataset = _dataset(root, segments={"enabled": True, "count": 8})
    sample = dataset[0]

    for modality in ("audio", "video", "text"):
        assert sample[modality].shape[0] == 8  # type: ignore[literal-required]
        valid = sample["quality"][modality][:, 0]
        assert valid.tolist() == [1.0] * 3 + [0.0] * 5
        # A padded segment carries no features either, not just a cleared flag.
        assert float(sample[modality][3:].abs().sum()) == 0.0  # type: ignore[literal-required]


def test_audio_takes_only_participant_speech(corpus: Path) -> None:
    """Audio frames come from the participant's turns, not the whole envelope.

    The fixture marks a frame voiced exactly while the participant speaks, so a
    voiced ratio below 1 would mean the interviewer's audio leaked in.
    """
    dataset = _dataset(corpus, segments={"enabled": True, "count": 4})
    quality = dataset[0]["quality"]["audio"]
    assert quality[:, 0].tolist() == [1.0] * 4
    assert quality[:, 1].tolist() == [1.0] * 4

    # 12 turns / 4 groups = 3 turns of 1 s each, at AUDIO_RATE_HZ rows per second.
    assert np.allclose(quality[:, 2].numpy(), np.log1p(3 * AUDIO_RATE_HZ))


def test_video_takes_the_whole_envelope(corpus: Path) -> None:
    """Video keeps the listening stretches too — a documented asymmetry (ADR-0027)."""
    dataset = _dataset(corpus, segments={"enabled": True, "count": 4})
    quality = dataset[0]["quality"]["video"]
    # Group 0 spans turn 0's start to turn 2's stop: 1.0 s .. 6.0 s.
    start, _ = turn_window(0)
    _, stop = turn_window(2)
    expected = int(round((stop - start) * VIDEO_RATE_HZ))
    assert quality[0, 0] == 1.0
    assert quality[0, 1] == pytest.approx(0.9)  # tracker confidence
    assert quality[0, 2] == 1.0  # success ratio
    assert quality[0, 3] == pytest.approx(np.log1p(expected), abs=0.05)


def test_all_modalities_share_one_segmentation(corpus: Path) -> None:
    dataset = _dataset(corpus, segments={"enabled": True, "count": 5})
    sample = dataset[0]
    counts = {m: sample[m].shape[0] for m in ("audio", "video", "text")}  # type: ignore[literal-required]
    assert set(counts.values()) == {5}
    assert {m: q.shape[1] for m, q in sample["quality"].items()} == QUALITY_DIMS


def test_empty_window_is_marked_invalid(tmp_path: Path) -> None:
    """A window with text but no frames says so instead of imputing zeros silently."""
    root = write_corpus(tmp_path / "daic", splits=SPLITS, turns=8, seconds=30.0)
    # Truncate one participant's audio to the first 4 seconds, so the later
    # groups' turn intervals contain no rows at all.
    path = root / "300_P" / "300_COVAREP.csv"
    kept = path.read_text(encoding="utf-8").splitlines()[: 4 * AUDIO_RATE_HZ]
    write_file(path, "\n".join(kept) + "\n")

    dataset = _dataset(root, segments={"enabled": True, "count": 4})
    audio_valid = dataset[0]["quality"]["audio"][:, 0]
    assert audio_valid[0] == 1.0
    assert audio_valid[-1] == 0.0
    # Text still covers the whole session, so the segment itself stays usable.
    assert dataset[0]["quality"]["text"][:, 0].tolist() == [1.0] * 4


def test_malformed_row_does_not_shift_later_timestamps(tmp_path: Path) -> None:
    """A skipped row must not move every following frame earlier in time.

    Deriving a COVAREP timestamp from its position in the retained matrix would
    do exactly that, and the misalignment would be invisible — every segment
    would still be full, just of the wrong frames.
    """
    root = write_corpus(tmp_path / "daic", splits=SPLITS, turns=4, seconds=10.0)
    path = root / "300_P" / "300_COVAREP.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[5] = "not,a,number,here"
    write_file(path, "\n".join(lines) + "\n")

    parsed = _load_feature_matrix(
        path,
        delimiter=",",
        has_header=False,
        drop_columns=[],
        max_frames=10_000,
        frame_stride=1,
        quality_columns={"voiced": 1},
    )
    assert parsed.skipped == 1
    # Row 6 of the file is still reported as row 6, not as row 5.
    assert 5 not in parsed.source_rows.tolist()
    assert parsed.source_rows.tolist()[5] == 6
    assert parsed.values.shape[0] == parsed.source_rows.shape[0]
    assert parsed.columns["voiced"].shape[0] == parsed.values.shape[0]


def test_rows_within_unions_intervals() -> None:
    timestamps = np.arange(0.0, 1.0, 0.1)
    mask = rows_within(timestamps, [(0.0, 0.2), (0.5, 0.7)])
    assert mask.tolist() == [True, True, False, False, False, True, True, False, False, False]


def test_functionals_match_the_session_level_definition() -> None:
    matrix = np.asarray([[1.0, 4.0], [3.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    out = segment_functionals(matrix)
    assert out.shape == (10,)
    assert out[:2].tolist() == pytest.approx([3.0, 4.0])  # mean
    assert out[4:6].tolist() == [1.0, 2.0]  # min
    assert out[6:8].tolist() == [5.0, 6.0]  # max
    assert out[8:].tolist() == pytest.approx([2.0, 3.0])  # mean |first difference|


def test_functionals_of_an_empty_segment_are_zero() -> None:
    assert segment_functionals(np.zeros((0, 3), dtype=np.float32)).tolist() == [0.0] * 15


def test_damaged_timestamps_yield_a_zero_length_interval(tmp_path: Path) -> None:
    """A half-readable pair must not be completed from its surviving endpoint.

    A turn with an unreadable start and a stop of 120 would otherwise claim the
    interview's first two minutes of audio, and the mis-attribution would be
    invisible: every segment would still look full.
    """
    root = write_corpus(tmp_path / "daic", splits=SPLITS, turns=4, seconds=20.0)
    path = root / "300_P" / "300_TRANSCRIPT.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    # Damage the first participant turn's start time, and the second turn's stop.
    lines[2] = "\t".join(["", "120.0", "Participant", "answer number 0 words here"])
    lines[4] = "\t".join(["5.0", "nan", "Participant", "answer number 1 words here"])
    write_file(path, "\n".join(lines) + "\n")

    dataset = _dataset(root, segments={"enabled": True, "count": 4})
    audio_valid = dataset[0]["quality"]["audio"][:, 0]

    # The two damaged turns select no frames at all; the intact ones still do.
    assert audio_valid.tolist() == [0.0, 0.0, 1.0, 1.0]
    # Text survives regardless — the utterance was readable, only its clock was not.
    assert dataset[0]["quality"]["text"][:, 0].tolist() == [1.0] * 4


@pytest.mark.parametrize("raw", ["", "   ", "not-a-time", "nan", "inf", "-inf"])
def test_unusable_timestamps_are_rejected(raw: str) -> None:
    from privchain.data.daic_woz import _parse_seconds

    assert _parse_seconds(raw) is None


def test_backwards_interval_collapses_rather_than_inverting() -> None:
    from privchain.data.daic_woz import _timed_turn

    turn = _timed_turn(5.0, 2.0, "text")
    assert (turn.start, turn.stop) == (5.0, 5.0)
