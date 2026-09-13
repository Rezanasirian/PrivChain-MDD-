"""Tests for concatenating several feature files into one modality (ADR-0031).

DAIC-WOZ ships more per-session exports than the pipeline read: the video branch
saw only the action units while gaze and head pose sat unused beside them, and
the acoustic side ignored ``FORMANT`` entirely. A modality may now declare
several ``sources``; these tests pin the alignment rule, which is the part that
can be silently wrong — two exports of the same session need not have the same
number of rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from privchain.data.daic_woz import DaicWozDataset, FeatureConfigError
from privchain.data.text_vectorizers import HashingTextVectorizer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _openface(path: Path, rows: range, *, columns: tuple[str, ...], offset: float) -> None:
    """Write an OpenFace-style export with metadata columns and a header."""
    header = "frame, timestamp, confidence, success, " + ", ".join(columns)
    lines = [header]
    for row in rows:
        values = ", ".join(str(offset + row + index) for index in range(len(columns)))
        lines.append(f"{row}, {row * 0.1}, 0.99, 1, {values}")
    _write(path, "\n".join(lines) + "\n")


@pytest.fixture
def daic_root(tmp_path: Path) -> Path:
    """One participant with three OpenFace exports of unequal length."""
    root = tmp_path / "daic"
    _write(root / "train.csv", "Participant_ID,PHQ8_Binary,PHQ8_Score\n300,1,15\n")
    _write(root / "300_P" / "300_COVAREP.csv", "\n".join(f"{r},{r + 1}" for r in range(8)) + "\n")
    # The action units run the full session; gaze stops two frames early, which
    # is the case that makes positional alignment wrong rather than merely short.
    _openface(root / "300_P" / "300_CLNF_AUs.txt", range(8), columns=("AU01", "AU02"), offset=0.0)
    _openface(root / "300_P" / "300_CLNF_gaze.txt", range(6), columns=("x", "y"), offset=100.0)
    _openface(root / "300_P" / "300_CLNF_pose.txt", range(8), columns=("Tx",), offset=200.0)
    transcript = "start_time\tstop_time\tspeaker\tvalue\n"
    transcript += "0.0\t1.0\tParticipant\ti feel tired\n"
    _write(root / "300_P" / "300_TRANSCRIPT.csv", transcript)
    return root


def _config(root: Path, **video: Any) -> dict[str, Any]:
    return {
        "root": str(root),
        "phq8_max": 24,
        "participant_dir_template": "{pid}_P",
        "feature_cache_dir": None,
        "splits": {"train": "train.csv"},
        "label_columns": {
            "participant_id": "Participant_ID",
            "phq_binary": "PHQ8_Binary",
            "phq_score": "PHQ8_Score",
        },
        "audio": {
            "file_template": "{pid}_COVAREP.csv",
            "has_header": False,
            "delimiter": ",",
            "max_frames": 100,
            "frame_stride": 1,
            "normalization": "none",
            "sample_rate_hz": 100.0,
        },
        "video": {
            "file_template": "{pid}_CLNF_AUs.txt",
            "has_header": True,
            "delimiter": ",",
            "drop_columns": ["frame", "timestamp", "confidence", "success"],
            "max_frames": 100,
            "frame_stride": 1,
            "normalization": "none",
            "timestamp_column": "timestamp",
            "quality_columns": ["timestamp", "confidence", "success"],
            **video,
        },
        "text": {
            "file_template": "{pid}_TRANSCRIPT.csv",
            "delimiter": "\t",
            "speaker_column": "speaker",
            "value_column": "value",
            "participant_speaker": "Participant",
            "start_column": "start_time",
            "stop_column": "stop_time",
            "dim": 8,
        },
    }


def _dataset(config: dict[str, Any]) -> DaicWozDataset:
    return DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))


def test_a_single_file_modality_is_unchanged(daic_root: Path) -> None:
    """The one-source case is the section itself, so old configs behave the same."""
    dataset = _dataset(_config(daic_root))
    assert dataset.feature_dims["video"] == 2
    assert int(dataset[0]["video"].shape[0]) == 8


def test_sources_concatenate_channel_wise(daic_root: Path) -> None:
    """Gaze and pose widen the branch instead of needing a fourth modality."""
    config = _config(
        daic_root,
        sources=[
            {"file_template": "{pid}_CLNF_AUs.txt"},
            {"file_template": "{pid}_CLNF_gaze.txt", "drop_columns": ["frame", "timestamp"]},
            {"file_template": "{pid}_CLNF_pose.txt", "drop_columns": ["frame", "timestamp"]},
        ],
    )
    dataset = _dataset(config)
    # AUs contribute 2 channels; gaze keeps confidence/success too (4); pose 3.
    assert dataset.feature_dims["video"] == 2 + 4 + 3


def test_alignment_is_on_the_source_row_not_the_position(daic_root: Path) -> None:
    """Only frames every source kept survive, so a channel is never shifted.

    Gaze ends two frames early. Concatenating by position would pair the action
    units of frames 6 and 7 with nothing, or worse, silently truncate a different
    file's rows onto them; intersecting the source rows keeps the six frames all
    three exports actually recorded.
    """
    config = _config(
        daic_root,
        sources=[
            {"file_template": "{pid}_CLNF_AUs.txt"},
            {
                "file_template": "{pid}_CLNF_gaze.txt",
                "drop_columns": ["frame", "timestamp", "confidence", "success"],
            },
        ],
    )
    sample = _dataset(config)[0]
    assert int(sample["video"].shape[0]) == 6
    # Row k holds AU01 == k from the first source and gaze x == 100 + k from the
    # second: the two files' frame k, not two different moments.
    for row in range(6):
        assert float(sample["video"][row, 0].item()) == pytest.approx(row)
        assert float(sample["video"][row, 2].item()) == pytest.approx(100.0 + row)


def test_quality_columns_come_from_the_first_source(daic_root: Path) -> None:
    """The gate and the segment alignment already read the primary export."""
    config = _config(
        daic_root,
        segments={"enabled": True, "count": 2},
        sources=[
            {"file_template": "{pid}_CLNF_AUs.txt"},
            {
                "file_template": "{pid}_CLNF_gaze.txt",
                "drop_columns": ["frame", "timestamp", "confidence", "success"],
            },
        ],
    )
    # Segment mode needs the timestamp/confidence/success columns to survive the
    # merge; building the dataset at all is what proves they did.
    assert _dataset(config).feature_dims["video"] > 0


def test_a_source_without_a_file_template_is_rejected(daic_root: Path) -> None:
    """A source that names no file is a config fault, not a missing session."""
    config = _config(daic_root, sources=[{"drop_columns": ["frame"]}])
    with pytest.raises(FeatureConfigError, match="file_template"):
        _dataset(config)


def test_sources_that_share_no_frames_are_rejected(daic_root: Path) -> None:
    """Concatenating exports with no frame in common must fail, not fabricate one.

    The second export here has eight unparseable rows followed by two good ones,
    so the only frames it keeps are ones the action units — capped at eight rows
    — never reached. Falling back to positional alignment would pair its frame 8
    with the action units' frame 0.
    """
    lines = ["frame, timestamp, confidence, success, Q"]
    lines += [f"{row}, {row * 0.1}, 0.99, 1, damaged" for row in range(8)]
    lines += [f"{row}, {row * 0.1}, 0.99, 1, {row}" for row in range(8, 10)]
    _write(daic_root / "300_P" / "300_CLNF_late.txt", "\n".join(lines) + "\n")

    config = _config(
        daic_root,
        max_frames=8,
        sources=[
            {"file_template": "{pid}_CLNF_AUs.txt"},
            {
                "file_template": "{pid}_CLNF_late.txt",
                "drop_columns": ["frame", "timestamp", "confidence", "success"],
            },
        ],
    )
    # A damaged session is tolerated while feature dims are inferred, so on a
    # one-participant corpus it surfaces as "nothing readable" rather than the
    # per-session message.
    with pytest.raises(RuntimeError, match="no readable participant"):
        _dataset(config)
