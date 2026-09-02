"""A tiny fabricated corpus in the DAIC-WOZ layout (tests only).

The real corpus is access-controlled and must never enter the repo, yet the
parser, the segment alignment and the experiment scripts all need files in the
canonical shape to run against. This writes them: COVAREP without a header,
OpenFace AUs with the metadata columns, tab-separated transcripts carrying
``start_time``/``stop_time``, and the split label CSVs.

Timings are chosen so the alignment is checkable by hand: participant turn ``i``
runs from ``2i + 1`` to ``2i + 2`` seconds, and the interviewer speaks in the odd
gaps between them. Audio is written at ``AUDIO_RATE_HZ`` rows per second and
video at ``VIDEO_RATE_HZ``, so which frame belongs to which turn is arithmetic
rather than guesswork.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Rows per second in the fabricated COVAREP files.
AUDIO_RATE_HZ = 10
#: Rows per second in the fabricated OpenFace files.
VIDEO_RATE_HZ = 5
#: Feature widths of the fabricated files (real corpus: 74 and 20).
AUDIO_CHANNELS = 4
VIDEO_CHANNELS = 3


def write_file(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def turn_window(index: int) -> tuple[float, float]:
    """Return the ``(start, stop)`` seconds of participant turn ``index``."""
    return float(2 * index + 1), float(2 * index + 2)


def write_participant(root: Path, pid: int, *, turns: int, seconds: float) -> None:
    """Write one participant's three modality files.

    Args:
        root: Corpus root.
        pid: Participant id.
        turns: Number of participant turns to write.
        seconds: Session duration, which sets how many frames each file holds.
    """
    audio_rows = int(seconds * AUDIO_RATE_HZ)
    # Column 1 is the voiced flag, mirroring the COVAREP layout: voiced exactly
    # while the participant is speaking, so a segment's voiced ratio is known.
    covarep = []
    for row in range(audio_rows):
        time = row / AUDIO_RATE_HZ
        voiced = float(any(start <= time < stop for start, stop in map(turn_window, range(turns))))
        covarep.append(",".join([f"{time:.3f}", f"{voiced:.1f}", f"{row}", f"{row + 1}"]))
    write_file(root / f"{pid}_P" / f"{pid}_COVAREP.csv", "\n".join(covarep) + "\n")

    aus = ["frame, timestamp, confidence, success, AU01, AU02, AU03"]
    for row in range(int(seconds * VIDEO_RATE_HZ)):
        time = row / VIDEO_RATE_HZ
        aus.append(f"{row}, {time:.3f}, 0.9, 1, {row}, {row + 1}, {row + 2}")
    write_file(root / f"{pid}_P" / f"{pid}_CLNF_AUs.txt", "\n".join(aus) + "\n")

    lines = ["start_time\tstop_time\tspeaker\tvalue"]
    for index in range(turns):
        start, stop = turn_window(index)
        lines.append(f"{start - 1.0:.1f}\t{start:.1f}\tEllie\tquestion {index}")
        lines.append(f"{start:.1f}\t{stop:.1f}\tParticipant\tanswer number {index} words here")
    write_file(root / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv", "\n".join(lines) + "\n")


def write_corpus(
    root: Path,
    *,
    splits: dict[str, list[tuple[int, int, int]]],
    turns: int = 12,
    seconds: float = 30.0,
) -> Path:
    """Write a whole fabricated corpus.

    Args:
        root: Destination directory.
        splits: ``{split_name: [(pid, binary_label, phq8_score), ...]}``.
        turns: Participant turns per session.
        seconds: Session duration in seconds.

    Returns:
        ``root``, for chaining.
    """
    filenames = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev": "dev_split_Depression_AVEC2017.csv",
        "test": "full_test_split.csv",
    }
    for split, records in splits.items():
        header = (
            "Participant_ID,PHQ_Binary,PHQ_Score"
            if split == "test"
            else "Participant_ID,PHQ8_Binary,PHQ8_Score"
        )
        rows = "\n".join(f"{pid},{label},{score}" for pid, label, score in records)
        write_file(root / filenames[split], f"{header}\n{rows}\n")
        for pid, _, _ in records:
            write_participant(root, pid, turns=turns, seconds=seconds)
    return root


def corpus_config(root: Path, **overrides: Any) -> dict[str, Any]:
    """Return a ``daic_woz`` config mapping pointing at a fabricated corpus.

    Args:
        root: Corpus root written by :func:`write_corpus`.
        **overrides: Per-section overrides merged into the defaults, e.g.
            ``segments={"enabled": True, "count": 4}``.

    Returns:
        The config mapping, ready for :class:`~privchain.data.daic_woz.DaicWozDataset`.
    """
    config: dict[str, Any] = {
        "root": str(root),
        "phq8_max": 24,
        "participant_dir_template": "{pid}_P",
        "feature_cache_dir": None,
        "segments": {"enabled": False, "count": 4},
        "splits": {
            "train": "train_split_Depression_AVEC2017.csv",
            "dev": "dev_split_Depression_AVEC2017.csv",
            "test": "full_test_split.csv",
        },
        "label_columns": {
            "participant_id": "Participant_ID",
            "phq_binary": "PHQ8_Binary",
            "phq_score": "PHQ8_Score",
        },
        "split_label_columns": {
            "test": {
                "participant_id": "Participant_ID",
                "phq_binary": "PHQ_Binary",
                "phq_score": "PHQ_Score",
            }
        },
        "audio": {
            "file_template": "{pid}_COVAREP.csv",
            "has_header": False,
            "delimiter": ",",
            "max_frames": 10_000,
            "frame_stride": 1,
            "normalization": "none",
            "sample_rate_hz": AUDIO_RATE_HZ,
            "quality_columns": {"voiced": 1},
            "voiced_column": "voiced",
        },
        "video": {
            "file_template": "{pid}_CLNF_AUs.txt",
            "has_header": True,
            "delimiter": ",",
            "drop_columns": ["frame", "timestamp", "confidence", "success"],
            "max_frames": 10_000,
            "frame_stride": 1,
            "normalization": "none",
            "timestamp_column": "timestamp",
            "quality_columns": ["timestamp", "confidence", "success"],
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
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section] = {**config[section], **values}
        else:
            config[section] = values
    return config
