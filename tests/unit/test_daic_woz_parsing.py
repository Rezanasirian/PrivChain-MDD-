"""Parser tests for the real DAIC-WOZ loader using a tiny on-disk fixture.

This exercises the real parsing/assembly logic (COVAREP / OpenFace-AU /
transcript / split-label) against fabricated files in the canonical layout, so
we get confidence without the access-controlled 300 GB corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from privchain.data.daic_woz import DaicWozDataset
from privchain.data.mock_daic_woz import collate_fn
from privchain.data.text_vectorizers import HashingTextVectorizer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def daic_root(tmp_path: Path) -> Path:
    root = tmp_path / "daic"
    # Split label file.
    _write(
        root / "train_split_Depression_AVEC2017.csv",
        "Participant_ID,PHQ8_Binary,PHQ8_Score\n300,0,4\n301,1,15\n",
    )
    for pid, label_text in ((300, "hello there i feel fine"), (301, "i feel sad and tired")):
        # COVAREP: no header, 4 features, 6 frames.
        covarep = "\n".join(",".join(str(c + r) for c in range(4)) for r in range(6))
        _write(root / f"{pid}_P" / f"{pid}_COVAREP.csv", covarep + "\n")
        # OpenFace AUs: header with metadata cols to be dropped + 3 AU cols.
        aus = ["frame, timestamp, confidence, success, AU01, AU02, AU03"]
        aus += [f"{r}, {r * 0.1}, 0.99, 1, {r}, {r + 1}, {r + 2}" for r in range(5)]
        _write(root / f"{pid}_P" / f"{pid}_CLNF_AUs.txt", "\n".join(aus) + "\n")
        # Transcript: tab-separated, both speakers.
        transcript = "start_time\tstop_time\tspeaker\tvalue\n"
        transcript += "0.0\t1.0\tEllie\thow are you\n"
        transcript += f"1.0\t2.0\tParticipant\t{label_text}\n"
        _write(root / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv", transcript)
    return root


def _config(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "phq8_max": 24,
        "participant_dir_template": "{pid}_P",
        "splits": {"train": "train_split_Depression_AVEC2017.csv"},
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
            "standardize": False,
        },
        "video": {
            "file_template": "{pid}_CLNF_AUs.txt",
            "has_header": True,
            "delimiter": ",",
            "drop_columns": ["frame", "timestamp", "confidence", "success"],
            "max_frames": 100,
            "frame_stride": 1,
            "standardize": False,
        },
        "text": {
            "file_template": "{pid}_TRANSCRIPT.csv",
            "delimiter": "\t",
            "speaker_column": "speaker",
            "value_column": "value",
            "participant_speaker": "Participant",
            "dim": 8,
        },
    }


def test_dataset_length_and_labels(daic_root: Path) -> None:
    ds = DaicWozDataset(
        _config(daic_root), split="train", text_vectorizer=HashingTextVectorizer(8)
    )
    assert len(ds) == 2
    assert int(ds[0]["label"].item()) == 0
    assert int(ds[0]["phq8_score"].item()) == 4
    assert int(ds[1]["label"].item()) == 1
    assert int(ds[1]["phq8_score"].item()) == 15


def test_feature_dims_and_shapes(daic_root: Path) -> None:
    ds = DaicWozDataset(
        _config(daic_root), split="train", text_vectorizer=HashingTextVectorizer(8)
    )
    assert ds.feature_dims == {"audio": 4, "video": 3, "text": 8}

    sample = ds[0]
    assert sample["audio"].shape == (6, 4)  # 6 COVAREP frames, 4 features
    assert sample["video"].shape == (5, 3)  # 5 AU frames, metadata dropped
    assert sample["text"].shape == (1, 8)  # length-1 text sequence


def test_samples_collate(daic_root: Path) -> None:
    ds = DaicWozDataset(
        _config(daic_root), split="train", text_vectorizer=HashingTextVectorizer(8)
    )
    batch = collate_fn([ds[0], ds[1]])
    assert batch["audio"].shape == (2, 6, 4)
    assert batch["label"].shape == (2,)
    assert batch["text_lengths"].tolist() == [1, 1]
