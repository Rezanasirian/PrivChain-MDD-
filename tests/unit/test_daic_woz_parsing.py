"""Parser tests for the real DAIC-WOZ loader using a tiny on-disk fixture.

This exercises the real parsing/assembly logic (COVAREP / OpenFace-AU /
transcript / split-label) against fabricated files in the canonical layout, so
we get confidence without the access-controlled 300 GB corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

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
    ds = DaicWozDataset(_config(daic_root), split="train", text_vectorizer=HashingTextVectorizer(8))
    assert len(ds) == 2
    assert int(ds[0]["label"].item()) == 0
    assert int(ds[0]["phq8_score"].item()) == 4
    assert int(ds[1]["label"].item()) == 1
    assert int(ds[1]["phq8_score"].item()) == 15


def test_feature_dims_and_shapes(daic_root: Path) -> None:
    ds = DaicWozDataset(_config(daic_root), split="train", text_vectorizer=HashingTextVectorizer(8))
    assert ds.feature_dims == {"audio": 4, "video": 3, "text": 8}

    sample = ds[0]
    assert sample["audio"].shape == (6, 4)  # 6 COVAREP frames, 4 features
    assert sample["video"].shape == (5, 3)  # 5 AU frames, metadata dropped
    assert sample["text"].shape == (1, 8)  # length-1 text sequence


def test_samples_collate(daic_root: Path) -> None:
    ds = DaicWozDataset(_config(daic_root), split="train", text_vectorizer=HashingTextVectorizer(8))
    batch = collate_fn([ds[0], ds[1]])
    assert batch["audio"].shape == (2, 6, 4)
    assert batch["label"].shape == (2,)
    assert batch["text_lengths"].tolist() == [1, 1]


def test_loader_honours_the_configured_vectorizer(daic_root: Path) -> None:
    """The `text.vectorizer` key selects the vectorizer; it used to be ignored."""
    config = _config(daic_root)
    config["text"]["vectorizer"] = "tfidf"  # documented once, never implemented

    with pytest.raises(ValueError, match="tfidf"):
        DaicWozDataset(config, split="train")


def test_explicit_vectorizer_overrides_the_config(daic_root: Path) -> None:
    """An injected vectorizer wins over the config, so tests stay offline."""
    config = _config(daic_root)
    config["text"]["vectorizer"] = "transformer"  # would need network weights

    ds = DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))

    assert ds.feature_dims["text"] == 8


def test_feature_cache_is_written_and_reused(daic_root: Path) -> None:
    """Parsed features are memoized, and a stride change invalidates the entry."""
    config = _config(daic_root)
    config["feature_cache_dir"] = "_feature_cache"
    cache_dir = daic_root / "_feature_cache"

    first = DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))
    baseline_audio = first[0]["audio"].clone()
    assert cache_dir.is_dir()
    entries = {p.name for p in cache_dir.glob("*.npy")}
    assert entries, "expected cached feature matrices"

    # A second dataset reads the cache and yields identical tensors...
    second = DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))
    assert torch.equal(second[0]["audio"], baseline_audio)
    assert {p.name for p in cache_dir.glob("*.npy")} == entries  # no new entries

    # ...while changing a parsing option keys to a different entry.
    config["audio"]["frame_stride"] = 2
    DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))[0]
    assert {p.name for p in cache_dir.glob("*.npy")} > entries


def test_feature_cache_can_be_disabled(daic_root: Path) -> None:
    """``feature_cache_dir: null`` writes nothing to disk."""
    config = _config(daic_root)
    config["feature_cache_dir"] = None

    DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))[0]

    assert not (daic_root / "_feature_cache").exists()


def test_missing_label_column_raises_instead_of_zeroing_labels(daic_root: Path) -> None:
    """A header mismatch must fail loudly, not silently read every label as 0.

    The real AVEC2017 test split names its columns ``PHQ_Binary``/``PHQ_Score``
    rather than ``PHQ8_*`` (ADR-0010).
    """
    config = _config(daic_root)
    config["label_columns"]["phq_binary"] = "PHQ_Binary"  # not in the fixture header

    with pytest.raises(ValueError, match="PHQ_Binary"):
        DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))


def test_split_label_columns_override_per_split(tmp_path: Path, daic_root: Path) -> None:
    """A per-split override resolves the train/dev vs test header difference."""
    _write(
        daic_root / "full_test_split.csv",
        "Participant_ID,PHQ_Binary,PHQ_Score\n300,1,11\n301,0,3\n",
    )
    config = _config(daic_root)
    config["splits"]["test"] = "full_test_split.csv"
    config["split_label_columns"] = {
        "test": {
            "participant_id": "Participant_ID",
            "phq_binary": "PHQ_Binary",
            "phq_score": "PHQ_Score",
        }
    }

    ds = DaicWozDataset(config, split="test", text_vectorizer=HashingTextVectorizer(8))
    assert [int(ds[i]["label"].item()) for i in range(len(ds))] == [1, 0]
    assert [int(ds[i]["phq8_score"].item()) for i in range(len(ds))] == [11, 3]


def test_excluded_participants_are_dropped(daic_root: Path) -> None:
    """Sessions listed in ``exclude_participants`` never reach the split.

    Participant 440's published archive is truncated at source and has no
    transcript, so it is excluded from the dev split (ADR-0010).
    """
    config = _config(daic_root)
    config["exclude_participants"] = [301]

    ds = DaicWozDataset(config, split="train", text_vectorizer=HashingTextVectorizer(8))
    assert len(ds) == 1
    assert int(ds[0]["phq8_score"].item()) == 4  # participant 300 only
