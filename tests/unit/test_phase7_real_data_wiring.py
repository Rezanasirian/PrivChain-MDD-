"""Phase 7 (H5): the real-data wiring of the Chapter-4 evaluation harness.

Phase 7 was written against mock data only and its numbers were placeholders.
These tests pin the two things that changed when it was pointed at DAIC-WOZ
(ADR-0023), using the same tiny on-disk fixture as the loader tests so CI never
needs the access-controlled corpus:

* folds are drawn from the **pooled official train+dev split** while the
  AVEC2017 test split is appended but kept out of the fold pool;
* each fold's selection split is carved out of that fold's *training* indices,
  so nothing the model selects on is also scored.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_final_evaluation import _build_corpus, _carve_fold  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def daic_config_file(tmp_path: Path) -> Path:
    """A three-split corpus: 3 train, 2 dev, 2 test participants."""
    root = tmp_path / "daic"
    splits = {
        "train_split_Depression_AVEC2017.csv": (
            "Participant_ID,PHQ8_Binary,PHQ8_Score\n300,0,4\n301,1,15\n302,0,2\n"
        ),
        "dev_split_Depression_AVEC2017.csv": (
            "Participant_ID,PHQ8_Binary,PHQ8_Score\n303,1,18\n304,0,5\n"
        ),
        # The test split ships PHQ_* rather than PHQ8_* headers (ADR-0010).
        "full_test_split.csv": "Participant_ID,PHQ_Binary,PHQ_Score\n305,1,12\n306,0,3\n",
    }
    for name, text in splits.items():
        _write(root / name, text)

    for pid in range(300, 307):
        covarep = "\n".join(",".join(str(c + r) for c in range(4)) for r in range(6))
        _write(root / f"{pid}_P" / f"{pid}_COVAREP.csv", covarep + "\n")
        aus = ["frame, timestamp, confidence, success, AU01, AU02, AU03"]
        aus += [f"{r}, {r * 0.1}, 0.99, 1, {r}, {r + 1}, {r + 2}" for r in range(5)]
        _write(root / f"{pid}_P" / f"{pid}_CLNF_AUs.txt", "\n".join(aus) + "\n")
        transcript = "start_time\tstop_time\tspeaker\tvalue\n"
        transcript += "0.0\t1.0\tEllie\thow are you\n"
        transcript += f"1.0\t2.0\tParticipant\tsession {pid} talking\n"
        _write(root / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv", transcript)

    config_path = tmp_path / "daic_woz.yaml"
    config_path.write_text(
        f"""
seed: 42
daic_woz:
  root: {root.as_posix()}
  phq8_max: 24
  participant_dir_template: "{{pid}}_P"
  feature_cache_dir: null
  splits:
    train: train_split_Depression_AVEC2017.csv
    dev: dev_split_Depression_AVEC2017.csv
    test: full_test_split.csv
  label_columns:
    participant_id: Participant_ID
    phq_binary: PHQ8_Binary
    phq_score: PHQ8_Score
  split_label_columns:
    test:
      phq_binary: PHQ_Binary
      phq_score: PHQ_Score
  audio:
    file_template: "{{pid}}_COVAREP.csv"
    has_header: false
    delimiter: ","
    max_frames: 100
    frame_stride: 1
    normalization: none
  video:
    file_template: "{{pid}}_CLNF_AUs.txt"
    has_header: true
    delimiter: ","
    drop_columns: [frame, timestamp, confidence, success]
    max_frames: 100
    frame_stride: 1
    normalization: none
  text:
    file_template: "{{pid}}_TRANSCRIPT.csv"
    delimiter: "\t"
    speaker_column: speaker
    value_column: value
    participant_speaker: Participant
    dim: 8
    vectorizer: hashing
""",
        encoding="utf-8",
    )
    return config_path


def test_folds_pool_train_and_dev_but_reserve_the_official_test(
    daic_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fold pool is train+dev; the official test sits outside it."""
    monkeypatch.chdir(daic_config_file.parent)
    dataset, pool_labels, input_dims, official_idx = _build_corpus(None, daic_config_file)

    # 3 train + 2 dev sessions are poolable; the 2 test sessions are not.
    assert len(pool_labels) == 5
    assert pool_labels == [0, 1, 0, 1, 0]
    assert official_idx == [5, 6]
    # Every session is still reachable, so the same index-based runners can
    # score the official split without a second code path.
    assert len(dataset) == 7
    assert input_dims == {"audio": 4, "video": 3, "text": 8}


def test_official_test_indices_resolve_to_the_test_split(daic_config_file: Path) -> None:
    """Indices past the pool address the AVEC2017 test sessions, PHQ_* headers and all."""
    dataset, pool_labels, _, official_idx = _build_corpus(None, daic_config_file)

    scores = [int(dataset[i]["phq8_score"].item()) for i in official_idx]
    assert scores == [12, 3]  # read via the split_label_columns override
    assert [int(dataset[i]["label"].item()) for i in official_idx] == [1, 0]


def test_selection_split_is_disjoint_from_fold_training_data() -> None:
    """Nothing used to early-stop or pick the threshold is also trained on."""
    corpus: list[dict[str, Any]] = [{"i": i} for i in range(20)]
    labels = [i % 2 for i in range(20)]
    train_idx = list(range(16))

    fold_train, selection = _carve_fold(corpus, train_idx, labels, selection_fraction=0.25, seed=42)

    trained = {fold_train[i]["i"] for i in range(len(fold_train))}  # type: ignore[arg-type]
    selected = {selection[i]["i"] for i in range(len(selection))}  # type: ignore[arg-type]
    assert not trained & selected
    assert trained | selected == set(train_idx)
    # The four indices left out of train_idx are the fold's scored split and
    # must not have leaked into either side.
    assert not (trained | selected) & {16, 17, 18, 19}
