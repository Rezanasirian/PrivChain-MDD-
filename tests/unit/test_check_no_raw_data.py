"""Unit tests for the pre-commit raw-data guard (CLAUDE.md §7)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_no_raw_data import is_blocked, main  # noqa: E402


def test_raw_media_blocked_anywhere() -> None:
    assert is_blocked("notebooks/sample.wav")
    assert is_blocked("src/privchain/clip.mp4")
    assert is_blocked("whatever/RECORDING.MOV")


def test_anything_under_data_is_blocked() -> None:
    assert is_blocked("data/mock/session_001/audio.npy")
    assert is_blocked("data/train_split_Depression_AVEC2017.csv")
    assert is_blocked("data\\mock\\session_001\\audio.npy")  # windows separators


def test_data_placeholders_allowed() -> None:
    assert not is_blocked("data/.gitkeep")
    assert not is_blocked("data/README.md")


def test_result_csv_outside_data_is_allowed() -> None:
    """Chapter-4 tables must remain committable — the old rule blocked them."""
    assert not is_blocked("experiments/phase7/run/cv_results.csv")
    assert not is_blocked("docs/architecture/table.csv")


def test_main_exit_codes() -> None:
    assert main(["src/privchain/config.py", "experiments/run/results.csv"]) == 0
    assert main(["data/secret.wav"]) == 1
