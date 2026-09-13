"""Smoke test for the train-only CTD federated comparison."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from privchain.data.ctd_reference import CTD_REFERENCE_DIM


def test_train_cv_writes_paired_oof_without_dev_or_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    output = tmp_path / "output"
    artifact.mkdir()
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * 10, dtype=np.int64)
    features = rng.normal(size=(len(labels), CTD_REFERENCE_DIM)).astype(np.float32)
    np.savez(
        artifact / "ctd_train.npz",
        session_id=np.arange(100, 100 + len(labels)),
        emb=features,
        label=labels,
    )
    argv = [
        "run_ctd_train_cv.py",
        "--artifact-dir", str(artifact),
        "--output-dir", str(output),
        "--seeds", "42",
        "--folds", "2",
        "--num-clients", "2",
        "--rounds", "2",
        "--patience", "2",
        "--bootstrap-resamples", "20",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path("scripts/run_ctd_train_cv.py", run_name="__main__")

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["official_dev_read"] is False
    assert summary["official_test_read"] is False
    assert len((output / "oof_predictions.jsonl").read_text().splitlines()) == 2
    assert len((output / "results.jsonl").read_text().splitlines()) == 4
