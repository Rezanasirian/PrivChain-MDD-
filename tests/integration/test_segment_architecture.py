"""End-to-end run of the segment-architecture ladder on a fabricated corpus.

No real data and no model download: the corpus is written by
:mod:`tests.fixtures.fake_corpus` and the text branch uses the offline hashing
vectorizer. The point is not the numbers — they are noise at this size — but that
all six arms build, train, score, and write the artifacts the protocol promises,
including out-of-fold predictions and a manifest saying dev was never read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.fixtures.fake_corpus import corpus_config, write_corpus

ARMS = [
    "document+concat",
    "segments+attn",
    "aligned+quality_gated",
    "aligned+huber",
    "aligned+dropout",
]


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a fabricated corpus plus the two configs a run needs."""
    root = write_corpus(
        tmp_path / "daic",
        splits={
            "train": [(300 + i, i % 2, 4 + 11 * (i % 2)) for i in range(12)],
            "dev": [(400 + i, i % 2, 4 + 11 * (i % 2)) for i in range(6)],
            "test": [(500 + i, i % 2, 4 + 11 * (i % 2)) for i in range(4)],
        },
        turns=8,
        seconds=20.0,
    )
    daic_cfg = corpus_config(root)
    daic_cfg["text"]["vectorizer"] = "hashing"
    daic_path = tmp_path / "daic.yaml"
    daic_path.write_text(yaml.safe_dump({"seed": 42, "daic_woz": daic_cfg}), encoding="utf-8")

    base = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    base["train"].update(
        {
            "epochs": 2,
            "batch_size": 4,
            "early_stopping_patience": 1,
            "seeds": [42],
            "device": "cpu",
            "output_dir": str(tmp_path / "experiments"),
        }
    )
    base_path = tmp_path / "baseline.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return base_path, daic_path, tmp_path / "experiments"


def test_ladder_runs_and_writes_its_artifacts(workspace: tuple[Path, Path, Path]) -> None:
    base_path, daic_path, output_dir = workspace
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_segment_architecture.py",
            "--config",
            str(base_path),
            "--daic-config",
            str(daic_path),
            "--inner-folds",
            "2",
            "--seeds",
            "42",
            "--num-segments",
            "3",
            "--arms",
            *ARMS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    runs = list((output_dir / "phase1").glob("phase1_segment_architecture_*"))
    assert len(runs) == 1
    run_dir = runs[0]

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert {row["arm"] for row in rows} == set(ARMS)
    assert all("pr_auc" in row and "roc_auc" in row for row in rows)

    # Out-of-fold predictions cover every pooled participant exactly once.
    oof_lines = (run_dir / "oof_predictions.jsonl").read_text().splitlines()
    oof = [json.loads(line) for line in oof_lines]
    assert {entry["arm"] for entry in oof} == set(ARMS)
    pool_size = len(oof[0]["labels"])
    assert all(len(entry["scores"]) == pool_size for entry in oof)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    # The dev dataset is constructed by `build_splits`; the claim the protocol
    # rests on is that it is never scored or selected on, and the manifest says
    # exactly that rather than the flattering "untouched".
    assert manifest["official_dev_scored"] is False
    assert manifest["official_dev_selected_on"] is False
    assert manifest["official_dev_constructed"] is True
    assert manifest["official_test_read"] is False
    assert manifest["pool_size"] == pool_size
    assert manifest["bootstrap_unit"].startswith("participant")

    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    for arm in ARMS:
        assert arm in summary
    assert "95% CI" in summary
