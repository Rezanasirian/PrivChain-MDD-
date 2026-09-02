"""End-to-end run of the capability comparison on a fabricated corpus (ADR-0028).

No real data and no model download. The numbers are noise at this size; what is
asserted is that the run makes exactly one inferential claim, about the
capability named before the run, and records the provenance that claim rests on:
which pattern list was used, and that the gate prior was not swept.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.fixtures.fake_corpus import corpus_config, write_corpus

CAPABILITIES = {"full", "audio_text", "audio_only", "text_only"}
ARMS = {"baseline_gated_fusion", "capability_moe"}


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a fabricated corpus plus the baseline config a run needs."""
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


def _run(workspace: tuple[Path, Path, Path]) -> Path:
    base_path, daic_path, output_dir = workspace
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_capability_moe.py",
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
            "--bootstrap-resamples",
            "50",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    runs = list((output_dir / "phase1").glob("phase1_capability_moe_*"))
    assert len(runs) == 1
    return runs[0]


def test_comparison_scores_every_arm_under_every_capability(
    workspace: tuple[Path, Path, Path],
) -> None:
    run_dir = _run(workspace)

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]

    assert {row["arm"] for row in rows} == ARMS
    assert {row["capability"] for row in rows} == CAPABILITIES
    # Counterfactual evaluation: every arm sees every capability, so no
    # capability's estimate rests on a quarter of the fold.
    for arm in ARMS:
        seen = {row["capability"] for row in rows if row["arm"] == arm}
        assert seen == CAPABILITIES


def test_exactly_one_inferential_claim_is_made(workspace: tuple[Path, Path, Path]) -> None:
    """The segment ladder's lesson: eight questions of 107 participants is seven too many."""
    run_dir = _run(workspace)

    primary = json.loads((run_dir / "primary_test.json").read_text())

    assert primary["primary_capability"] == "audio_only"
    assert primary["contrast"] == "capability_moe - baseline_gated_fusion"
    assert {"delta", "ci_low", "ci_high", "significant"} <= set(primary)
    # The min-over-capabilities statistic is reported, but not as a test.
    assert "significant" not in primary["secondary_min_capability"]

    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.count("significant") == 1
    assert "no significance claimed" in summary


def test_manifest_records_what_the_claim_rests_on(workspace: tuple[Path, Path, Path]) -> None:
    run_dir = _run(workspace)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())

    assert manifest["primary_capability"] == "audio_only"
    assert manifest["primary_capability_fixed_before_run"] is True
    # The gate prior came from an earlier run's numbers; a reader must be able to
    # see that it was not tuned here.
    assert manifest["gate_bias_swept"] is False
    assert manifest["gate_bias_initialization"]["text"] > 0
    assert manifest["capability_pattern_source"].endswith("federated.yaml")
    assert {p["name"] for p in manifest["capability_patterns"]} == CAPABILITIES
    assert manifest["official_dev_scored"] is False
    assert manifest["official_test_read"] is False


def test_gate_weights_are_reported_for_both_arms(workspace: tuple[Path, Path, Path]) -> None:
    """An MoE that learned to ignore audio should be visible, not inferred."""
    run_dir = _run(workspace)

    primary = json.loads((run_dir / "primary_test.json").read_text())
    weights = primary["gate_weights_full_modality"]["capability_moe"]

    assert set(weights) == {"audio", "video", "text"}
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
