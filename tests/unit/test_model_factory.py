"""The model factory, and the promise that no script bypasses it.

``model.architecture`` is inert unless every path that builds a model honours it.
The worst outcome is not a crash but a silent one: a config that says
``segment_gated`` while a script constructs the old model and reports its numbers
under the new name. The source check below is what keeps that from creeping back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from privchain.config import load_baseline_config
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.fusion.factory import build_depression_model, require_baseline_architecture
from privchain.fusion.segment_model import SegmentGatedNetwork

INPUT_DIMS = {"audio": 6, "video": 5, "text": 4}

#: Scripts that legitimately construct the baseline model directly, because they
#: reach into its structure. Each must refuse a `segment_gated` config instead.
GUARDED_SCRIPTS = {
    "run_attack_eval.py",
    "run_reid_risk.py",
    "run_text_representation.py",
}


def _model_config(**updates: object):  # noqa: ANN202 - test helper
    base = load_baseline_config("configs/baseline.yaml")
    return base.model.model_copy(update=updates)


def test_factory_builds_the_configured_architecture() -> None:
    baseline = build_depression_model(INPUT_DIMS, _model_config(architecture="encode_then_fuse"))
    assert isinstance(baseline, MultimodalDepressionModel)

    segment = build_depression_model(
        INPUT_DIMS,
        _model_config(architecture="segment_gated"),
        {"audio": 3, "video": 4, "text": 3},
    )
    assert isinstance(segment, SegmentGatedNetwork)


def test_unknown_architecture_is_rejected() -> None:
    config = _model_config()
    object.__setattr__(config, "architecture", "telepathy")
    with pytest.raises(ValueError, match="unknown model architecture"):
        build_depression_model(INPUT_DIMS, config)


def test_guard_rejects_the_segment_architecture() -> None:
    require_baseline_architecture(_model_config(), "a test")  # no raise
    with pytest.raises(ValueError, match="encode_then_fuse"):
        require_baseline_architecture(_model_config(architecture="segment_gated"), "a test")


def test_segment_model_requires_one_embedding_width() -> None:
    """Summing the modalities needs them to agree on a width; say so, don't broadcast."""
    config = _model_config(
        architecture="segment_gated",
        encoder_overrides={"text": {"out_dim": 64}},
    )
    with pytest.raises(ValueError, match="share one out_dim"):
        build_depression_model(INPUT_DIMS, config)


def test_no_script_builds_the_baseline_model_behind_the_factory() -> None:
    offenders = []
    for script in sorted(Path("scripts").glob("*.py")):
        if script.name in GUARDED_SCRIPTS:
            continue
        source = script.read_text(encoding="utf-8")
        if re.search(r"\bMultimodalDepressionModel\s*\(", source):
            offenders.append(script.name)
    assert not offenders, (
        f"{offenders} construct the baseline model directly; use build_depression_model, "
        "or add the script to GUARDED_SCRIPTS and call require_baseline_architecture"
    )


def test_guarded_scripts_actually_call_the_guard() -> None:
    missing = [
        name
        for name in sorted(GUARDED_SCRIPTS)
        if "require_baseline_architecture(" not in Path("scripts", name).read_text(encoding="utf-8")
    ]
    assert not missing, f"{missing} skip the architecture guard"
