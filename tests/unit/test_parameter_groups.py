"""Parameter→group mapping: one rule for DP budgets and for aggregation.

A modality-specific parameter that lands in ``shared`` is charged to the wrong
privacy budget *and* averaged over clients that never held that modality. The
gated fusion's per-modality gate heads used to do exactly that, so the rule now
recognizes them by prefix — deliberately by prefix rather than by moving the
parameters, which would rename ``state_dict`` keys and orphan every checkpoint
written before ADR-0027.
"""

from __future__ import annotations

import torch

from privchain.config import load_baseline_config
from privchain.federated.capability import SHARED_GROUP, param_group
from privchain.fusion.factory import build_depression_model
from privchain.privacy.dp_sgd import map_parameter_groups

INPUT_DIMS = {"audio": 6, "video": 5, "text": 4}


def _model(fusion_type: str):  # noqa: ANN202 - test helper
    base = load_baseline_config("configs/baseline.yaml")
    config = base.model.model_copy(
        update={"fusion": base.model.fusion.model_copy(update={"type": fusion_type})}
    )
    return build_depression_model(INPUT_DIMS, config)


def test_encoder_parameters_map_to_their_modality() -> None:
    assert param_group("encoders.audio.proj.weight") == "audio"
    assert param_group("encoders.video.out.bias") == "video"
    assert param_group("encoders.text.attention.score.0.weight") == "text"


def test_gated_fusion_gates_map_to_their_modality() -> None:
    assert param_group("fusion.gates.audio.weight") == "audio"
    assert param_group("fusion.gates.text.bias") == "text"


def test_shared_parameters_stay_shared() -> None:
    assert param_group("fusion.net.0.weight") == SHARED_GROUP
    assert param_group("classifier.0.bias") == SHARED_GROUP


def test_gated_model_puts_no_modality_parameter_in_the_shared_group() -> None:
    model = _model("gated")
    shared = [name for name, _ in model.named_parameters() if param_group(name) == SHARED_GROUP]
    assert not any(".gates." in name for name in shared)
    groups = map_parameter_groups(model)
    # Each gate head adds a weight and a bias to its modality's group.
    for modality in ("audio", "video", "text"):
        assert len(groups[modality]) > 0


def test_capability_restricted_grouping_drops_absent_modalities() -> None:
    groups = map_parameter_groups(_model("gated"), capability=(0, 1, 1))
    assert "audio" not in groups
    assert {"video", "text", SHARED_GROUP} <= set(groups)


def test_existing_gated_checkpoints_still_load() -> None:
    """The rule change renames nothing, so a saved state_dict is still strict-loadable."""
    saved = _model("gated").state_dict()
    reloaded = _model("gated")
    reloaded.load_state_dict(saved, strict=True)
    assert torch.equal(
        reloaded.state_dict()["fusion.gates.audio.weight"], saved["fusion.gates.audio.weight"]
    )
