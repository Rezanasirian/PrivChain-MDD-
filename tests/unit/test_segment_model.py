"""The segment-gated network: shapes, isolation, DP compatibility, grouping.

Three properties matter beyond "it runs":

* a sample's output must not depend on its batch mates (the same invariant the
  encoders already assert — without it a DP guarantee is void and an evaluation
  result depends on batching);
* Opacus must be able to take per-sample gradients of it, or Phase 3 cannot use
  the architecture at all;
* every modality-specific parameter must land in its own DP/aggregation group.
"""

from __future__ import annotations

import pytest
import torch
from opacus import GradSampleModule
from opacus.validators import ModuleValidator

from privchain.config import load_baseline_config
from privchain.data.mock_daic_woz import MODALITIES, Batch
from privchain.federated.capability import SHARED_GROUP, param_group
from privchain.fusion.factory import build_depression_model
from privchain.fusion.segment_model import SegmentGatedNetwork
from privchain.privacy.dp_sgd import map_parameter_groups

INPUT_DIMS = {"audio": 20, "video": 15, "text": 8}
QUALITY_DIMS = {"audio": 3, "video": 4, "text": 3}
SEGMENTS = 4


def _model() -> SegmentGatedNetwork:
    torch.manual_seed(0)
    base = load_baseline_config("configs/baseline.yaml")
    config = base.model.model_copy(
        update={
            "architecture": "segment_gated",
            "fusion": base.model.fusion.model_copy(update={"type": "quality_gated"}),
        }
    )
    model = build_depression_model(INPUT_DIMS, config, QUALITY_DIMS)
    assert isinstance(model, SegmentGatedNetwork)
    return model


def _batch(size: int = 3, *, quality: bool = True, seed: int = 0) -> Batch:
    torch.manual_seed(seed)
    batch: Batch = {  # type: ignore[typeddict-item]
        modality: torch.randn(size, SEGMENTS, INPUT_DIMS[modality]) for modality in MODALITIES
    }
    batch.update(
        {
            f"{m}_lengths": torch.full((size,), SEGMENTS, dtype=torch.long) for m in MODALITIES
        }  # type: ignore[typeddict-item]
    )
    batch["presence"] = {m: torch.ones(size, dtype=torch.long) for m in MODALITIES}
    batch["phq8_score"] = torch.randint(0, 24, (size,))
    batch["label"] = torch.randint(0, 2, (size,))
    if quality:
        batch["quality"] = {
            m: torch.cat(
                [torch.ones(size, SEGMENTS, 1), torch.rand(size, SEGMENTS, QUALITY_DIMS[m] - 1)],
                dim=-1,
            )
            for m in MODALITIES
        }
    return batch


def test_forward_shapes() -> None:
    outputs = _model()(_batch())
    assert outputs["logit"].shape == (3,)
    assert outputs["phq_pred"].shape == (3,)


def test_runs_without_quality() -> None:
    """Batches with no measured quality (distillation anchors) still work."""
    outputs = _model()(_batch(quality=False))
    assert torch.isfinite(outputs["logit"]).all()


def test_mismatched_segment_counts_are_rejected() -> None:
    batch = _batch()
    batch["audio"] = batch["audio"][:, :2]
    with pytest.raises(ValueError, match="one segment count"):
        _model()(batch)


def test_sample_output_is_independent_of_batch_mates() -> None:
    model = _model().eval()
    batch = _batch(size=4)
    full = model(batch)["logit"]

    single: Batch = {  # type: ignore[typeddict-item]
        key: (
            {name: tensor[:1] for name, tensor in value.items()}
            if isinstance(value, dict)
            else value[:1]
        )
        for key, value in batch.items()
    }
    assert torch.allclose(model(single)["logit"], full[:1], atol=1e-6)


def test_a_sample_with_no_valid_segment_stays_finite() -> None:
    model = _model()
    batch = _batch()
    for modality in MODALITIES:
        batch["quality"][modality][0, :, 0] = 0.0
    outputs = model(batch)
    outputs["logit"].sum().backward()

    assert torch.isfinite(outputs["logit"]).all()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_modality_specific_parameters_stay_in_their_own_group() -> None:
    model = _model()
    groups = map_parameter_groups(model)
    for modality in MODALITIES:
        assert groups[modality], f"{modality} has no parameters of its own"
    shared = [name for name, _ in model.named_parameters() if param_group(name) == SHARED_GROUP]
    # Only genuinely cross-modal parts may be shared: the post-fusion projection,
    # the temporal pooler, and the heads.
    assert all(
        name.startswith(("fusion.project", "temporal", "classifier", "regressor"))
        for name in shared
    ), shared


def test_opacus_accepts_the_architecture() -> None:
    model = _model()
    assert ModuleValidator.validate(model, strict=False) == []

    wrapped = GradSampleModule(model)
    batch = _batch(size=3)
    # Both heads, so every parameter is on some path to the loss (the regression
    # head is not reached by the classification logit alone).
    outputs = wrapped(batch)
    (outputs["logit"].sum() + outputs["phq_pred"].sum()).backward()

    named = dict(wrapped.named_parameters())
    missing = [name for name, p in named.items() if getattr(p, "grad_sample", None) is None]
    assert not missing, f"no per-sample gradient for {missing}"
    for name, param in named.items():
        assert param.grad_sample.shape[0] == 3, name
