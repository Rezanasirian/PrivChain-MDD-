"""Invariants of the capability-conditioned logit MoE (ADR-0028).

The architecture's whole claim is about what happens when a modality is absent,
so these tests pin that behaviour directly rather than checking that a forward
pass returns a tensor of the right shape.
"""

from __future__ import annotations

import math

import pytest
import torch

from privchain.config import load_baseline_config
from privchain.data.mock_daic_woz import MODALITIES, Sample, collate_fn
from privchain.encoders.sequence_encoder import sinusoidal_positions
from privchain.federated.capability import SHARED_GROUP, param_group
from privchain.fusion.factory import build_depression_model

SEGMENTS = 4
DIMS = {"audio": 6, "video": 5, "text": 8}
QUALITY_DIMS = {"audio": 3, "video": 4, "text": 3}


def _sample(*, valid: dict[str, float] | None = None) -> Sample:
    """One segment-aligned sample; ``valid`` overrides a modality's valid flag."""
    flags = valid or {}
    sample: Sample = {  # type: ignore[typeddict-item]
        modality: torch.randn(SEGMENTS, DIMS[modality]) for modality in MODALITIES
    }
    sample["presence"] = {m: torch.tensor(1, dtype=torch.long) for m in MODALITIES}
    sample["phq8_score"] = torch.tensor(10, dtype=torch.long)
    sample["label"] = torch.tensor(1, dtype=torch.long)
    sample["quality"] = {
        m: torch.cat(
            [
                torch.full((SEGMENTS, 1), flags.get(m, 1.0)),
                torch.ones(SEGMENTS, QUALITY_DIMS[m] - 1),
            ],
            dim=-1,
        )
        for m in MODALITIES
    }
    return sample


def _model(**overrides: object) -> torch.nn.Module:
    base = load_baseline_config("configs/baseline.yaml")
    update: dict[str, object] = {
        "architecture": "capability_moe",
        "use_phq_regression": False,
        **overrides,
    }
    # model_validate, not model_copy: the latter skips validation, so a test
    # asserting that a bad gate_bias is rejected would pass vacuously.
    config = type(base.model).model_validate({**base.model.model_dump(), **update})
    model = build_depression_model(DIMS, config, QUALITY_DIMS)
    model.eval()  # dropout off, so two forwards of the same input agree
    return model


def _presence(**flags: int) -> dict[str, torch.Tensor]:
    return {m: torch.tensor([flags.get(m, 0)]) for m in MODALITIES}


# ── The mixture ──────────────────────────────────────────────────────────────


def test_absent_modality_gets_exactly_zero_weight() -> None:
    """Not "small": zero. A residual weight is how a masked branch leaks back in."""
    model = _model()
    batch = collate_fn([_sample()])

    model(batch, _presence(text=1))

    assert model.last_weights["audio"].item() == 0.0
    assert model.last_weights["video"].item() == 0.0
    assert model.last_weights["text"].item() == pytest.approx(1.0)


def test_present_weights_renormalize_to_one() -> None:
    model = _model()
    batch = collate_fn([_sample()])

    model(batch, _presence(audio=1, text=1))

    total = sum(model.last_weights[m].item() for m in MODALITIES)
    assert total == pytest.approx(1.0)
    assert model.last_weights["video"].item() == 0.0


def test_single_modality_prediction_is_that_expert_alone() -> None:
    """An audio-only client must be scored by the audio expert, not by a remnant."""
    model = _model()
    batch = collate_fn([_sample()])

    fused = model(batch, _presence(audio=1))["logit"]

    quality = batch["quality"]["audio"]
    # The model adds position codes before the branch when configured to, so the
    # reference has to see the same input or this compares two different things.
    features = batch["audio"]
    if model.config.temporal.positional:
        features = features + sinusoidal_positions(
            features.shape[1], features.shape[2], features.device
        )
    expert_logit, _, _ = model.encoders["audio"](features, quality, quality[..., 0])
    assert fused.item() == pytest.approx(expert_logit.item(), abs=1e-5)


def test_a_sample_with_no_usable_modality_scores_zero_without_nan() -> None:
    """Reachable: modality dropout, or a session whose every segment is empty."""
    model = _model()
    batch = collate_fn([_sample(valid={m: 0.0 for m in MODALITIES})])

    outputs = model(batch, _presence(audio=1, video=1, text=1))

    assert not torch.isnan(outputs["logit"]).any()
    assert outputs["logit"].item() == 0.0
    assert all(model.last_weights[m].item() == 0.0 for m in MODALITIES)


def test_a_modality_with_no_valid_segment_is_dropped_even_when_present() -> None:
    """Presence says the client holds audio; the data says this session has none."""
    model = _model()
    batch = collate_fn([_sample(valid={"audio": 0.0})])

    model(batch, _presence(audio=1, text=1))

    assert model.last_weights["audio"].item() == 0.0
    assert model.last_weights["text"].item() == pytest.approx(1.0)


def test_one_modality_cannot_change_another_branch_logit() -> None:
    """Late fusion's point: no shared projection for a weak branch to disturb."""
    model = _model()
    batch = collate_fn([_sample()])
    quality = batch["quality"]["text"]

    alone, _, _ = model.encoders["text"](batch["text"], quality, quality[..., 0])
    noisy = collate_fn([_sample()])
    noisy["text"] = batch["text"]
    noisy["quality"]["text"] = quality
    together, _, _ = model.encoders["text"](noisy["text"], quality, quality[..., 0])

    assert alone.item() == pytest.approx(together.item())


# ── The recorded prior ───────────────────────────────────────────────────────


def test_gate_bias_prior_starts_the_mixture_where_the_ladder_left_it() -> None:
    """text=+2, audio=video=0 is ~0.79/0.11/0.11 before any training."""
    expected = math.exp(2.0) / (math.exp(2.0) + 2.0)
    model = _model(moe={"gate_bias": {"audio": 0.0, "video": 0.0, "text": 2.0}})
    # Zero the scorer's weights so only the bias speaks; the prior is the bias.
    for modality in MODALITIES:
        torch.nn.init.zeros_(model.encoders[modality].gate[-1].weight)
        torch.nn.init.zeros_(model.encoders[modality].gate[0].weight)
        torch.nn.init.zeros_(model.encoders[modality].gate[0].bias)
    batch = collate_fn([_sample()])

    model(batch, _presence(audio=1, video=1, text=1))

    assert model.last_weights["text"].item() == pytest.approx(expected, abs=1e-4)
    assert model.last_weights["audio"].item() == pytest.approx((1 - expected) / 2, abs=1e-4)


def test_gate_bias_rejects_an_unknown_modality() -> None:
    with pytest.raises(ValueError, match="unknown modalities"):
        _model(moe={"gate_bias": {"txet": 1.0}})


# ── Contracts the rest of the system relies on ───────────────────────────────


def test_every_parameter_is_owned_by_a_modality() -> None:
    """Late fusion has no cross-modal weights, so nothing should land in `shared`."""
    model = _model()

    groups = {param_group(name) for name, _ in model.named_parameters()}

    assert groups == set(MODALITIES)
    assert SHARED_GROUP not in groups


def test_masked_branch_receives_no_gradient() -> None:
    """A zero weight must also mean no update, or DP charges the wrong budget."""
    model = _model()
    model.train()
    batch = collate_fn([_sample()])

    model(batch, _presence(text=1))["logit"].sum().backward()

    assert model.encoders["video"].expert.weight.grad is None or torch.all(
        model.encoders["video"].expert.weight.grad == 0
    )
    assert model.encoders["text"].expert.weight.grad is not None
    assert torch.any(model.encoders["text"].expert.weight.grad != 0)


def test_single_modality_mask_still_trains_that_expert() -> None:
    """The weak experts learn from the masks where they stand alone."""
    model = _model()
    model.train()
    batch = collate_fn([_sample()])

    model(batch, _presence(audio=1))["logit"].sum().backward()

    grad = model.encoders["audio"].expert.weight.grad
    assert grad is not None and torch.any(grad != 0)


def test_mismatched_segment_counts_are_refused() -> None:
    """Silent misalignment would make "capability" mean a different span per branch."""
    model = _model()
    batch = collate_fn([_sample()])
    batch["audio"] = batch["audio"][:, :2]

    with pytest.raises(ValueError, match="one segment count"):
        model(batch)


def test_phq_head_is_mixed_by_the_same_weights() -> None:
    model = _model(use_phq_regression=True)
    batch = collate_fn([_sample()])

    outputs = model(batch, _presence(text=1))

    assert "phq_pred" in outputs
    assert not torch.isnan(outputs["phq_pred"]).any()
