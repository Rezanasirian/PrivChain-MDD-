"""Quality-aware fusion: competition, exact zeros, and no NaN anywhere.

The failure this module has to rule out is silent: a softmax over three ``-inf``
scores is ``NaN``, and a segment where no modality is valid is reachable on real
data (a silent participant) and on purpose (modality dropout).
"""

from __future__ import annotations

import torch

from privchain.fusion.quality_gated_fusion import QualityGatedFusion

MODALITIES = ["audio", "video", "text"]


def _fusion(embed_dim: int = 4, hidden_dim: int = 3) -> QualityGatedFusion:
    torch.manual_seed(0)
    return QualityGatedFusion(MODALITIES, embed_dim, hidden_dim)


def _inputs(batch: int = 2, steps: int = 3, dim: int = 4) -> tuple[dict, dict]:
    torch.manual_seed(1)
    embeddings = {m: torch.randn(batch, steps, dim) for m in MODALITIES}
    scores = {m: torch.randn(batch, steps) for m in MODALITIES}
    return embeddings, scores


def test_weights_sum_to_one_over_valid_modalities() -> None:
    fusion = _fusion()
    embeddings, scores = _inputs()
    valid = {m: torch.ones(2, 3) for m in MODALITIES}
    fused, any_valid = fusion(embeddings, scores, valid)

    assert fused.shape == (2, 3, 3)
    assert bool(any_valid.all())
    total = sum(fusion.last_gates[m] for m in MODALITIES)
    assert torch.allclose(total, torch.ones(2, 3), atol=1e-6)


def test_absent_modality_gets_exactly_zero_and_others_renormalize() -> None:
    fusion = _fusion()
    embeddings, scores = _inputs()
    valid = {m: torch.ones(2, 3) for m in MODALITIES}
    valid["audio"] = torch.zeros(2, 3)
    fusion(embeddings, scores, valid)

    assert float(fusion.last_gates["audio"].abs().max()) == 0.0
    remaining = fusion.last_gates["video"] + fusion.last_gates["text"]
    assert torch.allclose(remaining, torch.ones(2, 3), atol=1e-6)


def test_all_invalid_segment_fuses_to_zero_without_nan() -> None:
    fusion = _fusion()
    embeddings, scores = _inputs()
    valid = {m: torch.ones(2, 3) for m in MODALITIES}
    for modality in MODALITIES:
        valid[modality][0, 1] = 0.0  # an interior segment with nothing valid

    fused, any_valid = fusion(embeddings, scores, valid)

    assert torch.isfinite(fused).all()
    assert not bool(any_valid[0, 1])
    assert float(fused[0, 1].abs().sum().item()) == 0.0
    for modality in MODALITIES:
        assert float(fusion.last_gates[modality][0, 1]) == 0.0
    # The other segments of the same sample are unaffected.
    assert bool(any_valid[0, 0]) and bool(any_valid[0, 2])


def test_gradients_stay_finite_when_everything_is_invalid() -> None:
    """A sample stripped of every modality must not poison the batch's gradients."""
    fusion = _fusion()
    embeddings, scores = _inputs()
    for tensor in embeddings.values():
        tensor.requires_grad_(True)
    valid = {m: torch.ones(2, 3) for m in MODALITIES}
    for modality in MODALITIES:
        valid[modality][0] = 0.0

    fused, _ = fusion(embeddings, scores, valid)
    fused.sum().backward()

    for tensor in embeddings.values():
        assert torch.isfinite(tensor.grad).all()


def test_a_better_score_wins_weight_from_the_others() -> None:
    """The softmax makes the modalities compete rather than each score alone."""
    fusion = _fusion()
    embeddings, scores = _inputs()
    valid = {m: torch.ones(2, 3) for m in MODALITIES}
    fusion(embeddings, scores, valid)
    before = float(fusion.last_gates["text"][0, 0])

    scores["text"] = scores["text"] + 5.0
    fusion(embeddings, scores, valid)
    after = float(fusion.last_gates["text"][0, 0])
    assert after > before
    assert float(fusion.last_gates["audio"][0, 0]) < 0.5
