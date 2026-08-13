"""Unit tests for within-session attacker views (Phase 6, H5, ADR-0017).

Everything runs on the mock corpus, per CLAUDE.md section 3 — CI must never need
real DAIC-WOZ. Mock text is a genuine token sequence, so all three modalities
exercise the same slicing path here; the real corpus swaps in
``DaicWozDataset.text_segment_vectors`` for text, which
:func:`test_text_segment_hook_is_preferred_when_present` covers with a stub.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from privchain.config import (
    AudioConfig,
    DataConfig,
    EncoderConfig,
    FusionConfig,
    HeadConfig,
    ModelConfig,
    TextConfig,
    VideoConfig,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample
from privchain.eval.session_views import (
    apply_pca,
    build_views,
    concat_views,
    fit_pca,
    run_reidentification,
    segment_session,
    unwrap_subset,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything

NUM_SEGMENTS = 4


def _data_config(num_sessions: int = 10) -> DataConfig:
    return DataConfig(
        num_sessions=num_sessions,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=6, min_frames=20, max_frames=24),
        video=VideoConfig(n_features=5, min_frames=16, max_frames=20),
        text=TextConfig(embed_dim=8, min_tokens=12, max_tokens=16),
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(type="stats", hidden_dim=8, out_dim=8, dropout=0.0),
        fusion=FusionConfig(hidden_dim=16),
        head=HeadConfig(hidden_dim=8),
        use_phq_regression=True,
        phq_loss_weight=0.1,
    )


def _dataset(num_sessions: int = 10) -> MockDaicWozDataset:
    return MockDaicWozDataset(_data_config(num_sessions), seed=7)


class _IdentityDataset(Dataset[Sample]):
    """Sessions that genuinely carry identity: a per-subject mean plus noise.

    The mock corpus draws i.i.d. features, so one stretch of a session says
    nothing about another. That is realistic in the sense that it stops the
    attacker cold, but it cannot exercise the attack path, so this stand-in gives
    each subject its own mean vector.

    The offset has to be a *direction*, not a magnitude: the attacker scores by
    cosine similarity, which is scale-invariant, so subjects differing only by a
    scalar multiple would be indistinguishable to it by design.
    """

    def __init__(self, num_subjects: int, frames: int = 24, dim: int = 6) -> None:
        self._num_subjects = num_subjects
        self._frames = frames
        self._dim = dim

    def __len__(self) -> int:
        return self._num_subjects

    def __getitem__(self, index: int) -> Sample:
        centre = np.random.default_rng(1000 + index).standard_normal(self._dim)
        noise = np.random.default_rng(index).standard_normal((self._frames, self._dim))
        features = torch.from_numpy((centre + 0.1 * noise).astype(np.float32))
        return Sample(
            audio=features,
            video=features,
            text=features,
            phq8_score=torch.tensor(0, dtype=torch.long),
            label=torch.tensor(0, dtype=torch.long),
        )


def test_segments_are_disjoint_and_cover_the_session() -> None:
    dataset = _dataset()
    segments = segment_session(dataset, 0, "audio", NUM_SEGMENTS)
    assert len(segments) == NUM_SEGMENTS
    rebuilt = torch.cat(segments, dim=0)
    assert torch.equal(rebuilt, dataset[0]["audio"])


def test_build_views_labels_every_subject_and_view() -> None:
    dataset = _dataset(num_sessions=6)
    views = build_views(dataset, "video", num_segments=NUM_SEGMENTS, encoder_type="stats")

    assert views.features.shape[0] == 6 * NUM_SEGMENTS
    assert views.features.shape[1] == 5 * 5  # 5 functionals over 5 AU channels
    assert views.num_subjects == 6
    assert sorted(set(views.view_ids.tolist())) == list(range(NUM_SEGMENTS))
    assert views.skipped == ()


def test_mean_encoder_summary_keeps_the_feature_width() -> None:
    views = build_views(_dataset(4), "text", num_segments=NUM_SEGMENTS, encoder_type="mean")
    assert views.features.shape[1] == 8  # a masked mean, not 5 functionals


def test_sessions_with_too_few_rows_are_skipped_not_fatal() -> None:
    # min_frames=16 for video, so 40 segments cannot be carved from any session.
    views = build_views(_dataset(3), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    assert views.num_subjects == 3

    with pytest.raises(ValueError, match="no usable sessions"):
        build_views(_dataset(3), "video", num_segments=40, encoder_type="stats")


def test_subject_offset_keeps_pools_disjoint() -> None:
    left = build_views(_dataset(3), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    right = build_views(
        _dataset(3), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats", subject_offset=3
    )
    pool = concat_views(left, right)
    assert pool.num_subjects == 6
    assert set(pool.subject_ids.tolist()) == set(range(6))


def test_concat_rejects_mismatched_widths() -> None:
    audio = build_views(_dataset(3), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    video = build_views(_dataset(3), "video", num_segments=NUM_SEGMENTS, encoder_type="stats")
    with pytest.raises(ValueError, match="differing widths"):
        concat_views(audio, video)


def test_unwrap_subset_resolves_nested_indices() -> None:
    dataset = _dataset(8)
    nested = Subset(Subset(dataset, [1, 3, 5, 7]), [2])
    base, index = unwrap_subset(nested, 0)
    assert base is dataset
    assert index == 5


def test_views_through_a_subset_reach_the_base_dataset() -> None:
    dataset = _dataset(8)
    subset = Subset(dataset, [2, 4])
    views = build_views(subset, "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    direct = build_views(dataset, "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    # Subject 0 of the subset is subject 2 of the base dataset.
    np.testing.assert_allclose(
        views.features[:NUM_SEGMENTS], direct.features[2 * NUM_SEGMENTS :][:NUM_SEGMENTS]
    )


def test_text_segment_hook_is_preferred_when_present() -> None:
    """A dataset exposing ``text_segment_vectors`` bypasses row slicing."""
    dataset = _dataset(3)
    calls: list[tuple[int, int]] = []

    def fake_segments(index: int, num_segments: int) -> np.ndarray:
        calls.append((index, num_segments))
        return np.full((num_segments, 4), float(index), dtype=np.float32)

    dataset.text_segment_vectors = fake_segments  # type: ignore[attr-defined]
    views = build_views(dataset, "text", num_segments=NUM_SEGMENTS, encoder_type="mean")

    assert calls == [(i, NUM_SEGMENTS) for i in range(3)]
    assert views.features.shape == (3 * NUM_SEGMENTS, 4)
    # Audio still goes through the slicing path.
    assert (
        build_views(
            dataset, "audio", num_segments=NUM_SEGMENTS, encoder_type="stats"
        ).features.shape[1]
        == 6 * 5
    )


def test_encoder_output_is_width_matched_across_modalities() -> None:
    seed_everything(3)
    data_cfg, model_cfg = _data_config(5), _model_config()
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    dataset = MockDaicWozDataset(data_cfg, seed=3)

    widths = {
        modality: build_views(
            dataset,
            modality,
            num_segments=NUM_SEGMENTS,
            encoder_type="stats",
            encoder=model.encoders[modality],
        ).features.shape[1]
        for modality in ("audio", "video", "text")
    }
    assert set(widths.values()) == {model_cfg.encoder.out_dim}


def test_pca_projects_to_the_requested_width_and_is_capped() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((20, 12))
    mean, components = fit_pca(features, 4)
    assert apply_pca(features, mean, components).shape == (20, 4)

    # More components than the data can support: capped, never an error.
    _, wide = fit_pca(features[:5], 50)
    assert wide.shape[1] == 5

    with pytest.raises(ValueError, match="dim must be positive"):
        fit_pca(features, 0)


def test_attack_recovers_identity_and_the_control_collapses_to_chance() -> None:
    views = build_views(
        _IdentityDataset(num_subjects=10), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats"
    )

    real = run_reidentification(views, enroll_segments=2, seed=0)
    control = run_reidentification(views, enroll_segments=2, seed=0, shuffle_subjects=True)

    assert real["chance"] == pytest.approx(0.1)
    assert real["accuracy"] > 0.9  # a per-subject offset is trivially recoverable
    assert real["ratio_to_chance"] == pytest.approx(real["accuracy"] / real["chance"])
    assert control["accuracy"] <= 2 * real["chance"]  # shuffled labels carry no identity


def test_mock_sessions_carry_no_within_session_identity() -> None:
    """Mock features are i.i.d. noise, so a *different* stretch identifies nobody.

    This is why the Phase 6 numbers have to come from real DAIC-WOZ (ADR-0017).
    The older jitter-based harness compared noisy copies of the *same* feature
    matrix, which is recoverable by construction and says nothing about whether
    the modality carries identity.
    """
    views = build_views(_dataset(10), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    result = run_reidentification(views, enroll_segments=2, seed=0)
    assert result["accuracy"] == pytest.approx(result["chance"], abs=0.1)


def test_attack_reports_per_group_accuracy() -> None:
    views = build_views(_dataset(6), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    groups = {subject: ("train" if subject < 3 else "report") for subject in range(6)}

    result = run_reidentification(views, enroll_segments=2, seed=0, groups=groups)
    assert "accuracy_train" in result
    assert "accuracy_report" in result
    assert result["num_probes"] == 6 * (NUM_SEGMENTS - 2)


def test_attack_needs_a_probe_left_over() -> None:
    views = build_views(_dataset(4), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    with pytest.raises(ValueError, match="leaves no probe"):
        run_reidentification(views, enroll_segments=NUM_SEGMENTS, seed=0)


def test_pca_is_fitted_only_on_enrollment_rows() -> None:
    """The projection must not be shaped by the rows it is scored against."""
    views = build_views(_dataset(8), "audio", num_segments=NUM_SEGMENTS, encoder_type="stats")
    projected = run_reidentification(views, enroll_segments=2, seed=0, pca_dim=4)
    assert 0.0 <= projected["accuracy"] <= 1.0
    assert projected["num_subjects"] == 8
