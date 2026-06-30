"""Typed configuration loading and validation.

Phase 0 (Environment & Data Setup). No hyperparameter is hardcoded in source;
everything is loaded from ``configs/*.yaml`` and validated with ``pydantic``
(see CLAUDE.md §3). This module currently models the pieces Phase 0 needs (the
mock-data config); later phases extend it with their own validated sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Base model that forbids unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ModalityShapeConfig(_Strict):
    """Sequence-length bounds for one modality's synthetic features."""

    min_frames: int = Field(gt=0)
    max_frames: int = Field(gt=0)

    def validated(self) -> ModalityShapeConfig:
        """Return self after checking ``min_frames <= max_frames``.

        Returns:
            The same instance, once validated.

        Raises:
            ValueError: If ``min_frames`` exceeds ``max_frames``.
        """
        if self.min_frames > self.max_frames:
            raise ValueError(
                f"min_frames ({self.min_frames}) must be <= max_frames ({self.max_frames})"
            )
        return self


class AudioConfig(ModalityShapeConfig):
    """Synthetic log-mel acoustic feature config."""

    n_mels: int = Field(gt=0)


class VideoConfig(ModalityShapeConfig):
    """Synthetic facial-feature config."""

    n_features: int = Field(gt=0)


class TextConfig(_Strict):
    """Synthetic transcript token-embedding config."""

    embed_dim: int = Field(gt=0)
    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)


class DataConfig(_Strict):
    """Mock DAIC-WOZ dataset configuration (Phase 0)."""

    num_sessions: int = Field(gt=0)
    root: str
    phq8_max: int = Field(gt=0)
    depression_cutoff: int = Field(ge=0)
    audio: AudioConfig
    video: VideoConfig
    text: TextConfig


class EncoderConfig(_Strict):
    """Per-modality sequence-encoder hyperparameters (Phase 1)."""

    type: Literal["mean", "gru"] = "gru"
    hidden_dim: int = Field(gt=0)
    out_dim: int = Field(gt=0)
    bidirectional: bool = True
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class FusionConfig(_Strict):
    """Multimodal fusion hyperparameters (Phase 1)."""

    type: Literal["concat"] = "concat"
    hidden_dim: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class HeadConfig(_Strict):
    """Prediction-head hyperparameters (Phase 1)."""

    hidden_dim: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelConfig(_Strict):
    """Multimodal baseline-model schema (Phase 1)."""

    encoder: EncoderConfig
    fusion: FusionConfig
    head: HeadConfig
    use_phq_regression: bool = True
    phq_loss_weight: float = Field(default=0.1, ge=0.0)


class TrainConfig(_Strict):
    """Centralized-training schema (Phase 1)."""

    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    val_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    num_workers: int = Field(default=0, ge=0)
    output_dir: str = "experiments"
    run_name: str = "phase1_centralized_baseline"


class BaselineConfig(_Strict):
    """Top-level schema for ``configs/baseline.yaml``."""

    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def modality_input_dims(data: DataConfig) -> dict[str, int]:
    """Return the per-modality input feature dimensions from the data config.

    Args:
        data: Validated data configuration.

    Returns:
        Mapping ``{"audio": ..., "video": ..., "text": ...}`` of input dims,
        matching the feature dimension produced by the dataset for each modality.
    """
    return {
        "audio": data.audio.n_mels,
        "video": data.video.n_features,
        "text": data.text.embed_dim,
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the document's top level is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Top level of {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def load_baseline_config(path: str | Path) -> BaselineConfig:
    """Load and validate ``configs/baseline.yaml``.

    Args:
        path: Path to the baseline config file.

    Returns:
        The validated :class:`BaselineConfig`.
    """
    return BaselineConfig.model_validate(load_yaml(path))


# ── Federated configuration (Phase 2) ────────────────────────────────────────

# Capability-vector order used project-wide.
CAPABILITY_MODALITIES: tuple[str, str, str] = ("audio", "video", "text")


class ModalityPattern(_Strict):
    """A client modality-access pattern and its share of the population."""

    name: str
    capability: list[int]  # one-hot-ish [audio, video, text], values in {0, 1}
    fraction: float = Field(gt=0.0, le=1.0)

    @field_validator("capability")
    @classmethod
    def _check_capability(cls, value: list[int]) -> list[int]:
        """Validate the capability vector has length 3, is binary, and non-empty."""
        if len(value) != len(CAPABILITY_MODALITIES):
            raise ValueError(f"capability must have length {len(CAPABILITY_MODALITIES)}")
        if any(v not in (0, 1) for v in value):
            raise ValueError("capability entries must be 0 or 1")
        if sum(value) == 0:
            raise ValueError("a client must have at least one modality")
        return value


class FederationConfig(_Strict):
    """Federated-population schema (Phase 2)."""

    num_clients: int = Field(gt=0)
    num_rounds: int = Field(gt=0)
    clients_per_round: int = Field(gt=0)
    local_epochs: int = Field(gt=0)
    modality_patterns: list[ModalityPattern]


class AggregationConfig(_Strict):
    """Aggregation-strategy schema (Phase 2 baseline; extended in Phase 4)."""

    strategy: Literal["fedavg"] = "fedavg"
    reputation_weighting: bool = False
    federated_distillation: bool = False


class FederatedConfig(_Strict):
    """Top-level schema for ``configs/federated.yaml``."""

    seed: int
    federation: FederationConfig
    aggregation: AggregationConfig


def load_federated_config(path: str | Path) -> FederatedConfig:
    """Load and validate ``configs/federated.yaml``.

    Args:
        path: Path to the federated config file.

    Returns:
        The validated :class:`FederatedConfig`.
    """
    return FederatedConfig.model_validate(load_yaml(path))


# ── Privacy configuration (Phase 3, objective H1) ────────────────────────────


class ModalityPrivacy(_Strict):
    """Per-modality base budget and re-identification risk."""

    epsilon: float = Field(gt=0.0)
    reidentification_risk: float = Field(ge=0.0, le=1.0)


class AllocationConfig(_Strict):
    """How per-modality target budgets are derived."""

    mode: Literal["explicit", "inverse_risk"] = "explicit"
    total_epsilon: float = Field(default=14.0, gt=0.0)
    risk_sharpness: float = Field(default=1.0, ge=0.0)  # gamma


class PrivacySweepConfig(_Strict):
    """Accuracy-vs-epsilon sweep settings (Phase 3 Definition of Done)."""

    target_epsilons: list[float]
    epochs: int = Field(default=3, gt=0)

    @field_validator("target_epsilons")
    @classmethod
    def _positive_nonempty(cls, value: list[float]) -> list[float]:
        """Require a non-empty list of positive epsilon values."""
        if not value:
            raise ValueError("target_epsilons must be non-empty")
        if any(v <= 0 for v in value):
            raise ValueError("target_epsilons must all be positive")
        return value


class PrivacySettings(_Strict):
    """The ``privacy`` block of ``configs/privacy.yaml``."""

    delta: float = Field(gt=0.0, lt=1.0)
    accountant: Literal["rdp"] = "rdp"
    max_grad_norm: float = Field(gt=0.0)
    allocation: AllocationConfig
    per_modality: dict[str, ModalityPrivacy]
    sweep: PrivacySweepConfig


class PrivacyConfig(_Strict):
    """Top-level schema for ``configs/privacy.yaml``."""

    seed: int
    privacy: PrivacySettings


def load_privacy_config(path: str | Path) -> PrivacyConfig:
    """Load and validate ``configs/privacy.yaml``.

    Args:
        path: Path to the privacy config file.

    Returns:
        The validated :class:`PrivacyConfig`.
    """
    return PrivacyConfig.model_validate(load_yaml(path))
