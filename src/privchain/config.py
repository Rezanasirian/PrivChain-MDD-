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
from pydantic import BaseModel, ConfigDict, Field


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
