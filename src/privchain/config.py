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

    # stats: AVEC2017-style functionals over the whole session (mean/std/min/max/
    #        delta) then an MLP — the tractable choice at 107 training sessions.
    # mean:  learned projection then masked mean-pool.
    # gru:   bidirectional DPGRU over the sequence (needs far more data).
    type: Literal["mean", "gru", "stats"] = "gru"
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
    # Partial per-modality overrides layered onto `encoder`, e.g. giving text a
    # different encoder type. Text arrives as a document-level embedding (a
    # length-1 "sequence"), where session functionals are degenerate — see
    # `encoder_for` and ADR-0014.
    encoder_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fusion: FusionConfig
    head: HeadConfig
    use_phq_regression: bool = True
    phq_loss_weight: float = Field(default=0.1, ge=0.0)

    def encoder_for(self, modality: str) -> EncoderConfig:
        """Return the encoder config for one modality, applying any override.

        Args:
            modality: ``"audio"``, ``"video"``, or ``"text"``.

        Returns:
            The base encoder config, or a validated copy with the modality's
            overrides applied.
        """
        override = self.encoder_overrides.get(modality)
        if not override:
            return self.encoder
        return EncoderConfig.model_validate({**self.encoder.model_dump(), **override})


class TrainConfig(_Strict):
    """Centralized-training schema (Phase 1).

    ``val_fraction`` only applies when no dedicated validation set is supplied;
    on real DAIC-WOZ the official dev split is used instead (ADR-0011).
    """

    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    val_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    num_workers: int = Field(default=0, ge=0)
    output_dir: str = "experiments"
    run_name: str = "phase1_centralized_baseline"

    # ── Class imbalance ──────────────────────────────────────────────────────
    # DAIC-WOZ is ~28% positive; unweighted BCE collapses onto the majority
    # class (F1 = 0). When true, the BCE positive class is weighted by
    # neg/pos counted from the training split (ADR-0011).
    class_weighting: bool = False

    # ── Model selection / early stopping ─────────────────────────────────────
    # Metric used to pick the best checkpoint and to drive early stopping.
    selection_metric: Literal["roc_auc", "f1"] = "roc_auc"
    # Stop after this many epochs without improvement; None disables it.
    early_stopping_patience: int | None = Field(default=None, gt=0)

    # ``auto`` selects CUDA when available, else CPU.
    device: Literal["auto", "cpu", "cuda"] = "cpu"


class BaselineConfig(_Strict):
    """Top-level schema for ``configs/baseline.yaml``."""

    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def resolve_device(device: str) -> str:
    """Resolve a configured device string to a concrete torch device.

    Args:
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``. ``"auto"`` selects CUDA
            when it is available and falls back to CPU otherwise.

    Returns:
        Either ``"cuda"`` or ``"cpu"``.
    """
    if device != "auto":
        return device
    import torch  # imported here so config loading stays torch-free

    return "cuda" if torch.cuda.is_available() else "cpu"


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


class ReputationConfig(_Strict):
    """Reputation-scoring hyperparameters (Phase 4, objective H2).

    Reputation blends a client's relative data volume with the per-group
    consistency of its update (agreement with the subgraph consensus), smoothed
    across rounds by an EMA. See ADR-0005.
    """

    volume_weight: float = Field(default=0.5, ge=0.0, le=1.0)  # alpha
    ema_decay: float = Field(default=0.8, ge=0.0, lt=1.0)
    min_reputation: float = Field(default=0.05, gt=0.0, le=1.0)


class DistillationConfig(_Strict):
    """Federated-distillation hyperparameters (Phase 4, objective H2).

    The frozen global model at the start of a round acts as the teacher; a
    client's local training adds a soft-target term matching the teacher's
    predictions, transferring cross-modal knowledge to missing-modality clients.
    """

    weight: float = Field(default=0.5, ge=0.0)
    temperature: float = Field(default=2.0, gt=0.0)
    apply_to: Literal["missing_modality", "all"] = "missing_modality"


class AggregationConfig(_Strict):
    """Aggregation-strategy schema (Phase 2 baseline; extended in Phase 4)."""

    strategy: Literal["fedavg", "capability_aware"] = "fedavg"
    reputation_weighting: bool = False
    federated_distillation: bool = False
    byzantine_filter: bool = False  # drop shared-group outlier updates (Phase 5)
    byzantine_z: float = Field(default=2.5, gt=0.0)  # robust-std threshold
    reputation: ReputationConfig = Field(default_factory=ReputationConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)


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


# ── Blockchain / ledger configuration (Phase 5, objective H3) ─────────────────


class LedgerConfig(_Strict):
    """How the Python bridge talks to the audit ledger.

    ``mock`` uses the in-memory :class:`~privchain.chain_client.ledger.MockLedger`
    (offline, enforces the same invariants as the chaincode); ``fabric_rest``
    targets a live Fabric REST gateway fronting the ``privchain-cc`` chaincode.
    """

    backend: Literal["mock", "fabric_rest"] = "mock"
    channel: str = "privchain-channel"
    chaincode: str = "privchain-cc"
    gateway_url: str = "http://localhost:8801"
    timeout_seconds: float = Field(default=10.0, gt=0.0)


class BlockchainConfig(_Strict):
    """Top-level schema for ``configs/blockchain.yaml``."""

    ledger: LedgerConfig


def load_blockchain_config(path: str | Path) -> BlockchainConfig:
    """Load and validate ``configs/blockchain.yaml``.

    Args:
        path: Path to the blockchain config file.

    Returns:
        The validated :class:`BlockchainConfig`.
    """
    return BlockchainConfig.model_validate(load_yaml(path))


# ── Attacker-model / privacy-evaluation configuration (Phase 6, objective H5) ─


class MembershipInferenceConfig(_Strict):
    """Membership-inference attack settings."""

    enabled: bool = True


class AttackSettings(_Strict):
    """The ``attack`` block of ``configs/attack.yaml``.

    A re-identification attacker sees several "views" per subject; some are
    enrolled (build the subject's template) and the rest are probed.

    Two different privacy mechanisms are measured against the same attackers, and
    the config separates them deliberately (ADR-0007):

    * **DP-SGD training** — the model is trained at the swept ε. This is what
      bounds membership inference.
    * **Embedding release** — each released embedding is clipped to
      ``embedding_clip_norm`` (bounding its sensitivity) and perturbed by the
      Gaussian mechanism calibrated to the same ε. This is what bounds
      re-identification; DP-SGD alone does not, since an encoder may map an
      unseen subject to a distinctive point regardless of how it was trained.
    """

    num_views: int = Field(default=6, gt=1)
    enroll_views: int = Field(default=3, gt=0)
    jitter: float = Field(default=0.1, ge=0.0)  # intra-subject feature variability
    # L2 bound enforced on a released embedding: its sensitivity, and therefore
    # the scale the Gaussian mechanism's sigma multiplies.
    embedding_clip_norm: float = Field(default=1.0, gt=0.0)
    delta: float = Field(default=1.0e-5, gt=0.0, lt=1.0)
    target_epsilons: list[float]
    membership_inference: MembershipInferenceConfig = Field(
        default_factory=MembershipInferenceConfig
    )

    @field_validator("target_epsilons")
    @classmethod
    def _positive_nonempty(cls, value: list[float]) -> list[float]:
        """Require a non-empty list of positive epsilon values."""
        if not value:
            raise ValueError("target_epsilons must be non-empty")
        if any(v <= 0 for v in value):
            raise ValueError("target_epsilons must all be positive")
        return value

    def validated(self) -> AttackSettings:
        """Return self after checking ``enroll_views < num_views``.

        Returns:
            The same instance, once validated.

        Raises:
            ValueError: If there are not enough views left to probe.
        """
        if self.enroll_views >= self.num_views:
            raise ValueError(
                f"enroll_views ({self.enroll_views}) must be < num_views ({self.num_views})"
            )
        return self


class AttackConfig(_Strict):
    """Top-level schema for ``configs/attack.yaml``."""

    seed: int
    attack: AttackSettings


def load_attack_config(path: str | Path) -> AttackConfig:
    """Load and validate ``configs/attack.yaml``.

    Args:
        path: Path to the attack config file.

    Returns:
        The validated :class:`AttackConfig` (with view counts cross-checked).
    """
    config = AttackConfig.model_validate(load_yaml(path))
    config.attack.validated()
    return config


# ── Final-evaluation configuration (Phase 7, objective H5) ────────────────────


class EvaluationSettings(_Strict):
    """The ``evaluation`` block of ``configs/evaluation.yaml``.

    Controls the cross-validation protocol shared by every method variant, plus
    the inference-latency benchmark (the compute-budget axis). Epoch/round counts
    are kept modest so the whole comparison runs offline on mock data.
    """

    k_folds: int = Field(default=10, ge=2)
    held_out_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    epochs: int = Field(default=3, gt=0)  # centralized / DP-SGD epochs
    rounds: int = Field(default=5, gt=0)  # federated rounds
    latency_batch_sizes: list[int]
    latency_repeats: int = Field(default=20, gt=0)

    @field_validator("latency_batch_sizes")
    @classmethod
    def _positive_nonempty(cls, value: list[int]) -> list[int]:
        """Require a non-empty list of positive batch sizes."""
        if not value:
            raise ValueError("latency_batch_sizes must be non-empty")
        if any(v <= 0 for v in value):
            raise ValueError("latency_batch_sizes must all be positive")
        return value


class EvaluationConfig(_Strict):
    """Top-level schema for ``configs/evaluation.yaml``."""

    seed: int
    evaluation: EvaluationSettings


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load and validate ``configs/evaluation.yaml``.

    Args:
        path: Path to the evaluation config file.

    Returns:
        The validated :class:`EvaluationConfig`.
    """
    return EvaluationConfig.model_validate(load_yaml(path))
