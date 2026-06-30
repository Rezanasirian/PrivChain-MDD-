"""Synthetic, DAIC-WOZ-shaped multimodal dataset.

Phase 0 (Environment & Data Setup), supporting objective H4 (prototype). The
real DAIC-WOZ corpus is access-controlled and must never enter the repo
(CLAUDE.md §7), yet the full audio/video/text pipeline needs *something* with
the right shapes to run against in CI and during early development.

This module generates random-but-reproducible sessions that mirror the three
DAIC-WOZ modalities and PHQ-8 labels:

* **audio** — log-mel acoustic features, shape ``(num_frames, n_mels)``
* **video** — facial features (landmarks/AUs), shape ``(num_frames, n_features)``
* **text**  — transcript token embeddings, shape ``(num_tokens, embed_dim)``
* **phq8_score** — integer in ``[0, phq8_max]``; ``label = score >= cutoff``

Sequence lengths vary per session (as in the real data); :func:`collate_fn`
pads a batch to its longest member and returns per-modality length masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from privchain.config import DataConfig

# Capability vector order used project-wide: [audio, video, text].
MODALITIES: tuple[str, str, str] = ("audio", "video", "text")


class Sample(TypedDict):
    """One synthetic session (variable-length, unbatched)."""

    audio: torch.Tensor  # (T_audio, n_mels)
    video: torch.Tensor  # (T_video, n_features)
    text: torch.Tensor  # (T_text, embed_dim)
    phq8_score: torch.Tensor  # scalar long
    label: torch.Tensor  # scalar long in {0, 1}


class Batch(TypedDict):
    """A padded, collated batch produced by :func:`collate_fn`."""

    audio: torch.Tensor  # (B, T_audio_max, n_mels)
    video: torch.Tensor  # (B, T_video_max, n_features)
    text: torch.Tensor  # (B, T_text_max, embed_dim)
    audio_lengths: torch.Tensor  # (B,) true (unpadded) lengths
    video_lengths: torch.Tensor  # (B,)
    text_lengths: torch.Tensor  # (B,)
    phq8_score: torch.Tensor  # (B,)
    label: torch.Tensor  # (B,)


@dataclass(frozen=True)
class _SessionSpec:
    """Resolved per-session sequence lengths and label, drawn from a seed."""

    audio_frames: int
    video_frames: int
    text_tokens: int
    phq8_score: int


class MockDaicWozDataset(Dataset[Sample]):
    """A reproducible synthetic stand-in for DAIC-WOZ.

    Samples are generated lazily from a per-session seed derived from the
    dataset seed, so ``dataset[i]`` is deterministic and independent of access
    order (important for stable tests and federated client splits).

    Args:
        config: Validated mock-data configuration.
        seed: Base seed; combined with the sample index per session.
    """

    def __init__(self, config: DataConfig, seed: int = 42) -> None:
        self._config = config
        self._seed = seed
        self._specs: list[_SessionSpec] = [
            self._draw_spec(index) for index in range(config.num_sessions)
        ]

    def _rng(self, index: int) -> np.random.Generator:
        """Return a per-session generator (stable across runs and ordering)."""
        return np.random.default_rng([self._seed, index])

    def _draw_spec(self, index: int) -> _SessionSpec:
        """Draw the variable sequence lengths and PHQ-8 score for a session."""
        cfg = self._config
        rng = self._rng(index)
        return _SessionSpec(
            audio_frames=int(rng.integers(cfg.audio.min_frames, cfg.audio.max_frames + 1)),
            video_frames=int(rng.integers(cfg.video.min_frames, cfg.video.max_frames + 1)),
            text_tokens=int(rng.integers(cfg.text.min_tokens, cfg.text.max_tokens + 1)),
            phq8_score=int(rng.integers(0, cfg.phq8_max + 1)),
        )

    def __len__(self) -> int:
        """Return the number of synthetic sessions."""
        return self._config.num_sessions

    def __getitem__(self, index: int) -> Sample:
        """Generate the synthetic multimodal sample at ``index``.

        Args:
            index: Session index in ``[0, len(self))``.

        Returns:
            A :class:`Sample` of float feature tensors plus integer labels.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} sessions")

        cfg = self._config
        spec = self._specs[index]
        # Use a fresh stream (offset seed) so feature values don't correlate
        # with the length-drawing stream above.
        rng = np.random.default_rng([self._seed, index, 1])

        audio = rng.standard_normal((spec.audio_frames, cfg.audio.n_mels), dtype=np.float32)
        video = rng.standard_normal((spec.video_frames, cfg.video.n_features), dtype=np.float32)
        text = rng.standard_normal((spec.text_tokens, cfg.text.embed_dim), dtype=np.float32)
        label = int(spec.phq8_score >= cfg.depression_cutoff)

        return Sample(
            audio=torch.from_numpy(audio),
            video=torch.from_numpy(video),
            text=torch.from_numpy(text),
            phq8_score=torch.tensor(spec.phq8_score, dtype=torch.long),
            label=torch.tensor(label, dtype=torch.long),
        )


def _pad_stack(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length ``(T, D)`` tensors and stack to ``(B, T_max, D)``.

    Args:
        sequences: List of 2-D tensors sharing the same feature dim ``D``.

    Returns:
        A tuple ``(padded, lengths)`` where ``padded`` has shape
        ``(B, T_max, D)`` and ``lengths`` has shape ``(B,)``.
    """
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    feat_dim = sequences[0].shape[1]
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(sequences), max_len, feat_dim), dtype=sequences[0].dtype)
    for row, seq in enumerate(sequences):
        padded[row, : seq.shape[0]] = seq
    return padded, lengths


def collate_fn(samples: list[Sample]) -> Batch:
    """Collate variable-length samples into a padded batch.

    Args:
        samples: A list of :class:`Sample` items from the dataset.

    Returns:
        A :class:`Batch` with each modality padded to the batch's longest
        sequence, accompanied by true per-modality length tensors.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("collate_fn received an empty batch")

    audio, audio_lengths = _pad_stack([s["audio"] for s in samples])
    video, video_lengths = _pad_stack([s["video"] for s in samples])
    text, text_lengths = _pad_stack([s["text"] for s in samples])

    return Batch(
        audio=audio,
        video=video,
        text=text,
        audio_lengths=audio_lengths,
        video_lengths=video_lengths,
        text_lengths=text_lengths,
        phq8_score=torch.stack([s["phq8_score"] for s in samples]),
        label=torch.stack([s["label"] for s in samples]),
    )


def build_dataloader(
    config: DataConfig,
    *,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = False,
) -> DataLoader[Sample]:
    """Build a :class:`~torch.utils.data.DataLoader` over the mock dataset.

    Args:
        config: Validated mock-data configuration.
        batch_size: Number of sessions per batch.
        seed: Base seed for reproducible generation.
        shuffle: Whether to shuffle session order each epoch.

    Returns:
        A DataLoader yielding padded :class:`Batch` objects.
    """
    dataset = MockDaicWozDataset(config, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def write_mock_dataset(config: DataConfig, *, seed: int = 42) -> Path:
    """Write the mock dataset to disk in a DAIC-WOZ-like per-session layout.

    Each session gets ``data/mock/session_XXX/`` containing ``audio.npy``,
    ``video.npy``, ``text.npy``, and ``label.txt``. This is purely for manual
    inspection — tests use the in-memory dataset and do not require these files.
    The output directory is git-ignored (CLAUDE.md §7).

    Args:
        config: Validated mock-data configuration (``config.root`` is the
            destination directory).
        seed: Base seed for reproducible generation.

    Returns:
        The root path the dataset was written to.
    """
    dataset = MockDaicWozDataset(config, seed=seed)
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)

    for index in range(len(dataset)):
        sample = dataset[index]
        session_dir = root / f"session_{index:03d}"
        session_dir.mkdir(exist_ok=True)
        np.save(session_dir / "audio.npy", sample["audio"].numpy())
        np.save(session_dir / "video.npy", sample["video"].numpy())
        np.save(session_dir / "text.npy", sample["text"].numpy())
        score = int(sample["phq8_score"].item())
        label = int(sample["label"].item())
        (session_dir / "label.txt").write_text(
            f"phq8_score={score}\nlabel={label}\n", encoding="utf-8"
        )

    return root
