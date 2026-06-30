"""Train/validation DataLoader construction (Phase 1).

Builds a reproducible train/val split over a multimodal dataset and wraps each
split in a DataLoader using the project's padding :func:`collate_fn`.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from privchain.config import DataConfig, TrainConfig
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn


def split_dataset(
    dataset: Dataset[Sample], val_fraction: float, seed: int
) -> tuple[Dataset[Sample], Dataset[Sample]]:
    """Split a dataset into train/val subsets reproducibly.

    Args:
        dataset: The full dataset (must support ``len``).
        val_fraction: Fraction assigned to validation, in ``(0, 1)``.
        seed: Seed for the split generator.

    Returns:
        ``(train_subset, val_subset)``.

    Raises:
        ValueError: If the split would leave either subset empty.
    """
    total = len(dataset)  # type: ignore[arg-type]
    val_size = int(round(total * val_fraction))
    train_size = total - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"val_fraction={val_fraction} on {total} samples yields an empty split"
        )
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def build_train_val_loaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    seed: int,
    dataset: Dataset[Sample] | None = None,
) -> tuple[DataLoader[Sample], DataLoader[Sample]]:
    """Build train/val DataLoaders over the mock dataset (or a provided one).

    Args:
        data_config: Validated mock-data config (used to build the default
            dataset when ``dataset`` is not supplied).
        train_config: Training config (batch size, val fraction, workers).
        seed: Base seed for generation and the split.
        dataset: Optional pre-built dataset (e.g., a real DAIC-WOZ dataset that
            follows the same :class:`Sample` contract). When ``None``, a
            :class:`MockDaicWozDataset` is constructed.

    Returns:
        ``(train_loader, val_loader)`` yielding padded batches.
    """
    if dataset is None:
        dataset = MockDaicWozDataset(data_config, seed=seed)

    train_subset, val_subset = split_dataset(dataset, train_config.val_fraction, seed)

    train_loader: DataLoader[Sample] = DataLoader(
        train_subset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader: DataLoader[Sample] = DataLoader(
        val_subset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers,
    )
    return train_loader, val_loader
