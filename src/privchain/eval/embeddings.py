"""Per-modality subject-embedding extraction for attacker evaluation (Phase 6).

A re-identification attacker needs several embedding "views" per subject. On the
mock corpus each session is one subject whose random features are a genuine
identity signal (unlike the random depression label), so we synthesize
intra-subject variability by adding small feature-space jitter and encoding each
jittered view through the trained modality encoder. On real DAIC-WOZ the natural
multiple utterances/frames per subject replace the jitter.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from privchain.data.mock_daic_woz import Sample
from privchain.fusion.baseline_model import MultimodalDepressionModel


@torch.no_grad()
def extract_subject_embeddings(
    model: MultimodalDepressionModel,
    dataset: Dataset[Sample],
    modality: str,
    *,
    num_views: int,
    jitter: float,
    seed: int,
    device: torch.device,
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]]:
    """Encode ``num_views`` jittered views per subject for one modality.

    Args:
        model: The trained multimodal model (its ``encoders[modality]`` is used).
        dataset: Dataset of :class:`Sample` items; each item is one subject.
        modality: ``"audio"``, ``"video"``, or ``"text"``.
        num_views: Number of jittered views to generate per subject.
        jitter: Standard deviation of the per-view feature-space noise.
        seed: Base seed for reproducible jitter.
        device: Torch device to run the encoder on.

    Returns:
        ``(embeddings, subject_ids, view_ids)`` where ``embeddings`` has shape
        ``(num_subjects * num_views, out_dim)`` and the two id arrays are aligned
        with its rows.
    """
    model.eval()
    encoder = model.encoders[modality]
    num_subjects = len(dataset)  # type: ignore[arg-type]

    embeddings: list[NDArray[np.float64]] = []
    subject_ids: list[int] = []
    view_ids: list[int] = []
    for subject in range(num_subjects):
        feature = dataset[subject][modality]  # type: ignore[literal-required]
        frames, feat_dim = feature.shape
        generator = torch.Generator().manual_seed(seed + subject)
        noise = torch.randn((num_views, frames, feat_dim), generator=generator) * jitter
        views = feature.unsqueeze(0) + noise
        lengths = torch.full((num_views,), frames, dtype=torch.long)

        encoded = encoder(views.to(device), lengths.to(device))
        embeddings.append(encoded.cpu().numpy().astype(np.float64))
        subject_ids.extend([subject] * num_views)
        view_ids.extend(range(num_views))

    return (
        np.concatenate(embeddings, axis=0),
        np.asarray(subject_ids, dtype=np.int_),
        np.asarray(view_ids, dtype=np.int_),
    )


def split_enroll_probe(
    embeddings: NDArray[np.float64],
    subject_ids: NDArray[np.int_],
    view_ids: NDArray[np.int_],
    enroll_views: int,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    """Split views into enrollment (``view_id < enroll_views``) and probe sets.

    Args:
        embeddings: All embeddings, shape ``(N, D)``.
        subject_ids: Subject id per row.
        view_ids: View index per row.
        enroll_views: Number of leading views per subject used for enrollment.

    Returns:
        ``(enroll_embeddings, enroll_subjects, probe_embeddings, probe_subjects)``.
    """
    enroll_mask = view_ids < enroll_views
    probe_mask = ~enroll_mask
    return (
        embeddings[enroll_mask],
        subject_ids[enroll_mask],
        embeddings[probe_mask],
        subject_ids[probe_mask],
    )
