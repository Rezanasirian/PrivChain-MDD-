"""Within-session views for the re-identification attacker (Phase 6, H5).

The attacker in :mod:`privchain.eval.attackers` enrols a template per subject and
probes it with *other* views of the same subject. On the mock corpus those views
are manufactured by jittering the features (:mod:`privchain.eval.embeddings`).
Real DAIC-WOZ gives one recording per participant, so the views have to be carved
out of that recording: ``num_segments`` disjoint, contiguous stretches, enrolled
on some and probed with the rest (ADR-0017).

Each segment is turned into a fixed-size vector at one of two levels:

* ``raw`` — the summary the configured encoder consumes *before* its first
  projection (functionals for a ``stats`` encoder, a masked mean for a ``mean``
  one). This is what the modality's features themselves carry, and it is the
  quantity the per-modality ε allocation claims to be calibrated against.
* ``encoded`` — the trained encoder's output. Every modality shares ``out_dim``,
  so this comparison is width-matched by construction.

``raw`` widths differ across modalities (audio 74x5, video 20x5, text 768), and
nearest-centroid accuracy grows with width, so :func:`fit_pca` provides the
width-matched control for the raw level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset

from privchain.data.mock_daic_woz import Sample
from privchain.encoders.sequence_encoder import SequenceEncoder, masked_mean, masked_statistics
from privchain.eval.attackers import ReidentificationAttacker
from privchain.segmentation import contiguous_spans


@dataclass(frozen=True)
class ModalityViews:
    """Segment views of one modality, ready for the attacker.

    Attributes:
        features: Row per view, shape ``(num_kept * num_segments, D)``.
        subject_ids: Subject index per row (indexes the *pool*, not the split).
        view_ids: Segment index within the session, ``0 .. num_segments - 1``.
        skipped: Subject indices dropped because the session had fewer items
            than ``num_segments`` (too few transcript turns, in practice).
    """

    features: NDArray[np.float64]
    subject_ids: NDArray[np.int_]
    view_ids: NDArray[np.int_]
    skipped: tuple[int, ...]

    @property
    def num_subjects(self) -> int:
        """Number of distinct subjects with usable views."""
        return int(np.unique(self.subject_ids).size)


def unwrap_subset(dataset: Dataset[Sample], index: int) -> tuple[Dataset[Sample], int]:
    """Resolve a possibly-nested :class:`Subset` to its base dataset and index.

    The protocol splits (:mod:`privchain.training.protocol`) are ``Subset``
    views, and ``Subset`` does not forward attribute lookups — so without this,
    a real-data split would lose the ``text_segment_vectors`` method and fall
    back to slicing text that has already been collapsed to a single row.

    Args:
        dataset: A dataset, possibly wrapped in one or more ``Subset`` layers.
        index: Index into ``dataset``.

    Returns:
        ``(base_dataset, base_index)``.
    """
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    return dataset, index


def segment_session(
    dataset: Dataset[Sample], index: int, modality: str, num_segments: int
) -> list[torch.Tensor]:
    """Cut one session's modality into ``num_segments`` contiguous views.

    Text on the real corpus is a special case: :class:`DaicWozDataset` collapses
    a whole transcript to a single ``(1, D)`` row, so there is nothing to slice.
    When the dataset offers ``text_segment_vectors`` the stretches are embedded
    from the turn list instead; otherwise — the mock corpus, whose text is a
    genuine token sequence — every modality is sliced the same way.

    Args:
        dataset: Dataset of :class:`Sample` items, one per subject.
        index: Session index.
        modality: ``"audio"``, ``"video"`` or ``"text"``.
        num_segments: Number of contiguous views to produce.

    Returns:
        ``num_segments`` tensors of shape ``(T_i, D)``, in chronological order.

    Raises:
        ValueError: If the session has fewer rows/turns than ``num_segments``.
    """
    base, base_index = unwrap_subset(dataset, index)
    segmenter = getattr(base, "text_segment_vectors", None)
    if modality == "text" and segmenter is not None:
        vectors: NDArray[np.float32] = segmenter(base_index, num_segments)
        return [torch.from_numpy(row).unsqueeze(0) for row in vectors]

    matrix = base[base_index][modality]  # type: ignore[literal-required]
    return [matrix[start:stop] for start, stop in contiguous_spans(len(matrix), num_segments)]


def _summarize(segments: list[torch.Tensor], encoder_type: str) -> torch.Tensor:
    """Reduce padded segments to the summary the encoder consumes pre-projection."""
    lengths = torch.tensor([len(segment) for segment in segments], dtype=torch.long)
    padded = pad_sequence(segments, batch_first=True)
    if encoder_type == "mean":
        return masked_mean(padded, lengths)
    # `stats` explicitly, and `gru` for want of a meaningful pre-projection summary.
    return masked_statistics(padded, lengths)


@torch.no_grad()
def build_views(
    dataset: Dataset[Sample],
    modality: str,
    *,
    num_segments: int,
    encoder_type: str,
    encoder: SequenceEncoder | None = None,
    device: torch.device | None = None,
    subject_offset: int = 0,
) -> ModalityViews:
    """Build one modality's segment views over every session in ``dataset``.

    Args:
        dataset: Dataset of :class:`Sample` items, one per subject.
        modality: ``"audio"``, ``"video"`` or ``"text"``.
        num_segments: Contiguous views per session.
        encoder_type: The configured encoder type for this modality, which
            decides the ``raw`` summary. Ignored when ``encoder`` is given.
        encoder: Trained encoder to run each segment through. ``None`` yields the
            ``raw`` representation instead.
        device: Device for the encoder forward pass.
        subject_offset: Added to every subject id, so several splits can be
            concatenated into one candidate pool without colliding.

    Returns:
        The views, with the subjects that had too few items recorded in
        ``skipped`` rather than raising.
    """
    target = device or torch.device("cpu")
    if encoder is not None:
        encoder.eval()

    rows: list[NDArray[np.float64]] = []
    subject_ids: list[int] = []
    view_ids: list[int] = []
    skipped: list[int] = []

    for index in range(len(dataset)):  # type: ignore[arg-type]
        try:
            segments = segment_session(dataset, index, modality, num_segments)
        except ValueError:
            skipped.append(subject_offset + index)
            continue

        if encoder is None:
            summary = _summarize(segments, encoder_type)
        else:
            lengths = torch.tensor([len(s) for s in segments], dtype=torch.long)
            padded = pad_sequence(segments, batch_first=True)
            summary = encoder(padded.to(target), lengths.to(target)).cpu()

        rows.append(summary.numpy().astype(np.float64))
        subject_ids.extend([subject_offset + index] * num_segments)
        view_ids.extend(range(num_segments))

    if not rows:
        raise ValueError(f"no usable sessions for modality {modality!r}")

    return ModalityViews(
        features=np.concatenate(rows, axis=0),
        subject_ids=np.asarray(subject_ids, dtype=np.int_),
        view_ids=np.asarray(view_ids, dtype=np.int_),
        skipped=tuple(skipped),
    )


def concat_views(*views: ModalityViews) -> ModalityViews:
    """Merge views from several splits into one candidate pool.

    Args:
        *views: Per-split views, each built with a distinct ``subject_offset``.

    Returns:
        The concatenation.

    Raises:
        ValueError: If nothing was passed, or the feature widths disagree.
    """
    if not views:
        raise ValueError("at least one ModalityViews is required")
    widths = {view.features.shape[1] for view in views}
    if len(widths) != 1:
        raise ValueError(f"cannot concatenate views of differing widths: {sorted(widths)}")
    return ModalityViews(
        features=np.concatenate([view.features for view in views], axis=0),
        subject_ids=np.concatenate([view.subject_ids for view in views]),
        view_ids=np.concatenate([view.view_ids for view in views]),
        skipped=tuple(index for view in views for index in view.skipped),
    )


def run_reidentification(
    views: ModalityViews,
    *,
    enroll_segments: int,
    seed: int,
    pca_dim: int | None = None,
    shuffle_subjects: bool = False,
    groups: Mapping[int, str] | None = None,
) -> dict[str, float]:
    """Enrol on a random subset of each subject's segments and probe the rest.

    Args:
        views: The segment views for one modality.
        enroll_segments: Segments per subject used to build the template.
        seed: Chooses which segments enrol; repeats over seeds give the spread.
        pca_dim: Project to this width first, fitted on the enrollment rows only.
            ``None`` attacks the features as they are.
        shuffle_subjects: Negative control — permute the probe labels, which must
            drive accuracy down to chance if the pipeline is sound.
        groups: Optional subject id → group name, for a per-group breakdown
            (e.g. subjects the encoder was fitted on vs unseen ones).

    Returns:
        ``accuracy``, ``chance``, ``ratio_to_chance``, ``num_subjects``,
        ``num_probes``, and ``accuracy_<group>`` per group.

    Raises:
        ValueError: If ``enroll_segments`` leaves no probes for some subject.
    """
    rng = np.random.default_rng(seed)
    enroll_mask = np.zeros(len(views.subject_ids), dtype=bool)
    for subject in np.unique(views.subject_ids):
        positions = np.flatnonzero(views.subject_ids == subject)
        if enroll_segments >= len(positions):
            raise ValueError(
                f"enroll_segments={enroll_segments} leaves no probe for subject "
                f"{subject}, which has {len(positions)} segments"
            )
        enroll_mask[rng.choice(positions, size=enroll_segments, replace=False)] = True

    enroll_features = views.features[enroll_mask]
    probe_features = views.features[~enroll_mask]
    if pca_dim is not None:
        mean, components = fit_pca(enroll_features, pca_dim)
        enroll_features = apply_pca(enroll_features, mean, components)
        probe_features = apply_pca(probe_features, mean, components)

    enroll_subjects = views.subject_ids[enroll_mask]
    probe_subjects = views.subject_ids[~enroll_mask]
    if shuffle_subjects:
        probe_subjects = rng.permutation(probe_subjects)

    attacker = ReidentificationAttacker()
    attacker.enroll(enroll_features, enroll_subjects)
    correct = attacker.predict(probe_features) == probe_subjects

    num_subjects = views.num_subjects
    chance = ReidentificationAttacker.chance_accuracy(num_subjects)
    accuracy = float(np.mean(correct))
    result = {
        "accuracy": accuracy,
        "chance": chance,
        "ratio_to_chance": accuracy / chance,
        "num_subjects": float(num_subjects),
        "num_probes": float(len(correct)),
    }
    if groups is not None:
        names = np.array([groups.get(int(s), "other") for s in probe_subjects])
        for name in sorted(set(names.tolist())):
            selected = names == name
            result[f"accuracy_{name}"] = float(np.mean(correct[selected]))
    return result


def fit_pca(
    train_features: NDArray[np.float64], dim: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a PCA projection, for comparing modalities at matched width.

    Fitted on the **enrollment** rows only and applied to the probes, so the
    attacker never gets a projection shaped by the data it is scored on.

    Args:
        train_features: Rows to fit on, shape ``(N, D)``.
        dim: Target width; capped at ``min(N, D)``.

    Returns:
        ``(mean, components)`` where ``components`` has shape ``(D, k)``.

    Raises:
        ValueError: If ``dim`` is not positive.
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    mean = train_features.mean(axis=0, keepdims=True)
    centered = train_features - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[: min(dim, vt.shape[0])].T


def apply_pca(
    features: NDArray[np.float64], mean: NDArray[np.float64], components: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project features onto a fitted PCA basis.

    Args:
        features: Rows to project, shape ``(N, D)``.
        mean: Fitted mean, shape ``(1, D)``.
        components: Fitted basis, shape ``(D, k)``.

    Returns:
        The projected rows, shape ``(N, k)``.
    """
    projected: NDArray[np.float64] = (features - mean) @ components
    return projected
