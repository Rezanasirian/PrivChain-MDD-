"""The optional ``quality`` field: collation, device moves, and capability masks.

``quality`` was added to a contract that four places construct and that the DP,
federated and distillation paths all consume. These tests pin the parts that
would fail quietly: a device move that skips it, a capability mask that collapses
a segment sequence to one row, and a batch that carries no quality at all.
"""

from __future__ import annotations

import pytest
import torch

from privchain.data.mock_daic_woz import MODALITIES, Sample, collate_fn
from privchain.federated.partition import ModalityMaskedDataset
from privchain.training.objective import move_batch_to_device

SEGMENTS = 4
DIMS = {"audio": 5, "video": 4, "text": 3}
QUALITY_DIMS = {"audio": 3, "video": 4, "text": 3}


def _sample(*, quality: bool, segments: int = SEGMENTS) -> Sample:
    sample: Sample = {  # type: ignore[typeddict-item]
        modality: torch.ones(segments, DIMS[modality]) for modality in MODALITIES
    }
    sample["presence"] = {m: torch.tensor(1, dtype=torch.long) for m in MODALITIES}
    sample["phq8_score"] = torch.tensor(10, dtype=torch.long)
    sample["label"] = torch.tensor(1, dtype=torch.long)
    if quality:
        sample["quality"] = {
            m: torch.ones(segments, QUALITY_DIMS[m]) for m in MODALITIES
        }
    return sample


class _Corpus:
    """Minimal dataset of identical samples."""

    def __init__(self, *, quality: bool, size: int = 3) -> None:
        self._samples = [_sample(quality=quality) for _ in range(size)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]


def test_collate_pads_quality_alongside_features() -> None:
    batch = collate_fn([_sample(quality=True), _sample(quality=True, segments=2)])
    assert batch["quality"]["audio"].shape == (2, SEGMENTS, QUALITY_DIMS["audio"])
    # Padding zeroes the `valid` channel, which is what the length mask says too.
    assert batch["quality"]["audio"][1, 2:, 0].tolist() == [0.0, 0.0]


def test_collate_omits_quality_when_no_sample_has_it() -> None:
    assert "quality" not in collate_fn([_sample(quality=False)])


def test_collate_rejects_a_mixed_batch() -> None:
    with pytest.raises(ValueError, match="mix of samples"):
        collate_fn([_sample(quality=True), _sample(quality=False)])


def test_move_batch_to_device_moves_quality() -> None:
    batch = move_batch_to_device(collate_fn([_sample(quality=True)]), torch.device("cpu"))
    assert batch["quality"]["audio"].device == torch.device("cpu")


def test_capability_mask_preserves_the_segment_count() -> None:
    """An absent modality must keep K rows, or segment k stops meaning one thing."""
    masked = ModalityMaskedDataset(_Corpus(quality=True), [0, 1], (0, 1, 1))
    sample = masked[0]

    assert sample["audio"].shape == (SEGMENTS, DIMS["audio"])
    assert float(sample["audio"].abs().sum()) == 0.0
    assert float(sample["quality"]["audio"].abs().sum()) == 0.0
    assert int(sample["presence"]["audio"].item()) == 0
    assert int(sample["presence"]["video"].item()) == 1
    assert float(sample["video"].abs().sum()) > 0.0


def test_capability_mask_still_collapses_frame_level_samples() -> None:
    """Without alignment there is nothing to preserve, and 3000 zero frames cost real time."""
    masked = ModalityMaskedDataset(_Corpus(quality=False), [0], (0, 1, 1))
    sample = masked[0]
    assert sample["audio"].shape == (1, DIMS["audio"])
    assert int(sample["presence"]["audio"].item()) == 0


def test_capability_mask_does_not_mutate_the_base_sample() -> None:
    corpus = _Corpus(quality=True)
    ModalityMaskedDataset(corpus, [0], (0, 1, 1))[0]
    assert float(corpus[0]["audio"].abs().sum()) > 0.0
    assert int(corpus[0]["presence"]["audio"].item()) == 1


# ── Ablation wrapper ─────────────────────────────────────────────────────────


class _EchoModel(torch.nn.Module):
    """Records what presence flags and features the wrapped model actually saw."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, torch.Tensor] = {}
        self.features: dict[str, torch.Tensor] = {}

    def forward(
        self, batch: Sample, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        self.seen = dict(presence or {})
        self.features = {m: batch[m] for m in MODALITIES}  # type: ignore[literal-required]
        return {"logit": torch.zeros(batch["label"].shape[0])}


def test_ablation_zeroes_features_and_presence_together() -> None:
    from privchain.eval.modality_masking import MaskedModalityModel

    inner = _EchoModel()
    batch = collate_fn([_sample(quality=True), _sample(quality=True)])
    MaskedModalityModel(inner, frozenset({"text"}))(batch)

    assert inner.seen["audio"].tolist() == [0.0, 0.0]
    assert inner.seen["text"].tolist() == [1.0, 1.0]
    assert float(inner.features["audio"].abs().sum()) == 0.0
    assert float(inner.features["text"].abs().sum()) > 0.0


def test_ablation_cannot_resurrect_a_modality_the_sample_never_had() -> None:
    """The arm removes modalities; it must not add one back for a partial client."""
    from privchain.eval.modality_masking import MaskedModalityModel

    inner = _EchoModel()
    batch = collate_fn([_sample(quality=True), _sample(quality=True)])
    batch["presence"]["video"] = torch.tensor([0, 1])

    MaskedModalityModel(inner, frozenset({"video", "text"}))(batch)

    assert inner.seen["video"].tolist() == [0.0, 1.0]


def test_ablation_does_not_mutate_the_incoming_batch() -> None:
    from privchain.eval.modality_masking import MaskedModalityModel

    batch = collate_fn([_sample(quality=True)])
    before = batch["audio"].clone()
    MaskedModalityModel(_EchoModel(), frozenset({"text"}))(batch)

    assert torch.equal(batch["audio"], before)
    assert float(batch["quality"]["audio"].abs().sum()) > 0.0
