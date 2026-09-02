"""The reported evaluation loss must not depend on how the loader batched things.

`evaluate_model` accumulated per-batch means and divided by the batch count, so a
short trailing batch counted as much as a full one. That is invisible while the
loss is only logged, and decisive once it selects a checkpoint: ADR-0028's
selection set is every selection participant under every capability, ordered by
capability, so at 17 participants and batch size 32 the 68 rows split 32/32/4 and
the 4-row tail is entirely `text_only` — which then carried ~47% of the criterion
instead of the intended 25%.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from privchain.data.mock_daic_woz import MODALITIES, Sample, collate_fn
from privchain.training.objective import DepressionObjective, evaluate_model

DIMS = {"audio": 4, "video": 3, "text": 5}
PHQ8_MAX = 24


class _Fixed(nn.Module):
    """Returns each sample's own stored logit, so the loss is fully determined."""

    def forward(
        self, batch: Sample, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        return {"logit": batch["audio"][:, 0, 0]}


class _Corpus(Dataset[Sample]):
    """Samples whose logits and labels vary, so batching could plausibly matter."""

    def __init__(self, logits: list[float], labels: list[int]) -> None:
        self._logits = logits
        self._labels = labels

    def __len__(self) -> int:
        return len(self._logits)

    def __getitem__(self, index: int) -> Sample:
        sample: Sample = {  # type: ignore[typeddict-item]
            modality: torch.full((2, DIMS[modality]), self._logits[index])
            for modality in MODALITIES
        }
        sample["presence"] = {m: torch.tensor(1, dtype=torch.long) for m in MODALITIES}
        sample["phq8_score"] = torch.tensor(10, dtype=torch.long)
        sample["label"] = torch.tensor(self._labels[index], dtype=torch.long)
        return sample


def _loss(corpus: Dataset[Sample], batch_size: int) -> float:
    loader: DataLoader[Sample] = DataLoader(
        corpus, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    objective = DepressionObjective(PHQ8_MAX, phq_loss_weight=0.0)
    return evaluate_model(_Fixed(), loader, objective, torch.device("cpu"))["loss"]


# 68 rows: four blocks of 17, as ConcatDataset lays the capabilities out.
LOGITS = [float(i % 7) - 3.0 for i in range(68)]
LABELS = [i % 2 for i in range(68)]


@pytest.mark.parametrize("batch_size", [68, 32, 17, 7, 1])
def test_loss_is_identical_however_the_loader_batches(batch_size: int) -> None:
    """The failing case is 32: 68 rows split 32/32/4 with the tail one capability."""
    corpus = _Corpus(LOGITS, LABELS)

    assert _loss(corpus, batch_size) == pytest.approx(_loss(corpus, 68), abs=1e-6)


def test_reversing_the_capability_order_does_not_move_the_loss() -> None:
    """Under batch-count averaging, whichever block lands in the tail is favoured."""
    forward = _Corpus(LOGITS, LABELS)
    reversed_corpus = _Corpus(LOGITS[::-1], LABELS[::-1])

    assert _loss(forward, 32) == pytest.approx(_loss(reversed_corpus, 32), abs=1e-6)


def test_each_block_carries_its_own_share() -> None:
    """The whole point: 4 equal blocks contribute 25% each, not 17.7/17.7/17.7/46.9."""
    blocks = [
        _Corpus(LOGITS[i * 17 : (i + 1) * 17], LABELS[i * 17 : (i + 1) * 17]) for i in range(4)
    ]
    per_block = [_loss(block, 17) for block in blocks]

    combined = _loss(_Corpus(LOGITS, LABELS), 32)

    assert combined == pytest.approx(sum(per_block) / 4, abs=1e-6)
