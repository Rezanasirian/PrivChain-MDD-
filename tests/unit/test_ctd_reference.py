from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from privchain.data.ctd_reference import CtdLinearClassifier, CtdReferenceDataset
from privchain.data.mock_daic_woz import collate_fn


def _write_artifact(root: Path, *, width: int = 24) -> None:
    np.savez(
        root / "ctd_train.npz",
        session_id=np.array([300, 301]),
        emb=np.arange(2 * width, dtype=np.float32).reshape(2, width),
        label=np.array([0, 1]),
    )


def test_reference_dataset_preserves_sessions_and_marks_other_modalities_absent(
    tmp_path: Path,
) -> None:
    _write_artifact(tmp_path)
    dataset = CtdReferenceDataset(tmp_path, "train")

    assert dataset.session_ids.tolist() == [300, 301]
    assert dataset[0]["audio"].shape == (1, 24)
    assert {name: int(flag) for name, flag in dataset[0]["presence"].items()} == {
        "audio": 1,
        "video": 0,
        "text": 0,
    }


def test_reference_model_uses_audio_dp_parameter_group(tmp_path: Path) -> None:
    pytest.importorskip("opacus")
    from privchain.privacy.dp_sgd import map_parameter_groups

    _write_artifact(tmp_path)
    batch = collate_fn([CtdReferenceDataset(tmp_path, "train")[0]])
    model = CtdLinearClassifier()

    assert model(batch)["logit"].shape == (1,)
    groups = map_parameter_groups(model, CtdReferenceDataset.capability)
    assert len(groups["audio"]) == 2
    assert groups["shared"] == []


def test_reference_dataset_rejects_wrong_feature_width(tmp_path: Path) -> None:
    _write_artifact(tmp_path, width=12)
    with pytest.raises(ValueError, match="expected CTD features"):
        CtdReferenceDataset(tmp_path, "train")


def test_reference_model_rejects_missing_ctd(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    batch = collate_fn([CtdReferenceDataset(tmp_path, "train")[0]])
    batch["presence"]["audio"] = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="requires the CTD/audio capability"):
        CtdLinearClassifier()(batch)
