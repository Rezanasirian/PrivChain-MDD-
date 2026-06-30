"""Real DAIC-WOZ dataset loader (Phase 1, objective H4).

Loads the canonical DAIC-WOZ / AVEC2017 distribution
(https://dcapswoz.ict.usc.edu/) into the same ``Sample``/``Batch`` contract as
the mock dataset, so the Phase 1 model/trainer run unchanged on real data:

* **audio** — COVAREP features (one row per ~10 ms frame), shape ``(T, D_audio)``
* **video** — OpenFace facial features/AUs (metadata columns dropped), ``(T, D_video)``
* **text**  — participant transcript turns vectorized into a single ``(1, D_text)`` row
* **label** — ``PHQ8_Binary``; **phq8_score** — ``PHQ8_Score`` from the split file

To keep memory and IO bounded over ~15-minute interviews, feature rows are
subsampled (``frame_stride``) and truncated (``max_frames``), then optionally
z-score standardized per feature. All paths/columns/limits come from a config
dict (``configs/daic_woz.yaml``) — nothing is hardcoded.

NOTE: This loader is written against the documented DAIC-WOZ layout but has not
been executed against the real 300 GB corpus in this environment. Validate the
file templates / column names in ``configs/daic_woz.yaml`` against your download
before relying on the numbers. See ADR-0002.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from privchain.data.mock_daic_woz import Sample
from privchain.data.text_vectorizers import HashingTextVectorizer, TextVectorizer


def _load_feature_matrix(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
    drop_columns: list[str],
    max_frames: int,
    frame_stride: int,
    standardize: bool,
) -> NDArray[np.float32]:
    """Stream a CSV/TXT feature file into a subsampled ``(T, D)`` matrix.

    Args:
        path: Feature file path.
        delimiter: Field delimiter.
        has_header: Whether the first row is a header (used to resolve
            ``drop_columns`` by name).
        drop_columns: Header names of metadata columns to drop (requires
            ``has_header``).
        max_frames: Maximum number of (subsampled) frames to keep.
        frame_stride: Keep every ``frame_stride``-th row.
        standardize: Z-score each feature column over time.

    Returns:
        Float32 matrix of shape ``(T, D)`` with ``T >= 1``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no usable feature rows are found.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DAIC-WOZ feature file not found: {path}")

    keep_idx: list[int] | None = None
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, skipinitialspace=True)
        if has_header:
            header = [col.strip() for col in next(reader)]
            drop = {name.strip() for name in drop_columns}
            keep_idx = [i for i, name in enumerate(header) if name not in drop]

        for row_num, raw in enumerate(reader):
            if frame_stride > 1 and row_num % frame_stride != 0:
                continue
            fields = [f for f in raw if f != ""]
            if not fields:
                continue
            selected = [raw[i] for i in keep_idx] if keep_idx is not None else fields
            try:
                rows.append([float(value) for value in selected])
            except ValueError:
                continue  # skip malformed lines defensively
            if len(rows) >= max_frames:
                break

    if not rows:
        raise ValueError(f"No usable feature rows parsed from {path}")

    matrix = np.asarray(rows, dtype=np.float32)
    if standardize:
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        matrix = (matrix - mean) / (std + 1e-6)
    return matrix


def _read_participant_transcript(
    path: Path, *, delimiter: str, speaker_column: str, value_column: str, participant_speaker: str
) -> str:
    """Concatenate a participant's transcript turns into one document.

    Args:
        path: Transcript file path.
        delimiter: Field delimiter (DAIC-WOZ transcripts are tab-separated).
        speaker_column: Header name of the speaker column.
        value_column: Header name of the utterance-text column.
        participant_speaker: Speaker label identifying the participant's turns.

    Returns:
        The participant's concatenated utterances (possibly empty string).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DAIC-WOZ transcript not found: {path}")
    parts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            speaker = (row.get(speaker_column) or "").strip()
            if speaker == participant_speaker:
                parts.append((row.get(value_column) or "").strip())
    return " ".join(p for p in parts if p)


def _read_split_labels(
    path: Path, *, participant_col: str, binary_col: str, score_col: str
) -> list[dict[str, int]]:
    """Read a split CSV into per-participant label records.

    Args:
        path: Split label CSV (e.g., ``train_split_Depression_AVEC2017.csv``).
        participant_col: Header name of the participant-ID column.
        binary_col: Header name of the PHQ-8 binary label column.
        score_col: Header name of the PHQ-8 score column.

    Returns:
        List of ``{"pid", "label", "score"}`` records.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DAIC-WOZ split file not found: {path}")
    records: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pid_raw = (row.get(participant_col) or "").strip()
            if not pid_raw:
                continue
            score_raw = (row.get(score_col) or "0").strip() or "0"
            records.append(
                {
                    "pid": int(float(pid_raw)),
                    "label": int(float((row.get(binary_col) or "0").strip() or "0")),
                    "score": int(float(score_raw)),
                }
            )
    return records


class DaicWozDataset(Dataset[Sample]):
    """Real DAIC-WOZ sessions exposed via the project ``Sample`` contract.

    Args:
        config: The ``daic_woz`` sub-mapping from ``configs/daic_woz.yaml``.
        split: One of the keys under ``config["splits"]`` (e.g., ``"train"``).
        text_vectorizer: Vectorizer for transcripts; defaults to a
            :class:`HashingTextVectorizer` sized by ``config["text"]["dim"]``.
        cache: If ``True``, cache parsed (subsampled) tensors per session.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        split: str,
        text_vectorizer: TextVectorizer | None = None,
        cache: bool = True,
    ) -> None:
        self._cfg = config
        self._root = Path(config["root"])
        self._audio_cfg = config["audio"]
        self._video_cfg = config["video"]
        self._text_cfg = config["text"]
        self._dir_template = config.get("participant_dir_template", "{pid}_P")

        label_cols = config["label_columns"]
        split_path = self._root / config["splits"][split]
        self._records = _read_split_labels(
            split_path,
            participant_col=label_cols["participant_id"],
            binary_col=label_cols["phq_binary"],
            score_col=label_cols["phq_score"],
        )

        self._vectorizer: TextVectorizer = text_vectorizer or HashingTextVectorizer(
            int(self._text_cfg["dim"]), seed=int(config.get("seed", 0))
        )
        self._cache: dict[int, Sample] | None = {} if cache else None
        self.phq8_max: int = int(config.get("phq8_max", 24))
        self.feature_dims: dict[str, int] = self._infer_feature_dims()

    def _participant_dir(self, pid: int) -> Path:
        """Return the participant's directory path."""
        return self._root / self._dir_template.format(pid=pid)

    def _file(self, pid: int, template: str) -> Path:
        """Resolve a per-participant file path from a ``{pid}`` template."""
        return self._participant_dir(pid) / template.format(pid=pid)

    def _load_audio(self, pid: int) -> NDArray[np.float32]:
        cfg = self._audio_cfg
        return _load_feature_matrix(
            self._file(pid, cfg["file_template"]),
            delimiter=cfg.get("delimiter", ","),
            has_header=cfg.get("has_header", False),
            drop_columns=cfg.get("drop_columns", []),
            max_frames=int(cfg["max_frames"]),
            frame_stride=int(cfg.get("frame_stride", 1)),
            standardize=cfg.get("standardize", True),
        )

    def _load_video(self, pid: int) -> NDArray[np.float32]:
        cfg = self._video_cfg
        return _load_feature_matrix(
            self._file(pid, cfg["file_template"]),
            delimiter=cfg.get("delimiter", ","),
            has_header=cfg.get("has_header", True),
            drop_columns=cfg.get("drop_columns", []),
            max_frames=int(cfg["max_frames"]),
            frame_stride=int(cfg.get("frame_stride", 1)),
            standardize=cfg.get("standardize", True),
        )

    def _load_text(self, pid: int) -> NDArray[np.float32]:
        cfg = self._text_cfg
        document = _read_participant_transcript(
            self._file(pid, cfg["file_template"]),
            delimiter=cfg.get("delimiter", "\t"),
            speaker_column=cfg.get("speaker_column", "speaker"),
            value_column=cfg.get("value_column", "value"),
            participant_speaker=cfg.get("participant_speaker", "Participant"),
        )
        return self._vectorizer.transform(document)

    def _infer_feature_dims(self) -> dict[str, int]:
        """Infer per-modality feature dims from the first available session."""
        for record in self._records:
            pid = record["pid"]
            try:
                audio = self._load_audio(pid)
                video = self._load_video(pid)
            except (FileNotFoundError, ValueError):
                continue
            return {
                "audio": int(audio.shape[1]),
                "video": int(video.shape[1]),
                "text": self._vectorizer.dim,
            }
        raise RuntimeError(
            "Could not infer feature dims: no readable participant found under "
            f"{self._root}. Check configs/daic_woz.yaml paths/templates."
        )

    def __len__(self) -> int:
        """Return the number of sessions in this split."""
        return len(self._records)

    def __getitem__(self, index: int) -> Sample:
        """Load and assemble the session at ``index`` as a :class:`Sample`.

        Args:
            index: Session index in ``[0, len(self))``.

        Returns:
            A :class:`Sample` with audio/video sequences, a length-1 text
            sequence, and integer labels.
        """
        if self._cache is not None and index in self._cache:
            return self._cache[index]

        record = self._records[index]
        pid = record["pid"]
        audio = self._load_audio(pid)
        video = self._load_video(pid)
        text = self._load_text(pid)  # (D_text,)

        sample = Sample(
            audio=torch.from_numpy(audio),
            video=torch.from_numpy(video),
            text=torch.from_numpy(text).unsqueeze(0),  # (1, D_text)
            phq8_score=torch.tensor(record["score"], dtype=torch.long),
            label=torch.tensor(record["label"], dtype=torch.long),
        )
        if self._cache is not None:
            self._cache[index] = sample
        return sample


def build_daic_woz_dataset(config: dict[str, Any], *, split: str) -> DaicWozDataset:
    """Build a :class:`DaicWozDataset` from a loaded ``daic_woz.yaml`` mapping.

    Args:
        config: The full YAML mapping (must contain a ``daic_woz`` section).
        split: Split key (e.g., ``"train"``, ``"dev"``, ``"test"``).

    Returns:
        A configured :class:`DaicWozDataset`.
    """
    daic_cfg = dict(config["daic_woz"])
    daic_cfg.setdefault("seed", config.get("seed", 0))
    return DaicWozDataset(daic_cfg, split=split)
