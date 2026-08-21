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
normalized under the configured scheme (``session``/``corpus``/``none``, see
:func:`apply_normalization` and ADR-0019). All paths/columns/limits come from a
config dict (``configs/daic_woz.yaml``) — nothing is hardcoded.

The file templates and column names in ``configs/daic_woz.yaml`` were verified
against the real AVEC2017 distribution on 2026-08-13 (see ADR-0010 for what the
download turned up, including the corrupt archive for participant 440).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from privchain.data.mock_daic_woz import MODALITIES, Sample
from privchain.data.text_vectorizers import TextVectorizer, build_text_vectorizer
from privchain.segmentation import contiguous_spans


def _load_feature_matrix(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
    drop_columns: list[str],
    max_frames: int,
    frame_stride: int,
) -> NDArray[np.float32]:
    """Stream a CSV/TXT feature file into a subsampled ``(T, D)`` matrix.

    Returns the values **as recorded**. Normalization is applied afterwards by
    :func:`apply_normalization`, so switching normalization mode does not force a
    re-parse of the 36 MB COVAREP files (ADR-0019).

    Args:
        path: Feature file path.
        delimiter: Field delimiter.
        has_header: Whether the first row is a header (used to resolve
            ``drop_columns`` by name).
        drop_columns: Header names of metadata columns to drop (requires
            ``has_header``).
        max_frames: Maximum number of (subsampled) frames to keep.
        frame_stride: Keep every ``frame_stride``-th row.

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

    return np.asarray(rows, dtype=np.float32)


def apply_normalization(
    matrix: NDArray[np.float32],
    mode: str,
    corpus_stats: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None,
) -> NDArray[np.float32]:
    """Normalize a session's feature matrix under the configured scheme.

    The choice is not cosmetic (ADR-0019). ``session`` z-scores each channel
    *within* the session, which forces every participant's features to per-channel
    mean 0 and std 1 — deleting absolute pitch level, formant positions and
    overall energy. Those are exactly the cues a speaker-identification attacker
    uses, and among the best-attested acoustic markers of depression, so the mode
    silently bounds both the utility and the leakage this project measures.

    Args:
        matrix: Raw ``(T, D)`` features as recorded.
        mode: ``"session"``, ``"corpus"`` or ``"none"``.
        corpus_stats: ``(mean, std)`` of shape ``(1, D)``, fitted on the training
            split only. Required for ``"corpus"``.

    Returns:
        The normalized matrix (a new array unless ``mode`` is ``"none"``).

    Raises:
        ValueError: If ``mode`` is unknown, or ``corpus`` is requested without
            statistics.
    """
    if mode == "none":
        return matrix
    if mode == "session":
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
    elif mode == "corpus":
        if corpus_stats is None:
            raise ValueError("corpus normalization requires statistics fitted on the train split")
        mean, std = corpus_stats
    else:
        raise ValueError(f"unknown normalization mode {mode!r}; expected session, corpus or none")
    normalized: NDArray[np.float32] = ((matrix - mean) / (std + 1e-6)).astype(np.float32)
    return normalized


def _cached_feature_matrix(
    path: Path, cache_dir: Path | None, **options: Any
) -> NDArray[np.float32]:
    """Load a feature matrix, memoizing the parsed result on disk.

    Parsing dominates runtime: a COVAREP file is ~36 MB and ~90k rows, and the
    subsampling loop must read every row to keep every ``frame_stride``-th one.
    Re-parsing on every run made experiment iteration cost minutes (ADR-0012),
    so the subsampled matrix is written to ``cache_dir`` once.

    The cache key includes all parsing options **and the source file's size and
    mtime**, so changing ``frame_stride``/``max_frames`` — or
    re-extracting the corpus — produces a different entry rather than silently
    reusing a stale one.

    Args:
        path: Feature file path.
        cache_dir: Directory for cached ``.npy`` files; ``None`` disables caching.
        **options: Parsing options forwarded to :func:`_load_feature_matrix`.

    Returns:
        Float32 matrix of shape ``(T, D)``.
    """
    if cache_dir is None:
        return _load_feature_matrix(path, **options)

    stat = path.stat()
    key = json.dumps(
        {**options, "_size": stat.st_size, "_mtime_ns": stat.st_mtime_ns},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"{path.stem}.{digest}.npy"

    if cache_path.is_file():
        cached: NDArray[np.float32] = np.load(cache_path)
        return cached

    matrix = _load_feature_matrix(path, **options)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write via a temp file so a crash mid-write cannot leave a truncated cache
    # entry that later runs would happily load.
    tmp_path = cache_path.with_suffix(".tmp.npy")
    np.save(tmp_path, matrix)
    tmp_path.replace(cache_path)
    return matrix


def _read_participant_turns(
    path: Path, *, delimiter: str, speaker_column: str, value_column: str, participant_speaker: str
) -> list[str]:
    """Read the participant's transcript turns in chronological order.

    DAIC-WOZ transcripts are written in time order, so file order is turn order;
    no timestamp parsing is needed to slice a session into contiguous stretches.

    Args:
        path: Transcript file path.
        delimiter: Field delimiter (DAIC-WOZ transcripts are tab-separated).
        speaker_column: Header name of the speaker column.
        value_column: Header name of the utterance-text column.
        participant_speaker: Speaker label identifying the participant's turns.

    Returns:
        The participant's non-empty utterances, in chronological order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DAIC-WOZ transcript not found: {path}")
    turns: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            speaker = (row.get(speaker_column) or "").strip()
            if speaker != participant_speaker:
                continue
            value = (row.get(value_column) or "").strip()
            if value:
                turns.append(value)
    return turns


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
    return " ".join(
        _read_participant_turns(
            path,
            delimiter=delimiter,
            speaker_column=speaker_column,
            value_column=value_column,
            participant_speaker=participant_speaker,
        )
    )


def _read_split_labels(
    path: Path,
    *,
    participant_col: str,
    binary_col: str,
    score_col: str,
    exclude: frozenset[int] = frozenset(),
) -> list[dict[str, int]]:
    """Read a split CSV into per-participant label records.

    The configured columns must actually be present in the file. The AVEC2017
    distribution names them inconsistently across splits (``PHQ8_Binary`` in
    train/dev vs ``PHQ_Binary`` in ``full_test_split.csv``), and a missing
    column silently defaulting to ``0`` would zero out every test label — so a
    mismatch is raised rather than absorbed (ADR-0010).

    Args:
        path: Split label CSV (e.g., ``train_split_Depression_AVEC2017.csv``).
        participant_col: Header name of the participant-ID column.
        binary_col: Header name of the PHQ-8 binary label column.
        score_col: Header name of the PHQ-8 score column.
        exclude: Participant IDs to drop (e.g., sessions with corrupt media).

    Returns:
        List of ``{"pid", "label", "score"}`` records.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a configured label column is absent from the CSV header.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DAIC-WOZ split file not found: {path}")
    records: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [name.strip() for name in (reader.fieldnames or [])]
        missing = [c for c in (participant_col, binary_col, score_col) if c not in header]
        if missing:
            raise ValueError(
                f"{path.name}: configured label column(s) {missing} not found. "
                f"Header is {header}. Fix `label_columns` (or add a "
                f"`split_label_columns` override) in configs/daic_woz.yaml."
            )

        for row in reader:
            pid_raw = (row.get(participant_col) or "").strip()
            if not pid_raw:
                continue
            pid = int(float(pid_raw))
            if pid in exclude:
                continue
            score_raw = (row.get(score_col) or "0").strip() or "0"
            records.append(
                {
                    "pid": pid,
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
        # Parsed-feature memoization; set `feature_cache_dir: null` to disable.
        cache_name: str | None = config.get("feature_cache_dir", "_feature_cache")
        self._cache_dir = self._root / cache_name if cache_name else None

        # Per-split header overrides layer on top of the defaults: the AVEC2017
        # test split uses PHQ_Binary/PHQ_Score rather than PHQ8_* (ADR-0010).
        label_cols = dict(config["label_columns"])
        label_cols.update(dict(config.get("split_label_columns", {}).get(split, {})))
        self._excluded = frozenset(int(p) for p in config.get("exclude_participants", []))

        split_path = self._root / config["splits"][split]
        self._records = _read_split_labels(
            split_path,
            participant_col=label_cols["participant_id"],
            binary_col=label_cols["phq_binary"],
            score_col=label_cols["phq_score"],
            exclude=self._excluded,
        )

        # An explicitly supplied vectorizer wins (tests inject one); otherwise the
        # `text.vectorizer` config key decides. That key used to be ignored, so a
        # config asking for anything but hashing silently got hashing.
        self._vectorizer: TextVectorizer = text_vectorizer or build_text_vectorizer(
            self._text_cfg, seed=int(config.get("seed", 0))
        )
        self._cache: dict[int, Sample] | None = {} if cache else None
        # Corpus normalization statistics are fitted on the *train* split, whatever
        # split this dataset is, and memoized per modality (ADR-0019).
        self._corpus_stats: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
        self.phq8_max: int = int(config.get("phq8_max", 24))
        self.feature_dims: dict[str, int] = self._infer_feature_dims()

    def _parse_options(self, cfg: dict[str, Any], *, default_header: bool) -> dict[str, Any]:
        """Parsing options for one modality, shared by loading and stat-fitting."""
        return {
            "delimiter": cfg.get("delimiter", ","),
            "has_header": cfg.get("has_header", default_header),
            "drop_columns": cfg.get("drop_columns", []),
            "max_frames": int(cfg["max_frames"]),
            "frame_stride": int(cfg.get("frame_stride", 1)),
        }

    def _corpus_statistics(
        self, modality: str, cfg: dict[str, Any], options: dict[str, Any]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Per-channel mean/std over the training split, for corpus normalization.

        Fitted on **train only** and reused for dev and test, so normalization
        never carries information from an evaluation split back into training.

        Args:
            modality: Modality name, for the cache key.
            cfg: That modality's config section.
            options: Parsing options, so a stride change invalidates the stats.

        Returns:
            ``(mean, std)``, each of shape ``(1, D)``.

        Raises:
            RuntimeError: If no training session could be read.
        """
        if modality in self._corpus_stats:
            return self._corpus_stats[modality]

        digest = hashlib.sha256(
            json.dumps({**options, "modality": modality}, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        cache_path = (
            self._cache_dir / f"corpus_stats.{modality}.{digest}.npz" if self._cache_dir else None
        )
        if cache_path is not None and cache_path.is_file():
            cached = np.load(cache_path)
            stats = (cached["mean"], cached["std"])
            self._corpus_stats[modality] = stats
            return stats

        label_cols = dict(self._cfg["label_columns"])
        label_cols.update(dict(self._cfg.get("split_label_columns", {}).get("train", {})))
        train_records = _read_split_labels(
            self._root / self._cfg["splits"]["train"],
            participant_col=label_cols["participant_id"],
            binary_col=label_cols["phq_binary"],
            score_col=label_cols["phq_score"],
            exclude=self._excluded,
        )

        total = count = 0.0
        sums: NDArray[np.float64] | None = None
        squares: NDArray[np.float64] | None = None
        for record in train_records:
            try:
                matrix = _cached_feature_matrix(
                    self._file(record["pid"], cfg["file_template"]), self._cache_dir, **options
                ).astype(np.float64)
            except (FileNotFoundError, ValueError):
                continue  # a session missing this modality contributes nothing
            sums = matrix.sum(axis=0) if sums is None else sums + matrix.sum(axis=0)
            squares = (
                (matrix**2).sum(axis=0) if squares is None else squares + (matrix**2).sum(axis=0)
            )
            count += matrix.shape[0]
            total += 1

        if sums is None or squares is None or count == 0:
            raise RuntimeError(f"no readable training sessions for corpus stats of {modality!r}")

        mean = (sums / count).astype(np.float32)[None, :]
        variance = np.maximum(squares / count - (sums / count) ** 2, 0.0)
        std = np.sqrt(variance).astype(np.float32)[None, :]
        if cache_path is not None and self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez(tmp_path, mean=mean, std=std)
            tmp_path.replace(cache_path)
        self._corpus_stats[modality] = (mean, std)
        return mean, std

    def _load_normalized(
        self, pid: int, modality: str, cfg: dict[str, Any], *, default_header: bool
    ) -> NDArray[np.float32]:
        """Load one modality's raw matrix and apply the configured normalization."""
        options = self._parse_options(cfg, default_header=default_header)
        matrix = _cached_feature_matrix(
            self._file(pid, cfg["file_template"]), self._cache_dir, **options
        )
        mode = str(cfg.get("normalization", "session"))
        stats = self._corpus_statistics(modality, cfg, options) if mode == "corpus" else None
        return apply_normalization(matrix, mode, stats)

    def _participant_dir(self, pid: int) -> Path:
        """Return the participant's directory path."""
        participant_dir: Path = self._root / self._dir_template.format(pid=pid)
        return participant_dir

    def _file(self, pid: int, template: str) -> Path:
        """Resolve a per-participant file path from a ``{pid}`` template."""
        return self._participant_dir(pid) / template.format(pid=pid)

    def _load_audio(self, pid: int) -> NDArray[np.float32]:
        return self._load_normalized(pid, "audio", self._audio_cfg, default_header=False)

    def _load_video(self, pid: int) -> NDArray[np.float32]:
        return self._load_normalized(pid, "video", self._video_cfg, default_header=True)

    def _text_cache_path(self, path: Path, kind: str) -> Path | None:
        """Cache path for a text vector, or ``None`` when caching is disabled.

        The key covers the vectorizer identity, so switching model — or back to
        hashing — does not reuse the wrong vectors.

        Args:
            path: The transcript this vector was derived from.
            kind: Distinguishes the whole-session vector from the segmented one.

        Returns:
            The ``.npy`` path, or ``None``.
        """
        if self._cache_dir is None:
            return None
        cfg = self._text_cfg
        identity = {
            "vectorizer": cfg.get("vectorizer", "hashing"),
            "dim": self._vectorizer.dim,
            "options": cfg.get("transformer", {}),
            "speaker": cfg.get("participant_speaker", "Participant"),
        }
        key = json.dumps(identity, sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return self._cache_dir / f"{path.stem}.{kind}.{digest}.npy"

    def _write_text_cache(self, cache_path: Path | None, array: NDArray[np.float32]) -> None:
        """Memoize a text vector, writing via a temp file so a crash cannot truncate it."""
        if cache_path is None or self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp.npy")
        np.save(tmp_path, array)
        tmp_path.replace(cache_path)

    def _participant_turns(self, pid: int) -> list[str]:
        """Read one participant's utterances in chronological order."""
        cfg = self._text_cfg
        return _read_participant_turns(
            self._file(pid, cfg["file_template"]),
            delimiter=cfg.get("delimiter", "\t"),
            speaker_column=cfg.get("speaker_column", "speaker"),
            value_column=cfg.get("value_column", "value"),
            participant_speaker=cfg.get("participant_speaker", "Participant"),
        )

    def _load_text(self, pid: int) -> NDArray[np.float32]:
        cfg = self._text_cfg
        path = self._file(pid, cfg["file_template"])

        # Transformer embeddings cost a GPU forward pass per session, so the
        # result is memoized like the audio/video matrices.
        cache_path = self._text_cache_path(path, "text")
        if cache_path is not None and cache_path.is_file():
            cached: NDArray[np.float32] = np.load(cache_path)
            return cached

        vector = self._vectorizer.transform(" ".join(self._participant_turns(pid)))
        self._write_text_cache(cache_path, vector)
        return vector

    def text_segment_vectors(self, index: int, num_segments: int) -> NDArray[np.float32]:
        """Embed ``num_segments`` contiguous stretches of one participant's speech.

        The re-identification attacker needs several views per subject, and each
        DAIC-WOZ participant appears in exactly one session (ADR-0017). Audio and
        video get their views by slicing the frame matrix; text gets them by
        slicing the turn list and embedding each stretch on its own, so all three
        modalities are attacked under the same enrol/probe protocol.

        Args:
            index: Session index in ``[0, len(self))``.
            num_segments: Number of contiguous stretches to embed.

        Returns:
            Array of shape ``(num_segments, dim)``, one row per stretch, in
            chronological order.

        Raises:
            ValueError: If the participant has fewer turns than ``num_segments``.
        """
        pid = self._records[index]["pid"]
        path = self._file(pid, self._text_cfg["file_template"])
        cache_path = self._text_cache_path(path, f"textseg{num_segments}")
        if cache_path is not None and cache_path.is_file():
            cached: NDArray[np.float32] = np.load(cache_path)
            return cached

        turns = self._participant_turns(pid)
        try:
            spans = contiguous_spans(len(turns), num_segments)
        except ValueError as error:
            raise ValueError(f"participant {pid}: {error}") from error

        vectors = np.stack(
            [self._vectorizer.transform(" ".join(turns[start:stop])) for start, stop in spans]
        ).astype(np.float32)
        self._write_text_cache(cache_path, vectors)
        return vectors

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
            presence={m: torch.tensor(1, dtype=torch.long) for m in MODALITIES},
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
