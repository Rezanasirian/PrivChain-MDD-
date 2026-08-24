"""Transcript text vectorizers (Phase 1).

Turns a participant's transcript into a fixed-size float vector so text fits the
same float-sequence contract as audio/video (as a length-1 sequence).

Two implementations, selected by ``daic_woz.text.vectorizer``:

* :class:`HashingTextVectorizer` — pure NumPy bag-of-words, no network or
  pretrained model. The default, and what CI and the mock path use.
* :class:`TransformerTextVectorizer` — contextual embeddings from a pretrained
  language model. Text is the strongest modality on DAIC-WOZ, and a hashed
  bag-of-words discards the word order and context that carry the signal, so
  this is the real-data default (ADR-0014). It needs the model weights, which
  are downloaded once and cached.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens.

    Args:
        text: Raw transcript text.

    Returns:
        List of lowercase tokens.
    """
    return _TOKEN_RE.findall(text.lower())


class TextVectorizer(Protocol):
    """A callable mapping a transcript string to a fixed-size vector."""

    @property
    def dim(self) -> int:
        """The output vector dimension."""
        ...

    def transform(self, text: str) -> NDArray[np.float32]:
        """Vectorize one transcript into a ``(dim,)`` float32 vector."""
        ...

    def transform_many(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Vectorize several transcripts into a ``(len(texts), dim)`` array."""
        ...


class HashingTextVectorizer:
    """Hashing-trick bag-of-words vectorizer (offline, deterministic).

    Tokens are hashed into ``dim`` buckets and L2-normalized. No vocabulary
    fitting is required, so the same instance works across train/dev/test.

    Args:
        dim: Number of hash buckets / output dimension.
        seed: Salt mixed into the token hash for reproducibility.
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        """The output vector dimension."""
        return self._dim

    def _bucket(self, token: str) -> int:
        """Map a token to a bucket index in ``[0, dim)`` deterministically."""
        salted = f"{self._seed}:{token}"
        return _stable_hash(salted) % self._dim

    def transform(self, text: str) -> NDArray[np.float32]:
        """Vectorize one transcript into a normalized ``(dim,)`` vector.

        Args:
            text: Raw transcript text.

        Returns:
            L2-normalized float32 vector of shape ``(dim,)``.
        """
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            vec[self._bucket(token)] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def transform_many(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Vectorize several transcripts.

        Args:
            texts: Transcript strings.

        Returns:
            Array of shape ``(len(texts), dim)``. Empty input yields ``(0, dim)``.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self.transform(text) for text in texts]).astype(np.float32)


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


class TransformerTextVectorizer:
    """Fixed-size transcript embedding from a pretrained language model.

    A DAIC-WOZ interview runs to a couple of thousand participant words, far past
    any encoder's context window, so the document is split into token chunks.
    Each chunk is embedded and its tokens are attention-masked mean-pooled; the
    chunk vectors are then averaged and the result is L2-normalized. Averaging
    (rather than taking the first chunk) keeps the whole session in the
    representation, matching the full-session coverage the audio/video modalities
    already have (ADR-0011).

    The model is loaded once per ``(model_name, device)`` and shared, so building
    separate train/dev/test datasets does not reload the weights.

    Args:
        model_name: Hugging Face model id.
        max_length: Tokens per chunk; must not exceed the model's limit.
        batch_size: Chunks encoded per forward pass.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 384,
        batch_size: int = 16,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "TransformerTextVectorizer needs the `nlp` extra: uv pip install -e '.[nlp]'"
            ) from exc

        from privchain.config import resolve_device

        self._torch = torch
        self._device = resolve_device(device)
        self._max_length = max_length
        self._batch_size = batch_size
        self.model_name = model_name

        key = (model_name, self._device)
        if key not in _MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(self._device).eval()
            _MODEL_CACHE[key] = (tokenizer, model)
        self._tokenizer, self._model = _MODEL_CACHE[key]
        self._dim = int(self._model.config.hidden_size)

    @property
    def dim(self) -> int:
        """The output vector dimension (the model's hidden size)."""
        return self._dim

    def transform(self, text: str) -> NDArray[np.float32]:
        """Embed one transcript into a normalized ``(dim,)`` vector.

        Args:
            text: The participant's concatenated transcript turns.

        Returns:
            L2-normalized float32 vector of shape ``(dim,)``. An empty transcript
            yields a zero vector rather than an error, so one unusable session
            cannot abort a whole run.
        """
        torch = self._torch
        if not text.strip():
            return np.zeros(self._dim, dtype=np.float32)

        # `return_overflowing_tokens` splits the document into as many
        # max_length-sized windows as it needs, with the model's special tokens
        # and padding handled by the tokenizer itself.
        encoded = self._tokenizer(
            text,
            max_length=self._max_length,
            truncation=True,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        vectors: list[Any] = []
        with torch.no_grad():
            for start in range(0, input_ids.shape[0], self._batch_size):
                stop = start + self._batch_size
                ids = input_ids[start:stop].to(self._device)
                mask_2d = attention_mask[start:stop].to(self._device)
                hidden = self._model(input_ids=ids, attention_mask=mask_2d).last_hidden_state
                mask = mask_2d.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                vectors.append(pooled.float().cpu())

        document = torch.cat(vectors, dim=0).mean(dim=0)
        norm = float(document.norm())
        if norm > 0.0:
            document = document / norm
        return np.asarray(document.numpy(), dtype=np.float32)

    def transform_many(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed several transcripts, batching chunks across them.

        Segment- and turn-level representations ask for many short texts per
        session instead of one long one. Embedding them one call at a time would
        run a forward pass per segment; this pools every text's chunks into
        shared batches, so the cost stays close to the single-document case.

        Averaging and L2 normalization match :meth:`transform` exactly, so the
        two paths cannot drift apart.

        Args:
            texts: Transcript strings, one per segment or turn.

        Returns:
            Array of shape ``(len(texts), dim)``. Empty strings map to zero rows.
        """
        torch = self._torch
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        chunk_ids: list[Any] = []
        chunk_masks: list[Any] = []
        owner: list[int] = []  # which text each chunk came from
        for position, text in enumerate(texts):
            if not text.strip():
                continue
            encoded = self._tokenizer(
                text,
                max_length=self._max_length,
                truncation=True,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids = encoded["input_ids"]
            chunk_ids.append(ids)
            chunk_masks.append(encoded["attention_mask"])
            owner.extend([position] * int(ids.shape[0]))

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        if not chunk_ids:
            return out

        all_ids = torch.cat(chunk_ids, dim=0)
        all_masks = torch.cat(chunk_masks, dim=0)
        pooled_chunks: list[Any] = []
        with torch.no_grad():
            for start in range(0, all_ids.shape[0], self._batch_size):
                stop = start + self._batch_size
                ids = all_ids[start:stop].to(self._device)
                mask_2d = all_masks[start:stop].to(self._device)
                hidden = self._model(input_ids=ids, attention_mask=mask_2d).last_hidden_state
                mask = mask_2d.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled_chunks.append(pooled.float().cpu())

        chunk_vectors = torch.cat(pooled_chunks, dim=0)
        for position in range(len(texts)):
            rows = [i for i, source in enumerate(owner) if source == position]
            if not rows:
                continue  # empty text keeps its zero row
            document = chunk_vectors[rows].mean(dim=0)
            norm = float(document.norm())
            if norm > 0.0:
                document = document / norm
            out[position] = document.numpy()
        return out


def build_text_vectorizer(text_config: dict[str, Any], *, seed: int = 0) -> TextVectorizer:
    """Construct the vectorizer named by a ``daic_woz.text`` config block.

    Args:
        text_config: The ``text`` sub-mapping from ``configs/daic_woz.yaml``.
        seed: Seed for the hashing vectorizer's token salt.

    Returns:
        The configured :class:`TextVectorizer`.

    Raises:
        ValueError: If ``vectorizer`` names an unknown or unimplemented option.
    """
    kind = str(text_config.get("vectorizer", "hashing")).lower()

    if kind == "hashing":
        return HashingTextVectorizer(int(text_config["dim"]), seed=seed)

    if kind == "transformer":
        options = dict(text_config.get("transformer", {}))
        return TransformerTextVectorizer(
            str(options.get("model_name", "sentence-transformers/all-mpnet-base-v2")),
            max_length=int(options.get("max_length", 384)),
            batch_size=int(options.get("batch_size", 16)),
            device=str(options.get("device", "auto")),
        )

    # `tfidf` appears in older config comments but was never implemented; failing
    # loudly beats silently falling back to a different feature space.
    raise ValueError(f"Unknown text vectorizer {kind!r}. Supported: 'hashing', 'transformer'.")


def _stable_hash(text: str) -> int:
    """A small, process-stable string hash (FNV-1a, 64-bit).

    Python's built-in ``hash`` is salted per process, which would make features
    non-reproducible across runs; FNV-1a is deterministic.

    Args:
        text: Input string.

    Returns:
        A non-negative 64-bit hash.
    """
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
