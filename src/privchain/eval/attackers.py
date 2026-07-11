"""Privacy-attacker models (Phase 6, objective H5).

Empirical attackers that quantify how much a *released* (DP-noised) embedding
leaks about the subject it came from — the evidence that the per-modality DP of
H1 actually protects privacy:

* :class:`ReidentificationAttacker` — the shared engine behind the three named
  attackers in the thesis (speaker-identification on audio, face-recognition on
  video, named-entity de-anonymization on text). It enrols a template per
  subject and re-identifies held-out probe embeddings by nearest cosine
  centroid; **top-1 accuracy is the re-identification success rate**.
* :class:`MembershipInferenceAttacker` — a loss/score-threshold attack deciding
  whether a sample was in the training set; its advantage over chance is the
  membership-inference success.

Everything is pure NumPy so the evaluation path stays dependency-light and runs
in CI. Noise is added to embeddings via :func:`add_gaussian_noise`, with the
standard deviation derived from the target ε upstream (see the run script).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from privchain.eval.metrics import roc_auc_score


def add_gaussian_noise(
    embeddings: NDArray[np.float64], std: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return ``embeddings`` perturbed by i.i.d. Gaussian noise of scale ``std``.

    Args:
        embeddings: Array of shape ``(N, D)``.
        std: Noise standard deviation (0 leaves the input unchanged).
        rng: Source of randomness (for reproducibility).

    Returns:
        A new array of the same shape with added noise.
    """
    if std <= 0.0:
        return embeddings.astype(np.float64, copy=True)
    noise = rng.standard_normal(embeddings.shape) * std
    return embeddings.astype(np.float64, copy=True) + noise


def _l2_normalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise L2-normalize, guarding against zero-norm rows."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / norms


class ReidentificationAttacker:
    """Nearest-centroid re-identification attacker (cosine similarity).

    The attacker enrols one template (centroid) per subject from enrollment
    embeddings, then classifies each probe embedding as the subject of its
    nearest template. Top-1 accuracy is the re-identification success rate; the
    chance baseline is ``1 / num_subjects``.
    """

    def __init__(self) -> None:
        """Create an unfitted attacker."""
        self._subjects: NDArray[np.int_] | None = None
        self._centroids: NDArray[np.float64] | None = None

    def enroll(
        self, embeddings: NDArray[np.float64], subject_ids: NDArray[np.int_]
    ) -> None:
        """Build one L2-normalized centroid template per subject.

        Args:
            embeddings: Enrollment embeddings, shape ``(N, D)``.
            subject_ids: Subject label per row, shape ``(N,)``.

        Raises:
            ValueError: If inputs are empty or mismatched.
        """
        if len(embeddings) != len(subject_ids):
            raise ValueError("embeddings and subject_ids must have the same length")
        if len(embeddings) == 0:
            raise ValueError("cannot enroll on empty embeddings")
        subjects = np.unique(subject_ids)
        centroids = np.stack(
            [embeddings[subject_ids == subject].mean(axis=0) for subject in subjects]
        )
        self._subjects = subjects
        self._centroids = _l2_normalize(centroids.astype(np.float64))

    def attack(
        self, probes: NDArray[np.float64], subject_ids: NDArray[np.int_]
    ) -> float:
        """Return the top-1 re-identification accuracy on probe embeddings.

        Args:
            probes: Probe embeddings, shape ``(M, D)``.
            subject_ids: True subject label per probe, shape ``(M,)``.

        Returns:
            Fraction of probes assigned to their true subject.

        Raises:
            RuntimeError: If called before :meth:`enroll`.
            ValueError: If inputs are empty or mismatched.
        """
        if self._centroids is None or self._subjects is None:
            raise RuntimeError("attacker must be enrolled before attacking")
        if len(probes) != len(subject_ids):
            raise ValueError("probes and subject_ids must have the same length")
        if len(probes) == 0:
            raise ValueError("cannot attack on empty probes")
        similarities = _l2_normalize(probes.astype(np.float64)) @ self._centroids.T
        predicted = self._subjects[np.argmax(similarities, axis=1)]
        return float(np.mean(predicted == subject_ids))

    @staticmethod
    def chance_accuracy(num_subjects: int) -> float:
        """Return the random-guess baseline ``1 / num_subjects``.

        Args:
            num_subjects: Number of enrolled subjects.

        Returns:
            The chance top-1 accuracy.

        Raises:
            ValueError: If ``num_subjects`` is not positive.
        """
        if num_subjects <= 0:
            raise ValueError("num_subjects must be positive")
        return 1.0 / num_subjects


class MembershipInferenceAttacker:
    """Loss/score-threshold membership-inference attacker.

    Members (training samples) usually score higher (lower loss) than
    non-members. The attack ranks samples by their membership score and reports
    the ROC-AUC, the best-threshold balanced accuracy, and the membership
    *advantage* ``2·AUC − 1`` (0 = no leakage, 1 = perfect).
    """

    def attack(
        self,
        member_scores: NDArray[np.float64],
        nonmember_scores: NDArray[np.float64],
    ) -> dict[str, float]:
        """Score a membership-inference attempt.

        Args:
            member_scores: Membership signal for training samples (higher ⇒ more
                member-like, e.g. negative loss or model confidence).
            nonmember_scores: The same signal for held-out samples.

        Returns:
            Mapping with ``auc``, ``accuracy`` (best threshold), and
            ``advantage``.

        Raises:
            ValueError: If either group is empty.
        """
        if len(member_scores) == 0 or len(nonmember_scores) == 0:
            raise ValueError("both member and non-member scores are required")
        scores = np.concatenate([member_scores, nonmember_scores]).astype(np.float64)
        labels = np.concatenate(
            [np.ones(len(member_scores), dtype=int), np.zeros(len(nonmember_scores), dtype=int)]
        )
        auc = roc_auc_score(labels, scores)
        if np.isnan(auc):
            auc = 0.5
        # Attackers can flip their decision rule, so leakage is symmetric about 0.5.
        auc = max(auc, 1.0 - auc)
        return {
            "auc": auc,
            "accuracy": _best_threshold_accuracy(scores, labels),
            "advantage": 2.0 * auc - 1.0,
        }


def _best_threshold_accuracy(
    scores: NDArray[np.float64], labels: NDArray[np.int_]
) -> float:
    """Best balanced accuracy over candidate thresholds (both decision signs)."""
    thresholds = np.unique(scores)
    best = 0.5
    n = len(labels)
    for threshold in thresholds:
        for predicted in ((scores >= threshold).astype(int), (scores <= threshold).astype(int)):
            accuracy = float(np.sum(predicted == labels)) / n
            best = max(best, accuracy)
    return best
