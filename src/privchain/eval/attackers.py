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


def clip_rows(embeddings: NDArray[np.float64], max_norm: float) -> NDArray[np.float64]:
    """Scale each row down so its L2 norm is at most ``max_norm``.

    Clipping is what gives a released embedding a *bounded sensitivity*, without
    which no Gaussian-mechanism noise level can claim an ``(ε, δ)`` guarantee.

    Args:
        embeddings: Array of shape ``(N, D)``.
        max_norm: The L2 bound ``C_e``.

    Returns:
        A new array whose rows all have norm ``<= max_norm``.

    Raises:
        ValueError: If ``max_norm`` is not positive.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    matrix = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.where(norms > 0.0, norms, 1.0))
    return matrix * scale


def release_embeddings_dp(
    embeddings: NDArray[np.float64],
    *,
    target_epsilon: float,
    delta: float,
    clip_norm: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Release embeddings under the ``(ε, δ)`` Gaussian mechanism.

    Each row is clipped to ``clip_norm`` — bounding the change one subject can
    cause — and perturbed with Gaussian noise of standard deviation
    ``σ(ε, δ) · clip_norm``, where ``σ`` comes from the same RDP accountant used
    for DP-SGD, evaluated as a single unsubsampled application.

    This is the mechanism that actually bounds *re-identification*. DP-SGD does
    not: it constrains how much a trained model depends on any one training
    record, not how distinctive the encoder's output is for a given input.

    Args:
        embeddings: Array of shape ``(N, D)`` to release.
        target_epsilon: Privacy budget ``ε`` for the release.
        delta: Target ``δ``.
        clip_norm: L2 bound applied before noising (the sensitivity).
        rng: Source of randomness.

    Returns:
        The clipped, noised embeddings.
    """
    from privchain.privacy.accountant import get_noise_multiplier

    clipped = clip_rows(embeddings, clip_norm)
    sigma = get_noise_multiplier(target_epsilon, sample_rate=1.0, steps=1, delta=delta)
    return add_gaussian_noise(clipped, sigma * clip_norm, rng)


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

    def enroll(self, embeddings: NDArray[np.float64], subject_ids: NDArray[np.int_]) -> None:
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

    def attack(self, probes: NDArray[np.float64], subject_ids: NDArray[np.int_]) -> float:
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
    non-members. The attack calibrates a decision rule on one part of the data
    and *reports* on the rest, which matters because the two obvious shortcuts
    both inflate the result:

    * picking the best threshold on the same scores it is scored against turns
      noise into apparent leakage, and
    * folding the ROC-AUC about 0.5 (``max(auc, 1-auc)``) makes the reported
      advantage non-negative by construction, so a model that leaks nothing still
      looks like it leaks something.

    Here the threshold *and* the direction of the decision rule are chosen on a
    calibration split; ``auc``, ``accuracy`` and ``advantage`` are then computed
    on the held-out remainder and may legitimately come out at (or below) chance.

    Args:
        calibration_fraction: Share of each group used to pick the decision rule.
    """

    def __init__(self, calibration_fraction: float = 0.5) -> None:
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        self.calibration_fraction = calibration_fraction

    def attack(
        self,
        member_scores: NDArray[np.float64],
        nonmember_scores: NDArray[np.float64],
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Score a membership-inference attempt.

        Args:
            member_scores: Membership signal for training samples (higher ⇒ more
                member-like, e.g. negative loss or model confidence).
            nonmember_scores: The same signal for held-out samples.
            rng: Source of randomness for the calibration split (defaults to a
                fixed seed so runs are reproducible).

        Returns:
            Mapping with ``auc`` (on the evaluation split, not folded about 0.5),
            ``accuracy`` (calibrated threshold applied to the evaluation split),
            ``advantage`` (``2·AUC − 1``, which may be negative), and
            ``direction`` (``+1`` if higher scores were taken to mean "member").

        Raises:
            ValueError: If either group is empty or too small to split.
        """
        if len(member_scores) == 0 or len(nonmember_scores) == 0:
            raise ValueError("both member and non-member scores are required")
        if len(member_scores) < 2 or len(nonmember_scores) < 2:
            raise ValueError("each group needs at least 2 scores to calibrate and evaluate")

        generator = rng if rng is not None else np.random.default_rng(0)
        cal_members, eval_members = _split_scores(
            member_scores, self.calibration_fraction, generator
        )
        cal_nonmembers, eval_nonmembers = _split_scores(
            nonmember_scores, self.calibration_fraction, generator
        )

        direction, threshold = _calibrate_rule(cal_members, cal_nonmembers)

        eval_scores = np.concatenate([eval_members, eval_nonmembers]).astype(np.float64)
        eval_labels = np.concatenate(
            [np.ones(len(eval_members), dtype=int), np.zeros(len(eval_nonmembers), dtype=int)]
        )
        auc = roc_auc_score(eval_labels, direction * eval_scores)
        if np.isnan(auc):
            auc = 0.5
        predicted = (direction * eval_scores >= direction * threshold).astype(int)
        accuracy = float(np.mean(predicted == eval_labels))

        return {
            "auc": float(auc),
            "accuracy": accuracy,
            "advantage": 2.0 * float(auc) - 1.0,
            "direction": float(direction),
        }


def _split_scores(
    scores: NDArray[np.float64], calibration_fraction: float, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Randomly split one group's scores into calibration and evaluation parts."""
    shuffled = np.asarray(scores, dtype=np.float64)[rng.permutation(len(scores))]
    cut = int(round(len(shuffled) * calibration_fraction))
    cut = min(max(cut, 1), len(shuffled) - 1)
    return shuffled[:cut], shuffled[cut:]


def _calibrate_rule(
    member_scores: NDArray[np.float64], nonmember_scores: NDArray[np.float64]
) -> tuple[int, float]:
    """Choose the decision direction and threshold on the calibration split.

    Args:
        member_scores: Calibration scores for members.
        nonmember_scores: Calibration scores for non-members.

    Returns:
        ``(direction, threshold)`` where ``direction`` is ``+1`` when higher
        scores are treated as more member-like and ``-1`` otherwise.
    """
    scores = np.concatenate([member_scores, nonmember_scores]).astype(np.float64)
    labels = np.concatenate(
        [np.ones(len(member_scores), dtype=int), np.zeros(len(nonmember_scores), dtype=int)]
    )
    best = (-1.0, 1, float(scores[0]))
    for direction in (1, -1):
        oriented = direction * scores
        for threshold in np.unique(scores):
            predicted = (oriented >= direction * threshold).astype(int)
            accuracy = float(np.mean(predicted == labels))
            if accuracy > best[0]:
                best = (accuracy, direction, float(threshold))
    return best[1], best[2]
