"""Global reproducibility seeding.

Phase 0 (Environment & Data Setup). Every training/eval entry point must call
:func:`seed_everything` with the ``seed`` read from config so that ``torch``,
``numpy``, and ``random`` are all deterministic — reproducibility is required
for the Chapter 4 results (see CLAUDE.md §3).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed all sources of randomness used in the project.

    Args:
        seed: The integer seed (sourced from a ``configs/*.yaml`` file).
        deterministic_torch: If ``True``, also request deterministic cuDNN
            behaviour. This can slow training but removes nondeterminism.

    Returns:
        None.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(*parts: int) -> int:
    """Combine several integers into one seed that cannot collide by addition.

    Repetition harnesses need a distinct seed per (repetition, fold) pair. The
    obvious ``seed + fold`` collides whenever two repetition seeds sit closer
    together than the fold count: with 5 folds, seeds 42 and 45 share three fold
    seeds, so three of the two arms' "independent" repetitions are the same run.
    Nothing errors and the spread over seeds simply comes out too small.

    A wide multiplicative mix keeps every pair distinct regardless of how the
    caller chose its seeds, and stays deterministic across processes and
    platforms (unlike :func:`hash`, which is salted per process).

    Args:
        *parts: The coordinates identifying this run, most significant first.

    Returns:
        A seed in ``[0, 2**31)``.

    Raises:
        ValueError: If no parts are given.
    """
    if not parts:
        raise ValueError("derive_seed needs at least one part")
    # FNV-1a over the parts' 64-bit two's-complement representations.
    digest = 0xCBF29CE484222325
    for part in parts:
        for byte in int(part).to_bytes(8, "little", signed=True):
            digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest % (2**31)
