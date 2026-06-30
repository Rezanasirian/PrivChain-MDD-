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
