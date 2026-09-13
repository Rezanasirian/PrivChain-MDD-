"""Run the published CTD baseline without modifying its upstream checkout.

The USC archive available to this project has no transcript/audio for participant
440.  The reference repository's cleaned dev cohort retains 440, so its stock
entry point crashes before feature extraction.  This wrapper records 440 as an
additional corpus-availability exclusion and then executes the upstream script
unchanged.  Consequently our honest cohort is train/dev/test = 102/32/45 rather
than the paper's 102/33/45.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Apply the documented local exclusion and run upstream ``ml_splits.py``."""
    parser = argparse.ArgumentParser(description="Published CTD baseline adapter")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    ctd_dir = args.repo.resolve() / "src" / "ctd"
    source_root = args.repo.resolve() / "src"
    entrypoint = ctd_dir / "ml_splits.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"CTD entry point not found: {entrypoint}")

    os.environ["DAIC_WOZ_ROOT"] = str(args.data_root.resolve())
    sys.path[:0] = [str(ctd_dir), str(source_root)]

    from common import daic_cleaning

    daic_cleaning.KNOWN_ERRORS[440] = {
        "reason": "USC archive copy has no transcript/audio for participant 440",
        "action": "exclude",
        "source": "local corpus availability audit 2026-09-04",
    }
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
