"""Execute a published reference entry point with local corpus exclusions.

The licensed USC archive on this server lacks participant 440's transcript and
audio.  The published CTD checkout otherwise retains that development session.
This adapter leaves the upstream checkout untouched, injects the documented
availability exclusion, then forwards all remaining CLI arguments.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Run one upstream Python entry point under the local corpus contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    args, forwarded = parser.parse_known_args()

    repo = args.repo.resolve()
    source_root = repo / "src"
    entrypoint = (repo / args.entrypoint).resolve()
    if repo not in entrypoint.parents or not entrypoint.is_file():
        raise FileNotFoundError(f"entry point is outside/missing from repository: {entrypoint}")

    os.environ["DAIC_WOZ_ROOT"] = str(args.data_root.resolve())
    sys.path[:0] = [str(entrypoint.parent), str(source_root)]

    from common import daic_cleaning

    daic_cleaning.KNOWN_ERRORS[440] = {
        "reason": "USC archive copy has no transcript/audio for participant 440",
        "action": "exclude",
        "source": "local corpus availability audit 2026-09-04",
    }
    sys.argv = [str(entrypoint), *forwarded]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
