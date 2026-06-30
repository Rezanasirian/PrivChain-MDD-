"""Pre-commit guard: block raw data / media files from being committed.

Enforces CLAUDE.md §7. Fails if any staged file is a raw-data extension
(``.wav``, ``.mp4``, ``.avi``, ``.mov``, ``.csv``) or lives under ``data/``
(other than the allowlisted ``.gitkeep`` / ``README.md`` placeholders).

Invoked by ``.pre-commit-config.yaml`` with the staged filenames as arguments.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

BLOCKED_SUFFIXES = {".wav", ".mp4", ".avi", ".mov", ".csv"}
ALLOWLIST = {"data/.gitkeep", "data/README.md"}


def is_blocked(path_str: str) -> bool:
    """Return ``True`` if a staged path must not be committed.

    Args:
        path_str: A repo-relative staged file path (any OS separator).

    Returns:
        Whether the path is blocked by the raw-data policy.
    """
    path = PurePosixPath(path_str.replace("\\", "/"))
    normalized = path.as_posix()
    if normalized in ALLOWLIST:
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return True
    return normalized.startswith("data/")


def main(argv: list[str]) -> int:
    """Check staged files; return a non-zero exit code if any are blocked.

    Args:
        argv: Staged filenames passed by pre-commit.

    Returns:
        Process exit code (0 = clean, 1 = blocked files found).
    """
    blocked = [p for p in argv if is_blocked(p)]
    if blocked:
        print("ERROR: refusing to commit raw data / secrets (CLAUDE.md §7):")
        for path in blocked:
            print(f"  - {path}")
        print("If this is mock data, keep it under data/ (git-ignored) and do not stage it.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
