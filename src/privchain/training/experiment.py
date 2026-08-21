"""Experiment run directories and metric logging (Phase 1).

Implements the minimum experiment-logging standard from CLAUDE.md §3: every run
writes to ``experiments/<phase>/<run-id>/`` the config used, metrics as JSONL,
and checkpoints. Run IDs follow the ``phaseN_<description>_<date>`` convention
(CLAUDE.md §5).
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import yaml


def create_run_dir(output_dir: str | Path, phase: str, run_name: str) -> Path:
    """Create and return a timestamped run directory.

    Args:
        output_dir: Base experiments directory (e.g., ``experiments``).
        phase: Phase sub-folder, e.g. ``phase1``.
        run_name: Human-readable run name; a UTC timestamp is appended.

    Returns:
        The created run directory ``<output_dir>/<phase>/<run_name>_<timestamp>``.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / phase / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_state() -> dict[str, Any]:
    """Return the current commit and dirty flag without failing outside Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, check=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _dependency_versions() -> dict[str, str]:
    """Return versions of dependencies that materially affect results."""
    resolved: dict[str, str] = {}
    for package in ("numpy", "opacus", "pydantic", "scikit-learn", "torch"):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def save_config(
    run_dir: Path, config: dict[str, Any], *, manifest_extra: dict[str, Any] | None = None
) -> None:
    """Write the resolved config snapshot to ``<run_dir>/config.yaml``.

    Args:
        run_dir: Target run directory.
        config: The fully resolved configuration mapping.
        manifest_extra: Optional run-specific provenance such as dataset/split
            digests, excluded-session counts, stop reason, and accountant state.
    """
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "git": _git_state(),
        "dependencies": _dependency_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


class JsonlMetricLogger:
    """Append-only JSONL metric logger (one JSON object per line).

    Args:
        path: Destination ``metrics.jsonl`` file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        """Append one metrics record as a JSON line.

        Args:
            record: A JSON-serializable mapping (e.g., epoch + metric values).
        """
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
