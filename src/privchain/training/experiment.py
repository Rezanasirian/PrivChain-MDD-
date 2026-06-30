"""Experiment run directories and metric logging (Phase 1).

Implements the minimum experiment-logging standard from CLAUDE.md §3: every run
writes to ``experiments/<phase>/<run-id>/`` the config used, metrics as JSONL,
and checkpoints. Run IDs follow the ``phaseN_<description>_<date>`` convention
(CLAUDE.md §5).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / phase / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: dict[str, Any]) -> None:
    """Write the resolved config snapshot to ``<run_dir>/config.yaml``.

    Args:
        run_dir: Target run directory.
        config: The fully resolved configuration mapping.
    """
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


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
