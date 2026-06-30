"""CLI: generate the mock DAIC-WOZ dataset to disk (Phase 0).

Reads ``configs/baseline.yaml`` for the mock-data spec and writes one folder per
synthetic session under ``data/mock/`` (git-ignored). Useful for manual
inspection; tests do not depend on these files.

Usage:
    python scripts/generate_mock_data.py [--config configs/baseline.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privchain.config import load_baseline_config
from privchain.data.mock_daic_woz import write_mock_dataset
from privchain.seeding import seed_everything


def main() -> None:
    """Parse args, seed, and write the mock dataset to disk."""
    parser = argparse.ArgumentParser(description="Generate mock DAIC-WOZ data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the baseline config (default: configs/baseline.yaml).",
    )
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    seed_everything(config.seed)
    root = write_mock_dataset(config.data, seed=config.seed)
    print(f"Wrote {config.data.num_sessions} mock sessions to {root.resolve()}")


if __name__ == "__main__":
    main()
