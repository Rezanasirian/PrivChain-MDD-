"""Run published late fusion against this server's retrained artifacts.

The licensed corpus copy lacks participant 440, and the retrained RoBERTa
checkpoint has its own dev-selected threshold.  This adapter keeps the
published checkout unchanged while applying those two local facts.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import pandas as pd


def configure(repo: Path, data_root: Path) -> Path:
    repo = repo.resolve()
    source_root = repo / "src"
    os.environ["DAIC_WOZ_ROOT"] = str(data_root.resolve())
    sys.path[:0] = [str(source_root), str(source_root / "ctd")]
    from common import daic_cleaning

    daic_cleaning.KNOWN_ERRORS[440] = {
        "reason": "USC archive copy has no transcript/audio for participant 440",
        "action": "exclude",
        "source": "local corpus availability audit 2026-09-04",
    }
    return source_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--extract-ctd", action="store_true")
    parser.add_argument("--wavlm-seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_root = configure(args.repo, args.data_root)
    sys.path.insert(0, str(source_root / "fusion"))

    if args.extract_ctd:
        module = importlib.import_module("extract_ctd_roberta")
        module.OUT.mkdir(parents=True, exist_ok=True)
        module.extract_ctd()

    if args.wavlm_seeds:
        module = importlib.import_module("run_fusion_wavlm_seeds")
        dev_predictions = (
            source_root / "semantic-depr-roberta" / "results" / "runs"
            / "roberta-large_frozen_seed43" / "dev_predictions_best.csv"
        )
        thresholds = pd.read_csv(dev_predictions)["threshold"].dropna().unique()
        if len(thresholds) != 1:
            raise ValueError(f"Expected one RoBERTa threshold; got {thresholds}")
        module.ROBERTA_THR = float(thresholds[0])
        for seed in args.wavlm_seeds:
            module.run_fusion_for_seed(seed, device=args.device)


if __name__ == "__main__":
    main()
