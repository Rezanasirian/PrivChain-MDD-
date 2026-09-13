"""Paired analysis and W&B reporting for locally reproduced reference models."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fusion-seed", type=int, default=44)
    parser.add_argument("--wandb-project", default="privchain-mdd-reference-baselines")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    source = repo / "src"
    os.environ["DAIC_WOZ_ROOT"] = str(args.data_root.resolve())
    sys.path[:0] = [str(source), str(source / "ctd"), str(source / "fusion")]

    from common import daic_cleaning

    daic_cleaning.KNOWN_ERRORS[440] = {
        "reason": "USC archive copy has no transcript/audio for participant 440",
        "action": "exclude",
        "source": "local corpus availability audit 2026-09-04",
    }

    paired = importlib.import_module("paired_delta_analysis")
    fusion_path = (
        source / "ctd" / "outputs" / "fusion_by_wavlm_seed"
        / f"fusion_wavlm_seed{args.fusion_seed}.json"
    )
    fusion = json.loads(fusion_path.read_text())
    rule_name = "wconvex[roberta+ctd]"
    rule = fusion["results"][rule_name]
    weights = rule["weights"]

    dev_csv = source / "semantic-depr-roberta" / "results" / "runs" / (
        "roberta-large_frozen_seed43/dev_predictions_best.csv"
    )
    roberta_thresholds = pd.read_csv(dev_csv)["threshold"].dropna().unique()
    if len(roberta_thresholds) != 1:
        raise ValueError(f"Expected one RoBERTa threshold; got {roberta_thresholds}")

    paired.ROBERTA_THR = float(roberta_thresholds[0])
    paired.ROBERTA_WEIGHT = float(weights["roberta"])
    paired.CTD_WEIGHT = float(weights["ctd"])
    paired.FUSION_THR = float(rule["threshold"])
    ctd = paired.ctd_probs()

    report: dict[str, object] = {
        "selection_contract": "weights and threshold selected on dev; test read once",
        "fusion_seed_artifact": args.fusion_seed,
        "rule": rule_name,
        "weights": weights,
        "thresholds": {
            "roberta": paired.ROBERTA_THR,
            "ctd": paired.CTD_THR,
            "fusion": paired.FUSION_THR,
        },
        "splits": {},
    }
    for split in ("dev", "test"):
        ids, y, p_roberta, p_ctd = paired.align(paired.read_roberta(split), ctd[split])
        p_fused = paired.ROBERTA_WEIGHT * p_roberta + paired.CTD_WEIGHT * p_ctd
        predictions = {
            "roberta": (p_roberta >= paired.ROBERTA_THR).astype(int),
            "ctd": (p_ctd >= paired.CTD_THR).astype(int),
            "fusion": (p_fused >= paired.FUSION_THR).astype(int),
        }
        split_report: dict[str, object] = {
            "n": len(ids),
            "depressed": int(y.sum()),
            "macro_f1": {k: paired.macro_f1(y, v) for k, v in predictions.items()},
            "paired_delta_fusion_minus_baseline": {},
        }
        deltas = split_report["paired_delta_fusion_minus_baseline"]
        assert isinstance(deltas, dict)
        for baseline in ("roberta", "ctd"):
            point, low, high = paired.paired_ci(y, predictions["fusion"], predictions[baseline])
            changed, corrected, worsened, neutral = paired.change_counts(
                y, predictions[baseline], predictions["fusion"]
            )
            deltas[baseline] = {
                "point": point,
                "ci_low": low,
                "ci_high": high,
                "changed": changed,
                "corrected": corrected,
                "worsened": worsened,
                "neutral": neutral,
            }
        report["splits"][split] = split_report  # type: ignore[index]

    output = source / "ctd" / "outputs" / "reference_paired_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved -> {output}")

    if not args.disable_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name="reference-baselines-final")
        for result_path in sorted(fusion_path.parent.glob("fusion_wavlm_seed*.json")):
            payload = json.loads(result_path.read_text())
            seed = int(payload["wavlm_seed"])
            best = payload["dev_selected_best"]
            results = payload["results"]
            run.log(
                {
                    "wavlm_seed": seed,
                    "single/wavlm_test_macro_f1": (
                        results["single_wavlm"]["test"]["macro_f1"]["point"]
                    ),
                    "single/roberta_test_macro_f1": (
                        results["single_roberta"]["test"]["macro_f1"]["point"]
                    ),
                    "single/ctd_test_macro_f1": (
                        results["single_ctd"]["test"]["macro_f1"]["point"]
                    ),
                    "fusion/dev_selected_macro_f1": results[best]["dev"]["macro_f1"][
                        "point"
                    ],
                    "fusion/test_macro_f1": results[best]["test"]["macro_f1"]["point"],
                }
            )
        test = report["splits"]["test"]  # type: ignore[index]
        run.summary["paired_test"] = test
        run.summary["conclusion"] = (
            "Fusion does not significantly improve over CTD if delta CI includes zero."
        )
        run.finish()


if __name__ == "__main__":
    main()
