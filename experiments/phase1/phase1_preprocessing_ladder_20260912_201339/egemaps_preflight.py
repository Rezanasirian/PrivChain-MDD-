"""Preflight for ADR-0032: every eGeMAPS file is present/finite, both arms load
the same pool, and the audio encoders are capacity-matched."""
import json, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, "scripts")
from run_preprocessing_ladder import ARMS, ARM_ENCODER_OVERRIDES
from privchain.config import load_baseline_config
from privchain.training.protocol import build_splits, labels_of
from privchain.fusion.factory import build_depression_model

cfg = yaml.safe_load(Path("configs/daic_woz.yaml").read_text())["daic_woz"]
root = Path(cfg["root"]); tmpl = cfg.get("participant_dir_template", "{pid}_P")
excluded = {int(p) for p in cfg.get("exclude_participants", [])}
pids = sorted({int(d.name.split("_")[0]) for d in root.glob(tmpl.replace("{pid}", "*"))
               if d.name.split("_")[0].isdigit()} - excluded)
missing, bad = [], []
for pid in pids:
    f = root / tmpl.format(pid=pid) / f"{pid}_eGeMAPSv02.csv"
    if not f.is_file(): missing.append(pid); continue
    v = np.loadtxt(f, delimiter=",", ndmin=2)
    if v.shape != (1, 88) or not np.isfinite(v).all(): bad.append(pid)
print(f"participants={len(pids)} missing={missing} bad={bad}")
assert not missing and not bad

base = load_baseline_config(Path("configs/baseline.yaml"))
report = {}
pool_labels = None
for arm in ("committed", "speech+egemaps"):
    arm_base = base
    if arm in ARM_ENCODER_OVERRIDES:
        eo = dict(base.model.encoder_overrides); eo.update(ARM_ENCODER_OVERRIDES[arm])
        arm_base = base.model_copy(update={"model": base.model.model_copy(update={"encoder_overrides": eo})})
    splits, dims = build_splits(arm_base, Path("configs/daic_woz.yaml"), daic_overrides=ARMS[arm])
    labels = labels_of(splits.train) + labels_of(splits.selection)
    if pool_labels is not None: assert labels == pool_labels, "pool mismatch"
    pool_labels = labels
    model = build_depression_model(dims, arm_base.model, None)
    audio = sum(p.numel() for n, p in model.named_parameters() if "audio" in n)
    total = sum(p.numel() for p in model.parameters())
    report[arm] = {"dims": dims, "pool": len(labels), "dev": len(splits.report),
                   "audio_params": audio, "total_params": total}
    print(arm, report[arm])
a, b = report["committed"]["audio_params"], report["speech+egemaps"]["audio_params"]
print(f"audio capacity difference = {abs(a-b)/a*100:.2f}%")
Path("/workspace/egemaps_preflight.json").write_text(json.dumps(report, indent=2))
