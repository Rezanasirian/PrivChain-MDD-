"""Smoke-test the real transcript vectorizer and its cache."""

import time

import yaml

from privchain.data.daic_woz import build_daic_woz_dataset

with open("configs/daic_woz.yaml", encoding="utf-8") as config_file:
    cfg = yaml.safe_load(config_file)
t0 = time.time()
ds = build_daic_woz_dataset(cfg, split="train")
print("dims:", ds.feature_dims, f"({time.time() - t0:.1f}s to build)")

t0 = time.time()
s = ds[0]
txt = s["text"]
print(
    f"sample0 text {tuple(txt.shape)} norm={float(txt.norm()):.4f} "
    f"nonzero={int((txt != 0).sum())} ({time.time() - t0:.1f}s)"
)

t0 = time.time()
for i in range(len(ds)):
    ds[i]
print(f"embedded all {len(ds)} train sessions in {time.time() - t0:.1f}s")
