"""Cache the *raw-length* recordings of the injection holdout set.

S5 (duration drift) has to be applied before fixed-length resampling, so it
cannot be simulated from the already-resampled windows: stretching a length-512
window and resampling it back is a double interpolation whose error is far
larger than the real pipeline's, which would make P4 look violated for purely
numerical reasons.

Writes results/01_preprocess/holdout_raw.npz (object array, one (C, T) float32
array per holdout recording).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref" / "voraus-ad-dataset"))
from voraus_ad import Signals  # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
DATASET = ROOT / "data" / "voraus-ad-dataset-100hz.parquet"

meta = pd.read_csv(PRE / "meta.csv")
kept = json.load(open(PRE / "kept_channels.json"))
hold = json.load(open(PRE / "holdout_normal.json"))["holdout_normal_indices"]

want_samples = set(meta.loc[meta["idx"].isin(hold), "sample"].tolist())
print(f"[raw] extracting {len(want_samples)} holdout recordings at native length ...")

machine_cols = list(Signals.machine())
df = pd.read_parquet(DATASET, columns=machine_cols + ["sample"])
df = df[df["sample"].isin(want_samples)]
kept_names = kept["kept_channel_names"]

out = {}
for s, grp in df.groupby("sample", sort=True):
    out[str(int(s))] = grp[kept_names].to_numpy(dtype=np.float32).T   # (C, T)

lens = [v.shape[1] for v in out.values()]
print(f"[raw] lengths min={min(lens)} median={int(np.median(lens))} max={max(lens)}")
np.savez_compressed(PRE / "holdout_raw.npz", **out)
print(f"[done] wrote {PRE/'holdout_raw.npz'}")
