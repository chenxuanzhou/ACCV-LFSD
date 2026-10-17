"""Where exactly does the imaging detour lose its accuracy?

Three candidate mechanisms have now been separated:

  invariance   provable, but results/17_affine shows only ~9 percentage points
               of a real anomaly's deviation is affine, so it cannot explain the
               real-data gap on this dataset
  footprint    results/08_diagnose shows concentrated faults fall below the
               rendering's resolution -- but results/17_affine also shows real
               voraus-AD anomalies occupy 33-82 of the 129 channels, so they are
               not concentrated either
  plain loss   what is left: rendering a 129x512 float array as a 224x224x3
               uint8 image downsamples time, quantises amplitude, and (for the
               heatmap family) discards per-channel scale

This script applies the rendering's information-destroying steps to the signal
*without leaving the signal domain*, one at a time, and scores each stage with
the same nearest-neighbour detector.  The resulting ladder attributes the loss
to a step instead of to "the image domain" as a whole.

Stages
  A0 raw, train z-scored                      (129, 512) float
  A1 + per-window per-channel min-max         (the P1 step)
  A2 + time downsampled 512 -> 224
  A3 + 8-bit quantisation
  A4 + channel axis resampled 129 -> 224 -> 129   (the bilinear row round trip)

Writes results/18_detour/detour_ladder.csv
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "results" / "01_preprocess"
OUT = ROOT / "results" / "18_detour"
OUT.mkdir(parents=True, exist_ok=True)

W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
stats = json.load(open(PRE / "norm_stats.json"))
fewshot = json.load(open(PRE / "fewshot_indices.json"))
ch_mean = np.array(stats["mean"])
ch_std = np.array(stats["std"])

test_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
test_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
test_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
CATS = sorted(set(test_cat[test_anom]))
SEEDS = [0, 1, 2, 3, 4]
K_ALL = ["1", "2", "4", "8", "16", "all"]


def resample_axis(A, n, axis):
    src = np.linspace(0.0, A.shape[axis] - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, A.shape[axis] - 1)
    w = src - i0
    lo = np.take(A, i0, axis=axis)
    hi = np.take(A, i1, axis=axis)
    shape = [1] * A.ndim
    shape[axis] = n
    return lo * (1 - w.reshape(shape)) + hi * w.reshape(shape)


def minmax_rows(A):
    mn = A.min(axis=2, keepdims=True)
    mx = A.max(axis=2, keepdims=True)
    rng = np.where(mx - mn < 1e-12, 1.0, mx - mn)
    return np.where(mx - mn < 1e-12, 0.5, (A - mn) / rng)


def stage(A, name):
    if name == "A0":
        return A
    B = minmax_rows(A)
    if name == "A1":
        return B
    B = resample_axis(B, 224, 2)
    if name == "A2":
        return B
    B = np.round(np.clip(B, 0, 1) * 255.0) / 255.0
    if name == "A3":
        return B
    B = resample_axis(resample_axis(B, 224, 1), A.shape[1], 1)
    return B


def auroc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def mean_cat_auroc(scores):
    sn = scores[~test_anom]
    vals = []
    for c in CATS:
        m = test_anom & (test_cat == c)
        y = np.r_[np.ones(m.sum(), bool), np.zeros(len(sn), bool)]
        vals.append(auroc(y, np.r_[scores[m], sn]))
    return float(np.nanmean(vals))


print("[load] z-scoring the full window set ...")
Z = ((np.asarray(W, np.float32) - ch_mean[None, :, None].astype(np.float32))
     / ch_std[None, :, None].astype(np.float32))
print(f"[load] {Z.shape}")

rows = []
for st in ["A0", "A1", "A2", "A3", "A4"]:
    S = stage(Z, st).reshape(len(Z), -1).astype(np.float32)
    # scale-free comparison: each stage lives on its own range, and a nearest
    # neighbour distance is not comparable across ranges unless normalised
    S /= max(float(np.sqrt((S ** 2).mean())), 1e-12)
    Q = S[test_idx]
    for k in K_ALL:
        for seed in SEEDS:
            key = f"k{k}_seed{seed}" if k != "all" else f"kall_seed{seed}"
            B = S[np.array(fewshot[key])]
            d = np.empty(len(Q), np.float32)
            for i in range(0, len(Q), 128):
                q = Q[i:i + 128]
                dd = ((q ** 2).sum(1)[:, None] - 2 * q @ B.T + (B ** 2).sum(1)[None])
                d[i:i + len(q)] = np.sqrt(np.maximum(dd.min(axis=1), 0))
            rows.append({"stage": st, "k": k, "seed": seed,
                         "auroc_mean_of_categories": mean_cat_auroc(d)})
    m = np.mean([r["auroc_mean_of_categories"] for r in rows
                 if r["stage"] == st and r["k"] == "4"])
    print(f"  {st}: k=4 mean-of-categories AUROC = {m:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "detour_ladder.csv", index=False)
piv = df.pivot_table(index="stage", columns="k", values="auroc_mean_of_categories")
piv = piv.reindex(columns=[c for c in K_ALL if c in piv.columns])
piv.to_csv(OUT / "detour_ladder_summary.csv")
print()
print("=== nearest-neighbour AUROC by pipeline stage (mean over 5 seeds) ===")
print(piv.round(3).to_string())
print("\nStages: A0 raw | A1 +per-window min-max | A2 +time 512->224 | "
      "A3 +8-bit | A4 +channel-axis round trip")
