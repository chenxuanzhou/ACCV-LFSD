"""Matched-channel signal-domain baselines (NEXT-STEPS 0.2).

The obvious attack on E11 is that its advantage comes from using 28 hand-picked
channels rather than from the imaging step.  This runs signal-domain detectors
on *exactly the same 28 channels*, with the same 30 frozen few-shot index sets,
so the only difference left is the rendering.

Four arms, crossed over two channel sets (the 28-signal core set and all 129):
  knn_z        nearest neighbour on train-z-scored windows      (E11's normalisation)
  pca_z        PCA reconstruction error on the same             (training-free, signal)
  knn_minmax   nearest neighbour after per-window min-max       (E10's normalisation)
  pca_minmax   PCA reconstruction error on the same

The two normalisations are included because the paper's claim is that channel
count and scale preservation are the axes that matter; if that is right, the
same ordering should appear in the signal domain, which would make the point
about design axes rather than about imaging.

Writes results/24_matched_channel/matched_channel.csv
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.encodings import CORE_CHANNEL_NAMES  # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
OUT = ROOT / "results" / "24_matched_channel"
OUT.mkdir(parents=True, exist_ok=True)

W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
stats = json.load(open(PRE / "norm_stats.json"))
fewshot = json.load(open(PRE / "fewshot_indices.json"))
kept = json.load(open(PRE / "kept_channels.json"))["kept_channel_names"]
CORE = np.array([kept.index(n) for n in CORE_CHANNEL_NAMES if n in kept])
ch_mean = np.array(stats["mean"])
ch_std = np.array(stats["std"])
print(f"[setup] core set = {len(CORE)} of {len(kept)} channels")

test_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
test_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
test_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
CATS = sorted(set(test_cat[test_anom]))
K_ALL = ["1", "2", "4", "8", "16", "all"]
SEEDS = [0, 1, 2, 3, 4]


def auroc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n = int(y.sum())
    return float((r[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def mean_cat(scores):
    sn = scores[~test_anom]
    return float(np.mean([
        auroc(np.r_[np.ones((test_anom & (test_cat == c)).sum(), bool),
                    np.zeros(len(sn), bool)],
              np.r_[scores[test_anom & (test_cat == c)], sn]) for c in CATS]))


def zscore(X):
    return (X - ch_mean[None, :, None]) / ch_std[None, :, None]


def minmax(X):
    mn = X.min(axis=2, keepdims=True)
    mx = X.max(axis=2, keepdims=True)
    rng = np.where(mx - mn < 1e-12, 1.0, mx - mn)
    return np.where(mx - mn < 1e-12, 0.5, (X - mn) / rng)


def knn_scores(bank, query):
    B = bank.reshape(len(bank), -1).astype(np.float32)
    Q = query.reshape(len(query), -1).astype(np.float32)
    out = np.empty(len(Q), np.float32)
    for i in range(0, len(Q), 128):
        q = Q[i:i + 128]
        d = (q ** 2).sum(1)[:, None] - 2 * q @ B.T + (B ** 2).sum(1)[None]
        out[i:i + len(q)] = np.sqrt(np.maximum(d.min(axis=1), 0))
    return out


def pca_scores(bank, query, n_comp=20):
    F = bank.transpose(0, 2, 1).reshape(-1, bank.shape[1])
    mu = F.mean(0)
    Fc = F - mu
    n_comp = min(n_comp, Fc.shape[0], Fc.shape[1])
    _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
    V = Vt[:n_comp]
    out = np.empty(len(query), np.float64)
    for i in range(0, len(query), 512):
        Q = query[i:i + 512].transpose(0, 2, 1) - mu
        rec = (Q @ V.T) @ V
        out[i:i + len(Q)] = ((Q - rec) ** 2).mean(axis=(1, 2))
    return out


Z_all = zscore(np.asarray(W, np.float64))
M_all = minmax(np.asarray(W, np.float64))
REPS = {
    ("129ch", "z-score"): Z_all,
    ("129ch", "min-max"): M_all,
    ("28ch", "z-score"): Z_all[:, CORE],
    ("28ch", "min-max"): M_all[:, CORE],
}

rows = []
for (chset, norm), X in REPS.items():
    Q = X[test_idx]
    for k in K_ALL:
        for seed in SEEDS:
            key = f"k{k}_seed{seed}" if k != "all" else f"kall_seed{seed}"
            bi = np.array(fewshot[key])
            rows.append({"channels": chset, "normalisation": norm, "detector": "knn",
                         "k": k, "seed": seed,
                         "auroc": mean_cat(knn_scores(X[bi], Q))})
            rows.append({"channels": chset, "normalisation": norm, "detector": "pca20",
                         "k": k, "seed": seed,
                         "auroc": mean_cat(pca_scores(X[bi], Q))})
    print(f"[done] {chset} / {norm}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "matched_channel.csv", index=False)

# Average over the five seeds first, then take the better of the two detectors.
# A max over (detector, seed) would report the luckiest memory bank.
best = (df.groupby(["channels", "normalisation", "k", "detector"])["auroc"].mean()
        .reset_index()
        .groupby(["channels", "normalisation", "k"])["auroc"].max()
        .reset_index()
        .pivot_table(index=["channels", "normalisation"], columns="k", values="auroc"))
best = best[[c for c in K_ALL if c in best.columns]]
best.to_csv(OUT / "matched_channel_best.csv")
print("\n=== signal-domain, best of {kNN, PCA-20}, mean-of-categories AUROC ===")
print(best.round(3).to_string())
print("\n=== per detector ===")
print(df.groupby(["channels", "normalisation", "detector", "k"])["auroc"].mean()
      .unstack()[[c for c in K_ALL]].round(3).to_string())
