"""Density-model controls for the image-vs-signal margin (added in revision).

E11's 0.909 comes from PaDiM, a position-wise Gaussian.  The signal-domain
reference was a nearest-neighbour detector, so the margin confounds three
things: the rendering, the pretrained features, and the density family.  Two
controls separate them, using the SAME PaDiM class, shrinkage, seeded 256-d
random projection and frozen index sets as the image pipeline:

  signal_padim   no rendering, no backbone: the 28 z-scored channels cut into
                 16 time bins of 32 samples; one Gaussian per bin over the
                 (28 x 32 = 896 -> 256)-d vector.  The classical golden-cycle
                 monitor, in PaDiM form.
  pixel_padim    rendering but no backbone: E11 image cut into the same 14 x 14
                 grid the backbones are pooled to; one Gaussian per cell over
                 the raw (16 x 16 x 3 = 768 -> 256)-d pixel vector.

Writes results/31_density_controls/density_controls.csv
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.detectors import PaDiM            # noqa: E402
from voraus_vision.encodings import CORE_CHANNEL_NAMES  # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
OUT = ROOT / "results" / "31_density_controls"
OUT.mkdir(parents=True, exist_ok=True)
KS = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "all"]
SEEDS = [0, 1, 2, 3, 4]
fewshot = {**json.load(open(PRE / "fewshot_indices.json")),
           **json.load(open(ROOT / "results" / "29_intermediate_k" / "fewshot_indices_ext.json"))}


def project(X, dim=256, seed=0):
    """Same seeded Gaussian projection as voraus_vision.backbones.Backbone._project."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = (torch.randn(X.shape[-1], dim, generator=g) / np.sqrt(dim)).numpy()
    return (X @ P).astype(np.float32)


def auroc(y, s):
    r = rankdata(s)
    n = int(y.sum())
    return float((r[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def mean_cat(s, anom, cat):
    sn = s[~anom]
    return float(np.mean([auroc(np.r_[np.ones((anom & (cat == c)).sum(), bool), np.zeros(len(sn), bool)],
                                np.r_[s[anom & (cat == c)], sn]) for c in sorted(set(cat[anom]))]))


meta = pd.read_csv(PRE / "meta.csv")
rows = []

# ---------------------------------------------------------------- signal_padim
W = np.load(PRE / "windows.npy", mmap_mode="r")
stats = json.load(open(PRE / "norm_stats.json"))
kept = json.load(open(PRE / "kept_channels.json"))["kept_channel_names"]
CORE = np.array([kept.index(n) for n in CORE_CHANNEL_NAMES if n in kept])
Z = ((np.asarray(W, np.float64) - np.array(stats["mean"])[None, :, None]) / np.array(stats["std"])[None, :, None])[:, CORE]
n, c, t = Z.shape
F = Z.reshape(n, c, 16, t // 16).transpose(0, 2, 1, 3).reshape(n, 16, c * (t // 16))     # (n, 16 bins, 896)
F = project(F)
t_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
t_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
t_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
for k in KS:
    for seed in SEEDS:
        bi = np.array(fewshot[f"k{k}_seed{seed}"])
        s, _ = PaDiM().fit(F[bi]).score(F[t_idx])
        rows.append({"control": "signal_padim", "k": k, "seed": seed, "auroc": mean_cat(np.asarray(s, np.float64), t_anom, t_cat)})
    print(f"[signal_padim] k={k}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "density_controls.csv", index=False)

# ----------------------------------------------------------------- pixel_padim
man = pd.read_csv(REN / "manifest.csv")
bank, test = man[man["kind"] == "bank"], man[man["kind"] == "test"]
used = np.sort(np.r_[bank["row"].to_numpy(), test["row"].to_numpy()])
imgs = np.asarray(np.load(REN / "E11" / "images.npy", mmap_mode="r")[used]).astype(np.float32) / 255.0
pos = {int(r): i for i, r in enumerate(used)}
G = imgs.reshape(len(used), 14, 16, 14, 16, 3).transpose(0, 1, 3, 2, 4, 5).reshape(len(used), 196, 768)
G = project(G)
src_to_pos = {int(s): pos[int(r)] for s, r in zip(bank["src_idx"], bank["row"])}
tp = np.array([pos[int(r)] for r in test["row"]])
p_anom, p_cat = test["anomaly"].to_numpy().astype(bool), test["category"].to_numpy()
for k in KS:
    for seed in SEEDS:
        bi = np.array([src_to_pos[int(s)] for s in fewshot[f"k{k}_seed{seed}"]])
        s, _ = PaDiM().fit(G[bi]).score(G[tp])
        rows.append({"control": "pixel_padim", "k": k, "seed": seed, "auroc": mean_cat(np.asarray(s, np.float64), p_anom, p_cat)})
    print(f"[pixel_padim] k={k}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "density_controls.csv", index=False)
print(df.groupby(["control", "k"]).auroc.agg(["mean", "std"]).round(3).unstack(0).reindex(KS).to_string())
