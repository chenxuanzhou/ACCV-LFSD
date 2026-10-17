"""k-shot normalisation statistics (added in revision).

In the main grid the z-score encodings (E2, E11) and the signal baseline use the
per-channel mean and s.d. of norm_stats.json, which is fitted on all 948
training recordings.  k then bounds only the memory bank, so "from a single
normal sample" overstates what was run: a new machine does not come with
channel statistics from 948 cycles.

Here the statistics are estimated from the k bank windows of each frozen index
set and nothing else; E11 is re-rendered for the bank and the 1174 test windows
with those statistics, passed through the same frozen WideResNet-50-2, and
scored by the two fixed detectors (PatchCore, PaDiM).  The 28-channel signal
kNN baseline gets the same treatment.  "pool" repeats the run with the
population statistics as a regression check against results/07_eval.

Writes results/32_kshot_statistics/kshot_statistics.csv
"""
import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.backbones import Backbone                      # noqa: E402
from voraus_vision.context import load_context                    # noqa: E402
from voraus_vision.detectors import PaDiM, PatchCore              # noqa: E402
from voraus_vision.encodings import ENCODERS                      # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
OUT = ROOT / "results" / "32_kshot_statistics"
OUT.mkdir(parents=True, exist_ok=True)
KS = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "all"]
SEEDS = [0, 1, 2, 3, 4]
fewshot = {**json.load(open(PRE / "fewshot_indices.json")),
           **json.load(open(ROOT / "results" / "29_intermediate_k" / "fewshot_indices_ext.json"))}

ctx0 = load_context(ROOT)
W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
t_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
t_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
t_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
CATS = sorted(set(t_cat[t_anom]))
W_test = np.asarray(W[t_idx], np.float64)
CORE = ctx0.core_channels
E11 = ENCODERS["E11"]


def auroc(y, s):
    r = rankdata(s)
    n = int(y.sum())
    return float((r[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def mean_cat(s):
    s = np.asarray(s, np.float64)
    sn = s[~t_anom]
    return float(np.mean([auroc(np.r_[np.ones((t_anom & (t_cat == c)).sum(), bool), np.zeros(len(sn), bool)],
                                np.r_[s[t_anom & (t_cat == c)], sn]) for c in CATS]))


def knn(bank, query):
    B = bank.reshape(len(bank), -1).astype(np.float32)
    Q = query.reshape(len(query), -1).astype(np.float32)
    out = np.empty(len(Q), np.float32)
    for i in range(0, len(Q), 128):
        q = Q[i:i + 128]
        d = (q ** 2).sum(1)[:, None] - 2 * q @ B.T + (B ** 2).sum(1)[None]
        out[i:i + len(q)] = np.sqrt(np.maximum(d.min(axis=1), 0))
    return out


bb = Backbone("wrn50")
out_csv = OUT / "kshot_statistics.csv"
rows = pd.read_csv(out_csv, dtype={"k": str}).to_dict("records") if out_csv.exists() else []
have = {(r["stats"], r["k"], r["seed"]) for r in rows}
for stats_mode in ["kshot", "pool"]:
    for k in KS:
        for seed in SEEDS:
            if (stats_mode, k, seed) in have or (k == "all" and seed > 0):
                continue
            t0 = time.time()
            bi = np.array(fewshot[f"k{k}_seed{seed}"])
            Wb = np.asarray(W[bi], np.float64)
            if stats_mode == "kshot":
                mu = Wb.mean(axis=(0, 2))
                sd = Wb.std(axis=(0, 2))
                sd = np.where(sd < 1e-9, 1.0, sd)       # a channel constant over the k windows
                ctx = dataclasses.replace(ctx0, ch_mean=mu, ch_std=sd)
            else:
                mu, sd, ctx = ctx0.ch_mean, ctx0.ch_std, ctx0
            imgs = np.stack([E11(x, ctx) for x in Wb] + [E11(x, ctx) for x in W_test])
            pf, _ = bb.forward(imgs)
            fb, ft = pf[:len(bi)], pf[len(bi):]
            res = {"E11_patchcore": mean_cat(PatchCore(0.01).fit(fb).score(ft)[0]),
                   "E11_padim": mean_cat(PaDiM().fit(fb).score(ft)[0])}
            Zb = ((Wb - mu[None, :, None]) / sd[None, :, None])[:, CORE]
            Zt = ((W_test - mu[None, :, None]) / sd[None, :, None])[:, CORE]
            res["signal_knn"] = mean_cat(knn(Zb, Zt))
            for m, a in res.items():
                rows.append({"stats": stats_mode, "method": m, "k": k, "seed": seed, "auroc": a})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[{stats_mode}] k={k} seed={seed} " + " ".join(f"{m}={a:.3f}" for m, a in res.items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()

df = pd.DataFrame(rows)
print(df.groupby(["stats", "method", "k"]).auroc.mean().unstack().reindex(columns=KS).round(3).to_string())
