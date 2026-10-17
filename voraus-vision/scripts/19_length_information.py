"""Is fixed-length resampling the dominant blind spot?

MVT-Flow reproduces at AUROC 0.922 through the official pipeline, which pads
every recording with zeros to the train maximum.  Retrained on the fixed-length
L=512 windows the encoders consume, the same model reaches only 0.651
(results/09_signal/r1_scores.parquet).  Padding preserves how long a recording
was; resampling destroys it -- which is exactly P4.  If that single difference
accounts for the gap, then length normalisation, not the more elaborate
invariances, is the costliest thing the imaging recipe does.

Three arms, identical model, identical 758-recording training set:

  pad_1164    native timing, zero-padded to the train maximum   (official)
  resamp_1164 time-warped to 1164 -- same tensor size as pad_1164, so temporal
              resolution is held constant and only duration is destroyed
  resamp_512  the encoders' setting

Comparing pad_1164 with resamp_1164 isolates duration information from temporal
resolution; comparing resamp_1164 with resamp_512 isolates resolution.

Writes results/19_length/length_arms.csv
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref" / "voraus-ad-dataset"))
PRE = ROOT / "results" / "01_preprocess"
OUT = ROOT / "results" / "19_length"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE ON TENSOR LAYOUT
# NormalizingFlow is built as InputNode(n_signals, n_times) and must be fed
# (batch, n_signals, n_times).  The official train.py writes
# `model.forward(tensors.transpose(2, 1))` because its dataset yields (T, C)
# tensors; the arrays here are already (C, T), so the transpose must NOT be
# repeated.  FrEIA does not validate the layout -- feeding (B, T, C) runs
# silently and returns garbage (verified: it produced AUROC 0.66 where the
# correct layout gives 0.92), so this is checked by an assert below.
from configuration import Configuration                                    # noqa: E402
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample  # noqa: E402
from voraus_ad import Signals                                             # noqa: E402

meta = pd.read_csv(PRE / "meta.csv")
kept = json.load(open(PRE / "kept_channels.json"))["kept_channel_names"]
hold = json.load(open(PRE / "holdout_normal.json"))
bank_src = np.array(hold["memory_bank_pool_indices"])

CACHE = OUT / "native.npz"
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    native = {int(k): z[k] for k in z.files}
else:
    print("[load] reading the parquet at native length ...")
    df = pd.read_parquet(ROOT / "data" / "voraus-ad-dataset-100hz.parquet",
                         columns=kept + ["sample"])
    native = {}
    for s, g in df.groupby("sample", sort=True):
        native[int(s)] = g[kept].to_numpy(np.float32).T
    np.savez_compressed(CACHE, **{str(k): v for k, v in native.items()})
    print(f"[load] cached {len(native)} recordings")

sample_of = dict(zip(meta["idx"], meta["sample"]))
train_max = max(native[int(sample_of[i])].shape[1] for i in bank_src)
print(f"[len] train max length = {train_max}")

# Train statistics from the bank pool only, matching the rest of the study.
cat = np.concatenate([native[int(sample_of[i])] for i in bank_src], axis=1)
mu = cat.mean(axis=1, keepdims=True)
sd = np.maximum(cat.std(axis=1, keepdims=True), 1e-8)
del cat


def resample(A, n):
    src = np.linspace(0.0, A.shape[1] - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, A.shape[1] - 1)
    w = src - i0
    return A[:, i0] * (1 - w) + A[:, i1] * w


def build(arm, idx):
    out = []
    for i in idx:
        A = (native[int(sample_of[i])] - mu) / sd
        if arm == "pad_1164":
            B = np.zeros((A.shape[0], train_max), np.float32)
            n = min(A.shape[1], train_max)
            B[:, :n] = A[:, :n]
        elif arm == "resamp_1164":
            B = resample(A, train_max).astype(np.float32)
        else:
            B = resample(A, 512).astype(np.float32)
        out.append(B)
    X = np.stack(out)
    if X.shape[1] % 2 == 1:                     # coupling blocks split in half
        X = np.concatenate([X, np.zeros((len(X), 1, X.shape[2]), np.float32)], axis=1)
    return X


CFG = Configuration(
    columns="machine", epochs=70, frequencyDivider=1, trainGain=1.0, seed=177,
    batchsize=32, nCouplingBlocks=4, clamp=1.2, learningRate=8e-4, normalize=True,
    pad=True, nHiddenLayers=0, scale=2, kernelSize1=13, dilation1=2, kernelSize2=1,
    dilation2=1, kernelSize3=1, dilation3=1, milestones=[11, 61], gamma=0.1,
)

test_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
test_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
test_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
CATS = sorted(set(test_cat[test_anom]))


def auroc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n = int(y.sum())
    return float((r[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def mean_cat(scores):
    sn = scores[~test_anom]
    return float(np.mean([auroc(np.r_[np.ones((test_anom & (test_cat == c)).sum(), bool),
                                      np.zeros(len(sn), bool)],
                                np.r_[scores[test_anom & (test_cat == c)], sn])
                          for c in CATS]))


rows = []
for arm in ["pad_1164", "resamp_1164", "resamp_512"]:
    Xtr = torch.from_numpy(build(arm, bank_src))
    Xte = torch.from_numpy(build(arm, test_idx))
    for seed in (0, 1, 2):
        torch.manual_seed(CFG.seed + seed)
        n_signals, n_times = Xtr.shape[1], Xtr.shape[2]
        model = NormalizingFlow((n_signals, n_times), CFG).float().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, CFG.milestones, CFG.gamma)
        t0 = time.time()
        model.train()
        for _ in range(CFG.epochs):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), CFG.batchsize):
                xb = Xtr[perm[i:i + CFG.batchsize]].to(DEV)
                assert xb.shape[1] == n_signals, "wrong tensor layout"
                opt.zero_grad()
                zz, jac = model.forward(xb)
                get_loss(zz, torch.sum(jac, dim=tuple(range(1, jac.dim())))).backward()
                opt.step()
            sch.step()
        model.eval()
        sc = np.empty(len(Xte), np.float32)
        with torch.no_grad():
            for i in range(0, len(Xte), 32):
                xb = Xte[i:i + 32].to(DEV)
                zz, jac = model.forward(xb)
                j = torch.sum(jac, dim=tuple(range(1, jac.dim())))
                sc[i:i + xb.shape[0]] = get_loss_per_sample(zz, j).cpu().numpy()
        a = mean_cat(sc)
        rows.append({"arm": arm, "seed": seed, "n_train": len(bank_src),
                     "length": int(Xtr.shape[2]), "auroc_mean_of_categories": a,
                     "train_seconds": round(time.time() - t0, 1)})
        print(f"  {arm} seed={seed} AUROC={a:.3f} ({time.time()-t0:.0f}s)", flush=True)
    del Xtr, Xte
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(OUT / "length_arms.csv", index=False)
print()
print(df.groupby("arm")["auroc_mean_of_categories"].agg(["mean", "std"]).round(4).to_string())
