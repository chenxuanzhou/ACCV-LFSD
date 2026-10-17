"""Intermediate sample budgets k in {32, 64, 128, 256}  (added in revision).

The few-shot grid jumps from k=16 to k=all (758), so the
claim that the trained reference needs "hundreds" of recordings rested on an
untested interval.  This script fills it in for the three method families of
the few-shot figure, on new frozen index sets drawn from the same bank pool:

  indices   results/29_intermediate_k/fewshot_indices_ext.json  (5 seeds x 4 k)
  signal    28-channel z-scored kNN / PCA-20 baselines           (as script 24)
  image     E11, E10, E2, E1 x detectors M1-M5                   (as script 06/07)
  mvtflow   MVT-Flow, official pipeline and hyper-parameters     (as script 22)

k=16 and k=all are re-run from the ORIGINAL index file as a regression check:
they must reproduce the numbers already in results/07_eval and results/24.

Usage:  python 29_intermediate_k.py indices|signal|image|mvtflow [E11 ...]
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
OUT = ROOT / "results" / "29_intermediate_k"
OUT.mkdir(parents=True, exist_ok=True)

K_NEW = [32, 64, 128, 256]
K_CHECK = ["16", "all"]
SEEDS = [0, 1, 2, 3, 4]
stage = sys.argv[1] if len(sys.argv) > 1 else "indices"


def load_indices():
    ext = json.load(open(OUT / "fewshot_indices_ext.json"))
    orig = json.load(open(PRE / "fewshot_indices.json"))
    return {**orig, **ext}


def auroc(y, s):
    r = rankdata(s)
    n = int(y.sum())
    return float((r[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


# ---------------------------------------------------------------------- indices
if stage == "indices":
    hold = json.load(open(PRE / "holdout_normal.json"))
    bank_pool = np.array(hold["memory_bank_pool_indices"])
    ext = {}
    for seed in SEEDS:
        rng = np.random.default_rng(29_000 + seed)     # independent of the original draws
        for k in K_NEW:
            ext[f"k{k}_seed{seed}"] = sorted(int(v) for v in rng.choice(bank_pool, size=k, replace=False))
    json.dump(ext, open(OUT / "fewshot_indices_ext.json", "w"), indent=1)
    print(f"[indices] {len(ext)} sets from a bank pool of {len(bank_pool)}")
    sys.exit()

fewshot = load_indices()
KS = [str(k) for k in K_NEW] + K_CHECK


def key(k, seed):
    return f"k{k}_seed{seed}"


# ----------------------------------------------------------------------- signal
if stage == "signal":
    from voraus_vision.encodings import CORE_CHANNEL_NAMES
    W = np.load(PRE / "windows.npy", mmap_mode="r")
    meta = pd.read_csv(PRE / "meta.csv")
    stats = json.load(open(PRE / "norm_stats.json"))
    kept = json.load(open(PRE / "kept_channels.json"))["kept_channel_names"]
    CORE = np.array([kept.index(n) for n in CORE_CHANNEL_NAMES if n in kept])
    Z = ((np.asarray(W, np.float64) - np.array(stats["mean"])[None, :, None])
         / np.array(stats["std"])[None, :, None])[:, CORE]
    t_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
    t_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
    t_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
    CATS = sorted(set(t_cat[t_anom]))

    def mean_cat(s):
        sn = s[~t_anom]
        return float(np.mean([auroc(np.r_[np.ones((t_anom & (t_cat == c)).sum(), bool), np.zeros(len(sn), bool)],
                                    np.r_[s[t_anom & (t_cat == c)], sn]) for c in CATS]))

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
        _, _, Vt = np.linalg.svd(F - mu, full_matrices=False)
        V = Vt[:min(n_comp, Vt.shape[0])]
        out = np.empty(len(query), np.float64)
        for i in range(0, len(query), 512):
            Qc = query[i:i + 512].transpose(0, 2, 1) - mu
            out[i:i + len(Qc)] = ((Qc - (Qc @ V.T) @ V) ** 2).mean(axis=(1, 2))
        return out

    Q = Z[t_idx]
    rows = []
    for k in KS:
        for seed in SEEDS:
            bi = np.array(fewshot[key(k, seed)])
            rows.append({"detector": "knn", "k": k, "seed": seed, "auroc": mean_cat(knn_scores(Z[bi], Q))})
            rows.append({"detector": "pca20", "k": k, "seed": seed, "auroc": mean_cat(pca_scores(Z[bi], Q))})
        print(f"[signal] k={k} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_28ch_zscore.csv", index=False)
    print(df.groupby(["k", "detector"]).auroc.agg(["mean", "std"]).round(4).to_string())
    sys.exit()

# ------------------------------------------------------------------------ image
if stage == "image":
    import torch
    from voraus_vision.backbones import Backbone
    from voraus_vision.detectors import KNNGlobal, MahalanobisGlobal, PaDiM, PatchCore

    man = pd.read_csv(REN / "manifest.csv")
    bank = man[man["kind"] == "bank"]
    test = man[man["kind"] == "test"]
    rows_used = np.r_[bank["row"].to_numpy(), test["row"].to_numpy()]
    pos_of_row = {int(r): i for i, r in enumerate(rows_used)}
    src_to_pos = {int(s): pos_of_row[int(r)] for s, r in zip(bank["src_idx"], bank["row"])}
    test_pos = np.array([pos_of_row[int(r)] for r in test["row"]])
    t_anom = test["anomaly"].to_numpy().astype(bool)
    t_cat = test["category"].to_numpy()
    CATS = sorted(set(t_cat[t_anom]))

    def mean_cat(s):
        sn = s[~t_anom]
        return float(np.mean([auroc(np.r_[np.ones((t_anom & (t_cat == c)).sum(), bool), np.zeros(len(sn), bool)],
                                    np.r_[s[t_anom & (t_cat == c)], sn]) for c in CATS]))

    DETECTORS = {
        "M1_patchcore_wrn50": ("wrn50", lambda: PatchCore(0.01), True),
        "M3_padim_wrn50": ("wrn50", lambda: PaDiM(), True),
        "M2_patchcore_dinov2": ("dinov2", lambda: PatchCore(0.01), True),
        "M4_knn_cls_dinov2": ("dinov2", lambda: KNNGlobal(1), False),
        "M5_mahal_cls_dinov2": ("dinov2", lambda: MahalanobisGlobal(), False),
    }
    out_csv = OUT / "image_intermediate_k.csv"
    done = pd.read_csv(out_csv, dtype={"k": str}) if out_csv.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for enc in (sys.argv[2:] or ["E11", "E10", "E2", "E1"]):
        imgs = np.asarray(np.load(REN / enc / "images.npy", mmap_mode="r")[np.sort(rows_used)])
        order = {int(r): i for i, r in enumerate(np.sort(rows_used))}
        imgs = imgs[[order[int(r)] for r in rows_used]]
        for bb_name in ["wrn50", "dinov2"]:
            dets = {d: v for d, v in DETECTORS.items() if v[0] == bb_name}
            if len(done) and all(((done.encoding == enc) & (done.detector == d)).sum() == len(KS) * len(SEEDS)
                                 for d in dets):
                print(f"[{enc}/{bb_name}] done already")
                continue
            t0 = time.time()
            bb = Backbone(bb_name)
            pf, gf = bb.forward(imgs)
            del bb
            torch.cuda.empty_cache()
            print(f"[{enc}/{bb_name}] features {pf.shape} in {time.time() - t0:.0f}s", flush=True)
            for det_id, (_, factory, use_patch) in dets.items():
                feats = pf if use_patch else gf
                t1 = time.time()
                for k in KS:
                    for seed in SEEDS:
                        bi = np.array([src_to_pos[int(s)] for s in fewshot[key(k, seed)]])
                        det = factory().fit(feats[bi])
                        s, _ = det.score(feats[test_pos])
                        rows.append({"encoding": enc, "detector": det_id, "k": k, "seed": seed,
                                     "auroc": mean_cat(np.asarray(s, np.float64))})
                        del det
                        torch.cuda.empty_cache()
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"  [{enc}/{det_id}] {time.time() - t1:.0f}s", flush=True)
            del pf, gf
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.groupby(["encoding", "k", "detector"]).auroc.mean().unstack().round(4).to_string())
    sys.exit()

# ---------------------------------------------------------------------- mvtflow
if stage in ("mvtflow", "mvtflow_iters"):
    import torch
    sys.path.insert(0, str(ROOT / "ref" / "voraus-ad-dataset"))
    from configuration import Configuration
    from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample
    from sklearn import metrics
    from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    CFG = Configuration(
        columns="machine", epochs=70, frequencyDivider=1, trainGain=1.0, seed=177,
        batchsize=32, nCouplingBlocks=4, clamp=1.2, learningRate=8e-4, normalize=True,
        pad=True, nHiddenLayers=0, scale=2, kernelSize1=13, dilation1=2, kernelSize2=1,
        dilation2=1, kernelSize3=1, dilation3=1, milestones=[11, 61], gamma=0.1,
    )
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)
    train_ds, _, _, test_dl = load_torch_dataloaders(
        dataset=ROOT / "data" / "voraus-ad-dataset-100hz.parquet",
        batch_size=CFG.batchsize, seed=CFG.seed, columns=Signals.groups()[CFG.columns],
        normalize=CFG.normalize, frequency_divider=CFG.frequency_divider,
        train_gain=CFG.train_gain, pad=CFG.pad)
    meta = pd.read_csv(PRE / "meta.csv")
    idx_to_sample = dict(zip(meta["idx"], meta["sample"]))
    sample_to_pos = {int(lab["sample"]): i for i, lab in enumerate(train_ds.labels)}
    n_signals, n_times = train_ds.tensors[0].shape[1], train_ds.tensors[0].shape[0]
    # "mvtflow_iters": hold the number of optimisation steps at the full-data count (70 epochs x
    # ceil(758/32) = 1680 steps), as the training-set-size ablation of the voraus-AD paper does,
    # instead of holding epochs constant.  Milestones keep their relative position (11/70, 61/70).
    ITERS = stage == "mvtflow_iters"
    TOTAL_STEPS = CFG.epochs * int(np.ceil(758 / CFG.batchsize))
    out_csv = OUT / ("mvtflow_const_iters.csv" if ITERS else "mvtflow_intermediate_k.csv")
    rows = pd.read_csv(out_csv, dtype={"k": str}).to_dict("records") if out_csv.exists() else []
    have = {(r["k"], r["seed"]) for r in rows}
    for k in ([str(v) for v in (2, 4, 8, 16)] if ITERS else []) + [str(v) for v in K_NEW]:
        for seed in SEEDS:
            if (k, seed) in have:
                continue
            pos = [sample_to_pos[int(idx_to_sample[i])] for i in fewshot[key(k, seed)]
                   if int(idx_to_sample[i]) in sample_to_pos]
            torch.manual_seed(CFG.seed + seed)
            model = NormalizingFlow((n_signals, n_times), CFG).float().to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
            X = torch.stack([train_ds.tensors[i] for i in pos])
            bs = min(CFG.batchsize, len(X))
            spe = int(np.ceil(len(X) / bs))
            n_epochs = int(np.ceil(TOTAL_STEPS / spe)) if ITERS else CFG.epochs
            ms = [int(round(m * n_epochs / CFG.epochs)) for m in CFG.milestones]
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, ms, CFG.gamma)
            t0 = time.time()
            model.train()
            for _ in range(n_epochs):
                perm = torch.randperm(len(X))
                for i in range(0, len(X), bs):
                    xb = X[perm[i:i + bs]].float().to(DEV)
                    opt.zero_grad()
                    z, jac = model.forward(xb.transpose(2, 1))
                    get_loss(z, torch.sum(jac, dim=tuple(range(1, jac.dim())))).backward()
                    opt.step()
                sch.step()
            model.eval()
            res = []
            with torch.no_grad():
                for tensors, labels in test_dl:
                    tensors = tensors.float().to(DEV)
                    z, jac = model.forward(tensors.transpose(2, 1))
                    lps = get_loss_per_sample(z, torch.sum(jac, dim=tuple(range(1, jac.dim()))))
                    for m in range(lps.shape[0]):
                        r = {kk: (v[m].item() if isinstance(v, torch.Tensor) else v[m]) for kk, v in labels.items()}
                        r["score"] = lps[m].item()
                        res.append(r)
            df = pd.DataFrame(res)
            aur = []
            for c in ANOMALY_CATEGORIES:
                d = df[(df["category"] == c.name) | (~df["anomaly"])]
                fpr, tpr, _ = metrics.roc_curve(d["anomaly"], d["score"].values, pos_label=True)
                aur.append(metrics.auc(fpr, tpr))
            rows.append({"k": k, "seed": seed, "n_train": len(pos), "epochs": n_epochs, "steps": n_epochs * spe,
                         "auroc": float(np.mean(aur))})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  k={k} seed={seed} n={len(pos)} AUROC={rows[-1]['auroc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            del model, X
            torch.cuda.empty_cache()
    print(pd.DataFrame(rows).groupby("k").auroc.agg(["mean", "std"]).round(4).to_string())
