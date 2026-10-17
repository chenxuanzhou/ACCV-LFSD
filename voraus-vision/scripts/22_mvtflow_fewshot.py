"""MVT-Flow's few-shot curve, through the official data pipeline.

The workshop's question is what happens when normal data is scarce, so the
signal-domain reference has to be measured at the same budgets the image
pipeline gets, not only at k=all.  Everything comes from the official loader
(native length, zero padding, official normalisation) so the numbers are
directly comparable to the published 0.922; only the training subset changes.

The few-shot indices are the same frozen ones every other detector uses
(results/01_preprocess/fewshot_indices.json), so a given (k, seed) means the
same recordings in the signal domain and in every encoding.

Writes results/22_mvtflow_fewshot/mvtflow_fewshot.csv
"""

import json
import random
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
OUT = ROOT / "results" / "22_mvtflow_fewshot"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from configuration import Configuration                                     # noqa: E402
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample  # noqa: E402
from sklearn import metrics                                                 # noqa: E402
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders   # noqa: E402

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
fewshot = json.load(open(PRE / "fewshot_indices.json"))
idx_to_sample = dict(zip(meta["idx"], meta["sample"]))
sample_to_pos = {int(lab["sample"]): i for i, lab in enumerate(train_ds.labels)}
n_signals = train_ds.tensors[0].shape[1]
n_times = train_ds.tensors[0].shape[0]
print(f"[setup] {len(train_ds)} train, model dims ({n_signals}, {n_times})")

K_ALL = ["1", "2", "4", "8", "16", "all"]
SEEDS = [0, 1, 2, 3, 4]


def positions(k, seed):
    key = f"k{k}_seed{seed}" if k != "all" else f"kall_seed{seed}"
    return [sample_to_pos[int(idx_to_sample[i])] for i in fewshot[key]
            if int(idx_to_sample[i]) in sample_to_pos]


rows = []
for k in K_ALL:
    for seed in SEEDS:
        pos = positions(k, seed)
        if len(pos) < 2:                    # a flow cannot be fitted on one window
            rows.append({"k": k, "seed": seed, "n_train": len(pos), "auroc": np.nan,
                         "note": "not fittable"})
            continue
        torch.manual_seed(CFG.seed + seed)
        model = NormalizingFlow((n_signals, n_times), CFG).float().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, CFG.milestones, CFG.gamma)
        X = torch.stack([train_ds.tensors[i] for i in pos])
        bs = min(CFG.batchsize, len(X))
        t0 = time.time()
        model.train()
        for _ in range(CFG.epochs):
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
                j = torch.sum(jac, dim=tuple(range(1, jac.dim())))
                lps = get_loss_per_sample(z, j)
                for m in range(lps.shape[0]):
                    r = {kk: (v[m].item() if isinstance(v, torch.Tensor) else v[m])
                         for kk, v in labels.items()}
                    r["score"] = lps[m].item()
                    res.append(r)
        df = pd.DataFrame(res)
        aur = []
        for c in ANOMALY_CATEGORIES:
            d = df[(df["category"] == c.name) | (~df["anomaly"])]
            fpr, tpr, _ = metrics.roc_curve(d["anomaly"], d["score"].values, pos_label=True)
            aur.append(metrics.auc(fpr, tpr))
        a = float(np.mean(aur))
        rows.append({"k": k, "seed": seed, "n_train": len(pos), "auroc": a, "note": ""})
        print(f"  k={k} seed={seed} n={len(pos)} AUROC={a:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        del model, X
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / "mvtflow_fewshot.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT / "mvtflow_fewshot.csv", index=False)
print()
print(df.groupby("k")["auroc"].agg(["mean", "std"]).reindex(K_ALL).round(4).to_string())
