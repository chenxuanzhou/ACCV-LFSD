"""Locate the gap between the official MVT-Flow reproduction (0.922) and this
study's retrained reference (0.66).

Two arms, both using the *official* data loader so that nothing in this repo's
own preprocessing can be responsible:

  official_all    all 948 PRE_A recordings   -- must land near 0.922, otherwise
                  the evaluation code here is what differs
  official_bank   the 758-recording memory-bank pool, i.e. the same training
                  budget every other detector in the study gets

If official_bank also reaches ~0.92, then this study's native-length path has a
bug.  If it drops to ~0.66, holding out 20% of the normals for fault injection
is what costs the reference its accuracy -- which would have to be stated
whenever MVT-Flow is quoted as the ceiling.

Writes results/20_r1_gap/r1_gap.csv
"""

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
OUT = ROOT / "results" / "20_r1_gap"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from configuration import Configuration                                     # noqa: E402
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample  # noqa: E402
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders   # noqa: E402

CFG = Configuration(
    columns="machine", epochs=70, frequencyDivider=1, trainGain=1.0, seed=177,
    batchsize=32, nCouplingBlocks=4, clamp=1.2, learningRate=8e-4, normalize=True,
    pad=True, nHiddenLayers=0, scale=2, kernelSize1=13, dilation1=2, kernelSize2=1,
    dilation2=1, kernelSize3=1, dilation3=1, milestones=[11, 61], gamma=0.1,
)
DATASET = ROOT / "data" / "voraus-ad-dataset-100hz.parquet"

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)

train_ds, test_ds, _, test_dl = load_torch_dataloaders(
    dataset=DATASET, batch_size=CFG.batchsize, seed=CFG.seed,
    columns=Signals.groups()[CFG.columns], normalize=CFG.normalize,
    frequency_divider=CFG.frequency_divider, train_gain=CFG.train_gain, pad=CFG.pad)
print(f"[loader] train {len(train_ds)}  test {len(test_ds)}  "
      f"sample shape {tuple(train_ds.tensors[0].shape)}")

# map the study's bank-pool recordings onto the official loader's train order
meta = pd.read_csv(PRE / "meta.csv")
import json                                                                 # noqa: E402
bank_idx = json.load(open(PRE / "holdout_normal.json"))["memory_bank_pool_indices"]
bank_samples = set(meta.set_index("idx").loc[bank_idx, "sample"].tolist())
train_samples = [lab["sample"] for lab in train_ds.labels]
bank_pos = [i for i, s in enumerate(train_samples) if int(s) in bank_samples]
print(f"[map] {len(bank_pos)} of {len(train_samples)} official train recordings "
      f"are in the bank pool")

n_signals = train_ds.tensors[0].shape[1]
n_times = train_ds.tensors[0].shape[0]
print(f"[model] n_signals={n_signals} n_times={n_times}")


def run(name, positions, seed):
    torch.manual_seed(CFG.seed + seed)
    model = NormalizingFlow((n_signals, n_times), CFG).float().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, CFG.milestones, CFG.gamma)
    X = torch.stack([train_ds.tensors[i] for i in positions])
    t0 = time.time()
    model.train()
    for _ in range(CFG.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), CFG.batchsize):
            xb = X[perm[i:i + CFG.batchsize]].float().to(DEV)
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
                r = {k: (v[m].item() if isinstance(v, torch.Tensor) else v[m])
                     for k, v in labels.items()}
                r["score"] = lps[m].item()
                res.append(r)
    df = pd.DataFrame(res)
    from sklearn import metrics
    aur = []
    for c in ANOMALY_CATEGORIES:
        d = df[(df["category"] == c.name) | (~df["anomaly"])]
        fpr, tpr, _ = metrics.roc_curve(d["anomaly"], d["score"].values, pos_label=True)
        aur.append(metrics.auc(fpr, tpr))
    a = float(np.mean(aur))
    print(f"  {name} seed={seed} n_train={len(positions)} AUROC={a:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del model, X
    torch.cuda.empty_cache()
    return a, aur


rows = []
for name, positions in [("official_all", list(range(len(train_ds)))),
                        ("official_bank", bank_pos)]:
    for seed in (0, 1):
        a, per_cat = run(name, positions, seed)
        rows.append({"arm": name, "seed": seed, "n_train": len(positions),
                     "auroc_mean_of_categories": a,
                     **{c.name: v for c, v in zip(ANOMALY_CATEGORIES, per_cat)}})

df = pd.DataFrame(rows)
df.to_csv(OUT / "r1_gap.csv", index=False)
print()
print(df.groupby("arm")[["n_train", "auroc_mean_of_categories"]].mean().round(4).to_string())
