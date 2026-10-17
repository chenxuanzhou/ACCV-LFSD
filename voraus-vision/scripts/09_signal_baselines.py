"""R1/R2: signal-domain reference points.

The paper's question is what the imaging detour costs, so it needs a detector
that never leaves the signal domain.  Without one, a reviewer cannot tell a
statement about encodings from a statement about this dataset being hard.

R1  MVT-Flow, the dataset's official SOTA, retrained here on the *same* 758
    memory-bank-pool windows at the *same* L=512 the encoders consume.  The
    official reproduction (results/04_mvtflow_repro, AUROC 0.922) uses the
    official pipeline instead -- native length, zero padding, all 948 train
    recordings -- and is reported separately as the sanity anchor.  Retraining
    on the bank pool matters because the holdout windows that receive synthetic
    faults must never have been seen during training.

R2  PCA reconstruction error: fit PCA on the timestep distribution of the
    few-shot bank, score = mean squared residual of the window.  Deliberately
    naive.  It uses the same *kind* of projection the image encodings rely on
    (a 3-component PCA over channels) but refit per bank, which is the honest
    few-shot setting; n_comp=20 is reported alongside so the comparison is not
    hostage to the rank-3 choice.

Writes results/09_signal/{r1,r2}_scores.parquet
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
sys.path.insert(0, str(ROOT / "voraus-vision"))

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
CTX = ROOT / "results" / "02_context"
OUT = ROOT / "results" / "09_signal"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

man = pd.read_csv(REN / "manifest.csv")
meta = pd.read_csv(PRE / "meta.csv")
stats = json.load(open(PRE / "norm_stats.json"))
fewshot = json.load(open(PRE / "fewshot_indices.json"))
W = np.load(PRE / "windows.npy", mmap_mode="r")
raw = np.load(PRE / "holdout_raw.npz")

sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.injection import inject, inject_duration, injection_seed  # noqa: E402

ch_mean = np.array(stats["mean"])
ch_std = np.array(stats["std"])
L = W.shape[2]
SEEDS = [0, 1, 2, 3, 4]
K_ALL = ["1", "2", "4", "8", "16", "all"]
K_SYNTH = "4"


def resample_T(arr, n):
    T = arr.shape[1]
    src = np.linspace(0.0, T - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = src - i0
    return arr[:, i0] * (1 - w) + arr[:, i1] * w


def build(r):
    """Rebuild the exact (C, L) signal behind manifest row ``r``.

    Uses the same seed recipe as 05_render.py so that R1/R2 see byte-identical
    inputs to the ones the encoders rendered.
    """
    i = int(r.src_idx)
    if r.kind != "injected":
        return np.asarray(W[i], dtype=np.float64)
    fault, param = r.fault, float(r.param)
    rng = np.random.default_rng(injection_seed(fault, param, i))
    if fault == "S5":
        s = int(meta.iloc[i]["sample"])
        return resample_T(inject_duration(np.asarray(raw[str(s)], np.float64), param), L)
    return inject(np.asarray(W[i], np.float64), fault, param, ch_std, ch_mean, rng)[0]


print("[build] reconstructing all 8962 signal windows ...")
t0 = time.time()
X = np.stack([build(r) for r in man.itertuples()]).astype(np.float32)
X = (X - ch_mean[None, :, None]) / ch_std[None, :, None]        # train z-score
print(f"[build] {X.shape} in {time.time()-t0:.0f}s")

bank_rows = man[man["kind"] == "bank"]
src_to_row = dict(zip(bank_rows["src_idx"], bank_rows["row"]))
test_rows = man.index[man["kind"] == "test"].to_numpy()
synth_rows = man.index[man["kind"].isin(["holdout", "injected"])].to_numpy()


def bank_idx(k, seed):
    key = f"k{k}_seed{seed}" if k != "all" else f"kall_seed{seed}"
    return np.array([src_to_row[s] for s in fewshot[key]])


# ---------------------------------------------------------------------- R2
def r2_scores(bank, query, n_comp):
    """Mean squared reconstruction residual under a PCA fitted on the bank."""
    F = bank.transpose(0, 2, 1).reshape(-1, bank.shape[1])       # (n*L, C)
    mu = F.mean(0)
    Fc = F - mu
    n_comp = min(n_comp, Fc.shape[0], Fc.shape[1])
    _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
    V = Vt[:n_comp]                                              # (k, C)
    out = np.empty(len(query), np.float64)
    for i in range(0, len(query), 512):                           # chunk: the
        Q = query[i:i + 512].transpose(0, 2, 1) - mu              # full synth set
        rec = (Q @ V.T) @ V                                       # would be ~4 GB
        out[i:i + len(Q)] = ((Q - rec) ** 2).mean(axis=(1, 2))
    return out


rows = []
if (OUT / 'r2_scores.parquet').exists():
    print('[R2] already done, skipping')
    rows = None
for n_comp in ((3, 20) if rows is not None else ()):
    for k in K_ALL:
        for seed in SEEDS:
            b = X[bank_idx(k, seed)]
            s = r2_scores(b, X[test_rows], n_comp)
            rows.append(pd.DataFrame({"row": test_rows, "k": k, "seed": seed, "score": s,
                                      "protocol": "P-A", "detector": f"R2_pca{n_comp}"}))
            if k == K_SYNTH:
                s2 = r2_scores(b, X[synth_rows], n_comp)
                rows.append(pd.DataFrame({"row": synth_rows, "k": k, "seed": seed,
                                          "score": s2, "protocol": "P-B",
                                          "detector": f"R2_pca{n_comp}"}))
        print(f"  [R2 pca{n_comp}] k={k} done", flush=True)
if rows:
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "r2_scores.parquet", index=False)
    print(f"[R2] -> {OUT/'r2_scores.parquet'}")

# ---------------------------------------------------------------------- R1
# NOTE ON TENSOR LAYOUT
# NormalizingFlow is built as InputNode(n_signals, n_times) and must be fed
# (batch, n_signals, n_times).  The official train.py writes
# `model.forward(tensors.transpose(2, 1))` because its dataset yields (T, C)
# tensors; the arrays here are already (C, T), so the transpose must NOT be
# repeated.  FrEIA does not validate the layout -- feeding (B, T, C) runs
# silently and returns garbage (verified: it produced AUROC 0.66 where the
# correct layout gives 0.92), so this is checked by an assert below.
from configuration import Configuration              # noqa: E402
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample  # noqa: E402

CFG = Configuration(
    columns="machine", epochs=70, frequencyDivider=1, trainGain=1.0, seed=177,
    batchsize=32, nCouplingBlocks=4, clamp=1.2, learningRate=8e-4, normalize=True,
    pad=True, nHiddenLayers=0, scale=2, kernelSize1=13, dilation1=2, kernelSize2=1,
    dilation2=1, kernelSize3=1, dilation3=1, milestones=[11, 61], gamma=0.1,
)

if (OUT / 'r1_scores.parquet').exists():
    print('[R1] already done, skipping')
    sys.exit(0)

# MVT-Flow's coupling blocks split the channel axis in half, so the count must be
# even.  D1 dropped one constant channel (computed_inertia_6), leaving 129.  Pad
# with a zero channel, which is exactly what that channel becomes under the
# official pipeline's StandardScaler, so the architecture matches the published
# one while the informative channels stay identical to the image pipeline's.
Xr1 = X
if Xr1.shape[1] % 2 == 1:
    Xr1 = np.concatenate([Xr1, np.zeros((len(Xr1), 1, Xr1.shape[2]), Xr1.dtype)], axis=1)
    print(f"[R1] padded {X.shape[1]} -> {Xr1.shape[1]} channels for the coupling split")

Xt = torch.from_numpy(Xr1).float()
rows = []
for k in K_ALL:
    for seed in SEEDS:
        bi = bank_idx(k, seed)
        if len(bi) < 2:                      # a flow cannot be fitted on 1 window
            continue
        torch.manual_seed(CFG.seed + seed)
        model = NormalizingFlow((Xr1.shape[1], Xr1.shape[2]), CFG).float().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, CFG.milestones, CFG.gamma)
        tr = Xt[bi].to(DEV)
        bs = min(CFG.batchsize, len(bi))
        model.train()
        t1 = time.time()
        for ep in range(CFG.epochs):
            perm = torch.randperm(len(tr), device=DEV)
            for i in range(0, len(tr), bs):
                xb = tr[perm[i:i + bs]]
                assert xb.shape[1] == Xr1.shape[1], "wrong tensor layout"
                opt.zero_grad()
                z, jac = model.forward(xb)
                loss = get_loss(z, torch.sum(jac, dim=tuple(range(1, jac.dim()))))
                loss.backward()
                opt.step()
            sch.step()
        model.eval()

        def score(idx):
            out = np.empty(len(idx), np.float32)
            with torch.no_grad():
                for i in range(0, len(idx), 64):
                    xb = Xt[idx[i:i + 64]].to(DEV)
                    z, jac = model.forward(xb)
                    j = torch.sum(jac, dim=tuple(range(1, jac.dim())))
                    out[i:i + xb.shape[0]] = get_loss_per_sample(z, j).cpu().numpy()
            return out

        rows.append(pd.DataFrame({"row": test_rows, "k": k, "seed": seed,
                                  "score": score(test_rows), "protocol": "P-A",
                                  "detector": "R1_mvtflow"}))
        if k == K_SYNTH:
            rows.append(pd.DataFrame({"row": synth_rows, "k": k, "seed": seed,
                                      "score": score(synth_rows), "protocol": "P-B",
                                      "detector": "R1_mvtflow"}))
        print(f"  [R1] k={k} seed={seed} trained in {time.time()-t1:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
pd.concat(rows, ignore_index=True).to_parquet(OUT / "r1_scores.parquet", index=False)
print(f"[R1] -> {OUT/'r1_scores.parquet'}")
