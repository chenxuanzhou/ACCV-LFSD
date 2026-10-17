"""Is the single-channel insensitivity a property of the encoding, or an
artefact of pooling patch features to a 14x14 grid?

PatchCore's standard configuration on a 224 input is a 28x28 grid; 14x14 was
chosen here to fit the memory budget.  If the finer grid recovers
single-channel sensitivity, the finding in results/08_diagnose is about this
implementation rather than about the encoding, and the grid must be changed
before anything is claimed.

Writes results/08_diagnose/grid_ablation.csv
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.context import load_context
from voraus_vision.detectors import PatchCore
from voraus_vision.encodings import ENCODERS

PRE = ROOT / "results" / "01_preprocess"
OUT = ROOT / "results" / "08_diagnose"
ctx = load_context(ROOT)
W = np.load(PRE / "windows.npy", mmap_mode="r")
hold = json.load(open(PRE / "holdout_normal.json"))["holdout_normal_indices"]
bank_idx = json.load(open(PRE / "fewshot_indices.json"))["k4_seed0"]
L = W.shape[2]
DEV = "cuda"

import timm
model = timm.create_model("wide_resnet50_2", pretrained=True,
                          features_only=True, out_indices=(1, 2)).eval().to(DEV)
for p in model.parameters():
    p.requires_grad_(False)
MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEV).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=DEV).view(1, 3, 1, 1)
PROJ = {}


def proj_for(d):
    if d not in PROJ:
        g = torch.Generator().manual_seed(0)
        PROJ[d] = (torch.randn(d, 256, generator=g) / 16.0).to(DEV)
    return PROJ[d]


@torch.no_grad()
def feats(imgs, grid, batch=16):
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + batch])).to(DEV)
        x = (x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD
        f2, f3 = model(x)
        f2 = F.adaptive_avg_pool2d(F.avg_pool2d(f2, 3, 1, 1), grid)
        f3 = F.adaptive_avg_pool2d(F.avg_pool2d(f3, 3, 1, 1), grid)
        cat = torch.cat([f2, f3], 1).flatten(2).transpose(1, 2)
        p = cat @ proj_for(cat.shape[-1])
        out.append(p.half().cpu().numpy())
    return np.concatenate(out)


def auroc(y, s):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum()); return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def spike(X, amp, w, nch, rng):
    Y = X.copy()
    ch = rng.choice(X.shape[0], size=nch, replace=False)
    t0 = int(rng.integers(0, L - w))
    for c in ch:
        Y[c, t0:t0 + w] += amp * ctx.ch_std[c]
    return Y


rows = []
for enc in ["E1", "E10", "E3"]:
    fn = ENCODERS[enc]
    clean = np.stack([fn(np.asarray(W[i], np.float64), ctx) for i in hold])
    bank = np.stack([fn(np.asarray(W[i], np.float64), ctx) for i in bank_idx])
    inj = {}
    for amp in (3, 30):
        for nch in (1, 8):
            rng = np.random.default_rng(7)
            inj[(amp, nch)] = np.stack([fn(spike(np.asarray(W[i], np.float64), amp, 25, nch, rng), ctx)
                                        for i in hold])
    for grid in (14, 28, 56):
        det = PatchCore(0.01).fit(feats(bank, grid))
        sc, _ = det.score(feats(clean, grid))
        for (amp, nch), im in inj.items():
            si, _ = det.score(feats(im, grid))
            y = np.r_[np.ones(len(si), bool), np.zeros(len(sc), bool)]
            a = auroc(y, np.r_[si, sc])
            rows.append({"encoding": enc, "grid": grid, "amp_sigma": amp,
                         "n_channels_hit": nch, "auroc": a})
            print(f"  {enc} grid={grid:2d} amp={amp:3d} nch={nch} AUROC={a:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "grid_ablation.csv", index=False)
print()
print(df.pivot_table(index=["encoding", "amp_sigma", "n_channels_hit"],
                     columns="grid", values="auroc").round(3).to_string())
