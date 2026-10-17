"""Does the choice of WideResNet stage change the single-channel picture?

The main runs use timm out_indices=(1,2) -- layer1 (56x56x256) and layer2
(28x28x512).  PatchCore's published configuration is layer2+layer3
(28x28x512 + 14x14x1024), i.e. deeper and coarser.  Early layers are more
local and higher-resolution, so they should if anything be *more* sensitive to
a thin single-channel stripe; the main runs therefore sit on the conservative
side of a null result.  This script checks that directly instead of asserting it.

Writes results/08_diagnose/layer_ablation.csv
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
MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEV).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=DEV).view(1, 3, 1, 1)
PROJ = {}


def proj_for(d):
    if d not in PROJ:
        g = torch.Generator().manual_seed(0)
        PROJ[d] = (torch.randn(d, 256, generator=g) / 16.0).to(DEV)
    return PROJ[d]


import timm
MODELS = {}
for name, idx in {"layer1+2": (1, 2), "layer2+3": (2, 3)}.items():
    m = timm.create_model("wide_resnet50_2", pretrained=True,
                          features_only=True, out_indices=idx).eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    MODELS[name] = m


@torch.no_grad()
def feats(model, imgs, grid=28, batch=16):
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + batch])).to(DEV)
        x = (x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD
        fa, fb = model(x)
        fa = F.adaptive_avg_pool2d(F.avg_pool2d(fa, 3, 1, 1), grid)
        fb = F.adaptive_avg_pool2d(F.avg_pool2d(fb, 3, 1, 1), grid)
        cat = torch.cat([fa, fb], 1).flatten(2).transpose(1, 2)
        out.append((cat @ proj_for(cat.shape[-1])).half().cpu().numpy())
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
        for nch in (1, 129):
            rng = np.random.default_rng(7)
            inj[(amp, nch)] = np.stack([fn(spike(np.asarray(W[i], np.float64), amp, 25, nch, rng), ctx)
                                        for i in hold])
    for lname, model in MODELS.items():
        det = PatchCore(0.01).fit(feats(model, bank))
        sc, _ = det.score(feats(model, clean))
        for (amp, nch), im in inj.items():
            si, _ = det.score(feats(model, im))
            y = np.r_[np.ones(len(si), bool), np.zeros(len(sc), bool)]
            a = auroc(y, np.r_[si, sc])
            rows.append({"encoding": enc, "layers": lname, "amp_sigma": amp,
                         "n_channels_hit": nch, "auroc": a})
            print(f"  {enc} {lname} amp={amp:3d} nch={nch:3d} AUROC={a:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "layer_ablation.csv", index=False)
print()
print(df.pivot_table(index=["encoding", "amp_sigma", "n_channels_hit"],
                     columns="layers", values="auroc").round(3).to_string())
