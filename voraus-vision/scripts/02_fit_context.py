"""Fit and freeze every train-derived statistic the encoders need.

Fitted on the memory-bank pool only (train normal minus the 20% injection
holdout), so nothing an encoder knows can leak from the windows that later
receive synthetic faults.

Writes results/02_context/context.npz and context_summary.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.encodings import _decimate, _stft_logmag  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "voraus-vision" / "configs" / "base.yaml"))
PRE = ROOT / "results" / "01_preprocess"
OUT = ROOT / "results" / "02_context"
OUT.mkdir(parents=True, exist_ok=True)

L_PAIR = CFG["data"]["L_pair"]
NPERSEG = CFG["encodings"]["stft"]["nperseg"]
NOVERLAP = CFG["encodings"]["stft"]["noverlap"]
TOPK = CFG["encodings"]["gaf_tile"]["top_k_channels"]

W = np.load(PRE / "windows.npy", mmap_mode="r")
stats = json.load(open(PRE / "norm_stats.json"))
hold = json.load(open(PRE / "holdout_normal.json"))
pool = np.array(hold["memory_bank_pool_indices"])
print(f"[ctx] fitting on {len(pool)} memory-bank-pool windows, shape {W.shape}")

ch_mean = np.array(stats["mean"])
ch_std = np.array(stats["std"])

# ------------------------------------------------------------------ PCA
# Standardise with train stats first, then PCA. Both are affine, which is what
# keeps P1/P2 exact under the projection (see EncoderContext.project).
sub = np.asarray(W[pool], dtype=np.float64)                # (n, C, L)
n, C, L = sub.shape
U = (sub - ch_mean[None, :, None]) / ch_std[None, :, None]
feats = U.transpose(0, 2, 1).reshape(-1, C)                # (n*L, C)
pca_mean = feats.mean(axis=0)
Xc = feats - pca_mean
print(f"[pca] SVD on {Xc.shape} ...")
# randomised range finder keeps this to seconds at (388k, 129)
rng = np.random.default_rng(0)
Om = rng.standard_normal((C, 16))
Y = Xc @ Om
Q, _ = np.linalg.qr(Y)
B = Q.T @ Xc
_, S, Vt = np.linalg.svd(B, full_matrices=False)
comps = Vt[:3]                                             # (3, C)
evr = (S[:3] ** 2) / (Xc ** 2).sum()
print(f"[pca] explained variance ratio (3 comps): {evr.round(4).tolist()}  sum={evr.sum():.4f}")

Z = np.einsum("kc,ntc->nkt", comps, (U.transpose(0, 2, 1) - pca_mean))   # (n, 3, L)
pca_min = Z.min(axis=(0, 2))
pca_max = Z.max(axis=(0, 2))

# ------------------------------------------------- RP distance scale (E6)
idx = rng.choice(n, size=min(64, n), replace=False)
dists = []
for i in idx:
    Zi = _decimate(Z[i], L_PAIR).T                         # (L_pair, 3)
    d = Zi[:, None, :] - Zi[None, :, :]
    dists.append(np.sqrt((d ** 2).sum(-1)))
rp_dist_scale = float(np.median(np.stack(dists)))
print(f"[rp ] train median pairwise distance = {rp_dist_scale:.4f}")

# --------------------------------------------------- STFT normalisation
logmags = []
for i in idx:
    for c in range(3):
        logmags.append(_stft_logmag(Z[i, c], NPERSEG, NOVERLAP))
logmags = np.concatenate([m.ravel() for m in logmags])
stft_mean = float(logmags.mean())
stft_std = float(max(logmags.std(), 1e-8))
print(f"[stft] log-magnitude mean={stft_mean:.4f} std={stft_std:.4f}")

# ------------------------------------------------ top-variance channels (E4)
ch_var = sub.var(axis=(0, 2))
top = np.argsort(ch_var)[::-1][:TOPK]
names = stats["channel_names"]
print(f"[e4 ] rendered channels: {[names[i] for i in top]}")

np.savez(
    OUT / "context.npz",
    ch_mean=ch_mean, ch_std=ch_std,
    pca_mean=pca_mean, pca_components=comps,
    pca_min=pca_min, pca_max=pca_max,
    rp_dist_scale=rp_dist_scale,
    stft_mean=stft_mean, stft_std=stft_std,
    top_var_channels=top,
    pca_explained_variance_ratio=evr,
    channel_variance=ch_var,
)
json.dump(
    {
        "n_fit_windows": int(n),
        "n_channels": int(C),
        "pca_explained_variance_ratio": evr.tolist(),
        "pca_evr_sum": float(evr.sum()),
        "rp_dist_scale": rp_dist_scale,
        "stft_mean": stft_mean,
        "stft_std": stft_std,
        "e4_rendered_channels": [names[i] for i in top],
        "e4_rendered_channel_indices": top.tolist(),
        "L_pair": L_PAIR,
    },
    open(OUT / "context_summary.json", "w"),
    indent=2,
)
print(f"[done] wrote {OUT/'context.npz'}")
