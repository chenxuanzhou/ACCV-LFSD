"""Diagnose the positive-control failure (plan sec. 10: stop and fix the
pipeline before writing anything).

Observation that triggered this: on E1 the global fault S4b reaches AUROC 0.967
while the single-channel positive controls S7/S8 sit at 0.548/0.508.  The
detector chain therefore works; what fails is specific to single-channel
faults.  A 129-channel window rendered at 224x224 gives each channel ~1.7 pixel
rows, so a single-channel fault is a sub-pixel-row perturbation.

The question this script answers: is the single-channel insensitivity a
*severity* effect (fixable by a stronger control) or a *structural* one (no
severity makes it detectable)?  The answer decides whether S1/S2 nulls can be
attributed to the invariance group at all, or whether they are confounded with
a resolution limit that would suppress any single-channel fault.

Sweeps spike amplitude and width, and adds all-channel variants whose spatial
footprint matches the global faults.

Writes results/08_diagnose/control_sweep.csv (this invocation only) and
updates results/08_diagnose/control_sweep_all.csv (all encodings run so far).
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.backbones import Backbone                 # noqa: E402
from voraus_vision.detectors import PatchCore                # noqa: E402
from voraus_vision.context import load_context
from voraus_vision.encodings import ENCODERS  # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
CTX = ROOT / "results" / "02_context"
OUT = ROOT / "results" / "08_diagnose"
OUT.mkdir(parents=True, exist_ok=True)

c = np.load(CTX / "context.npz")
ctx = load_context(ROOT)
W = np.load(PRE / "windows.npy", mmap_mode="r")
hold = json.load(open(PRE / "holdout_normal.json"))
holdout = np.array(hold["holdout_normal_indices"])
bankpool = np.array(hold["memory_bank_pool_indices"])
L = W.shape[2]


def spike(X, amp, width, n_ch_hit, rng):
    Y = X.copy()
    chans = rng.choice(X.shape[0], size=n_ch_hit, replace=False)
    t0 = int(rng.integers(0, L - width))
    for ch in chans:
        Y[ch, t0:t0 + width] += amp * ctx.ch_std[ch]
    return Y


def freeze(X, width, n_ch_hit, rng):
    Y = X.copy()
    chans = rng.choice(X.shape[0], size=n_ch_hit, replace=False)
    t0 = int(rng.integers(0, L - width))
    for ch in chans:
        Y[ch, t0:t0 + width] = Y[ch, t0]
    return Y


def auroc(y, s):
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


CASES = []
for amp in (3, 10, 30, 100):
    for w in (5, 25):
        for nch in (1, 8, 129):
            CASES.append(("spike", amp, w, nch))
for w in (50, 150):
    for nch in (1, 8, 129):
        CASES.append(("freeze", 0, w, nch))

ENCS = sys.argv[1:] or ["E1", "E2", "E5", "E7"]
rows = []
for enc in ENCS:
    fn = ENCODERS[enc]
    clean = np.stack([fn(np.asarray(W[i], np.float64), ctx) for i in holdout])
    bank_idx = json.load(open(PRE / "fewshot_indices.json"))["k4_seed0"]
    bank_img = np.stack([fn(np.asarray(W[i], np.float64), ctx) for i in bank_idx])
    bb = Backbone("wrn50")
    bank_f, _ = bb.forward(bank_img)
    clean_f, _ = bb.forward(clean)
    det = PatchCore(0.01).fit(bank_f)
    s_clean, _ = det.score(clean_f)

    for kind, amp, w, nch in CASES:
        rng = np.random.default_rng(7)
        imgs = []
        for i in holdout:
            X = np.asarray(W[i], np.float64)
            Y = spike(X, amp, w, nch, rng) if kind == "spike" else freeze(X, w, nch, rng)
            imgs.append(fn(Y, ctx))
        f, _ = bb.forward(np.stack(imgs))
        s_inj, _ = det.score(f)
        y = np.r_[np.ones(len(s_inj), bool), np.zeros(len(s_clean), bool)]
        a = auroc(y, np.r_[s_inj, s_clean])
        px = np.abs(np.stack(imgs).astype(np.int16) - clean.astype(np.int16))
        rows.append({"encoding": enc, "kind": kind, "amp_sigma": amp, "width": w,
                     "n_channels_hit": nch, "auroc": a,
                     "frac_pixels_changed": float((px > 0).mean()),
                     "mean_abs_diff_u8": float(px.mean())})
        print(f"  {enc} {kind:6s} amp={amp:3d} w={w:3d} nch={nch:3d} "
              f"AUROC={a:.3f}  pix={float((px>0).mean()):.4%}", flush=True)
    del bb

df = pd.DataFrame(rows)
df.to_csv(OUT / "control_sweep.csv", index=False)

# control_sweep.csv holds only this invocation.  The all-encoding table read by
# later analyses is maintained here as well: rows of the encodings just run
# replace any earlier rows for them, other encodings are kept in place.  The
# released file was produced by three invocations in this order:
#   08_diagnose_controls.py E1
#   08_diagnose_controls.py E2 E4 E5 E7 E3 E6 E8 E9
#   08_diagnose_controls.py E10 E11
ALL = OUT / "control_sweep_all.csv"
if ALL.exists():
    old = pd.read_csv(ALL)
    df_all = pd.concat([old[~old["encoding"].isin(ENCS)], df], ignore_index=True)
else:
    df_all = df
df_all.to_csv(ALL, index=False)
print(f"\n[done] -> {OUT/'control_sweep.csv'} and {ALL} "
      f"({df_all['encoding'].nunique()} encodings)")
