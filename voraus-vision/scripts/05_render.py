"""Render every window that any experiment needs, once, per encoding.

Image roster (identical across encodings, same row order, so results are
directly comparable):

    test        1174  official test split (755 anomalous + 419 normal)   -> P-A
    bank         758  memory-bank pool (train normal minus holdout)      -> memory
    holdout      190  clean held-out normal                              -> P-B negatives
    injected    6840  190 holdout x 36 (fault, param) cells              -> P-B positives

Outputs per encoding under results/05_render/<E>/images.npy (uint8, N,224,224,3)
plus a shared manifest.csv describing the rows.

Usage:  python 05_render.py [E1 E2 ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.context import load_context
from voraus_vision.encodings import ENCODERS          # noqa: E402
from voraus_vision.injection import (FAULT_PARAMS, inject, inject_duration,
                                      injection_seed)  # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
CTX = ROOT / "results" / "02_context"
OUT = ROOT / "results" / "05_render"
OUT.mkdir(parents=True, exist_ok=True)

c = np.load(CTX / "context.npz")
ctx = load_context(ROOT)
RENDERED_CH = set(int(x) for x in c["top_var_channels"])

W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
hold = json.load(open(PRE / "holdout_normal.json"))
holdout = hold["holdout_normal_indices"]
bank = hold["memory_bank_pool_indices"]
raw = np.load(PRE / "holdout_raw.npz")
L = W.shape[2]
test_idx = meta.loc[meta["split"] == "test", "idx"].tolist()


def resample_T(arr, n):
    T = arr.shape[1]
    src = np.linspace(0.0, T - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = src - i0
    return arr[:, i0] * (1 - w) + arr[:, i1] * w


# ------------------------------------------------------------------ manifest
MANIFEST = OUT / "manifest.csv"
if MANIFEST.exists():
    man = pd.read_csv(MANIFEST)
    print(f"[manifest] reusing {len(man)} rows")
else:
    rows = []
    for i in test_idx:
        m = meta.iloc[i]
        rows.append({"kind": "test", "src_idx": int(i), "fault": "", "param": np.nan,
                     "anomaly": bool(m["anomaly"]), "category": m["category"],
                     "variant": m["variant"], "channels": "", "t_start": -1, "t_end": -1,
                     "injected_ch_rendered_by_e4": ""})
    for i in bank:
        rows.append({"kind": "bank", "src_idx": int(i), "fault": "", "param": np.nan,
                     "anomaly": False, "category": "NORMAL_OPERATION",
                     "variant": "PRE_A", "channels": "", "t_start": -1, "t_end": -1,
                     "injected_ch_rendered_by_e4": ""})
    for i in holdout:
        rows.append({"kind": "holdout", "src_idx": int(i), "fault": "", "param": np.nan,
                     "anomaly": False, "category": "NORMAL_OPERATION",
                     "variant": "PRE_A", "channels": "", "t_start": -1, "t_end": -1,
                     "injected_ch_rendered_by_e4": ""})
    for fault, params in FAULT_PARAMS.items():
        for p in params:
            for j, i in enumerate(holdout):
                rows.append({"kind": "injected", "src_idx": int(i), "fault": fault,
                             "param": float(p), "anomaly": True, "category": fault,
                             "variant": "PRE_A", "channels": "", "t_start": -1,
                             "t_end": -1, "injected_ch_rendered_by_e4": ""})
    man = pd.DataFrame(rows)
    man.insert(0, "row", np.arange(len(man)))
    man.to_csv(MANIFEST, index=False)
    print(f"[manifest] wrote {len(man)} rows -> {MANIFEST}")

N = len(man)
inj_mask = (man["kind"] == "injected").to_numpy()


def build_signal(r):
    """Return the (C, L) window for manifest row ``r``, injection applied."""
    i = int(r.src_idx)
    if r.kind != "injected":
        return np.asarray(W[i], dtype=np.float64), None
    fault, param = r.fault, float(r.param)
    # deterministic per (fault, param, source window): identical across encodings
    rng = np.random.default_rng(injection_seed(fault, param, i))
    if fault == "S5":
        s = int(meta.iloc[i]["sample"])
        R = np.asarray(raw[str(s)], dtype=np.float64)
        return resample_T(inject_duration(R, param), L), None
    X = np.asarray(W[i], dtype=np.float64)
    return inject(X, fault, param, ctx.ch_std, ctx.ch_mean, rng)


targets = sys.argv[1:] or list(ENCODERS)
for enc_id in targets:
    d = OUT / enc_id
    d.mkdir(exist_ok=True)
    path = d / "images.npy"
    if path.exists():
        print(f"[{enc_id}] already rendered, skipping")
        continue
    fn = ENCODERS[enc_id]
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(N, 224, 224, 3))
    gt_rows = []
    t0 = time.time()
    for r in man.itertuples():
        X, gt = build_signal(r)
        arr[r.row] = fn(X, ctx)
        if gt is not None:
            gt_rows.append({"row": r.row, **gt.as_row(),
                            "injected_ch_rendered_by_e4":
                                all(ch in RENDERED_CH for ch in gt.channels)})
        if r.row % 1000 == 0 and r.row:
            el = time.time() - t0
            print(f"  [{enc_id}] {r.row}/{N}  {el:.0f}s  eta {el/r.row*(N-r.row):.0f}s", flush=True)
    arr.flush()
    del arr
    if gt_rows:
        gtp = OUT / "injection_gt.csv"
        if not gtp.exists():
            pd.DataFrame(gt_rows).to_csv(gtp, index=False)
    print(f"[{enc_id}] done in {time.time()-t0:.0f}s -> {path}", flush=True)
