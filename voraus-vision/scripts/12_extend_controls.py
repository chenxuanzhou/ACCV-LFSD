"""Add the footprint-matched positive controls to the main protocol.

results/08_diagnose/control_sweep.csv established that an encoding's
sensitivity is set by how many channels a fault touches, not by its severity.
The plan's controls (S7/S8) touch one channel, so they can only validate the
pipeline for single-channel faults -- and on the heatmap encodings they do not
even do that.  Global faults (S3/S3b/S4b/S5) need a control that touches every
channel.

This script adds S7m/S8m (8 channels) and S7g/S8g (all channels) to the P-B
protocol, at the same k=4 x 5 seeds and the same five detectors as the main run,
so the pre-registered blind/visible rule can be applied to a matched control.

It renders and scores only the new rows: the memory banks are unchanged, so the
detectors are refitted from bank images already present in images.npy.

Writes results/05_render/<E>/images_ext.npy, manifest_ext.csv,
injection_gt_ext.csv and results/06_scores/<E>/<M>_ext.parquet
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
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.backbones import Backbone                       # noqa: E402
from voraus_vision.detectors import (KNNGlobal, MahalanobisGlobal,  # noqa: E402
                                     PaDiM, PatchCore)
from voraus_vision.context import load_context
from voraus_vision.encodings import ENCODERS        # noqa: E402
from voraus_vision.injection import inject, injection_seed           # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
CTX = ROOT / "results" / "02_context"
SCO = ROOT / "results" / "06_scores"

NEW_FAULTS = ["S7m", "S8m", "S7g", "S8g"]
K_SYNTH = "4"
SEEDS = [0, 1, 2, 3, 4]
DETECTORS = {
    "M1_patchcore_wrn50":   ("wrn50",  lambda: PatchCore(0.01), True,  True),
    "M3_padim_wrn50":       ("wrn50",  lambda: PaDiM(),         True,  True),
    "M2_patchcore_dinov2":  ("dinov2", lambda: PatchCore(0.01), True,  True),
    "M4_knn_cls_dinov2":    ("dinov2", lambda: KNNGlobal(1),    False, False),
    "M5_mahal_cls_dinov2":  ("dinov2", lambda: MahalanobisGlobal(), False, False),
}

c = np.load(CTX / "context.npz")
ctx = load_context(ROOT)
RENDERED_CH = set(int(x) for x in c["top_var_channels"])

W = np.load(PRE / "windows.npy", mmap_mode="r")
man = pd.read_csv(REN / "manifest.csv")
hold = json.load(open(PRE / "holdout_normal.json"))
holdout = hold["holdout_normal_indices"]
fewshot = json.load(open(PRE / "fewshot_indices.json"))
bank_rows_df = man[man["kind"] == "bank"]
src_to_row = dict(zip(bank_rows_df["src_idx"], bank_rows_df["row"]))

# ------------------------------------------------------------ ext manifest
EXT = REN / "manifest_ext.csv"
if EXT.exists():
    ext = pd.read_csv(EXT)
else:
    rows = []
    for fault in NEW_FAULTS:
        for i in holdout:
            rows.append({"kind": "injected_ext", "src_idx": int(i), "fault": fault,
                         "param": 1.0, "anomaly": True, "category": fault})
    ext = pd.DataFrame(rows)
    ext.insert(0, "ext_row", np.arange(len(ext)))
    ext.to_csv(EXT, index=False)
print(f"[ext] {len(ext)} extension rows ({len(NEW_FAULTS)} faults x {len(holdout)} windows)")

targets = sys.argv[1:] or list(ENCODERS)
for enc in targets:
    d = REN / enc
    ext_path = d / "images_ext.npy"
    if not ext_path.exists():
        fn = ENCODERS[enc]
        arr = np.lib.format.open_memmap(ext_path, mode="w+", dtype=np.uint8,
                                        shape=(len(ext), 224, 224, 3))
        gt_rows = []
        t0 = time.time()
        for r in ext.itertuples():
            i = int(r.src_idx)
            rng = np.random.default_rng(injection_seed(r.fault, 1.0, i))
            Y, gt = inject(np.asarray(W[i], np.float64), r.fault, 1.0,
                           ctx.ch_std, ctx.ch_mean, rng)
            arr[r.ext_row] = fn(Y, ctx)
            gt_rows.append({"ext_row": r.ext_row, **gt.as_row(),
                            "n_channels_hit": len(gt.channels),
                            "injected_ch_rendered_by_e4":
                                all(ch in RENDERED_CH for ch in gt.channels)})
        arr.flush()
        del arr
        p = REN / "injection_gt_ext.csv"
        if not p.exists():
            pd.DataFrame(gt_rows).to_csv(p, index=False)
        print(f"[{enc}] rendered {len(ext)} ext images in {time.time()-t0:.0f}s", flush=True)

    imgs_main = np.load(d / "images.npy", mmap_mode="r")
    imgs_ext = np.load(ext_path, mmap_mode="r")
    dest = SCO / enc
    for bb_name in ["wrn50", "dinov2"]:
        dets = {k: v for k, v in DETECTORS.items() if v[0] == bb_name}
        if all((dest / f"{k}_ext.parquet").exists() for k in dets):
            continue
        bb = Backbone(bb_name)
        bank_all = np.array([src_to_row[s] for s in fewshot["kall_seed0"]])
        bpf, bgf = bb.forward(np.asarray(imgs_main[bank_all]))
        epf, egf = bb.forward(np.asarray(imgs_ext))
        del bb
        torch.cuda.empty_cache()
        row_of = {int(r): j for j, r in enumerate(bank_all)}
        for det_id, (_, factory, use_patch, _) in dets.items():
            if (dest / f"{det_id}_ext.parquet").exists():
                continue
            bf = bpf if use_patch else bgf
            qf = epf if use_patch else egf
            out = []
            for seed in SEEDS:
                bi = np.array([row_of[src_to_row[s]]
                               for s in fewshot[f"k{K_SYNTH}_seed{seed}"]])
                det = factory().fit(bf[bi])
                s, _ = det.score(qf)
                out.append(pd.DataFrame({"ext_row": ext["ext_row"].to_numpy(),
                                         "fault": ext["fault"].to_numpy(),
                                         "k": K_SYNTH, "seed": seed, "score": s,
                                         "protocol": "P-B-ext"}))
                del det
                torch.cuda.empty_cache()
            pd.concat(out, ignore_index=True).to_parquet(
                dest / f"{det_id}_ext.parquet", index=False)
            print(f"  [{enc}/{det_id}] ext scores written", flush=True)
    print(f"[{enc}] extension complete", flush=True)
