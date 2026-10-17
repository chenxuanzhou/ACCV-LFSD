"""Main experiment: 9 encodings x 5 vision detectors x few-shot sweep.

Protocols (plan sec. 5):
  P-A  real anomalies, official test split, k in {1,2,4,8,16,all} x 5 seeds
  P-B  synthetic faults, holdout normals vs injected, k=4 x 5 seeds

One backbone pass per (encoding, backbone); every detector and every few-shot
configuration then reuses those features, so the GPU cost is 2 passes per
encoding rather than 5.

Outputs results/06_scores/<E>/<M>.parquet with one row per scored image, and
maps.npz holding the k=4 anomaly maps needed by the attribution protocols.

Usage:  python 06_run_detectors.py [E1 E2 ...]
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.backbones import Backbone                       # noqa: E402
from voraus_vision.detectors import (KNNGlobal, MahalanobisGlobal,  # noqa: E402
                                     PaDiM, PatchCore)
from voraus_vision.encodings import ENCODERS                        # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
OUT = ROOT / "results" / "06_scores"
OUT.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(REN / "manifest.csv")
fewshot = json.load(open(PRE / "fewshot_indices.json"))
SEEDS = [0, 1, 2, 3, 4]
K_ALL = ["1", "2", "4", "8", "16", "all"]
K_SYNTH = "4"          # the synthetic sweep is run at a single k to keep it affordable

bank_rows = man[man["kind"] == "bank"]
src_to_row = dict(zip(bank_rows["src_idx"], bank_rows["row"]))
test_rows = man.index[man["kind"] == "test"].to_numpy()
synth_rows = man.index[man["kind"].isin(["holdout", "injected"])].to_numpy()

# detector id -> (backbone, factory, uses patch features, produces a map)
DETECTORS = {
    "M1_patchcore_wrn50":   ("wrn50",  lambda: PatchCore(0.01), True,  True),
    "M3_padim_wrn50":       ("wrn50",  lambda: PaDiM(),         True,  True),
    "M2_patchcore_dinov2":  ("dinov2", lambda: PatchCore(0.01), True,  True),
    "M4_knn_cls_dinov2":    ("dinov2", lambda: KNNGlobal(1),    False, False),
    "M5_mahal_cls_dinov2":  ("dinov2", lambda: MahalanobisGlobal(), False, False),
}


def bank_indices(k: str, seed: int) -> np.ndarray:
    key = f"k{k}_seed{seed}" if k != "all" else f"kall_seed{seed}"
    return np.array([src_to_row[s] for s in fewshot[key]])


targets = sys.argv[1:] or list(ENCODERS)
for enc in targets:
    imgs = np.load(REN / enc / "images.npy", mmap_mode="r")
    dest = OUT / enc
    dest.mkdir(parents=True, exist_ok=True)
    for bb_name in ["wrn50", "dinov2"]:
        dets = {k: v for k, v in DETECTORS.items() if v[0] == bb_name}
        if all((dest / f"{d}.parquet").exists() for d in dets):
            print(f"[{enc}/{bb_name}] done already, skipping")
            continue
        t0 = time.time()
        bb = Backbone(bb_name)
        pf, gf = bb.forward(np.asarray(imgs))
        del bb
        import torch
        torch.cuda.empty_cache()
        print(f"[{enc}/{bb_name}] features {pf.shape} in {time.time()-t0:.0f}s", flush=True)

        for det_id, (_, factory, use_patch, has_map) in dets.items():
            if (dest / f"{det_id}.parquet").exists():
                continue
            t1 = time.time()
            feats = pf if use_patch else gf
            rows, maps = [], {}
            for k in K_ALL:
                for seed in SEEDS:
                    bi = bank_indices(k, seed)
                    det = factory().fit(feats[bi])
                    s, m = det.score(feats[test_rows])
                    rows.append(pd.DataFrame({"row": test_rows, "k": k, "seed": seed,
                                              "score": s, "protocol": "P-A"}))
                    if k == K_SYNTH:
                        s2, m2 = det.score(feats[synth_rows])
                        rows.append(pd.DataFrame({"row": synth_rows, "k": k, "seed": seed,
                                                  "score": s2, "protocol": "P-B"}))
                        if has_map:
                            maps[f"seed{seed}"] = m2.astype(np.float16)
                    del det
                    torch.cuda.empty_cache()
            df = pd.concat(rows, ignore_index=True)
            df.to_parquet(dest / f"{det_id}.parquet", index=False)
            if maps:
                np.savez_compressed(dest / f"{det_id}_maps.npz", rows=synth_rows, **maps)
            print(f"  [{enc}/{det_id}] {len(df)} scores in {time.time()-t1:.0f}s", flush=True)
        del pf, gf
    print(f"[{enc}] complete", flush=True)
