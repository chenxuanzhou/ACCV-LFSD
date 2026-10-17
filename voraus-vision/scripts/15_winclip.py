"""M6: WinCLIP zero-shot and WinCLIP+ few-shot over every encoding.

Emits the same parquet layout as 06_run_detectors.py so the evaluation code does
not need to special-case it.  Zero-shot has no memory bank, so its score does
not depend on k or seed; it is written once per encoding under k="0".

Window features are extracted once per encoding and reused across all 31
configurations.  Re-encoding the images for every (k, seed) would multiply the
CLIP forward passes by ~30 and turn a 20-minute job into a four-hour one.

Usage:  python 15_winclip.py [E1 E2 ...]
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
from voraus_vision.encodings import ENCODERS   # noqa: E402
from voraus_vision.winclip import WinCLIP      # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
OUT = ROOT / "results" / "06_scores"

man = pd.read_csv(REN / "manifest.csv")
fewshot = json.load(open(PRE / "fewshot_indices.json"))
SEEDS = [0, 1, 2, 3, 4]
K_ALL = ["1", "2", "4", "8", "16", "all"]
K_SYNTH = "4"
bank_rows = man[man["kind"] == "bank"]
src_to_row = dict(zip(bank_rows["src_idx"], bank_rows["row"]))
test_rows = man.index[man["kind"] == "test"].to_numpy()
synth_rows = man.index[man["kind"].isin(["holdout", "injected"])].to_numpy()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

wc = WinCLIP()


def few_shot_scores(Wq: torch.Tensor, Wb: torch.Tensor,
                    chunk: int = 64, bank_chunk: int = 16384) -> np.ndarray:
    """Max over windows of the cosine distance to the nearest bank window.

    Both axes are chunked: at k=all the bank holds 758*313 = 237k vectors, and
    materialising the full query-by-bank similarity would ask for ~18 GB.
    """
    bank = Wb.reshape(-1, Wb.shape[-1])
    out = np.empty(len(Wq), np.float32)
    for i in range(0, len(Wq), chunk):
        q = Wq[i:i + chunk].to(DEV).float()
        best = torch.full((q.shape[0], q.shape[1]), -2.0, device=DEV)
        for j in range(0, len(bank), bank_chunk):
            b = bank[j:j + bank_chunk].to(DEV).float()
            best = torch.maximum(best, torch.einsum("bmd,kd->bmk", q, b).max(dim=2).values)
            del b
        out[i:i + q.shape[0]] = ((1.0 - best) / 2.0).max(dim=1).values.cpu().numpy()
        del q, best
    torch.cuda.empty_cache()
    return out


targets = sys.argv[1:] or list(ENCODERS)
for enc in targets:
    dest = OUT / enc
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "M6_winclip.parquet").exists():
        print(f"[{enc}] done already")
        continue
    imgs = np.load(REN / enc / "images.npy", mmap_mode="r")
    t0 = time.time()

    G, Wf = [], []
    for i in range(0, len(imgs), 512):
        g, w = wc._window_features(np.asarray(imgs[i:i + 512]))
        G.append(g)
        Wf.append(w)
    G = torch.cat(G)
    Wf = torch.cat(Wf)
    print(f"[{enc}] features {tuple(Wf.shape)} in {time.time()-t0:.0f}s", flush=True)

    rows = []
    g_dev = G.to(DEV).float()
    logits = 100.0 * torch.cat([g_dev @ wc.t_normal.T, g_dev @ wc.t_anom.T], dim=1)
    zs = logits.softmax(dim=1)[:, 1].cpu().numpy()
    del g_dev
    rows.append(pd.DataFrame({"row": test_rows, "k": "0", "seed": 0,
                              "score": zs[test_rows], "protocol": "P-A"}))
    rows.append(pd.DataFrame({"row": synth_rows, "k": "0", "seed": 0,
                              "score": zs[synth_rows], "protocol": "P-B"}))

    for k in K_ALL:
        for seed in SEEDS:
            key = f"k{k}_seed{seed}" if k != "all" else f"kall_seed{seed}"
            bi = np.array([src_to_row[s] for s in fewshot[key]])
            Wb = Wf[bi]
            rows.append(pd.DataFrame({"row": test_rows, "k": k, "seed": seed,
                                      "score": few_shot_scores(Wf[test_rows], Wb),
                                      "protocol": "P-A"}))
            if k == K_SYNTH:
                rows.append(pd.DataFrame({"row": synth_rows, "k": k, "seed": seed,
                                          "score": few_shot_scores(Wf[synth_rows], Wb),
                                          "protocol": "P-B"}))
        print(f"[{enc}] k={k} done ({time.time()-t0:.0f}s)", flush=True)

    pd.concat(rows, ignore_index=True).to_parquet(dest / "M6_winclip.parquet", index=False)
    del Wf, G
    torch.cuda.empty_cache()
    print(f"[{enc}] complete in {time.time()-t0:.0f}s", flush=True)
