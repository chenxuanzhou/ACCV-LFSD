"""Ablation 2 of the plan: permute the channel rows of the heatmap encodings.

A signal heatmap gives the channel axis a spatial meaning it does not have --
`motor_position_1` is adjacent to `motor_velocity_1` only because of the order
the columns happen to appear in.  A convolutional or windowed-attention backbone
assumes locality along both axes, so if the channel order matters, the encoding
is injecting structure that is not in the data.

Run on the encodings whose channel axis is spatial (E1, E2, E10, E11) plus E4
(tile order).  Five permutations, plus the identity, reported as spread.

Writes results/14_permutation/permutation.csv
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
from voraus_vision.backbones import Backbone            # noqa: E402
from voraus_vision.context import load_context          # noqa: E402
from voraus_vision.detectors import PatchCore           # noqa: E402
from voraus_vision.encodings import ENCODERS            # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
OUT = ROOT / "results" / "14_permutation"
OUT.mkdir(parents=True, exist_ok=True)

ctx = load_context(ROOT)
W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
fewshot = json.load(open(PRE / "fewshot_indices.json"))
test_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
test_anom = meta.loc[meta["split"] == "test", "anomaly"].to_numpy().astype(bool)
test_cat = meta.loc[meta["split"] == "test", "category"].to_numpy()
CATS = sorted(set(test_cat[test_anom]))
N_CH = W.shape[1]


def auroc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def mean_cat_auroc(scores):
    """Official protocol: per-category AUROC against all test normals, then mean."""
    sn = scores[~test_anom]
    vals = []
    for c in CATS:
        m = test_anom & (test_cat == c)
        y = np.r_[np.ones(m.sum(), bool), np.zeros(len(sn), bool)]
        vals.append(auroc(y, np.r_[scores[m], sn]))
    return float(np.mean(vals))


ENCS = sys.argv[1:] or ["E1", "E2", "E10", "E11", "E4"]
N_PERM = 5
rows = []
bb = Backbone("wrn50")

for enc in ENCS:
    fn = ENCODERS[enc]
    for perm_id in range(N_PERM + 1):
        if perm_id == 0:
            perm = np.arange(N_CH)                       # identity
        else:
            perm = np.random.default_rng(100 + perm_id).permutation(N_CH)

        def render(i):
            return fn(np.asarray(W[i], np.float64)[perm], ctx)

        for seed in (0, 1, 2):
            bank = np.stack([render(i) for i in fewshot[f"k4_seed{seed}"]])
            bf, _ = bb.forward(bank)
            det = PatchCore(0.01).fit(bf)
            s = np.empty(len(test_idx), np.float32)
            for j in range(0, len(test_idx), 256):
                chunk = np.stack([render(i) for i in test_idx[j:j + 256]])
                f, _ = bb.forward(chunk)
                s[j:j + len(chunk)], _ = det.score(f)
            a = mean_cat_auroc(s)
            rows.append({"encoding": enc, "perm_id": perm_id, "seed": seed,
                         "identity": perm_id == 0, "auroc_mean_of_categories": a})
            print(f"  {enc} perm={perm_id} seed={seed} AUROC={a:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "permutation.csv", index=False)
summ = (df.groupby(["encoding", "identity"])["auroc_mean_of_categories"]
        .agg(["mean", "std", "min", "max", "count"]).reset_index())
summ.to_csv(OUT / "permutation_summary.csv", index=False)
print()
print(summ.round(4).to_string(index=False))
