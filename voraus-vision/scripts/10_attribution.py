"""P-C channel attribution and P-D temporal localisation on synthetic faults.

Both protocols read the anomaly maps saved by 06_run_detectors.py.  Those maps
live on a 14x14 patch grid, so the achievable resolution is bounded before any
detector is involved, and the bound has to be reported with the numbers:

  channel axis   129 channels rendered into 224 image rows, pooled to 14 grid
                 rows -> each grid row covers ~9.2 channels.  "top-1 channel"
                 therefore means "top-1 band of ~9 channels"; the honest random
                 baseline is 1/14 bands, not 1/129 channels, and both are
                 reported.
  time axis      512 samples into 14 grid columns -> ~36.6 samples per column.
                 An S7 spike is 5 samples (sub-cell) and an S8 freeze is 50
                 (~1.4 cells), so localisation IoU is quantisation-limited.

Only encodings with a channel-to-pixel bijection (E1, E2) or a channel-to-tile
mapping (E4) support P-C at all; for E3/E5/E6/E7/E8/E9 the PCA projection or the
full-dimensional distance destroys channel identity, and that fact is itself a
result (plan sec. 5).

Writes results/10_attribution/{pc_channel,pd_temporal}.csv
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
from voraus_vision.encodings import CHANNEL_ATTRIBUTABLE  # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
REN = ROOT / "results" / "05_render"
SCO = ROOT / "results" / "06_scores"
CTX = ROOT / "results" / "02_context"
OUT = ROOT / "results" / "10_attribution"
OUT.mkdir(parents=True, exist_ok=True)

GRID = 14
man = pd.read_csv(REN / "manifest.csv")
gt = pd.read_csv(REN / "injection_gt.csv").set_index("row")
kept = json.load(open(PRE / "kept_channels.json"))
N_CH = len(kept["kept_channel_names"])
L = 512
ctx = np.load(CTX / "context.npz")
E4_CH = list(ctx["top_var_channels"])
# E10/E11 render only the 28-signal core set, so their grid rows index into that
# subset rather than into all 129 channels.
from voraus_vision.encodings import CORE_CHANNEL_NAMES  # noqa: E402
_names = kept["kept_channel_names"]
CORE_CH = [_names.index(n) for n in CORE_CHANNEL_NAMES if n in _names]

# grid row -> channel interval, for the heatmap encodings
band_edges = np.linspace(0, N_CH, GRID + 1)
BAND_SIZE = N_CH / GRID


def channel_scores(m2d: np.ndarray, enc: str) -> np.ndarray:
    """Project a (14,14) anomaly map onto a per-channel score vector."""
    row_score = m2d.max(axis=1)                       # (14,)
    s = np.zeros(N_CH)
    if enc in ("E1", "E2", "E10", "E11"):
        rendered = list(range(N_CH)) if enc in ("E1", "E2") else CORE_CH
        edges = np.linspace(0, len(rendered), GRID + 1)
        for r in range(GRID):
            lo, hi = int(np.floor(edges[r])), int(np.ceil(edges[r + 1]))
            for ch in rendered[lo:hi]:
                s[ch] = max(s[ch], row_score[r])
    elif enc == "E4":
        # 3x3 tile grid over 14x14 cells: tile (i,j) covers roughly cells
        # [i*14/3,(i+1)*14/3) x [j*14/3,(j+1)*14/3)
        e = np.linspace(0, GRID, 4)
        for t, ch in enumerate(E4_CH):
            i, j = divmod(t, 3)
            blk = m2d[int(e[i]):int(np.ceil(e[i + 1])), int(e[j]):int(np.ceil(e[j + 1]))]
            s[int(ch)] = blk.max()
    return s


def otsu(x: np.ndarray) -> float:
    """Otsu threshold on a 1-D response."""
    hist, edges = np.histogram(x, bins=64)
    p = hist / max(hist.sum(), 1)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    m0 = np.cumsum(p * centres) / np.maximum(w0, 1e-12)
    m1 = (np.sum(p * centres) - np.cumsum(p * centres)) / np.maximum(w1, 1e-12)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.nanargmax(var))])


pc_rows, pd_rows = [], []
LOCAL_FAULTS = {"S6", "S7", "S8"}

for enc_dir in sorted(p for p in SCO.iterdir() if p.is_dir()):
    enc = enc_dir.name
    for f in sorted(enc_dir.glob("*_maps.npz")):
        det = f.name.replace("_maps.npz", "")
        z = np.load(f)
        rows = z["rows"]
        for key in [k for k in z.files if k.startswith("seed")]:
            seed = int(key[4:])
            M = z[key].astype(np.float32)              # (n_rows, 196)
            for i, row in enumerate(rows):
                if row not in gt.index:
                    continue
                g = gt.loc[row]
                fault = g["fault"]
                chans = [int(c) for c in str(g["channels"]).split(";") if c != ""]
                m2d = M[i].reshape(GRID, GRID)

                # ------------------------------------------------ P-C
                if CHANNEL_ATTRIBUTABLE.get(enc, False) and len(chans) == 1 \
                        and not bool(g["global_fault"]):
                    s = channel_scores(m2d, enc)
                    tgt = chans[0]
                    # Channels inside one grid row share a score, so a per-channel
                    # rank is decided by argsort's tie-breaking, not by the map.
                    # The resolvable unit is the band, and that is what is scored;
                    # the per-channel rank is kept only as an upper bound.
                    uniq = np.unique(s)[::-1]
                    band_rank = int(np.where(uniq == s[tgt])[0][0])
                    order = np.argsort(s, kind="stable")[::-1]
                    rank = int(np.where(order == tgt)[0][0])
                    pc_rows.append({
                        "encoding": enc, "detector": det, "seed": seed, "fault": fault,
                        "param": g["param"], "target_channel": tgt, "rank": rank,
                        "band_rank": band_rank,
                        "band_top1": band_rank == 0, "band_top3": band_rank < 3,
                        "top1": rank == 0, "top5": rank < 5,
                        "target_in_e4_grid": tgt in E4_CH,
                    })

                # ------------------------------------------------ P-D
                if fault in LOCAL_FAULTS and int(g["t_end"]) - int(g["t_start"]) < L:
                    resp = m2d.max(axis=0)                       # marginal over channels
                    up = np.interp(np.linspace(0, GRID - 1, L),
                                   np.arange(GRID), resp)
                    pred = up > otsu(up)
                    true = np.zeros(L, bool)
                    true[int(g["t_start"]):int(g["t_end"])] = True
                    inter = (pred & true).sum()
                    union = (pred | true).sum()
                    pd_rows.append({
                        "encoding": enc, "detector": det, "seed": seed, "fault": fault,
                        "param": g["param"], "iou": float(inter / max(union, 1)),
                        "pred_frac": float(pred.mean()), "true_frac": float(true.mean()),
                    })
    print(f"[attr] {enc} done", flush=True)

pc = pd.DataFrame(pc_rows)
pdf = pd.DataFrame(pd_rows)
pc.to_csv(OUT / "pc_channel.csv", index=False)
pdf.to_csv(OUT / "pd_temporal.csv", index=False)

meta = {
    "n_channels": N_CH,
    "grid": GRID,
    "channels_per_grid_row": BAND_SIZE,
    "samples_per_grid_col": L / GRID,
    "random_baseline_top1_per_channel": 1.0 / N_CH,
    "random_baseline_top1_per_band": 1.0 / GRID,
    "attributable_encodings": [k for k, v in CHANNEL_ATTRIBUTABLE.items() if v],
}
json.dump(meta, open(OUT / "attribution_meta.json", "w"), indent=2)

if len(pc):
    print("\n=== P-C top-1 / top-5 channel hit rate ===")
    print(pc.groupby(["encoding", "detector"])[["top1", "top5"]].mean().round(3).to_string())
    print(f"random baseline: top1 = {1/N_CH:.4f} per channel, {1/GRID:.4f} per band")
if len(pdf):
    print("\n=== P-D temporal IoU ===")
    print(pdf.groupby(["encoding", "fault"])["iou"].mean().round(3).unstack().to_string())
