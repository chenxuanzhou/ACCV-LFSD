"""How much of each *real* anomaly lies inside the invariance group?

The synthetic suite shows the mechanism, but the obvious reviewer question is
whether it matters for the actual voraus-AD anomalies.  This script measures,
per test recording, how much of its deviation from the normal template is a
per-channel affine map -- exactly the component that per-window min-max
annihilates (P1).

For each channel c the best-fitting affine map onto the normal template is

    argmin_{a,b} || a * T_c + b - X_c ||^2

and the deviation that survives it is the residual.  The affine-explained
fraction is then 1 - residual_energy / total_deviation_energy.  Normal test
recordings are measured the same way, so the number that matters is the
*excess* over normal variation, not the raw fraction.

If the theory transfers, categories with a high affine-explained fraction are
the ones where an affine-blind encoding (E1) loses most against its
affine-preserving control (E2).  That correlation is reported directly.

Writes results/17_affine/{affine_content.csv,affine_vs_gap.csv}
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "results" / "01_preprocess"
EVAL = ROOT / "results" / "07_eval"
OUT = ROOT / "results" / "17_affine"
OUT.mkdir(parents=True, exist_ok=True)

W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
stats = json.load(open(PRE / "norm_stats.json"))
hold = json.load(open(PRE / "holdout_normal.json"))
bank = np.array(hold["memory_bank_pool_indices"])

ch_std = np.array(stats["std"])
# Template from the bank pool only, in train-z-scored units so channels with
# large physical ranges do not dominate the energy accounting.
T = (np.asarray(W[bank], np.float64) - np.array(stats["mean"])[None, :, None]) / ch_std[None, :, None]
T = T.mean(axis=0)                                            # (C, L)
C, L = T.shape
print(f"[template] {T.shape} from {len(bank)} bank-pool recordings")

Tc = T - T.mean(axis=1, keepdims=True)
Tvar = (Tc ** 2).sum(axis=1)                                  # per-channel

rows = []
test_idx = meta.loc[meta["split"] == "test", "idx"].to_numpy()
for i in test_idx:
    X = (np.asarray(W[i], np.float64) - np.array(stats["mean"])[:, None]) / ch_std[:, None]
    Xc = X - X.mean(axis=1, keepdims=True)
    # per-channel least-squares affine fit onto the template
    a = np.where(Tvar > 1e-12, (Xc * Tc).sum(axis=1) / np.maximum(Tvar, 1e-12), 0.0)
    b = X.mean(axis=1) - a * T.mean(axis=1)
    fit = a[:, None] * T + b[:, None]
    total = ((X - T) ** 2).sum()
    resid = ((X - fit) ** 2).sum()
    m = meta.iloc[i]
    rows.append({
        "idx": int(i), "anomaly": bool(m["anomaly"]), "category": m["category"],
        "total_deviation": float(total), "residual_after_affine": float(resid),
        "affine_explained_fraction": float(1.0 - resid / max(total, 1e-12)),
        "gain_dispersion": float(np.std(a)), "offset_dispersion": float(np.std(b)),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "affine_content.csv", index=False)

norm = df[~df["anomaly"]]["affine_explained_fraction"].mean()
summ = (df[df["anomaly"]].groupby("category")["affine_explained_fraction"]
        .agg(["mean", "std", "count"]).reset_index())
summ["excess_over_normal"] = summ["mean"] - norm
summ = summ.sort_values("excess_over_normal", ascending=False)
summ.to_csv(OUT / "affine_by_category.csv", index=False)

print(f"\n=== affine-explained fraction of the deviation from the normal template ===")
print(f"test normals: {norm:.4f}")
print(summ.round(4).to_string(index=False))

# --------------------------------- how many channels does a real anomaly touch?
# The synthetic sweep (results/08_diagnose) shows sensitivity is governed by the
# number of channels a fault touches.  If real anomalies are concentrated in a
# few channels, the footprint limit -- not the invariance group -- is what binds
# on this dataset, and the paper must say so.
rows = []
for i in test_idx:
    X = (np.asarray(W[i], np.float64) - np.array(stats["mean"])[:, None]) / ch_std[:, None]
    D = np.abs(X - T)                                       # per-channel deviation
    peak = D.max(axis=1)                                    # (C,)
    energy = (D ** 2).sum(axis=1)
    order = np.argsort(energy)[::-1]
    csum = np.cumsum(energy[order]) / max(energy.sum(), 1e-12)
    m = meta.iloc[i]
    rows.append({
        "idx": int(i), "anomaly": bool(m["anomaly"]), "category": m["category"],
        "n_channels_peak_gt3": int((peak > 3.0).sum()),
        "n_channels_peak_gt5": int((peak > 5.0).sum()),
        "n_channels_for_50pct_energy": int(np.searchsorted(csum, 0.50) + 1),
        "n_channels_for_90pct_energy": int(np.searchsorted(csum, 0.90) + 1),
        "top1_channel_energy_share": float(energy[order[0]] / max(energy.sum(), 1e-12)),
    })
fp = pd.DataFrame(rows)
fp.to_csv(OUT / "channel_footprint.csv", index=False)
fpn = fp[~fp["anomaly"]]
fps = (fp[fp["anomaly"]].groupby("category")[
    ["n_channels_peak_gt3", "n_channels_for_50pct_energy",
     "n_channels_for_90pct_energy", "top1_channel_energy_share"]]
    .mean().reset_index())
print("\n=== how many of the 129 channels does the deviation actually occupy? ===")
print("test normals:", fpn[["n_channels_peak_gt3", "n_channels_for_50pct_energy",
                            "n_channels_for_90pct_energy",
                            "top1_channel_energy_share"]].mean().round(3).to_dict())
print(fps.round(3).to_string(index=False))
fps.to_csv(OUT / "channel_footprint_by_category.csv", index=False)

# ------------------------------------------------ link to the E1 vs E2 gap
pa_path = EVAL / "pa_percategory.csv"
if pa_path.exists():
    pa = pd.read_csv(pa_path)
    pa = pa[(pa["k"] == "4") & (pa["category"] != "__MEAN_OF_CATEGORIES__")]
    per = (pa.groupby(["encoding", "category"])["auroc"].mean().unstack(0))
    pairs = [("E1", "E2"), ("E10", "E11"), ("E3", "E9"), ("E5", "E6")]
    out = []
    for blind, ctrl in pairs:
        if blind not in per.columns or ctrl not in per.columns:
            continue
        gap = (per[ctrl] - per[blind]).rename("gap")
        j = summ.set_index("category").join(gap, how="inner")
        if len(j) < 3:
            continue
        r = float(np.corrcoef(j["excess_over_normal"], j["gap"])[0, 1])
        for cat, row in j.iterrows():
            out.append({"pair": f"{ctrl}-{blind}", "category": cat,
                        "affine_excess": row["excess_over_normal"],
                        "auroc_gap": row["gap"], "pearson_r_for_pair": r})
        print(f"\n[{ctrl} - {blind}]  Pearson r(affine excess, AUROC gap) = {r:+.3f} "
              f"over {len(j)} categories")
    if out:
        pd.DataFrame(out).to_csv(OUT / "affine_vs_gap.csv", index=False)
else:
    print("\n[skip] results/07_eval/pa_percategory.csv not present yet; "
          "rerun after 07_evaluate.py to get the correlation")
