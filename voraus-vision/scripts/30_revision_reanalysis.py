"""Zero-compute re-analyses of the main grid (added in revision).

  a  few-shot curve with seed dispersion and with ONE FIXED detector per domain
     (removes the best-of-six multiplicity from the image-vs-signal comparison)
  b  two-way ANOVA at every k with the interaction term reported
  c  encoding/detector ratio when the six PCA-3 projection encodings are dropped

Reads only existing CSVs.  Writes results/30_reanalysis/*.csv and prints a summary.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "30_reanalysis"
OUT.mkdir(exist_ok=True)
KS = ["1", "2", "4", "8", "16", "all"]

pa = pd.read_csv(R / "07_eval" / "pa_percategory.csv", dtype={"k": str})
main = pa[pa.category == "__MEAN_OF_CATEGORIES__"]
mc = pd.read_csv(R / "24_matched_channel" / "matched_channel.csv", dtype={"k": str})
sig = mc[(mc.channels == "28ch") & (mc.normalisation == "z-score")]

# ---- a: dispersion + fixed detector
rows = []
for enc in ["E11", "E10", "E2", "E1"]:
    g = main[main.encoding == enc]
    per = g.groupby(["detector", "k"]).auroc.agg(["mean", "std"]).reset_index()
    for k in KS:
        pk = per[per.k == k].sort_values("mean", ascending=False)
        b = pk.iloc[0]
        rows.append({"method": enc, "rule": "best-of-detectors", "k": k, "detector": b.detector,
                     "mean": b["mean"], "std": b["std"]})
        for det in ["M1_patchcore_wrn50", "M3_padim_wrn50"]:
            f = pk[pk.detector == det].iloc[0]
            rows.append({"method": enc, "rule": "fixed", "k": k, "detector": det, "mean": f["mean"], "std": f["std"]})
per = sig.groupby(["detector", "k"]).auroc.agg(["mean", "std"]).reset_index()
for k in KS:
    for det in ["knn", "pca20"]:
        f = per[(per.k == k) & (per.detector == det)].iloc[0]
        rows.append({"method": "signal28z", "rule": "fixed", "k": k, "detector": det, "mean": f["mean"], "std": f["std"]})
a = pd.DataFrame(rows)
a.to_csv(OUT / "fewshot_dispersion_fixed_detector.csv", index=False)
print("== E11 best-of vs fixed detectors vs signal kNN (mean +- std over 5 seeds)")
for rule, det, meth in [("best-of-detectors", None, "E11"), ("fixed", "M1_patchcore_wrn50", "E11"),
                        ("fixed", "M3_padim_wrn50", "E11"), ("fixed", "knn", "signal28z")]:
    q = a[(a.method == meth) & (a.rule == rule)]
    if det:
        q = q[q.detector == det]
    q = q.set_index("k").reindex(KS)
    print(f"{meth:10s} {rule:18s} {str(det):22s}", "  ".join(f"{m:.3f}+-{s:.3f}" for m, s in zip(q["mean"], q["std"])))
    if rule == "best-of-detectors":
        print(" " * 52, "  ".join(d.split("_")[0] + "_" + d.split("_")[1][:5] for d in q["detector"]))


# ---- b/c: ANOVA with interaction
def anova(df):
    gm = df.auroc.mean()
    sst = ((df.auroc - gm) ** 2).sum()
    n_e, n_d = df.encoding.nunique(), df.detector.nunique()
    cell = df.groupby(["encoding", "detector"]).auroc.mean()
    n_rep = len(df) / (n_e * n_d)
    em = df.groupby("encoding").auroc.mean()
    dm = df.groupby("detector").auroc.mean()
    ss_e = n_d * n_rep * ((em - gm) ** 2).sum()
    ss_d = n_e * n_rep * ((dm - gm) ** 2).sum()
    ss_cells = n_rep * ((cell - gm) ** 2).sum()
    ss_i = ss_cells - ss_e - ss_d
    return dict(eta_enc=ss_e / sst, eta_det=ss_d / sst, eta_int=ss_i / sst, eta_err=(sst - ss_cells) / sst,
                ratio=ss_e / ss_d, n_enc=n_e, n_det=n_d)


PROJ = ["E3", "E5", "E6", "E7", "E8", "E9"]
rows = []
for pool_name, encs in [("all 11", sorted(main.encoding.unique())),
                        ("5 channel-axis encodings", [e for e in sorted(main.encoding.unique()) if e not in PROJ])]:
    for k in ["1", "2", "4", "8", "16"]:
        r = anova(main[(main.k == k) & main.encoding.isin(encs)])
        rows.append({"pool": pool_name, "k": k, **r})
b = pd.DataFrame(rows)
b.to_csv(OUT / "anova_with_interaction.csv", index=False)
print("\n== two-way ANOVA, 12-category mean AUROC")
print(b.round(3).to_string(index=False))
