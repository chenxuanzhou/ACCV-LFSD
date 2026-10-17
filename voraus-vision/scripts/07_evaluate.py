"""Turn raw scores into the paper's tables.

P-A  real anomalies: per-category AUROC against all test normals, exactly the
     official protocol used by MVT-Flow's train.py, plus AUPRC and F1-max.
P-B  synthetic faults: AUROC of injected vs clean holdout, per (fault, param),
     with a bootstrap CI so that the blind / partial / visible verdict follows
     the pre-registered rule in plan sec. 8.

Outputs results/07_eval/{pa_percategory,pa_summary,pb_sweep,pb_verdict}.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
REN = ROOT / "results" / "05_render"
SCO = ROOT / "results" / "06_scores"
OUT = ROOT / "results" / "07_eval"
OUT.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(REN / "manifest.csv")
gt = pd.read_csv(REN / "injection_gt.csv")
CATEGORIES = sorted(man.loc[(man["kind"] == "test") & man["anomaly"], "category"].unique())
N_BOOT = 1000
# Pre-registered thresholds (plan sec. 8).
BLIND_UPPER = 0.55
VISIBLE_LOWER = 0.70


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s, pos_label=True)
    return auc(fpr, tpr)


def f1_max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(f.max())


def _auroc_rank(y_bool, s):
    """Mann-Whitney form of AUROC, vectorised over the last axis of a 2-D array.

    Used only for the bootstrap: it is ~100x faster than calling roc_curve in a
    Python loop and gives the same value (ties handled by average ranks).
    """
    n = s.shape[-1]
    order = np.argsort(s, axis=-1, kind="stable")
    v = np.take_along_axis(s, order, -1)
    pos = np.arange(n)
    # Bootstrap resampling draws duplicates, so ties are the rule, not the
    # exception; ranks inside a tied block must be averaged or AUROC is biased.
    starts = np.ones_like(v, dtype=bool)
    starts[:, 1:] = v[:, 1:] != v[:, :-1]
    lo = np.maximum.accumulate(np.where(starts, pos, -1), axis=-1)
    ends = np.ones_like(v, dtype=bool)
    ends[:, :-1] = v[:, 1:] != v[:, :-1]
    hi = np.minimum.accumulate(np.where(ends, pos, n)[:, ::-1], axis=-1)[:, ::-1]
    avg_rank_sorted = (lo + hi + 2) / 2.0
    ranks = np.empty_like(v, dtype=np.float64)
    np.put_along_axis(ranks, order, avg_rank_sorted, -1)
    npos = y_bool.sum(-1)
    nneg = n - npos
    rpos = (ranks * y_bool).sum(-1)
    out = (rpos - npos * (npos + 1) / 2.0) / np.maximum(npos * nneg, 1)
    return np.where((npos == 0) | (nneg == 0), np.nan, out)


def boot_ci(y, s, n=N_BOOT, seed=0):
    """Percentile bootstrap CI for AUROC, resampling test items."""
    rng = np.random.default_rng(seed)
    m = len(y)
    take = rng.integers(0, m, size=(n, m))
    vals = _auroc_rank(y[take], s[take])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


encodings = sys.argv[1:] or sorted(p.name for p in SCO.iterdir() if p.is_dir())
pa_rows, pb_rows = [], []

for enc in encodings:
    for f in sorted(x for x in (SCO / enc).glob("M*.parquet")
                if not x.stem.endswith("_ext")):
        det = f.stem
        df = pd.read_parquet(f)
        df = df.merge(man[["row", "kind", "anomaly", "category", "fault", "param"]],
                      on="row", how="left")

        # ---------------------------------------------------------- P-A
        pa = df[df["protocol"] == "P-A"]
        norm = pa[~pa["anomaly"]]
        for (k, seed), g in pa.groupby(["k", "seed"]):
            gn = g[~g["anomaly"]]
            per_cat = {}
            for cat in CATEGORIES:
                ga = g[g["anomaly"] & (g["category"] == cat)]
                if len(ga) == 0:
                    continue
                y = np.r_[np.ones(len(ga), bool), np.zeros(len(gn), bool)]
                s = np.r_[ga["score"].to_numpy(), gn["score"].to_numpy()]
                per_cat[cat] = auroc(y, s)
                pa_rows.append({"encoding": enc, "detector": det, "k": k, "seed": seed,
                                "category": cat, "n_anom": len(ga), "n_norm": len(gn),
                                "auroc": per_cat[cat],
                                "auprc": average_precision_score(y, s),
                                "f1max": f1_max(y, s)})
            ga = g[g["anomaly"]]
            y = np.r_[np.ones(len(ga), bool), np.zeros(len(gn), bool)]
            s = np.r_[ga["score"].to_numpy(), gn["score"].to_numpy()]
            pa_rows.append({"encoding": enc, "detector": det, "k": k, "seed": seed,
                            "category": "__MEAN_OF_CATEGORIES__", "n_anom": len(ga),
                            "n_norm": len(gn),
                            "auroc": float(np.nanmean(list(per_cat.values()))),
                            "auprc": average_precision_score(y, s), "f1max": f1_max(y, s)})

        # ---------------------------------------------------------- P-B
        # k is pinned: M6 also writes zero-shot P-B rows under k="0", and mixing
        # them with the k=4 few-shot rows would silently merge two score scales.
        pbd = df[(df["protocol"] == "P-B") & (df["k"].astype(str) == "4")]
        clean = pbd[pbd["kind"] == "holdout"]
        for (fault, param, seed), g in pbd[pbd["kind"] == "injected"].groupby(
                ["fault", "param", "seed"]):
            gc = clean[clean["seed"] == seed]
            y = np.r_[np.ones(len(g), bool), np.zeros(len(gc), bool)]
            s = np.r_[g["score"].to_numpy(), gc["score"].to_numpy()]
            lo, hi = boot_ci(y, s, seed=seed)
            pb_rows.append({"encoding": enc, "detector": det, "fault": fault,
                            "param": param, "seed": seed, "n_pos": len(g),
                            "n_neg": len(gc), "auroc": auroc(y, s),
                            "ci_lo": lo, "ci_hi": hi})

        # zero-shot (k="0") has no bank, so it is evaluated on its own
        zs = df[(df["protocol"] == "P-B") & (df["k"].astype(str) == "0")]
        if len(zs):
            zc = zs[zs["kind"] == "holdout"]
            for (fault, param), g in zs[zs["kind"] == "injected"].groupby(
                    ["fault", "param"]):
                y = np.r_[np.ones(len(g), bool), np.zeros(len(zc), bool)]
                sv = np.r_[g["score"].to_numpy(), zc["score"].to_numpy()]
                lo, hi = boot_ci(y, sv, seed=0)
                pb_rows.append({"encoding": enc, "detector": det + "_zeroshot",
                                "fault": fault, "param": param, "seed": 0,
                                "n_pos": len(g), "n_neg": len(zc),
                                "auroc": auroc(y, sv), "ci_lo": lo, "ci_hi": hi})

        # ------------------------------------- footprint-matched controls
        # Scored against the same clean holdout negatives as the main P-B rows.
        extf = SCO / enc / f"{det}_ext.parquet"
        if extf.exists():
            for (fault, seed), g in pd.read_parquet(extf).groupby(["fault", "seed"]):
                gc = clean[clean["seed"] == seed]
                y = np.r_[np.ones(len(g), bool), np.zeros(len(gc), bool)]
                s = np.r_[g["score"].to_numpy(), gc["score"].to_numpy()]
                lo, hi = boot_ci(y, s, seed=seed)
                pb_rows.append({"encoding": enc, "detector": det, "fault": fault,
                                "param": 1.0, "seed": seed, "n_pos": len(g),
                                "n_neg": len(gc), "auroc": auroc(y, s),
                                "ci_lo": lo, "ci_hi": hi})
    print(f"[eval] {enc} done", flush=True)

pa = pd.DataFrame(pa_rows)
pb = pd.DataFrame(pb_rows)
pa.to_csv(OUT / "pa_percategory.csv", index=False)
pb.to_csv(OUT / "pb_sweep.csv", index=False)

pa_sum = (pa[pa["category"] == "__MEAN_OF_CATEGORIES__"]
          .groupby(["encoding", "detector", "k"])
          .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
               auprc_mean=("auprc", "mean"), f1max_mean=("f1max", "mean"))
          .reset_index())
pa_sum.to_csv(OUT / "pa_summary.csv", index=False)

# ---------------------------------------------- pre-registered blind verdict
# Amended after the control diagnostic (results/08_diagnose/control_sweep.csv):
# a "blind" verdict is only interpretable as invariance if the positive control
# with the *same spatial footprint* passes on the same (encoding, detector).  An
# encoding that cannot see any single-channel fault tells us nothing about
# whether it is blind to single-channel *affine* faults specifically.
FOOTPRINT = {"S1": 1, "S2": 1, "S4": 1, "S6": 1, "S7": 1, "S8": 1,
             "S7m": 8, "S8m": 8,
             "S3": 129, "S3b": 129, "S4b": 129, "S5": 129, "S7g": 129, "S8g": 129}
CONTROLS = {1: ["S7", "S8"], 8: ["S7m", "S8m"], 129: ["S7g", "S8g"]}
CONTROL_PASS = 0.90

pb["footprint"] = pb["fault"].map(FOOTPRINT)
ctrl = (pb[pb["fault"].isin(["S7", "S8", "S7m", "S8m", "S7g", "S8g"])]
        .groupby(["encoding", "detector", "fault"])["auroc"].mean().reset_index())

verdicts = []
for (enc, fault), g in pb.groupby(["encoding", "fault"]):
    # strongest severity per detector: a cell is blind only if it stays blind
    # at the largest parameter too.
    per_det = g.groupby("detector").agg(ci_hi_max=("ci_hi", "max"),
                                        ci_lo_max=("ci_lo", "max"),
                                        auroc_max=("auroc", "max"))
    blind = bool((per_det["ci_hi_max"] < BLIND_UPPER).all())
    visible = bool((per_det["ci_lo_max"] > VISIBLE_LOWER).any())
    fp = FOOTPRINT.get(fault)
    names = CONTROLS.get(fp, [])
    cg = ctrl[(ctrl["encoding"] == enc) & (ctrl["fault"].isin(names))]
    ctrl_auroc = float(cg["auroc"].max()) if len(cg) else np.nan
    ctrl_ok = bool(ctrl_auroc >= CONTROL_PASS) if not np.isnan(ctrl_auroc) else False
    verdicts.append({
        "encoding": enc, "fault": fault, "footprint_channels": fp,
        "verdict": "blind" if blind else ("visible" if visible else "partial"),
        "matched_control": "+".join(names),
        "matched_control_auroc": ctrl_auroc,
        "control_passes": ctrl_ok,
        # Only a blind cell whose matched control passes is evidence of
        # invariance; otherwise the null is confounded with a resolution limit.
        "interpretable_as_invariance": bool(blind and ctrl_ok),
        "max_auroc_over_detectors": float(per_det["auroc_max"].max()),
        "max_ci_hi": float(per_det["ci_hi_max"].max()),
        "max_ci_lo": float(per_det["ci_lo_max"].max()),
        "n_detectors": int(len(per_det))})
pd.DataFrame(verdicts).to_csv(OUT / "pb_verdict.csv", index=False)

print("\n=== positive controls (must be > 0.9 on every encoding) ===")
ctrl_names = ["S7", "S8", "S7m", "S8m", "S7g", "S8g"]
pc = pb[pb["fault"].isin(ctrl_names)]
print(pc.pivot_table(index="encoding", columns="fault", values="auroc",
                     aggfunc="max").round(3).to_string())
print("(footprint: S7/S8 = 1 channel, S7m/S8m = 8, S7g/S8g = all 129)")
print("\n=== P-A mean-of-categories AUROC at k=4 ===")
print(pa_sum[pa_sum["k"] == "4"].pivot_table(index="encoding", columns="detector",
                                             values="auroc_mean").round(3).to_string())
