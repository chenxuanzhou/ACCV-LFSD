"""Assemble the paper's tables from the raw per-configuration results.

Reads results/07_eval (image encodings), results/09_signal (R1/R2 references)
and results/08_diagnose (footprint controls); writes one CSV per table plus a
console summary.

Includes ablation 3 from the plan -- score-level ensembling across encodings --
because it is pure post-processing on the scores that already exist.  The
prediction is that ensembling cannot close a blind spot: the information is
absent from every member that shares the invariance, so a max or mean over
members can only help if a member outside the invariance group is included.

Writes results/13_tables/*.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "07_eval"
SCO = ROOT / "results" / "06_scores"
SIG = ROOT / "results" / "09_signal"
REN = ROOT / "results" / "05_render"
DIAG = ROOT / "results" / "08_diagnose"
OUT = ROOT / "results" / "13_tables"
OUT.mkdir(parents=True, exist_ok=True)

ENC_ORDER = [f"E{i}" for i in range(1, 12)]
man = pd.read_csv(REN / "manifest.csv")


def order(df, col="encoding"):
    df = df.copy()
    df[col] = pd.Categorical(df[col], [e for e in ENC_ORDER if e in set(df[col])], ordered=True)
    return df.sort_values(col)


# ------------------------------------------------------------ T1 main table
pa = pd.read_csv(EVAL / "pa_percategory.csv")
main = pa[pa["category"] == "__MEAN_OF_CATEGORIES__"]
t1 = (main[main["k"] == "4"].groupby(["encoding", "detector"])["auroc"]
      .agg(["mean", "std"]).reset_index())
t1w = order(t1).pivot_table(index="encoding", columns="detector", values="mean", observed=True)
t1w.to_csv(OUT / "t1_main_k4.csv")

# ------------------------------------------------------------ T2 few-shot curve
# Average over the five seeds *first*, then take the best detector.  Taking a
# max over (detector, seed) would report the luckiest memory bank rather than
# the expected performance, and at k=1 that inflates the best encoding by 0.17.
t2 = (main.groupby(["encoding", "k", "detector"])["auroc"].mean().reset_index()
      .groupby(["encoding", "k"])["auroc"].max().reset_index()
      .pivot_table(index="k", columns="encoding", values="auroc", observed=True)
      .reindex(["1", "2", "4", "8", "16", "all"]))
t2.to_csv(OUT / "t2_fewshot_curve.csv")

# ------------------------------------------------------------ T3 per-category
t3 = (pa[(pa["k"] == "4") & (pa["category"] != "__MEAN_OF_CATEGORIES__")]
      .groupby(["encoding", "category"])["auroc"].mean().reset_index()
      .pivot_table(index="encoding", columns="category", values="auroc", observed=True))
t3.to_csv(OUT / "t3_per_category_k4.csv")

# ------------------------------------------------------------ T4 theory table
verd = pd.read_csv(EVAL / "pb_verdict.csv")
t4 = order(verd).pivot_table(index="encoding", columns="fault", values="verdict",
                             aggfunc="first", observed=True)
t4.to_csv(OUT / "t4_theory_verdicts.csv")
verd.to_csv(OUT / "t4_theory_verdicts_long.csv", index=False)

# ------------------------------------------------------------ T5 signal-domain
if (SIG / "r1_scores.parquet").exists() or (SIG / "r2_scores.parquet").exists():
    parts = [pd.read_parquet(p) for p in
             [SIG / "r1_scores.parquet", SIG / "r2_scores.parquet"] if p.exists()]
    ref = pd.concat(parts, ignore_index=True)
    ref = ref.merge(man[["row", "kind", "anomaly", "category", "fault", "param"]],
                    on="row", how="left")
    CATS = sorted(man.loc[(man["kind"] == "test") & man["anomaly"], "category"].unique())

    def auroc(y, s):
        o = np.argsort(s, kind="stable")
        r = np.empty(len(s))
        r[o] = np.arange(1, len(s) + 1)
        npos = int(y.sum())
        nneg = len(y) - npos
        if npos == 0 or nneg == 0:
            return np.nan
        return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    rows = []
    pa_ref = ref[ref["protocol"] == "P-A"]
    for (d, k, seed), g in pa_ref.groupby(["detector", "k", "seed"]):
        gn = g[~g["anomaly"].astype(bool)]
        vals = []
        for c in CATS:
            ga = g[g["anomaly"].astype(bool) & (g["category"] == c)]
            if not len(ga):
                continue
            y = np.r_[np.ones(len(ga), bool), np.zeros(len(gn), bool)]
            vals.append(auroc(y, np.r_[ga["score"].to_numpy(), gn["score"].to_numpy()]))
        rows.append({"detector": d, "k": k, "seed": seed, "auroc": float(np.nanmean(vals))})
    t5 = pd.DataFrame(rows).groupby(["detector", "k"])["auroc"].agg(["mean", "std"]).reset_index()
    t5.to_csv(OUT / "t5_signal_domain.csv", index=False)

    rows = []
    pb_ref = ref[ref["protocol"] == "P-B"]
    cl = pb_ref[pb_ref["kind"] == "holdout"]
    for (d, fault, param, seed), g in pb_ref[pb_ref["kind"] == "injected"].groupby(
            ["detector", "fault", "param", "seed"]):
        gc = cl[(cl["detector"] == d) & (cl["seed"] == seed)]
        y = np.r_[np.ones(len(g), bool), np.zeros(len(gc), bool)]
        rows.append({"detector": d, "fault": fault, "param": param, "seed": seed,
                     "auroc": auroc(y, np.r_[g["score"].to_numpy(), gc["score"].to_numpy()])})
    pd.DataFrame(rows).to_csv(OUT / "t5b_signal_domain_synthetic.csv", index=False)

# ------------------------------------------------------ T6 ensemble ablation
# Score-level fusion across encodings, per (detector, k, seed): z-score each
# member's scores over the evaluated set, then combine.
rows = []
enc_dirs = sorted(p.name for p in SCO.iterdir() if p.is_dir())
dets = sorted({f.stem for d in enc_dirs for f in (SCO / d).glob("M*.parquet")
               if not f.stem.endswith("_ext")})
GROUPS = {
    "all_encodings": enc_dirs,
    "affine_blind_only": [e for e in ["E1", "E3", "E4", "E5", "E10"] if e in enc_dirs],
    "blind_plus_control": [e for e in ["E1", "E3", "E4", "E5", "E10", "E2"] if e in enc_dirs],
}
CATS = sorted(man.loc[(man["kind"] == "test") & man["anomaly"], "category"].unique())


def auroc2(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


for det in dets:
    cache = {}
    for e in enc_dirs:
        f = SCO / e / f"{det}.parquet"
        if f.exists():
            cache[e] = pd.read_parquet(f)
    if not cache:
        continue
    for gname, members in GROUPS.items():
        members = [m for m in members if m in cache]
        if len(members) < 2:
            continue
        for seed in range(5):
            # synthetic protocol, where the blind spots live
            sub = {}
            for m in members:
                d = cache[m]
                d = d[(d["protocol"] == "P-B") & (d["seed"] == seed)
                      & (d["k"].astype(str) == "4")]
                sub[m] = d.set_index("row")["score"]
            idx = sub[members[0]].index
            Z = np.stack([((sub[m].loc[idx] - sub[m].mean()) / (sub[m].std() + 1e-12)).to_numpy()
                          for m in members])
            info = man.set_index("row").loc[idx]
            clean = (info["kind"] == "holdout").to_numpy()
            for agg, fn in (("max", Z.max(0)), ("mean", Z.mean(0))):
                for fault, gi in info.groupby("fault"):
                    if fault == "" or pd.isna(fault):
                        continue
                    m_pos = info["fault"].to_numpy() == fault
                    y = np.r_[np.ones(m_pos.sum(), bool), np.zeros(clean.sum(), bool)]
                    rows.append({"detector": det, "group": gname, "agg": agg,
                                 "seed": seed, "fault": fault, "n_members": len(members),
                                 "auroc": auroc2(y, np.r_[fn[m_pos], fn[clean]])})
if rows:
    t6 = pd.DataFrame(rows)
    t6.to_csv(OUT / "t6_ensemble.csv", index=False)
    print("\n=== T6 ensemble: AUROC by fault (mean over detectors and seeds) ===")
    print(t6.pivot_table(index="fault", columns=["group", "agg"], values="auroc").round(3).to_string())

print("\n=== T1 main table, k=4, mean-of-categories AUROC ===")
print(t1w.round(3).to_string())
print("\n=== T2 few-shot curve (best detector per encoding) ===")
print(t2.round(3).to_string())
print("\n=== T4 theory verdicts ===")
print(t4.to_string())
print(f"\n[done] tables -> {OUT}")
