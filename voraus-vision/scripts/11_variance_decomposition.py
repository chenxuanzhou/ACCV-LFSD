"""Two-factor ANOVA over the main matrix: how much of the spread in AUROC is
attributable to the encoding versus to the detector.

This is where the abstract's "encoding accounts for [X]x the variance of
detector" number comes from, so it is computed, never estimated.  The design is
balanced (every encoding x detector x seed cell is present), which is what makes
the plain sum-of-squares partition valid without a model-fitting library.

Reported per k, because the answer can legitimately change with sample budget:
at k=1 every detector is starved and the encoding may dominate trivially.

Writes results/11_anova/{anova_by_k,cell_means}.csv
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "07_eval"
OUT = ROOT / "results" / "11_anova"
OUT.mkdir(parents=True, exist_ok=True)

pa = pd.read_csv(EVAL / "pa_percategory.csv")
main = pa[pa["category"] == "__MEAN_OF_CATEGORIES__"].copy()


def two_way(df, a="encoding", b="detector", y="auroc"):
    """Balanced two-factor ANOVA with replication.  Returns the SS partition."""
    la = sorted(df[a].unique())
    lb = sorted(df[b].unique())
    cells = df.groupby([a, b])[y].apply(list)
    n = min(len(v) for v in cells)                # replications per cell
    M = np.array([[cells.loc[(i, j)][:n] for j in lb] for i in la])   # (A,B,n)
    grand = M.mean()
    A, B = len(la), len(lb)
    ss_a = B * n * ((M.mean(axis=(1, 2)) - grand) ** 2).sum()
    ss_b = A * n * ((M.mean(axis=(0, 2)) - grand) ** 2).sum()
    cell = M.mean(axis=2)
    ss_ab = n * ((cell - M.mean(axis=(1, 2))[:, None]
                  - M.mean(axis=(0, 2))[None, :] + grand) ** 2).sum()
    ss_err = ((M - cell[:, :, None]) ** 2).sum()
    ss_tot = ((M - grand) ** 2).sum()
    df_a, df_b = A - 1, B - 1
    df_ab, df_err = df_a * df_b, A * B * (n - 1)
    ms_err = ss_err / max(df_err, 1)
    return {
        "n_encodings": A, "n_detectors": B, "n_reps": n,
        "ss_encoding": ss_a, "ss_detector": ss_b, "ss_interaction": ss_ab,
        "ss_error": ss_err, "ss_total": ss_tot,
        "eta2_encoding": ss_a / ss_tot, "eta2_detector": ss_b / ss_tot,
        "eta2_interaction": ss_ab / ss_tot, "eta2_error": ss_err / ss_tot,
        "variance_ratio_encoding_over_detector": ss_a / ss_b if ss_b > 0 else np.inf,
        "F_encoding": (ss_a / df_a) / ms_err if ms_err > 0 else np.inf,
        "F_detector": (ss_b / df_b) / ms_err if ms_err > 0 else np.inf,
        "df_encoding": df_a, "df_detector": df_b, "df_error": df_err,
    }


rows = []
for k, g in main.groupby("k"):
    if g["encoding"].nunique() < 2 or g["detector"].nunique() < 2:
        continue
    r = two_way(g)
    r["k"] = k
    rows.append(r)
# k="all" uses the entire bank pool, so its five "seeds" are the same memory
# bank and the within-cell variance is exactly zero.  The F ratios and the
# pooled decomposition are meaningless once that row is included, so the pooled
# fit excludes it and the degeneracy is recorded rather than silently averaged.
r = two_way(main[main["k"] != "all"])
r["k"] = "pooled_excl_all"
rows.append(r)
r = two_way(main)
r["k"] = "pooled_incl_all_DEGENERATE"
rows.append(r)

anova = pd.DataFrame(rows)
cols = ["k", "n_encodings", "n_detectors", "n_reps", "eta2_encoding", "eta2_detector",
        "eta2_interaction", "eta2_error", "variance_ratio_encoding_over_detector",
        "F_encoding", "F_detector", "ss_encoding", "ss_detector", "ss_interaction",
        "ss_error", "ss_total", "df_encoding", "df_detector", "df_error"]
anova[cols].to_csv(OUT / "anova_by_k.csv", index=False)

cell = (main.groupby(["encoding", "detector", "k"])["auroc"]
        .agg(["mean", "std", "count"]).reset_index())
cell.to_csv(OUT / "cell_means.csv", index=False)

print("=== variance decomposition of mean-of-categories AUROC ===")
print("(k=all has zero within-cell variance: all five seeds share the full bank)")
print(anova[["k", "eta2_encoding", "eta2_detector", "eta2_interaction",
             "variance_ratio_encoding_over_detector"]].round(4).to_string(index=False))
print("\n=== cell means at k=4 ===")
print(main[main["k"] == "4"].pivot_table(index="encoding", columns="detector",
                                         values="auroc").round(3).to_string())

# ------------------------------------------------ BH correction, per-category
per_cat = pa[pa["category"] != "__MEAN_OF_CATEGORIES__"]
best = (per_cat[per_cat["k"] == "4"].groupby(["encoding", "category"])["auroc"]
        .agg(["mean", "std", "count"]).reset_index())
# One-sided test that a cell beats chance, then Benjamini-Hochberg over the
# 12 categories x N encodings family (plan sec. 8).
from scipy import stats                                            # noqa: E402
best["t"] = (best["mean"] - 0.5) / (best["std"] / np.sqrt(best["count"]))
best["p"] = 1 - stats.t.cdf(best["t"], df=best["count"] - 1)
o = np.argsort(best["p"].to_numpy())
m = len(best)
q = np.empty(m)
q[o] = np.minimum.accumulate((best["p"].to_numpy()[o] * m / np.arange(1, m + 1))[::-1])[::-1]
best["q_bh"] = q
best["significant_above_chance"] = best["q_bh"] < 0.05
best.to_csv(OUT / "per_category_bh.csv", index=False)
print(f"\n[BH] {int(best['significant_above_chance'].sum())} of {m} "
      f"(encoding, category) cells beat chance at q<0.05")
json.dump({"family_size": int(m)}, open(OUT / "bh_meta.json", "w"), indent=2)
