"""Corrected blind / partial / visible verdicts.

The planned rule (internal analysis plan, not published) declares a cell blind when the bootstrap
95% CI upper bound is below 0.55 on every detector.  With the holdout set this
study can afford -- 190 injected against 190 clean windows -- that rule is
unattainable by construction:

    SE(AUROC | H0) = sqrt((n1+n2+1) / (12 n1 n2)) = 0.0297
    upper bound at AUROC = 0.5  ->  0.5 + 1.96*0.0297 = 0.558  >  0.55

Every provably blind cell in this study returns AUROC exactly 0.5000 -- the
rendered images are byte-identical, so the two score sets are the same multiset
-- and still fails the planned test.  The threshold needed n >= 257 per
side, or should have been set at 0.56.  Both the planned verdict and a
corrected one are reported; the deviation is not hidden.

Three criteria, in decreasing strength:
  proved      byte-identical renderings (results/03_invariance) -- no detector,
              backbone or data budget can ever separate the classes
  blind       corrected empirical rule: CI upper < 0.56 on every detector, and
              the footprint-matched positive control passes
  untestable  the matched control fails, so a null says nothing specific

Writes results/21_verdicts/{verdicts.csv,power.json}
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
INV = ROOT / "results" / "03_invariance"
OUT = ROOT / "results" / "21_verdicts"
OUT.mkdir(parents=True, exist_ok=True)

PREREG_BLIND_UPPER = 0.55
CORRECTED_BLIND_UPPER = 0.56
VISIBLE_LOWER = 0.70
CONTROL_PASS = 0.90
FOOTPRINT = {"S1": 1, "S2": 1, "S4": 1, "S6": 1, "S7": 1, "S8": 1,
             "S7m": 8, "S8m": 8,
             "S3": 129, "S3b": 129, "S4b": 129, "S5": 129, "S7g": 129, "S8g": 129}
CONTROLS = {1: ["S7", "S8"], 8: ["S7m", "S8m"], 129: ["S7g", "S8g"]}

pb = pd.read_csv(EVAL / "pb_sweep.csv")
n_pos = int(pb["n_pos"].median())
n_neg = int(pb["n_neg"].median())
se = float(np.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg)))
json.dump({
    "n_pos": n_pos, "n_neg": n_neg, "se_auroc_under_null": se,
    "ci_half_width": 1.96 * se,
    "achievable_ci_upper_at_auroc_half": 0.5 + 1.96 * se,
    "prereg_threshold": PREREG_BLIND_UPPER,
    "prereg_threshold_attainable": bool(0.5 + 1.96 * se < PREREG_BLIND_UPPER),
    "n_per_side_needed_for_prereg": int(np.ceil(
        2 / (12 * ((PREREG_BLIND_UPPER - 0.5) / 1.96) ** 2))),
    "corrected_threshold": CORRECTED_BLIND_UPPER,
}, open(OUT / "power.json", "w"), indent=2)
print(f"[power] n_pos={n_pos} n_neg={n_neg}  SE={se:.4f}  "
      f"achievable CI upper at AUROC 0.5 = {0.5 + 1.96*se:.4f}")
print(f"[power] planned threshold {PREREG_BLIND_UPPER} is "
      f"{'attainable' if 0.5 + 1.96*se < PREREG_BLIND_UPPER else 'NOT attainable'}; "
      f"needs n >= {int(np.ceil(2 / (12 * ((PREREG_BLIND_UPPER - 0.5) / 1.96) ** 2)))} per side")

proved = set()
p = INV / "invariance_check.csv"
if p.exists():
    inv = pd.read_csv(p)
    if "channel_forced_into_e4_grid" in inv.columns:
        inv = inv[inv["channel_forced_into_e4_grid"]]
    for r in inv.itertuples():
        if getattr(r, "strictly_blind", False) and not (
                r.fault == "S6" and float(r.param) == 1.0):
            proved.add((r.encoding, r.fault))
print(f"[proved] {len(proved)} byte-identical (encoding, fault) cells")

ctrl = (pb[pb["fault"].isin(sum(CONTROLS.values(), []))]
        .groupby(["encoding", "fault"])["auroc"].mean().reset_index())

rows = []
for (enc, fault), g in pb.groupby(["encoding", "fault"]):
    per_det = g.groupby("detector").agg(hi=("ci_hi", "max"), lo=("ci_lo", "max"),
                                        a=("auroc", "max"))
    fp = FOOTPRINT.get(fault)
    names = CONTROLS.get(fp, [])
    cg = ctrl[(ctrl["encoding"] == enc) & (ctrl["fault"].isin(names))]
    ctrl_auroc = float(cg["auroc"].max()) if len(cg) else np.nan
    ctrl_ok = bool(ctrl_auroc >= CONTROL_PASS) if not np.isnan(ctrl_auroc) else False
    prereg = bool((per_det["hi"] < PREREG_BLIND_UPPER).all())
    corrected = bool((per_det["hi"] < CORRECTED_BLIND_UPPER).all())
    visible = bool((per_det["lo"] > VISIBLE_LOWER).any())
    if (enc, fault) in proved:
        verdict = "proved blind"
    elif corrected and ctrl_ok:
        verdict = "blind"
    elif corrected and not ctrl_ok:
        verdict = "untestable"
    elif visible:
        verdict = "visible"
    else:
        verdict = "partial"
    rows.append({
        "encoding": enc, "fault": fault, "footprint_channels": fp,
        "verdict": verdict, "auroc_max_over_detectors": float(per_det["a"].max()),
        "auroc_mean": float(g["auroc"].mean()),
        "max_ci_hi": float(per_det["hi"].max()),
        "prereg_blind": prereg, "corrected_blind": corrected,
        "matched_control": "+".join(names), "matched_control_auroc": ctrl_auroc,
        "control_passes": ctrl_ok,
        "byte_identical": (enc, fault) in proved,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "verdicts.csv", index=False)
order = [f"E{i}" for i in range(1, 12)]
df["encoding"] = pd.Categorical(df["encoding"], [e for e in order if e in set(df["encoding"])],
                                ordered=True)
piv = df.sort_values("encoding").pivot_table(index="encoding", columns="fault",
                                             values="verdict", aggfunc="first",
                                             observed=True)
cols = [c for c in ["S1", "S2", "S3", "S3b", "S4", "S4b", "S5", "S6",
                    "S7", "S7m", "S7g", "S8", "S8m", "S8g"] if c in piv.columns]
piv = piv[cols]
piv.to_csv(OUT / "verdict_matrix.csv")
print()
print(piv.to_string())
print()
print(df["verdict"].value_counts().to_string())
print(f"\n[note] planned rule would certify {int(df['prereg_blind'].sum())} cells; "
      f"corrected rule certifies {int(df['corrected_blind'].sum())}; "
      f"byte-identity proves {int(df['byte_identical'].sum())}")
