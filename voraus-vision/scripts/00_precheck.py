"""Step 0 pre-checks for the SmallData-CV 2026 plan.

Verifies dataset facts that every downstream design decision depends on:
channel count/semantics, sampling rate, recording length distribution,
official train/test split, per-category test counts, and whether
time-interval ground truth for real anomalies exists.

Writes results/00_precheck/precheck.json and precheck.md.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref" / "voraus-ad-dataset"))

from voraus_ad import ANOMALY_CATEGORIES, Category, Signals, Variant  # noqa: E402

DATASET = ROOT / "data" / "voraus-ad-dataset-100hz.parquet"
OUT = ROOT / "results" / "00_precheck"
OUT.mkdir(parents=True, exist_ok=True)

rep = {}

# ---------------------------------------------------------------- schema
import pyarrow.parquet as pq  # noqa: E402

pf = pq.ParquetFile(DATASET)
schema_cols = list(pf.schema_arrow.names)
rep["parquet_num_rows"] = pf.metadata.num_rows
rep["parquet_num_columns"] = len(schema_cols)
rep["parquet_columns"] = schema_cols

meta_cols = list(Signals.meta())
machine_cols = list(Signals.machine())
rep["n_meta_columns"] = len(meta_cols)
rep["n_machine_columns"] = len(machine_cols)
rep["meta_columns"] = meta_cols
rep["machine_columns_missing_from_file"] = [c for c in machine_cols if c not in schema_cols]
rep["file_columns_not_in_signals_all"] = [c for c in schema_cols if c not in list(Signals.all())]
rep["signal_groups_sizes"] = {k: len(v) for k, v in Signals.groups().items()}

print(f"[schema] rows={rep['parquet_num_rows']:,}  cols={rep['parquet_num_columns']}")
print(f"[schema] machine channels = {len(machine_cols)}, meta = {len(meta_cols)}")

# ---------------------------------------------------------------- load meta only
print("[load] reading meta columns ...")
meta = pd.read_parquet(DATASET, columns=["time", "sample", "anomaly", "category", "setting", "action", "active"])
print(f"[load] meta frame {meta.shape}")

# sampling rate from the time column of one sample
s0 = meta[meta["sample"] == meta["sample"].iloc[0]]
dt = np.diff(s0["time"].values)
rep["time_column_dtype"] = str(meta["time"].dtype)
rep["median_dt_seconds"] = float(np.median(dt))
rep["inferred_sampling_rate_hz"] = float(1.0 / np.median(dt)) if np.median(dt) > 0 else None
rep["dt_unique_count_in_first_sample"] = int(len(np.unique(np.round(dt, 9))))

# ---------------------------------------------------------------- per-sample table
g = meta.groupby("sample")
per_sample = pd.DataFrame({
    "length": g.size(),
    "anomaly": g["anomaly"].first(),
    "category": g["category"].first(),
    "setting": g["setting"].first(),
    "active_sum": g["active"].sum(),
    "active_any": g["active"].any(),
    "n_actions": g["action"].nunique(),
})
per_sample["category_name"] = [Category(int(c)).name for c in per_sample["category"]]
per_sample["variant_name"] = [Variant(int(v)).name for v in per_sample["setting"]]
per_sample["split"] = np.where(per_sample["setting"] == int(Variant.PRE_A), "train", "test")

rep["n_samples_total"] = int(len(per_sample))
rep["length_distribution"] = {
    "min": int(per_sample["length"].min()),
    "p05": float(per_sample["length"].quantile(0.05)),
    "median": float(per_sample["length"].median()),
    "mean": float(per_sample["length"].mean()),
    "p95": float(per_sample["length"].quantile(0.95)),
    "max": int(per_sample["length"].max()),
    "std": float(per_sample["length"].std()),
}
rep["length_distribution_seconds"] = {
    k: (v / rep["inferred_sampling_rate_hz"] if isinstance(v, (int, float)) else v)
    for k, v in rep["length_distribution"].items()
}

# split composition
for split in ["train", "test"]:
    sub = per_sample[per_sample["split"] == split]
    rep[f"{split}_n_samples"] = int(len(sub))
    rep[f"{split}_n_anomalous"] = int(sub["anomaly"].sum())
    rep[f"{split}_n_normal"] = int((~sub["anomaly"].astype(bool)).sum())
    rep[f"{split}_category_counts"] = {k: int(v) for k, v in
                                       Counter(sub["category_name"]).most_common()}
    rep[f"{split}_length_min_median_max"] = [int(sub["length"].min()),
                                             float(sub["length"].median()),
                                             int(sub["length"].max())]

# per-category test counts + small-category flag
test = per_sample[per_sample["split"] == "test"]
cat_rows = []
for cat in ANOMALY_CATEGORIES:
    n = int((test["category_name"] == cat.name).sum())
    n_var = int(test.loc[test["category_name"] == cat.name, "variant_name"].nunique())
    cat_rows.append({"category": cat.name, "code": int(cat), "n_test": n,
                     "n_variants": n_var, "small_flag": n < 20})
rep["anomaly_categories"] = cat_rows
rep["n_test_normal_operation"] = int((test["category_name"] == "NORMAL_OPERATION").sum())
rep["small_categories"] = [r["category"] for r in cat_rows if r["small_flag"]]

# ---------------------------------------------------------------- time-interval GT probe
# `active` and `action` are time-varying meta columns. If `active` marks the
# anomalous interval, anomalous samples should have 0 < active_sum < length
# while normal samples should have a different pattern.
probe = {}
for lab, sub in [("anomalous", per_sample[per_sample["anomaly"].astype(bool)]),
                 ("normal", per_sample[~per_sample["anomaly"].astype(bool)])]:
    frac = (sub["active_sum"] / sub["length"]).astype(float)
    probe[lab] = {
        "n": int(len(sub)),
        "active_frac_min": float(frac.min()),
        "active_frac_median": float(frac.median()),
        "active_frac_max": float(frac.max()),
        "n_with_partial_active": int(((frac > 0) & (frac < 1)).sum()),
        "n_all_active": int((frac == 1).sum()),
        "n_none_active": int((frac == 0).sum()),
    }
rep["active_column_probe"] = probe
rep["action_unique_values"] = sorted(int(a) for a in meta["action"].unique())

# ---------------------------------------------------------------- constant-channel probe on train
print("[channels] computing per-channel variance on train split ...")
train_samples = per_sample.index[per_sample["split"] == "train"].tolist()
chunk = pd.read_parquet(DATASET, columns=machine_cols + ["sample"])
chunk = chunk[chunk["sample"].isin(train_samples)].drop(columns=["sample"])
var = chunk.var(axis=0)
nuniq = chunk.nunique(axis=0)
const_var = var[var < 1e-8].index.tolist()
const_uniq = nuniq[nuniq < 3].index.tolist()
rep["train_channel_variance"] = {"min": float(var.min()), "median": float(var.median()),
                                 "max": float(var.max())}
rep["constant_channels_by_variance"] = const_var
rep["constant_channels_by_nunique"] = const_uniq
rep["constant_channels_union"] = sorted(set(const_var) | set(const_uniq))
rep["n_channels_kept"] = len(machine_cols) - len(set(const_var) | set(const_uniq))
rep["train_var_smallest_10"] = {k: float(v) for k, v in var.nsmallest(10).items()}

with open(OUT / "precheck.json", "w") as f:
    json.dump(rep, f, indent=2)
per_sample.to_csv(OUT / "per_sample_meta.csv")
print(f"[done] wrote {OUT/'precheck.json'} and per_sample_meta.csv")
