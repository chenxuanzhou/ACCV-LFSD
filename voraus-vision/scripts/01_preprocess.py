"""Steps D1-D4 of the plan: channel selection, length normalisation,
train-only normalisation statistics, and frozen few-shot index files.

Outputs (all under results/01_preprocess/, everything downstream reads these
and never touches the parquet again):

    windows.npy        float32 (N, n_kept, L)   resampled windows, raw units
    raw_lengths.npy    int32   (N,)             original recording length
    meta.csv                                    per-window labels and split
    kept_channels.json                          D1 decision, frozen
    norm_stats.json                             D3 statistics, train only
    fewshot_indices.json                        D4, 5 seeds x k in {1,2,4,8,16,all}
    holdout_normal.json                         20% of train-normal, injection pool
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref" / "voraus-ad-dataset"))
from voraus_ad import Category, Signals, Variant  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "voraus-vision" / "configs" / "base.yaml"))
DATASET = ROOT / "data" / "voraus-ad-dataset-100hz.parquet"
OUT = ROOT / "results" / "01_preprocess"
OUT.mkdir(parents=True, exist_ok=True)

L = CFG["data"]["L"]
VAR_TH = CFG["data"]["var_threshold"]
NUNIQ_TH = CFG["data"]["nunique_threshold"]


def resample(arr: np.ndarray, n: int) -> np.ndarray:
    """Linear resample (T, C) -> (n, C)."""
    T = arr.shape[0]
    src = np.linspace(0.0, T - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None]
    return arr[i0] * (1 - w) + arr[i1] * w


machine_cols = list(Signals.machine())
print(f"[load] reading {len(machine_cols)} machine channels + meta ...")
df = pd.read_parquet(DATASET, columns=machine_cols + ["sample", "anomaly", "category", "setting"])
print(f"[load] {df.shape}")

sample_ids = df["sample"].to_numpy()
order = np.argsort(sample_ids, kind="stable")
df = df.iloc[order].reset_index(drop=True)
sample_ids = df["sample"].to_numpy()

# sample boundaries
bounds = np.flatnonzero(np.diff(sample_ids)) + 1
starts = np.concatenate([[0], bounds])
ends = np.concatenate([bounds, [len(sample_ids)]])
uniq = sample_ids[starts]
print(f"[split] {len(uniq)} recordings")

settings = df["setting"].to_numpy()[starts]
anomaly = df["anomaly"].to_numpy()[starts].astype(bool)
category = df["category"].to_numpy()[starts]
is_train = settings == int(Variant.PRE_A)

# ------------------------------------------------------------------ D1
print("[D1] per-channel variance on the train split ...")
M = df[machine_cols].to_numpy(dtype=np.float64)
train_rows = np.zeros(len(M), dtype=bool)
for s, e, tr in zip(starts, ends, is_train):
    if tr:
        train_rows[s:e] = True
tr_block = M[train_rows]
var = tr_block.var(axis=0)
nuniq = np.array([len(np.unique(tr_block[:, c])) for c in range(tr_block.shape[1])])
drop = (var < VAR_TH) | (nuniq < NUNIQ_TH)
kept_idx = np.flatnonzero(~drop)
kept_names = [machine_cols[i] for i in kept_idx]
print(f"[D1] dropped {int(drop.sum())} of {len(machine_cols)} channels -> {len(kept_idx)} kept")
json.dump(
    {
        "kept_channel_names": kept_names,
        "kept_channel_indices_in_machine": kept_idx.tolist(),
        "dropped_channel_names": [machine_cols[i] for i in np.flatnonzero(drop)],
        "var_threshold": VAR_TH,
        "nunique_threshold": NUNIQ_TH,
    },
    open(OUT / "kept_channels.json", "w"),
    indent=2,
)

# ------------------------------------------------------------------ D2
print(f"[D2] resampling every recording to L={L} ...")
N = len(uniq)
W = np.empty((N, len(kept_idx), L), dtype=np.float32)
raw_len = np.empty(N, dtype=np.int32)
Mk = M[:, kept_idx]
for i, (s, e) in enumerate(zip(starts, ends)):
    raw_len[i] = e - s
    W[i] = resample(Mk[s:e], L).T.astype(np.float32)
np.save(OUT / "windows.npy", W)
np.save(OUT / "raw_lengths.npy", raw_len)
print(f"[D2] windows {W.shape}  ({W.nbytes/1e6:.0f} MB)")

meta = pd.DataFrame({
    "idx": np.arange(N),
    "sample": uniq,
    "split": np.where(is_train, "train", "test"),
    "anomaly": anomaly,
    "category": [Category(int(c)).name for c in category],
    "variant": [Variant(int(v)).name for v in settings],
    "raw_length": raw_len,
})
meta.to_csv(OUT / "meta.csv", index=False)

# ------------------------------------------------------------------ D3
print("[D3] train-only normalisation statistics ...")
train_W = W[is_train]                                  # (n_tr, C, L)
flat = train_W.transpose(1, 0, 2).reshape(len(kept_idx), -1).astype(np.float64)
stats = {
    "channel_names": kept_names,
    "mean": flat.mean(axis=1).tolist(),
    "std": np.maximum(flat.std(axis=1), 1e-8).tolist(),
    "min": flat.min(axis=1).tolist(),
    "max": flat.max(axis=1).tolist(),
    "n_train_recordings": int(is_train.sum()),
}
json.dump(stats, open(OUT / "norm_stats.json", "w"), indent=2)

# ------------------------------------------------------------------ D4
print("[D4] few-shot index files ...")
train_idx = np.flatnonzero(is_train)
rng_holdout = np.random.default_rng(12345)
perm = rng_holdout.permutation(train_idx)
n_hold = int(round(CFG["data"]["holdout_frac"] * len(train_idx)))
holdout = np.sort(perm[:n_hold])
bank_pool = np.sort(perm[n_hold:])
json.dump(
    {
        "holdout_normal_indices": holdout.tolist(),
        "memory_bank_pool_indices": bank_pool.tolist(),
        "note": "few-shot samples are drawn from memory_bank_pool only; "
                "the two sets never overlap (plan sec. 3)",
    },
    open(OUT / "holdout_normal.json", "w"),
    indent=2,
)

fewshot = {}
for seed in CFG["data"]["seeds"]:
    rng = np.random.default_rng(seed)
    for k in CFG["data"]["k_shots"]:
        fewshot[f"k{k}_seed{seed}"] = sorted(
            int(v) for v in rng.choice(bank_pool, size=min(k, len(bank_pool)), replace=False)
        )
    fewshot[f"kall_seed{seed}"] = [int(v) for v in bank_pool]
json.dump(fewshot, open(OUT / "fewshot_indices.json", "w"), indent=2)

print(f"[done] train={int(is_train.sum())} (bank pool {len(bank_pool)}, holdout {len(holdout)}), "
      f"test={int((~is_train).sum())}")
