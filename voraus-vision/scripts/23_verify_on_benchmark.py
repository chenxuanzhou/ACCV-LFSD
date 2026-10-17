"""Verify byte-identity on the released benchmark itself, not on a sample.

results/03_invariance checks 8 windows.  The blindness claim is the paper's
central result, so it is re-checked here against every rendered image that was
actually scored: for each (encoding, fault, param) cell, compare each injected
image with the clean render of the same source window.

This also catches floating-point cases the 8-window probe misses -- E10 x S3b,
for instance, scores AUROC 0.500042 rather than exactly 0.5, which can only come
from a rounding difference in a handful of pixels.

Writes results/23_benchmark_verify/byte_identity.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
REN = ROOT / "results" / "05_render"
OUT = ROOT / "results" / "23_benchmark_verify"
OUT.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(REN / "manifest.csv")
clean_row = dict(zip(man.loc[man["kind"] == "holdout", "src_idx"],
                     man.loc[man["kind"] == "holdout", "row"]))
inj = man[man["kind"] == "injected"]

encs = sorted((p.name for p in REN.iterdir() if p.is_dir() and (p / "images.npy").exists()),
              key=lambda e: int(e[1:]))
rows = []
for enc in encs:
    imgs = np.load(REN / enc / "images.npy", mmap_mode="r")
    for (fault, param), g in inj.groupby(["fault", "param"]):
        n_ident = 0
        max_diff = 0
        n_diff_px = 0
        for r in g.itertuples():
            a = np.asarray(imgs[clean_row[r.src_idx]])
            b = np.asarray(imgs[r.row])
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            if d.max() == 0:
                n_ident += 1
            else:
                max_diff = max(max_diff, int(d.max()))
                n_diff_px += int((d > 0).sum())
        rows.append({
            "encoding": enc, "fault": fault, "param": param, "n_windows": len(g),
            "n_byte_identical": n_ident,
            "frac_byte_identical": n_ident / len(g),
            "max_abs_diff_u8": max_diff,
            "mean_differing_pixels_per_nonidentical":
                (n_diff_px / max(len(g) - n_ident, 1)),
        })
    print(f"[verify] {enc} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "byte_identity.csv", index=False)

full = df[df["frac_byte_identical"] == 1.0]
near = df[(df["frac_byte_identical"] < 1.0) & (df["frac_byte_identical"] > 0.5)]
print(f"\n{len(full)} of {len(df)} (encoding, fault, param) cells are byte-identical "
      f"on ALL {int(df['n_windows'].iloc[0])} windows")
print("\n=== cells identical on every window (aggregated over the severity sweep) ===")
piv = df.pivot_table(index="encoding", columns="fault", values="frac_byte_identical",
                     aggfunc="min")
cols = [c for c in ["S1", "S2", "S3", "S3b", "S4", "S4b", "S5", "S6",
                    "S7", "S8"] if c in piv.columns]
piv = piv.reindex([e for e in encs if e in piv.index])[cols]
print(piv.round(3).to_string())
if len(near):
    print("\n=== cells identical on most but not all windows (floating-point edge) ===")
    print(near[["encoding", "fault", "param", "frac_byte_identical", "max_abs_diff_u8",
                "mean_differing_pixels_per_nonidentical"]].round(4).to_string(index=False))
