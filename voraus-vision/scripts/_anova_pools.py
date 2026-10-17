"""Encoding/detector variance ratio under different encoding and detector pools.

Prints, for k in {2, 4, 8, 16}, the eta^2 of encoding, detector and their
interaction, and the encoding/detector ratio, for three encoding pools: all 11
encodings, all but E11, and the planned grid E1-E8; plus k=4 without the CLIP
detector (M6).  These are the pool-sensitivity ratios quoted in Sec. 5.2 of
the paper (33.5x without CLIP, 4.2x without E11, 3.9x on the planned grid).
Zero-shot rows (k="0") are excluded.  Reads results/07_eval/pa_percategory.csv;
writes nothing.
"""
from pathlib import Path

import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
pa = pd.read_csv(ROOT / "results" / "07_eval" / "pa_percategory.csv", dtype={"k": str})
main = pa[pa.category == "__MEAN_OF_CATEGORIES__"]
main = main[main.k != "0"]
def anova(df):
    gm = df.auroc.mean(); sst = ((df.auroc - gm) ** 2).sum()
    ne, nd = df.encoding.nunique(), df.detector.nunique(); nrep = len(df) / (ne * nd)
    em = df.groupby("encoding").auroc.mean(); dm = df.groupby("detector").auroc.mean()
    cell = df.groupby(["encoding", "detector"]).auroc.mean()
    sse = nd * nrep * ((em - gm) ** 2).sum(); ssd = ne * nrep * ((dm - gm) ** 2).sum()
    ssi = nrep * ((cell - gm) ** 2).sum() - sse - ssd
    return sse / sst, ssd / sst, ssi / sst, sse / ssd
print(sorted(main.detector.unique()))
allE = sorted(main.encoding.unique())
pools = {"full": allE, "no E11": [e for e in allE if e != "E11"], "planned E1-E8": [f"E{i}" for i in range(1, 9)]}
for name, encs in pools.items():
    for k in ["2", "4", "8", "16"]:
        d = main[(main.k == k) & main.encoding.isin(encs)]
        print(name, k, [round(v, 3) for v in anova(d)])
d = main[(main.k == "4") & (~main.detector.str.contains("M6"))]
print("no CLIP k=4", [round(v, 3) for v in anova(d)])
