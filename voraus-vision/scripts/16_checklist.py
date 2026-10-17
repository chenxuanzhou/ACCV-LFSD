"""Generate the invariance-aware encoding checklist (plan sec. 9 deliverable).

The take-away artefact for an engineer choosing how to render machine signals:
given a fault type they care about, which encodings cannot see it, and why.

Rows come from three sources, in decreasing order of strength:
  * byte-level invariance  (results/03_invariance) -- provable, detector-free
  * measured AUROC verdict (results/07_eval/pb_verdict.csv)
  * footprint controls     (results/08_diagnose/control_sweep_all.csv)

A cell is marked "blind (proved)" only when the rendered images are byte
identical, which no detector can undo.  A cell is marked "blind (measured)"
when the AUROC verdict says blind *and* the footprint-matched control passes.
Where the control fails the cell is marked "untestable", because an encoding
that cannot see any fault of that footprint tells us nothing specific.

Writes results/16_checklist/{checklist.csv,checklist.md}
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "results" / "03_invariance"
EVAL = ROOT / "results" / "07_eval"
DIAG = ROOT / "results" / "08_diagnose"
OUT = ROOT / "results" / "16_checklist"
OUT.mkdir(parents=True, exist_ok=True)

FAULT_LABEL = {
    "S1": "per-channel constant offset",
    "S2": "per-channel gain",
    "S3": "uniform scaling (raw)",
    "S3b": "uniform scaling (about the mean)",
    "S4": "single-channel time shift",
    "S4b": "global time shift",
    "S5": "cycle-duration drift",
    "S6": "local per-channel offset",
    "S7": "transient spike, 1 channel",
    "S8": "freeze / hold, 1 channel",
    "S7m": "transient spike, 8 channels",
    "S8m": "freeze / hold, 8 channels",
    "S7g": "transient spike, all channels",
    "S8g": "freeze / hold, all channels",
}
ENC_LABEL = {
    "E10": "signal heatmap, per-window min-max (28 ch)",
    "E11": "signal heatmap, train z-score (28 ch)",
    "E1": "signal heatmap, per-window min-max (129 ch)",
    "E2": "signal heatmap, train z-score (129 ch)",
    "E3": "GASF on PCA-3, per-window min-max",
    "E4": "GASF of 9 channels, 3x3 tiles",
    "E5": "recurrence plot, percentile threshold",
    "E6": "recurrence plot, continuous distance",
    "E7": "STFT log-magnitude, DC kept",
    "E8": "STFT log-magnitude, DC dropped",
    "E9": "GASF on PCA-3, train-global min-max",
}

inv = pd.read_csv(INV / "invariance_check.csv") if (INV / "invariance_check.csv").exists() \
    else pd.DataFrame()
V = ROOT / "results" / "21_verdicts" / "verdicts.csv"
verd = pd.read_csv(V) if V.exists() else pd.read_csv(EVAL / "pb_verdict.csv")
strict = set()
if len(inv):
    sub = inv[inv.get("channel_forced_into_e4_grid", True) == True]  # noqa: E712
    for r in sub.itertuples():
        if getattr(r, "strictly_blind", False) and not (
                r.fault == "S6" and float(r.param) == 1.0):
            strict.add((r.encoding, r.fault))

rows = []
for r in verd.itertuples():
    if (r.encoding, r.fault) in strict or getattr(r, "byte_identical", False):
        status, basis = "blind (proved)", "rendered images are byte-identical"
    elif r.verdict in ("blind",) and bool(r.control_passes):
        status, basis = "blind (measured)", "AUROC CI upper bound < 0.55 on every detector"
    elif r.verdict in ("blind", "untestable") and not bool(r.control_passes):
        status, basis = "untestable", (
            f"matched control ({r.matched_control}) only reaches "
            f"AUROC {r.matched_control_auroc:.2f}")
    else:
        status, basis = r.verdict, f"best AUROC {getattr(r, 'auroc_max_over_detectors', float('nan')):.2f}"
    rows.append({
        "encoding": r.encoding, "encoding_description": ENC_LABEL.get(r.encoding, ""),
        "fault": r.fault, "fault_description": FAULT_LABEL.get(r.fault, ""),
        "channels_affected": r.footprint_channels, "status": status, "basis": basis,
        "best_auroc": getattr(r, "auroc_max_over_detectors", np.nan),
        "matched_control": r.matched_control,
        "matched_control_auroc": r.matched_control_auroc,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "checklist.csv", index=False)

lines = [
    "# Invariance-aware encoding checklist",
    "",
    "Which renderings of multivariate machine signals cannot represent which faults.",
    "",
    "`blind (proved)` means the rendered images are byte-identical with and without",
    "the fault: no detector, backbone or amount of data can recover it. `blind",
    "(measured)` means every detector tested stayed at chance while the",
    "footprint-matched positive control passed. `untestable` means the encoding is",
    "insensitive to *any* fault of that spatial footprint, so nothing specific can be",
    "concluded about this fault -- which is itself a reason not to use that encoding",
    "for faults of that shape.",
    "",
]
for enc in [e for e in ENC_LABEL if e in set(df["encoding"])]:
    g = df[df["encoding"] == enc]
    lines += [f"## {enc} — {ENC_LABEL[enc]}", ""]
    bad = g[g["status"].isin(["blind (proved)", "blind (measured)"])]
    unt = g[g["status"] == "untestable"]
    if len(bad):
        lines.append("**Do not use for:** " + ", ".join(
            f"{FAULT_LABEL.get(f, f)} ({s.split(' ')[1]})"
            for f, s in zip(bad["fault"], bad["status"])) + ".")
    else:
        lines.append("**No fault in the suite is provably invisible to this encoding.**")
    if len(unt):
        lines.append("")
        lines.append("**Cannot be assessed** (the encoding misses every fault of that "
                     "footprint): " + ", ".join(FAULT_LABEL.get(f, f) for f in unt["fault"]) + ".")
    lines += ["", g[["fault", "fault_description", "channels_affected", "status",
                     "best_auroc"]].to_markdown(index=False), ""]

(OUT / "checklist.md").write_text("\n".join(lines), encoding="utf-8")
print(df.pivot_table(index="encoding", columns="fault", values="status",
                     aggfunc="first").to_string())
print(f"\n[done] -> {OUT}")
