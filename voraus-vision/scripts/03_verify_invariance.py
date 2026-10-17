"""Numerically verify that the *implemented* encoders have the invariances
that P1-P4 claim.  This is a correctness gate, not an experiment: if an encoder
is not invariant where the proposition says it is, every downstream AUROC on
that cell measures an implementation artefact instead of the invariance.

Design notes, all of which came out of a first pass that produced misleading
"mismatches":

* S6 is checked at rho = 0.25.  At rho = 1.0 it degenerates into S1 (a
  full-window offset), so a "blind" verdict there says nothing new.
* S4/S4b are checked at tau = 2 and tau = 20.  P3's approximation argument only
  claims invariance for tau << nperseg (= 64), so both regimes are needed.
* E4 renders only 9 of the 129 kept channels.  Injections are therefore forced
  into a rendered channel; otherwise "blind" only means "the fault was not
  drawn".  The unforced case is reported separately as ``coverage_miss``.
* S5 is applied to the raw-length recording, not to the resampled window.

Writes results/03_invariance/invariance_check.csv and .json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "voraus-vision"))
from voraus_vision.context import load_context               # noqa: E402
from voraus_vision.encodings import ENCODERS                 # noqa: E402
from voraus_vision.injection import inject, inject_duration    # noqa: E402

PRE = ROOT / "results" / "01_preprocess"
CTX = ROOT / "results" / "02_context"
OUT = ROOT / "results" / "03_invariance"
OUT.mkdir(parents=True, exist_ok=True)

c = np.load(CTX / "context.npz")
ctx = load_context(ROOT)
RENDERED = list(c["top_var_channels"])

W = np.load(PRE / "windows.npy", mmap_mode="r")
meta = pd.read_csv(PRE / "meta.csv")
hold = json.load(open(PRE / "holdout_normal.json"))["holdout_normal_indices"]
raw = np.load(PRE / "holdout_raw.npz")
N_CHECK = 8
idxs = hold[:N_CHECK]
samples = meta.set_index("idx").loc[idxs, "sample"].tolist()

SINGLE_CHANNEL = {"S1", "S2", "S4", "S6", "S7", "S8"}
CHECKS = [
    ("S1", 5.0), ("S2", 0.25), ("S3", 0.25), ("S3b", 0.25),
    ("S4", 2), ("S4", 20), ("S4b", 2), ("S4b", 20),
    ("S6", 0.25), ("S6", 1.0), ("S7", 1.0), ("S8", 1.0),
]


def resample_T(arr, n):
    T = arr.shape[1]
    src = np.linspace(0.0, T - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = src - i0
    return arr[:, i0] * (1 - w) + arr[:, i1] * w


MAD_TOL = 1.0   # uint8 levels; below this the perturbation is under one quantisation step


def verdict(frac_changed: float, mad: float) -> str:
    """Three-level encoder-side verdict.

    ``strict``    byte-identical output: the fault lies in the invariance group.
    ``numerical`` output differs, but by less than one uint8 level on average --
                  the residual is discretisation/interpolation noise, not signal.
    ``visible``   the fault survives the encoding.
    """
    if frac_changed == 0.0:
        return "strict"
    if mad < MAD_TOL:
        return "numerical"
    return "visible"


class ForcedRNG:
    """Wraps a Generator so that the injected channel is always a rendered one.

    Both draw paths must be intercepted.  ``inject`` picks a single channel with
    ``integers`` for S1/S2/S4/S6 but with ``choice`` for the S7/S8 control
    family; overriding only ``integers`` silently let those two fall through to
    the base generator, so the "forced" run was not forcing anything and E4 x
    S7/S8 looked strictly blind when it was really just missing the nine
    rendered channels 93% of the time.
    """

    def __init__(self, base, forced):
        self._base = base
        self._forced = forced
        self._first = True

    def integers(self, *a, **kw):
        if self._first:
            self._first = False
            return self._forced
        return self._base.integers(*a, **kw)

    def choice(self, a, size=None, replace=True, **kw):
        if self._first and (size is None or size == 1):
            self._first = False
            return np.array([self._forced])
        return self._base.choice(a, size=size, replace=replace, **kw)

    def __getattr__(self, name):
        return getattr(self._base, name)


rows = []
for enc_id, fn in ENCODERS.items():
    for fault, param in CHECKS:
        for forced in ([True, False] if fault in SINGLE_CHANNEL else [True]):
            diffs, changed = [], []
            for j, i in enumerate(idxs):
                X = np.asarray(W[i], dtype=np.float64)
                base = np.random.default_rng(1000 + j)
                rng = ForcedRNG(base, int(RENDERED[j % len(RENDERED)])) if forced else base
                Y, _ = inject(X, fault, param, ctx.ch_std, ctx.ch_mean, rng)
                d = np.abs(fn(X, ctx).astype(np.int16) - fn(Y, ctx).astype(np.int16))
                diffs.append(d.mean())
                changed.append((d > 0).mean())
            rows.append({
                "encoding": enc_id, "fault": fault, "param": param,
                "channel_forced_into_e4_grid": forced,
                "mean_abs_diff_u8": float(np.mean(diffs)),
                "frac_pixels_changed": float(np.mean(changed)),
                "strictly_blind": bool(np.mean(changed) == 0.0),
                "verdict": verdict(np.mean(changed), np.mean(diffs)),
            })

# ------------------------------------------------------------------ S5
for enc_id, fn in ENCODERS.items():
    for gamma in (1.05, 1.1):
        diffs, changed = [], []
        for s in samples:
            R = np.asarray(raw[str(int(s))], dtype=np.float64)      # (C, T) native length
            X = resample_T(R, W.shape[2])
            Y = resample_T(inject_duration(R, gamma), W.shape[2])
            d = np.abs(fn(X, ctx).astype(np.int16) - fn(Y, ctx).astype(np.int16))
            diffs.append(d.mean())
            changed.append((d > 0).mean())
        rows.append({
            "encoding": enc_id, "fault": "S5", "param": gamma,
            "channel_forced_into_e4_grid": True,
            "mean_abs_diff_u8": float(np.mean(diffs)),
            "frac_pixels_changed": float(np.mean(changed)),
            "strictly_blind": bool(np.mean(changed) == 0.0),
            "verdict": verdict(np.mean(changed), np.mean(diffs)),
        })

df = pd.DataFrame(rows)

# Predicted-blind table, revised against the first pass (see report sec. "theory
# corrections").  Only cells where the *forced* variant is used are scored.
PREDICTED_BLIND = {
    ("E1", "S1"), ("E1", "S2"), ("E1", "S3"), ("E1", "S3b"),
    ("E3", "S1"), ("E3", "S3"), ("E3", "S3b"),
    ("E4", "S1"), ("E4", "S2"), ("E4", "S3"), ("E4", "S3b"),
    ("E5", "S1"), ("E5", "S3"), ("E5", "S3b"),
    ("E6", "S1"),                       # revised: the raw distance matrix is
                                        # translation-invariant too, only the
                                        # scale invariance is broken by E6
}
# P4: every encoding is blind to duration drift (up to resampling residual).
for e in ENCODERS:
    PREDICTED_BLIND.add((e, "S5"))

scored = df[df["channel_forced_into_e4_grid"]].copy()
# S6 at rho=1.0 degenerates to S1; only rho<1 is informative about locality.
scored = scored[~((scored["fault"] == "S6") & (scored["param"] == 1.0))]
scored["predicted_blind"] = [(r.encoding, r.fault) in PREDICTED_BLIND
                             for r in scored.itertuples()]
# P1-P3 claim *exact* invariance, so they are scored against byte-identity.
# Mean-absolute-difference is reported alongside but is deliberately not used as
# a detectability proxy: a 5-sample spike on 1 of 129 channels moves very few
# pixels yet may be perfectly detectable, while a diffuse shift moves every
# pixel and may not be.  Detectability is settled by the AUROC sweep (P-B), not
# here.  P4/S5 is inherently approximate in a discrete pipeline and is scored
# separately by the size of the resampling residual.
is_s5 = scored["fault"] == "S5"
scored["observed_blind"] = np.where(is_s5,
                                    scored["verdict"].isin(["strict", "numerical"]),
                                    scored["verdict"] == "strict")
scored["agrees"] = scored["predicted_blind"] == scored["observed_blind"]

df.to_csv(OUT / "invariance_check.csv", index=False)
scored.to_csv(OUT / "invariance_check_scored.csv", index=False)

mismatch = scored[~scored["agrees"]]
json.dump(
    {
        "n_cells_total": int(len(df)),
        "n_cells_scored": int(len(scored)),
        "verdict_counts": scored["verdict"].value_counts().to_dict(),
        "n_predicted_blind": int(scored["predicted_blind"].sum()),
        "n_mismatch": int(len(mismatch)),
        "mad_tolerance_u8": MAD_TOL,
        "mismatches": mismatch[["encoding", "fault", "param", "predicted_blind",
                                "verdict", "frac_pixels_changed",
                                "mean_abs_diff_u8"]].to_dict("records"),
        "e4_coverage_effect": df[(df["encoding"] == "E4") &
                                 (df["fault"].isin(["S7", "S8"]))]
        [["fault", "channel_forced_into_e4_grid", "frac_pixels_changed"]].to_dict("records"),
    },
    open(OUT / "invariance_check.json", "w"),
    indent=2,
)

print("mean |diff| in uint8 levels:")
print(scored.pivot_table(index="encoding", columns=["fault", "param"],
                         values="mean_abs_diff_u8").round(3).to_string())
print("\nverdict (strict / numerical / visible):")
print(scored.pivot_table(index="encoding", columns=["fault", "param"],
                         values="verdict", aggfunc="first").to_string())
print(f"\n[done] {len(scored)} scored cells; "
      f"{scored['verdict'].value_counts().to_dict()}; "
      f"{len(mismatch)} disagree with the propositions")
if len(mismatch):
    print(mismatch[["encoding", "fault", "param", "predicted_blind", "verdict",
                    "frac_pixels_changed", "mean_abs_diff_u8"]].to_string(index=False))
