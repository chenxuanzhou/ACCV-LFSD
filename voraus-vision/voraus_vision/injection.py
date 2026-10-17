"""Programmatic fault-injection suite: S1-S8, S3b, S4b and the footprint
variants S7m/S8m/S7g/S8g of the spike and hold controls.

Injections act on held-out *normal* windows only.  The injection parameters are
the ground truth: which channels were touched, over which sample range, and how
strongly.  No manual annotation is involved anywhere.

Two additions to the plan, both marked ``added``:

* ``S4b`` shifts every channel by the same tau.  P3 is exactly the statement
  that ``|DFT|`` is invariant to a *global* circular shift; S4 shifts a single
  channel, which after the PCA-3 projection is a channel-mixing perturbation
  and therefore only an approximate instance of P3.  S4b makes P3 testable in
  its exact form and S4 becomes the practical-fault variant.
* ``S3b`` applies a uniform scaling about the per-channel train mean.  S3 as
  written multiplies raw values, which for a channel with a large offset (e.g.
  a 48 V supply rail) is dominated by the induced offset rather than by the
  scaling.  S3b isolates the scaling itself.

S5 (duration drift) is special: it is applied to the raw recording *before*
fixed-length resampling, because that is the only place where the information
still exists.  After resampling, the injected window is numerically identical
to the clean one up to interpolation error -- which is precisely P4.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

FS = 100.0  # Hz


def injection_seed(fault: str, param: float, src_idx: int) -> int:
    """Deterministic seed for one (fault, param, source window) injection.

    Must not use the builtin ``hash``: CPython salts string hashing per process,
    so rendering E1 in one process and E2 in another would silently draw
    different injected channels for the same manifest row -- different encodings
    would then be compared on different data, and the recorded ground truth
    would match only whichever process happened to write it.
    """
    key = f"{fault}|{param:.6f}|{src_idx}".encode()
    return zlib.crc32(key) & 0x7FFFFFFF

FAULT_PARAMS: Dict[str, List[float]] = {
    "S1": [0.25, 0.5, 1.0, 2.0, 5.0],       # alpha, single-channel constant offset
    "S2": [0.02, 0.05, 0.1, 0.25],          # beta,  single-channel gain
    "S3": [0.02, 0.05, 0.1, 0.25],          # beta,  all-channel uniform scaling (raw)
    "S3b": [0.02, 0.05, 0.1, 0.25],         # added: uniform scaling about train mean
    "S4": [2, 5, 10, 20],                   # tau,   single-channel circular shift
    "S4b": [2, 5, 10, 20],                  # added: global circular shift
    "S5": [0.9, 0.95, 1.05, 1.1],           # gamma, overall duration scaling
    "S6": [0.05, 0.1, 0.25, 0.5, 1.0],      # rho,   local offset duration fraction
    "S7": [1.0],                            # transient spike, 1 channel   (control)
    "S8": [1.0],                            # freeze / hold,   1 channel   (control)
    "S7m": [1.0],                           # added: spike,  8 channels    (control)
    "S8m": [1.0],                           # added: freeze, 8 channels    (control)
    "S7g": [1.0],                           # added: spike,  all channels  (control)
    "S8g": [1.0],                           # added: freeze, all channels  (control)
}

# A positive control is only informative if its spatial footprint matches the
# fault it is controlling for.  Rendering 129 channels into 224 image rows gives
# each channel ~1.7 pixel rows, and results/08_diagnose/control_sweep_all.csv shows
# that a single-channel spike stays near chance even at 100 sigma while an
# all-channel spike reaches AUROC 1.000 at 3 sigma.  Sensitivity is governed by
# how many channels a fault touches, not by its severity, so single-channel
# controls cannot validate the pipeline for global faults and vice versa.
POSITIVE_CONTROLS = ("S7", "S8", "S7m", "S8m", "S7g", "S8g")
CONTROL_FOOTPRINT = {"S7": 1, "S8": 1, "S7m": 8, "S8m": 8, "S7g": None, "S8g": None}
# Which control each fault must be read against (matched footprint).
FAULT_FOOTPRINT = {
    "S1": 1, "S2": 1, "S6": 1, "S7": 1, "S8": 1,
    "S7m": 8, "S8m": 8,
    "S3": None, "S3b": None, "S4b": None, "S5": None, "S7g": None, "S8g": None,
    "S4": 1,
}
# Faults that must be applied before fixed-length resampling.
PRE_RESAMPLE_FAULTS = ("S5",)


@dataclass
class InjectionGT:
    """Exact ground truth produced by one injection."""

    fault: str
    param: float
    channels: List[int] = field(default_factory=list)  # indices into the kept-channel list
    t_start: int = 0                                   # inclusive, in resampled samples
    t_end: int = 0                                     # exclusive
    global_fault: bool = False                         # True when all channels are affected

    def as_row(self) -> dict:
        return {
            "fault": self.fault,
            "param": self.param,
            "channels": ";".join(str(c) for c in self.channels),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "global_fault": self.global_fault,
        }


def inject(
    X: np.ndarray,
    fault: str,
    param: float,
    ch_std: np.ndarray,
    ch_mean: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, InjectionGT]:
    """Apply one post-resampling fault to ``X`` of shape ``(n_ch, L)``.

    ``ch_std`` / ``ch_mean`` are the frozen train statistics used to make the
    severity parameters dimensionless.
    """
    n_ch, L = X.shape
    Y = X.copy()

    if fault == "S1":
        c = int(rng.integers(n_ch))
        Y[c] += param * ch_std[c]
        return Y, InjectionGT("S1", param, [c], 0, L)

    if fault == "S2":
        c = int(rng.integers(n_ch))
        Y[c] *= 1.0 + param
        return Y, InjectionGT("S2", param, [c], 0, L)

    if fault == "S3":
        Y *= 1.0 + param
        return Y, InjectionGT("S3", param, list(range(n_ch)), 0, L, global_fault=True)

    if fault == "S3b":
        Y = ch_mean[:, None] + (1.0 + param) * (Y - ch_mean[:, None])
        return Y, InjectionGT("S3b", param, list(range(n_ch)), 0, L, global_fault=True)

    if fault == "S4":
        c = int(rng.integers(n_ch))
        Y[c] = np.roll(Y[c], int(param))
        return Y, InjectionGT("S4", param, [c], 0, L)

    if fault == "S4b":
        Y = np.roll(Y, int(param), axis=1)
        return Y, InjectionGT("S4b", param, list(range(n_ch)), 0, L, global_fault=True)

    if fault == "S6":
        c = int(rng.integers(n_ch))
        span = max(1, int(round(param * L)))
        t0 = 0 if span >= L else int(rng.integers(0, L - span))
        Y[c, t0:t0 + span] += 1.0 * ch_std[c]
        return Y, InjectionGT("S6", param, [c], t0, t0 + span)

    if fault in ("S7", "S7m", "S7g"):       # positive control: transient spike
        n_hit = {"S7": 1, "S7m": 8, "S7g": n_ch}[fault]
        chans = rng.choice(n_ch, size=n_hit, replace=False)
        w = 5
        t0 = int(rng.integers(0, L - w))
        for c in chans:
            Y[c, t0:t0 + w] += 3.0 * ch_std[c]
        return Y, InjectionGT(fault, param, sorted(int(c) for c in chans),
                              t0, t0 + w, global_fault=(n_hit == n_ch))

    if fault in ("S8", "S8m", "S8g"):       # positive control: freeze / hold
        n_hit = {"S8": 1, "S8m": 8, "S8g": n_ch}[fault]
        chans = rng.choice(n_ch, size=n_hit, replace=False)
        w = int(round(0.5 * FS))            # 50 of the 512 resampled samples (~1 s of machine time)
        t0 = int(rng.integers(0, L - w))
        for c in chans:
            Y[c, t0:t0 + w] = Y[c, t0]
        return Y, InjectionGT(fault, param, sorted(int(c) for c in chans),
                              t0, t0 + w, global_fault=(n_hit == n_ch))

    raise ValueError(f"{fault} is not a post-resampling fault")


def inject_duration(raw: np.ndarray, gamma: float) -> np.ndarray:
    """S5: rescale the *raw* recording duration by ``gamma`` before resampling.

    Returns an array of length ``round(T * gamma)``.  Every encoder in the study
    consumes fixed-length windows, so this information cannot survive; the
    measured AUROC on S5 is a direct read-out of P4 plus interpolation noise.
    """
    n_ch, T = raw.shape
    T2 = max(2, int(round(T * gamma)))
    src = np.linspace(0.0, T - 1.0, T2)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = src - i0
    return raw[:, i0] * (1 - w) + raw[:, i1] * w
