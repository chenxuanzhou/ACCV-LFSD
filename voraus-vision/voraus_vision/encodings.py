"""Encoding family E1-E11: multivariate window -> 224x224x3 uint8 image.

Every encoder takes a window ``X`` of shape ``(n_ch, L)`` in *raw physical
units* (after channel selection and length resampling, before any
normalisation) plus a frozen ``EncoderContext`` holding train-split statistics.
Normalisation is part of the encoder, because the choice of normalisation is
exactly what the invariance propositions of the paper (Sec. 3) are about.

E9 is not in the original plan.  It is the missing control for P1 on the GAF
branch: identical to E3 except the min-max scaler is fitted on the train split
instead of on the window.  Without it, a null result on E3 cannot be attributed
to per-window normalisation rather than to the PCA-3 projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps

_VIRIDIS = colormaps["viridis"]
IMG = 224


# --------------------------------------------------------------------- utils
def _resize(mat: np.ndarray, size: int = IMG) -> np.ndarray:
    """Bilinear resize of a 2-D float array to ``size x size``."""
    h, w = mat.shape
    yi = np.linspace(0, h - 1, size)
    xi = np.linspace(0, w - 1, size)
    y0 = np.floor(yi).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    top = mat[np.ix_(y0, x0)] * (1 - wx) + mat[np.ix_(y0, x1)] * wx
    bot = mat[np.ix_(y1, x0)] * (1 - wx) + mat[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def _viridis_u8(mat01: np.ndarray) -> np.ndarray:
    """Map a [0,1] float image through viridis to uint8 RGB."""
    return (_VIRIDIS(np.clip(mat01, 0.0, 1.0))[..., :3] * 255.0).astype(np.uint8)


def _gray_u8(mat01: np.ndarray) -> np.ndarray:
    """Replicate a [0,1] float image to uint8 RGB."""
    g = (np.clip(mat01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _stack_u8(planes01: np.ndarray) -> np.ndarray:
    """Stack three [0,1] float planes (3,H,W) into uint8 RGB."""
    return (np.clip(planes01, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _minmax_rows(mat: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Per-row (per-channel) min-max over the window -> [lo,hi].

    This is the operator P1 is about: it annihilates any per-row affine map
    ``x -> a*x + b`` with ``a > 0``.
    """
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    rng = mx - mn
    flat = rng < 1e-12
    rng = np.where(flat, 1.0, rng)
    out = (mat - mn) / rng
    out = np.where(flat, 0.5, out)
    return lo + out * (hi - lo)


def _decimate(X: np.ndarray, n: int) -> np.ndarray:
    """Linear-interpolation resample along the time axis to ``n`` points."""
    L = X.shape[1]
    if L == n:
        return X
    src = np.linspace(0.0, L - 1.0, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, L - 1)
    w = src - i0
    return X[:, i0] * (1 - w) + X[:, i1] * w


# ------------------------------------------------------------------- context
@dataclass
class EncoderContext:
    """Frozen train-split statistics shared by every encoder."""

    ch_mean: np.ndarray           # (n_ch,)  train per-channel mean
    ch_std: np.ndarray            # (n_ch,)  train per-channel std
    pca_mean: np.ndarray          # (n_ch,)  mean of the standardised features
    pca_components: np.ndarray    # (3, n_ch)
    pca_min: np.ndarray           # (3,) train min of each component
    pca_max: np.ndarray           # (3,) train max of each component
    rp_dist_scale: float          # train median pairwise distance in PCA space
    stft_mean: float              # train mean of log-magnitude
    stft_std: float               # train std of log-magnitude
    top_var_channels: np.ndarray  # (9,) indices, highest train variance
    L_pair: int = 224
    stft_nperseg: int = 64
    stft_noverlap: int = 32
    rp_percentile: float = 10.0
    core_channels: np.ndarray = None  # (28,) indices for E10/E11, see below

    def zscore(self, X: np.ndarray) -> np.ndarray:
        return (X - self.ch_mean[:, None]) / self.ch_std[:, None]

    def project(self, X: np.ndarray) -> np.ndarray:
        """Standardise with train stats, then project to 3 PCA components.

        Both steps are affine in ``X``, which is what keeps P1/P2 exact under
        the dimensionality reduction: a window-constant per-channel offset maps
        to a window-constant offset in component space, and a uniform scaling
        ``X -> a*X`` maps to ``Z -> a*Z + const``.
        """
        U = self.zscore(X)
        return self.pca_components @ (U - self.pca_mean[:, None])


# --------------------------------------------------------------- primitives
def _gasf(series: np.ndarray) -> np.ndarray:
    """Gramian angular summation field of a series already scaled to [-1,1]."""
    s = np.clip(series, -1.0, 1.0)
    phi = np.arccos(s)
    return np.cos(phi[:, None] + phi[None, :])


def _stft_logmag(series: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """Magnitude spectrogram, |DFT| per frame, Hann window. Returns (F, T)."""
    step = nperseg - noverlap
    n_frames = 1 + max(0, (len(series) - nperseg) // step)
    win = np.hanning(nperseg)
    frames = np.stack([series[i * step: i * step + nperseg] * win for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, axis=1))           # (T, F)
    return np.log1p(spec).T                              # (F, T)


# ----------------------------------------------------------------- encoders
def e1_heatmap_minmax(X, ctx):
    """E1  signal heatmap, per-window per-channel min-max.  P1: blind to affine."""
    return _viridis_u8(_resize(_minmax_rows(X)))


def e2_heatmap_zscore(X, ctx):
    """E2  signal heatmap, train-global per-channel z-score.  Control for P1/P5."""
    Z = np.clip(ctx.zscore(X), -3.0, 3.0)
    return _viridis_u8(_resize((Z + 3.0) / 6.0))


def _gaf_pca(X, ctx, window_scaler):
    Z = _decimate(ctx.project(X), ctx.L_pair)            # (3, L_pair)
    if window_scaler:
        S = _minmax_rows(Z, -1.0, 1.0)                   # per-window  (E3)
    else:
        lo = ctx.pca_min[:, None]
        hi = ctx.pca_max[:, None]
        S = np.clip(2.0 * (Z - lo) / np.maximum(hi - lo, 1e-12) - 1.0, -1.0, 1.0)  # (E9)
    planes = np.stack([_resize(_gasf(S[c])) for c in range(3)])
    return _stack_u8((planes + 1.0) / 2.0)


def e3_gaf_pca_minmax(X, ctx):
    """E3  GASF on PCA-3, per-window min-max (the pyts default)."""
    return _gaf_pca(X, ctx, window_scaler=True)


def e9_gaf_pca_globalscale(X, ctx):
    """E9  GASF on PCA-3, train-global min-max.  Added control for P1."""
    return _gaf_pca(X, ctx, window_scaler=False)


def e4_gaf_tile(X, ctx):
    """E4  GASF of the 9 highest-variance raw channels, tiled 3x3.

    Only 9 of the retained channels are rendered, so a single-channel fault on
    any other channel is invisible for a reason that has nothing to do with the
    invariance group.  Evaluation must therefore split E4 results into
    "injected channel rendered" and "not rendered".
    """
    tile = IMG // 3                                       # 74
    sub = _decimate(X[ctx.top_var_channels], tile)        # (9, 74)
    S = _minmax_rows(sub, -1.0, 1.0)
    grid = np.zeros((tile * 3, tile * 3), dtype=np.float64)
    for i in range(9):
        r, c = divmod(i, 3)
        grid[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = _gasf(S[i])
    return _gray_u8(_resize((grid + 1.0) / 2.0))


def _pair_dist(X, ctx):
    Z = _decimate(ctx.project(X), ctx.L_pair).T           # (L_pair, 3) state vectors
    d = Z[:, None, :] - Z[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def e5_rp_threshold(X, ctx):
    """E5  recurrence plot, epsilon = 10th percentile of this window's distances.

    P2: strictly invariant to global translation, and -- because the threshold
    is a percentile of the same distances -- to uniform scaling as well.
    """
    D = _pair_dist(X, ctx)
    eps = np.percentile(D, ctx.rp_percentile)
    return _gray_u8(_resize((D <= eps).astype(np.float64)))


def e6_rp_continuous(X, ctx):
    """E6  continuous distance matrix, scaled by a frozen train constant.

    The scale must come from the train split, not from the window: a per-window
    min-max here would re-introduce the scale invariance that E6 exists to
    break, and the E5-vs-E6 contrast would be uninterpretable.
    """
    D = _pair_dist(X, ctx) / ctx.rp_dist_scale
    return _viridis_u8(_resize(np.clip(D / 3.0, 0.0, 1.0)))


def _stft_planes(X, ctx, drop_dc):
    Z = ctx.project(X)
    planes = []
    for c in range(3):
        S = _stft_logmag(Z[c], ctx.stft_nperseg, ctx.stft_noverlap)
        if drop_dc:
            S = S[1:]
        S = (S - ctx.stft_mean) / ctx.stft_std
        planes.append(_resize(np.clip(S, -3.0, 3.0)))
    return (np.stack(planes) + 3.0) / 6.0


def e7_stft_with_dc(X, ctx):
    """E7  log-magnitude spectrogram of PCA-3, DC bin kept.

    P3: near-invariant to intra-window time shift.  The DC bin still carries
    the window mean, so E7 should remain sensitive to a constant offset.
    """
    return _stack_u8(_stft_planes(X, ctx, drop_dc=False))


def e8_stft_no_dc(X, ctx):
    """E8  same as E7 with the DC bin discarded -> also blind to offset."""
    return _stack_u8(_stft_planes(X, ctx, drop_dc=True))


# A 28-signal "core sensor set": four measured quantities per axis plus the four
# robot-level electrical signals.  E10/E11 exist because rendering all 129
# channels into 224 image rows gives each channel ~1.7 pixel rows, and
# results/08_diagnose/control_sweep_all.csv shows that no severity makes a
# single-channel fault detectable at that density.  Any single-channel null on
# E1/E3/E5 is therefore confounded with a resolution limit rather than being
# evidence of invariance.  At 28 channels each signal gets 8 pixel rows and 0.5
# grid rows, the single-channel positive control becomes passable, and the
# invariance claim can be tested at single-channel footprint without the
# confound.  E10 carries the invariance-inducing normalisation, E11 is its
# control -- the same relationship E1/E2 have.
CORE_CHANNEL_NAMES = tuple(
    [f"{q}_{a}" for a in range(1, 7)
     for q in ("motor_position", "motor_velocity", "motor_torque", "torque_sensor_a")]
    + ["robot_voltage", "robot_current", "io_current", "system_current"]
)


def e10_core_heatmap_minmax(X, ctx):
    """E10  28-channel heatmap, per-window per-channel min-max.  Same invariance
    group as E1, but with a channel footprint the pipeline can actually see."""
    return _viridis_u8(_resize(_minmax_rows(X[ctx.core_channels])))


def e11_core_heatmap_zscore(X, ctx):
    """E11  28-channel heatmap, train-global z-score.  Control for E10."""
    Z = np.clip(ctx.zscore(X)[ctx.core_channels], -3.0, 3.0)
    return _viridis_u8(_resize((Z + 3.0) / 6.0))


ENCODERS = {
    "E1": e1_heatmap_minmax,
    "E2": e2_heatmap_zscore,
    "E3": e3_gaf_pca_minmax,
    "E4": e4_gaf_tile,
    "E5": e5_rp_threshold,
    "E6": e6_rp_continuous,
    "E7": e7_stft_with_dc,
    "E8": e8_stft_no_dc,
    "E9": e9_gaf_pca_globalscale,
    "E10": e10_core_heatmap_minmax,
    "E11": e11_core_heatmap_zscore,
}

# Whether the encoding supports per-channel attribution (script 10).
CHANNEL_ATTRIBUTABLE = {
    "E1": True, "E2": True, "E3": False, "E4": True,
    "E5": False, "E6": False, "E7": False, "E8": False, "E9": False,
    "E10": True, "E11": True,
}
