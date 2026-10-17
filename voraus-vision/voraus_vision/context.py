"""Single place where the frozen EncoderContext is assembled.

Every script used to build this inline, which is exactly how two scripts end up
rendering with slightly different statistics and nobody notices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .encodings import CORE_CHANNEL_NAMES, EncoderContext


def load_context(root: Path) -> EncoderContext:
    c = np.load(root / "results" / "02_context" / "context.npz")
    kept = json.load(open(root / "results" / "01_preprocess" / "kept_channels.json"))
    names = kept["kept_channel_names"]
    core = np.array([names.index(n) for n in CORE_CHANNEL_NAMES if n in names])
    return EncoderContext(
        ch_mean=c["ch_mean"], ch_std=c["ch_std"],
        pca_mean=c["pca_mean"], pca_components=c["pca_components"],
        pca_min=c["pca_min"], pca_max=c["pca_max"],
        rp_dist_scale=float(c["rp_dist_scale"]),
        stft_mean=float(c["stft_mean"]), stft_std=float(c["stft_std"]),
        top_var_channels=c["top_var_channels"],
        core_channels=core,
    )
