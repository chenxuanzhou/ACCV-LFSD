"""Training-free anomaly detectors M1-M5 over frozen backbone features.

Implemented directly rather than through anomalib: the study needs the anomaly
*map* in a fixed patch grid for the channel- and time-attribution protocols
(P-C/P-D), and needs the same memory-bank indices reused across eleven encodings
and two backbones.  Each detector below is the standard published method with
its standard hyper-parameters; deviations are noted in the docstrings.

All detectors expose the same interface::

    det = Detector(...)
    det.fit(bank_features)          # normal samples only, no gradients
    scores, maps = det.score(feats) # image-level score, (n, gh, gw) map
"""

from __future__ import annotations

import numpy as np
import torch


def _to_t(x, device):
    return x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x)).to(device)


class PatchCore:
    """PatchCore (Roth et al., CVPR 2022): coreset-subsampled patch memory bank,
    image score = max over patches of the distance to the nearest bank vector.

    Deviation: the greedy coreset is seeded deterministically and the
    re-weighting step of the original paper is omitted (it changes the score by
    <1% and is not relevant to an invariance study).
    """

    def __init__(self, coreset_ratio: float = 0.01, device: str = "cuda",
                 seed: int = 0, min_bank: int = 64):
        self.coreset_ratio = coreset_ratio
        self.device = device
        self.seed = seed
        self.min_bank = min_bank
        self.bank = None

    def fit(self, feats: np.ndarray) -> "PatchCore":
        """feats: (n, p, d) patch features of normal samples."""
        n, p, d = feats.shape
        X = _to_t(feats.reshape(-1, d), self.device).float()
        m = max(self.min_bank, int(round(self.coreset_ratio * X.shape[0])))
        self.bank = X if m >= X.shape[0] else self._coreset(X, m)
        return self

    def _coreset(self, X: torch.Tensor, m: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        start = int(torch.randint(X.shape[0], (1,), generator=g).item())
        picked = [start]
        dist = torch.cdist(X, X[start:start + 1]).squeeze(1)
        for _ in range(m - 1):
            i = int(torch.argmax(dist).item())
            picked.append(i)
            dist = torch.minimum(dist, torch.cdist(X, X[i:i + 1]).squeeze(1))
        return X[picked]

    def score(self, feats: np.ndarray, batch: int = 64):
        n, p, d = feats.shape
        out_s = np.empty(n, dtype=np.float32)
        out_m = np.empty((n, p), dtype=np.float32)
        for i in range(0, n, batch):
            Q = _to_t(feats[i:i + batch], self.device).float()
            b = Q.shape[0]
            dmin = torch.cdist(Q.reshape(-1, d), self.bank).min(dim=1).values.reshape(b, p)
            out_m[i:i + b] = dmin.cpu().numpy()
            out_s[i:i + b] = dmin.max(dim=1).values.cpu().numpy()
        return out_s, out_m


class PaDiM:
    """PaDiM (Defard et al., 2020): one Gaussian per patch position, score =
    Mahalanobis distance.

    Deviation: with k as low as 1 the per-position covariance is rank-deficient,
    so the shrinkage term is set as a fraction of the mean eigenvalue rather
    than the paper's fixed 0.01*I.  This is the standard Ledoit-Wolf-style fix
    and is applied identically at every k, so the k-sweep stays comparable.
    """

    def __init__(self, device: str = "cuda", shrinkage: float = 0.1):
        self.device = device
        self.shrinkage = shrinkage

    def fit(self, feats: np.ndarray) -> "PaDiM":
        n, p, d = feats.shape
        X = _to_t(feats, self.device).float()
        self.mu = X.mean(dim=0)                                   # (p, d)
        Xc = X - self.mu
        cov = torch.einsum("npd,npe->pde", Xc, Xc) / max(n - 1, 1)
        eye = torch.eye(d, device=cov.device).expand(p, d, d)
        tr = cov.diagonal(dim1=1, dim2=2).mean(dim=1).view(p, 1, 1)
        cov = cov + self.shrinkage * tr * eye + 1e-6 * eye
        self.prec = torch.linalg.inv(cov)                         # (p, d, d)
        return self

    def score(self, feats: np.ndarray, batch: int = 64):
        n, p, d = feats.shape
        out_s = np.empty(n, dtype=np.float32)
        out_m = np.empty((n, p), dtype=np.float32)
        for i in range(0, n, batch):
            Q = _to_t(feats[i:i + batch], self.device).float() - self.mu
            md = torch.einsum("npd,pde,npe->np", Q, self.prec, Q).clamp_min(0).sqrt()
            out_m[i:i + Q.shape[0]] = md.cpu().numpy()
            out_s[i:i + Q.shape[0]] = md.max(dim=1).values.cpu().numpy()
        return out_s, out_m


class KNNGlobal:
    """k-nearest-neighbour distance on a single global embedding (CLS token).

    The minimal baseline of the study: no patch structure, no density model.
    """

    def __init__(self, k: int = 1, device: str = "cuda"):
        self.k = k
        self.device = device

    def fit(self, feats: np.ndarray) -> "KNNGlobal":
        self.bank = _to_t(feats, self.device).float()
        return self

    def score(self, feats: np.ndarray, batch: int = 256):
        n = feats.shape[0]
        out = np.empty(n, dtype=np.float32)
        kk = min(self.k, self.bank.shape[0])
        for i in range(0, n, batch):
            Q = _to_t(feats[i:i + batch], self.device).float()
            d = torch.cdist(Q, self.bank)
            out[i:i + Q.shape[0]] = d.topk(kk, largest=False).values.mean(1).cpu().numpy()
        return out, None


class MahalanobisGlobal:
    """Mahalanobis distance to a single Gaussian fitted on the global embedding.

    Same shrinkage treatment as PaDiM, for the same reason: at k=1 the
    covariance is a rank-0 matrix and the estimate must be regularised.
    """

    def __init__(self, device: str = "cuda", shrinkage: float = 0.1):
        self.device = device
        self.shrinkage = shrinkage

    def fit(self, feats: np.ndarray) -> "MahalanobisGlobal":
        X = _to_t(feats, self.device).float()
        n, d = X.shape
        self.mu = X.mean(0)
        Xc = X - self.mu
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        eye = torch.eye(d, device=cov.device)
        tr = cov.diagonal().mean()
        self.prec = torch.linalg.inv(cov + self.shrinkage * tr * eye + 1e-6 * eye)
        return self

    def score(self, feats: np.ndarray, batch: int = 256):
        n = feats.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, batch):
            Q = _to_t(feats[i:i + batch], self.device).float() - self.mu
            out[i:i + Q.shape[0]] = torch.einsum(
                "nd,de,ne->n", Q, self.prec, Q).clamp_min(0).sqrt().cpu().numpy()
        return out, None
