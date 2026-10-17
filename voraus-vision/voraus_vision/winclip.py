"""WinCLIP-style vision-language anomaly detection (M6).

Follows Jeong et al., CVPR 2023: a compositional prompt ensemble over normal and
anomalous state words, multi-scale windows over the patch grid, and harmonic
aggregation; WinCLIP+ adds a memory of normal window features.

The prompts are the published ones, written for natural object defects.  They
are deliberately *not* tuned for rendered telemetry: whether an off-the-shelf
vision-language detector transfers to synthetic signal imagery is the question,
so re-engineering the prompts until it works would answer a different one.  The
object noun is left as the generic "object" for the same reason -- there is no
honest class name for a spectrogram.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "there is a {} in the scene.",
)
NORMAL_STATES = (
    "{}", "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
)
ANOMALY_STATES = (
    "damaged {}", "broken {}", "{} with flaw", "{} with defect",
    "{} with damage", "defective {}", "abnormal {}",
)
CLASS_NOUN = "object"


class WinCLIP:
    """Zero-shot (`fit` never called) and few-shot (WinCLIP+) scoring."""

    def __init__(self, device: str = "cuda", scales: tuple = (2, 3), grid: int = 14):
        import open_clip
        self.device = device
        self.grid = grid
        self.scales = scales
        # openai weights were trained with QuickGELU; the plain ViT-B-16 config
        # silently mismatches, so ask for the quickgelu variant explicitly.
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained="openai")
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
        self.t_normal, self.t_anom = self._text_embeddings()
        self.memory = None

    @torch.no_grad()
    def _embed_texts(self, states):
        prompts = [t.format(s.format(CLASS_NOUN)) for s in states for t in TEMPLATES]
        tok = self.tokenizer(prompts).to(self.device)
        e = self.model.encode_text(tok).float()
        e = F.normalize(e, dim=-1)
        return F.normalize(e.mean(0, keepdim=True), dim=-1)   # ensemble centroid

    def _text_embeddings(self):
        return self._embed_texts(NORMAL_STATES), self._embed_texts(ANOMALY_STATES)

    @torch.no_grad()
    def _window_features(self, imgs_u8, batch=32):
        """Project the patch grid through the visual head at several window sizes.

        WinCLIP slides windows over the patch grid and embeds each; here the
        windows are average-pooled patch tokens passed through the visual
        projection, which keeps the multi-scale structure at a fraction of the
        cost of re-encoding every crop.
        """
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073),
                            device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711),
                           device=self.device).view(1, 3, 1, 1)
        out_g, out_w = [], []
        for i in range(0, len(imgs_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(imgs_u8[i:i + batch])).to(self.device)
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            v = self.model.visual
            # open_clip's VisionTransformer.forward returns only the pooled
            # embedding unless output_tokens is set, so the three internal steps
            # are called directly to keep the patch tokens.
            h = v.transformer(v._embeds(x))
            pooled, tok = v._pool(h)                           # ln_post applied inside
            proj = getattr(v, "proj", None)
            g = pooled @ proj if proj is not None else pooled
            b, n, d = tok.shape
            s = int(round(n ** 0.5))
            grid = tok.transpose(1, 2).reshape(b, d, s, s)
            wins = []
            for k in self.scales:
                w = F.avg_pool2d(grid, k, 1)                       # (b, d, s-k+1, s-k+1)
                w = w.flatten(2).transpose(1, 2)                   # (b, m, d)
                if proj is not None:
                    w = w @ proj
                wins.append(F.normalize(w.float(), dim=-1))
            out_g.append(F.normalize(g.float(), dim=-1).cpu())
            out_w.append(torch.cat(wins, dim=1).half().cpu())
        return torch.cat(out_g), torch.cat(out_w)

    def fit(self, imgs_u8):
        """WinCLIP+: remember the window features of the normal reference images."""
        _, w = self._window_features(imgs_u8)
        self.memory = w.reshape(-1, w.shape[-1]).to(self.device).float()
        return self

    @torch.no_grad()
    def score(self, imgs_u8, batch=32):
        """Returns (zero_shot, few_shot_or_None, window_anomaly_map)."""
        g, w = self._window_features(imgs_u8)
        g = g.to(self.device)
        # zero-shot: softmax over the two text centroids, temperature 100 as in CLIP
        logits = 100.0 * torch.cat([g @ self.t_normal.T, g @ self.t_anom.T], dim=1)
        zs = logits.softmax(dim=1)[:, 1].cpu().numpy()
        fs, amap = None, None
        if self.memory is not None:
            n, m, d = w.shape
            fs = np.empty(n, np.float32)
            amap = np.empty((n, m), np.float32)
            for i in range(0, n, batch):
                q = w[i:i + batch].to(self.device).float()
                sim = torch.einsum("bmd,kd->bmk", q, self.memory).max(dim=2).values
                dist = (1.0 - sim) / 2.0                      # cosine distance in [0,1]
                amap[i:i + q.shape[0]] = dist.cpu().numpy()
                fs[i:i + q.shape[0]] = dist.max(dim=1).values.cpu().numpy()
        return zs, fs, amap
