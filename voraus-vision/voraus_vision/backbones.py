"""Frozen feature extractors.  No gradients, no fine-tuning anywhere.

Three backbones, chosen to separate "which pre-training" from "which detector":

    wrn50    WideResNet-50-2 (timm wide_resnet50_2, ImageNet-1k), stages 1+2
    dinov2   DINOv2 ViT-B/14, self-supervised       -- tests whether SSL
                                                       features cope better
                                                       with synthetic imagery
    clip     CLIP ViT-B/16, image-text              -- vision-language family

Patch features are pooled to a fixed 14x14 grid so that (a) memory stays within
the memory budget of one 8 GB GPU and (b) the anomaly map has a grid geometry that is identical
across backbones, which the attribution protocols need.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

GRID = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Backbone:
    """Wraps a frozen network and returns (patch_features, global_feature)."""

    def __init__(self, name: str, device: str = "cuda", proj_dim: int = 256, seed: int = 0):
        self.name = name
        self.device = device
        self.proj_dim = proj_dim
        self.seed = seed
        self._proj = None
        if name == "wrn50":
            import timm
            self.model = timm.create_model("wide_resnet50_2", pretrained=True,
                                           features_only=True, out_indices=(1, 2))
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        elif name == "dinov2":
            import timm
            # dynamic_img_size lets the 224x224 renders through without the
            # model's native 518x518 assumption; 224/14 = 16x16 patch tokens.
            self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                           pretrained=True, num_classes=0,
                                           dynamic_img_size=True)
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        elif name == "clip":
            # timm's port of the OpenAI CLIP image tower: same weights, but it
            # exposes patch tokens the way the other two backbones do.
            import timm
            self.model = timm.create_model("vit_base_patch16_clip_224.openai",
                                           pretrained=True, num_classes=0)
            self.mean, self.std = CLIP_MEAN, CLIP_STD
        else:
            raise ValueError(name)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed random Gaussian projection to ``proj_dim``.

        PatchCore itself uses a random projection for the coreset; applying one
        consistent projection to every backbone also equalises the feature
        dimension so that detector hyper-parameters mean the same thing across
        backbones.  The matrix is seeded, so it is identical for every encoding.
        """
        d = x.shape[-1]
        if d <= self.proj_dim:
            return x
        if self._proj is None or self._proj.shape[0] != d:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            P = torch.randn(d, self.proj_dim, generator=g) / np.sqrt(self.proj_dim)
            self._proj = P.to(self.device)
        return x @ self._proj

    @torch.no_grad()
    def forward(self, imgs_u8: np.ndarray, batch: int = 32):
        """imgs_u8: (n, 224, 224, 3) uint8.  Returns (n, 196, proj_dim), (n, dg)."""
        n = imgs_u8.shape[0]
        pf, gf = [], []
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        for i in range(0, n, batch):
            x = torch.from_numpy(np.ascontiguousarray(imgs_u8[i:i + batch]))
            x = x.to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            if self.name == "wrn50":
                f2, f3 = self.model(x)                       # 56x56x256, 28x28x512
                f2 = F.avg_pool2d(f2, 3, 1, 1)
                f3 = F.avg_pool2d(f3, 3, 1, 1)
                f2 = F.adaptive_avg_pool2d(f2, GRID)
                f3 = F.adaptive_avg_pool2d(f3, GRID)
                p = torch.cat([f2, f3], 1).flatten(2).transpose(1, 2)   # (b,196,768)
                g = torch.cat([f2, f3], 1).mean(dim=(2, 3))
            else:                                            # dinov2 / clip, both ViT
                t = self.model.forward_features(x)           # (b, prefix+np, d)
                npre = self.model.num_prefix_tokens
                g = t[:, 0]
                p = t[:, npre:]
                s = int(round(p.shape[1] ** 0.5))
                p = p.transpose(1, 2).reshape(t.shape[0], -1, s, s)
                p = F.adaptive_avg_pool2d(p, GRID).flatten(2).transpose(1, 2)
            pf.append(self._project(p).half().cpu().numpy())
            gf.append(g.float().cpu().numpy())
        return np.concatenate(pf), np.concatenate(gf)
