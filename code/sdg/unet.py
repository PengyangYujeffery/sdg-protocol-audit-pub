"""A plain 2D U-Net -- deliberately the textbook one, not a tuned variant.

Phase 1 is a re-benchmark: the backbone must be neutral, so that a difference between methods is a
difference between methods. Two knobs are exposed because both are known to interact with domain
shift and both are measured rather than assumed:

  * `norm` -- BatchNorm (the SDG-baseline default) vs InstanceNorm (nnU-Net's choice). IN normalises
               per-image statistics, which is itself a style-removal operation; using it silently
               would confound "my method fixes style shift".
  * `base` -- channel width. 32 is the usual U-Net; 16 halves the compute for sweeps.

Two hooks exist for the feature-space methods, and they are inert unless a method asks for them:
  * `style` modules after the first two encoder blocks (MixStyle / DSU / MaxStyle act on low-level
    features, which is where the published methods place them);
  * `forward(..., return_feat=True)` exposes the tensor entering the last convolution, which is what
    ADA's gradient-guided weakening operates on (paper eq. 8).
"""
import torch
import torch.nn as nn


def _norm(norm, c):
    return nn.BatchNorm2d(c) if norm == 'bn' else nn.InstanceNorm2d(c, affine=True)


class Block(nn.Module):
    def __init__(self, ci, co, norm):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), _norm(norm, co), nn.LeakyReLU(0.01, True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), _norm(norm, co), nn.LeakyReLU(0.01, True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, depth=4, norm='bn', style=None, n_style=2):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.enc.append(Block(c, ch, norm)); c = ch
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(Block(chs[i - 1] * 2, chs[i - 1], norm))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.styles = nn.ModuleList([style() for _ in range(n_style)]) if style is not None else None

    def forward(self, x, return_feat=False, feat_override=None):
        if feat_override is not None:                 # second pass of ADA's weakening step
            return self.head(feat_override)
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if self.styles is not None and i < len(self.styles):
                x = self.styles[i](x)
            if i < len(self.enc) - 1:
                skips.append(x); x = self.pool(x)
        for u, d, s in zip(self.up, self.dec, skips[::-1]):
            x = d(torch.cat([u(x), s], 1))
        return (self.head(x), x) if return_feat else self.head(x)
