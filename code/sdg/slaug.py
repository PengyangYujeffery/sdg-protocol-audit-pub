"""
SLAug — location-scale augmentation (Su et al., AAAI 2023, vol. 37, pp. 2366--2374).

Ported from the authors' `dataloaders/location_scale_augmentation.py`
(github.com/Kaiseem/SLAug), read on 2026-08-17. This is a re-implementation, not a copy: the
original is numpy + the `random` module operating on one image at a time, and everything in this
project must run on batched GPU tensors under a seeded torch generator, because the protocol is
fp32 + `--deterministic 1` and bit-reproducible. Drawing from `random` would put part of the
augmentation outside that control.

**Deviations from the original, stated because a re-implementation that hides them is worthless
as a baseline** (they also go in METHODS_REIMPLEMENTATION.md):

  1. Randomness source. `random.uniform` / `random.gauss` -> `torch.rand` / `torch.randn` on the
     run's generator. Distributions and bounds are identical; the stream is not, so our numbers will
     not equal the authors' even on their data.
  2. Value range. The original assumes inputs in `vrange=(0,1)`. Our prostate volumes are
     z-scored and RIGA+ is per-image standardised, so each image is mapped to [0,1] with the
     pipeline's existing `_to01`, transformed, and mapped back. This mirrors how `bigaug_intensity`
     is handled here and keeps the backbone comparison free of a preprocessing change.
  3. Bezier grid. `nTimes` (the interpolation grid) is 100000 in the original and kept here;
     `nPoints=4` control points, of which the two interior ones are drawn uniformly between the
     image min and max, exactly as in the source.
  4. Local variant. The original loops `for c in range(1, mask.max()+1)`. Our benchmarks are
     binary (gland / disc / cup / tumour region), so the loop runs over the single foreground
     class. This is the original's behaviour on a binary mask, not a simplification of it.
  5. Saliency-Balancing Fusion (SBF) is NOT in this module. The authors' `get_SBF_map` takes
     the network gradient as an argument, so it is a *training-loop* modification rather than an
     augmentation. Under this project's protocol -- one backbone, one schedule, one loss, only the
     augmentation policy differs -- SBF cannot be folded in without breaking the thing that makes
     the comparison readable. It is therefore run as a separate, explicitly labelled variant
     (`--method slaug_sbf`, see train.py) and never silently mixed with the augmentation-only arm.
     **Any number reported for `slaug` is SLAug's location-scale augmentation, not the full method,
     and must say so.**
"""
import torch

from aug import _to01, _from01, spatial


def _bernstein_basis(n_points, n_times, device, dtype):
    """(n_points, n_times) Bernstein polynomial basis — precomputed once per call site."""
    t = torch.linspace(0.0, 1.0, n_times, device=device, dtype=dtype)
    out = []
    n = n_points - 1
    for i in range(n_points):
        c = 1.0
        for k in range(i):                                   # binomial(n, i) without scipy
            c = c * (n - k) / (k + 1)
        out.append(c * (t ** (n - i)) * ((1 - t) ** i))
    return torch.stack(out)


def _interp(x, xp, fp):
    """torch equivalent of np.interp for a sorted xp."""
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-12)


def _nonlinear(v, rng, basis, inverse=False, inverse_prop=0.5):
    """Bezier intensity warp over the value range of `v` (a 1-D tensor of pixels)."""
    if v.numel() == 0:
        return v
    lo, hi = v.min(), v.max()
    if not torch.isfinite(lo) or (hi - lo) < 1e-8:
        return v
    dev, dt = v.device, v.dtype
    # two interior control points, uniform in [lo, hi]; order matches the original's insert(1, .)
    r = torch.rand(4, generator=rng, device=dev, dtype=dt)
    xs = torch.stack([lo, lo + (hi - lo) * r[0], lo + (hi - lo) * r[1], hi])
    ys = torch.stack([lo, lo + (hi - lo) * r[2], lo + (hi - lo) * r[3], hi])
    xv = xs @ basis
    yv = ys @ basis
    if inverse and torch.rand(1, generator=rng, device=dev).item() <= inverse_prop:
        xv = torch.sort(xv).values                            # y left unsorted -> inverted mapping
    else:
        xv, yv = torch.sort(xv).values, torch.sort(yv).values
    return _interp(v, xv, yv)


def _loc_scale(v, rng, vrange=(0.0, 1.0), slide_limit=20):
    if v.numel() == 0:
        return v
    dev, dt = v.device, v.dtype
    scale = (1.0 + 0.1 * torch.randn(1, generator=rng, device=dev, dtype=dt)).clamp(0.9, 1.1)
    loc = 0.5 * torch.randn(1, generator=rng, device=dev, dtype=dt)
    q = torch.quantile(v, torch.tensor([slide_limit / 100.0, 1 - slide_limit / 100.0],
                                       device=dev, dtype=dt))
    loc = loc.clamp(vrange[0] - q[0], vrange[1] - q[1])
    return (v * scale + loc).clamp(vrange[0], vrange[1])


def location_scale(x, y, rng, local=True, background_threshold=0.01):
    """SLAug GLA (local=False) / LLA (local=True) on a batch, in [0,1] space.

    x: (B,C,H,W) already mapped to [0,1]; y: (B,1,H,W) binary mask.
    Each image is transformed independently, as in the original.
    """
    basis = _bernstein_basis(4, 100000, x.device, x.dtype)
    out = x.clone()
    for b in range(x.shape[0]):
        img = x[b]
        if not local:
            v = _nonlinear(img.reshape(-1), rng, basis, inverse=False)
            out[b] = _loc_scale(v, rng).reshape(img.shape)
            continue
        m = (y[b, 0] > 0.5)
        o = torch.empty_like(img)
        for ch in range(img.shape[0]):
            plane, op = img[ch], o[ch]
            bg = plane[~m]
            if bg.numel():
                op[~m] = _loc_scale(_nonlinear(bg, rng, basis, inverse=True, inverse_prop=1.0), rng)
            fg = plane[m]
            if fg.numel():
                op[m] = _loc_scale(_nonlinear(fg, rng, basis, inverse=True, inverse_prop=0.5), rng)
        keep = img <= background_threshold          # very dark pixels bypass, as in the original
        o[keep] = img[keep]
        out[b] = o
    return out


def slaug_policy(x, y, rng, p_local=0.5):
    """One SLAug sample: spatial augmentation, then GLA or LLA on the intensity.

    The original draws the global and the local variant to be *fused* by saliency; without SBF
    (see the module docstring) the honest reduction is to draw one of the two per sample, which is
    what the authors' own ablation labels GLA / LLA.
    """
    x, y = spatial(x, y, rng)
    x01, lo, hi = _to01(x)
    local = torch.rand(1, generator=rng, device=x.device).item() < p_local
    x01 = location_scale(x01, y, rng, local=local)
    return _from01(x01, lo, hi), y
