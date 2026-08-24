"""
Augmentation policies for the SDG re-benchmark. Everything runs on GPU tensors, batch-wise.

Three policies, and the difference between them is the only thing that differs between the phase-1
baselines:

  erm spatial only (flip / rotate / scale). The lower bound every DG claim is measured against.
  bigaug ERM + the BigAug battery (Zhang et al., IEEE TMI 2020): three groups -- image *quality*
           (blur, sharpen, noise), image *appearance* (brightness, contrast, gamma), and *spatial*
           (already covered, plus elastic deformation).
  randconv random-convolution input perturbation (Xu et al., ICLR 2021): a randomly-initialised
           conv with k in {1,3,5,7} replaces the image, mixed back with weight alpha ~ U(0,1).
           This is the augmentation-only variant (RC_img); the consistency-loss variant is a
           separate method and is not silently folded in here.

Intensity operations are defined on a per-image [0,1] rescaling and mapped back afterwards, so the
same code is correct for z-scored prostate MRI and for [0,1] fundus RGB.
"""
import math
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ helpers
def _to01(x):
    b = x.shape[0]
    lo = x.reshape(b, -1).min(1).values.view(b, 1, 1, 1)
    hi = x.reshape(b, -1).max(1).values.view(b, 1, 1, 1)
    return (x - lo) / (hi - lo + 1e-8), lo, hi


def _from01(x, lo, hi):
    return x * (hi - lo + 1e-8) + lo


def _gauss_kernel(sigma, device, dtype):
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return (k / k.sum()), r


def _blur(x, sigma):
    k, r = _gauss_kernel(sigma, x.device, x.dtype)
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), ky, groups=c)


# ------------------------------------------------------------------ spatial
def spatial(x, y, rng, max_rot=20.0, scale=(0.9, 1.1), flip=True, elastic=False):
    """shared by every policy; y is resampled nearest so labels stay {0,1}."""
    b = x.shape[0]
    ang = (torch.rand(b, generator=rng, device=x.device) * 2 - 1) * max_rot * math.pi / 180
    sc = torch.rand(b, generator=rng, device=x.device) * (scale[1] - scale[0]) + scale[0]
    fl = torch.ones(b, device=x.device)
    if flip:
        fl = torch.where(torch.rand(b, generator=rng, device=x.device) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos * fl; theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin * fl; theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, x.shape, align_corners=False)

    if elastic:
        h, w = x.shape[-2:]
        d = torch.randn(b, 2, h // 16, w // 16, generator=rng, device=x.device, dtype=x.dtype)
        d = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
        d = _blur(d, 4.0) * 0.05                      # ~5% of the half-extent, smooth
        grid = grid + d.permute(0, 2, 3, 1)

    xo = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    yo = F.grid_sample(y, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    return xo, yo


# ------------------------------------------------------------------ BigAug
def bigaug_intensity(x, rng):
    b = x.shape[0]
    x, lo, hi = _to01(x)

    def coin(p):
        return (torch.rand(b, generator=rng, device=x.device) < p).view(b, 1, 1, 1)

    def u(a, c):
        return (torch.rand(b, generator=rng, device=x.device) * (c - a) + a).view(b, 1, 1, 1)

    # -- image quality: sharpen, blur, noise
    xs = x + (x - _blur(x, 1.0)) * u(10.0, 30.0) * 0.05
    x = torch.where(coin(0.5), xs.clamp(0, 1), x)
    xb = _blur(x, 1.0)
    x = torch.where(coin(0.5), xb, x)
    xn = x + torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype) * u(0.01, 0.1)
    x = torch.where(coin(0.5), xn.clamp(0, 1), x)

    # -- image appearance: brightness, contrast, gamma
    x = torch.where(coin(0.5), (x + u(-0.1, 0.1)).clamp(0, 1), x)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = torch.where(coin(0.5), ((x - m) * u(0.6, 1.5) + m).clamp(0, 1), x)
    x = torch.where(coin(0.5), x.clamp(1e-6, 1) ** u(0.6, 1.6), x)
    return _from01(x, lo, hi)


# ---------------------------------------------------------------- RandConv
def randconv(x, rng, alpha=None):
    b, c = x.shape[:2]
    k = int(torch.randint(0, 4, (1,), generator=rng, device=x.device).item())
    k = [1, 3, 5, 7][k]
    w = torch.randn(c, c, k, k, generator=rng, device=x.device, dtype=x.dtype) \
        * math.sqrt(1.0 / (c * k * k))
    xr = F.conv2d(F.pad(x, (k // 2,) * 4, mode='reflect'), w)
    # the official implementation re-standardises the conv output; without it the random weights set
    # an arbitrary scale and training diverges on z-scored MRI.
    xm, xs = x.mean(dim=(2, 3), keepdim=True), x.std(dim=(2, 3), keepdim=True)
    rm, rs = xr.mean(dim=(2, 3), keepdim=True), xr.std(dim=(2, 3), keepdim=True)
    xr = (xr - rm) / (rs + 1e-6) * (xs + 1e-6) + xm
    a = torch.rand(b, 1, 1, 1, generator=rng, device=x.device) if alpha is None \
        else torch.full((b, 1, 1, 1), alpha, device=x.device)
    return a * x + (1 - a) * xr


def noise_contrast(x, rng):
    """The static stage ADA sits on: the paper says its image augmentations follow C2SDG's settings,
    'Gaussian noise, contrast adjustment, etc.'. This is that subset of the BigAug battery and no
    more, so that ADA's own contribution is not confused with a full BigAug policy."""
    b = x.shape[0]
    x, lo, hi = _to01(x)

    def coin(p):
        return (torch.rand(b, generator=rng, device=x.device) < p).view(b, 1, 1, 1)

    def u(a, c):
        return (torch.rand(b, generator=rng, device=x.device) * (c - a) + a).view(b, 1, 1, 1)

    xn = x + torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype) * u(0.01, 0.1)
    x = torch.where(coin(0.5), xn.clamp(0, 1), x)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = torch.where(coin(0.5), ((x - m) * u(0.6, 1.5) + m).clamp(0, 1), x)
    return _from01(x, lo, hi)


# ------------------------------------------------------------------ policies
def apply_policy(name, x, y, rng):
    if name == 'erm':
        return spatial(x, y, rng)
    if name == 'bigaug':
        x, y = spatial(x, y, rng, elastic=True)
        return bigaug_intensity(x, rng), y
    if name == 'randconv':
        x, y = spatial(x, y, rng)
        return randconv(x, rng), y
    if name in ('slaug', 'slaug_sbf'):
        # SLAug (Su et al., AAAI 2023). `slaug` is the location-scale augmentation only; the
        # Saliency-Balancing Fusion component needs the network gradient and is therefore a
        # training-loop change, handled in train.py under `slaug_sbf`. See slaug.py.
        from slaug import slaug_policy
        return slaug_policy(x, y, rng)
    raise ValueError(name)
