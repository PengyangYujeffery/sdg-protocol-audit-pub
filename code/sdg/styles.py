"""
Feature-space style perturbation baselines: MixStyle, DSU and the MaxStyle core.

All three operate on channel-wise feature statistics (mu, sigma) of low-level features, which is the
mechanism the whole "style randomisation" family rests on. They are implemented here rather than
taken from the official repositories because our comparison must hold the backbone, the schedule and
the model-selection rule fixed; every deviation from the published implementation is recorded in
`METHODS_REIMPLEMENTATION.md` in the repo, and none of them is silent.

  MixStyle (Zhou et al., ICLR 2021) -- mixes the statistics of an image with those of another
                                          image in the batch, weight lambda ~ Beta(a, a).
  DSU (Li et al., ICLR 2022) -- treats the statistics as Gaussian random variables whose
                                          uncertainty is the batch standard deviation of those
                                          statistics, and resamples them.
  MaxStyle (Chen et al., MICCAI 2022) -- mixes statistics *and* adds noise, with the mixing weight
                                          and the noise optimised adversarially to maximise the
                                          segmentation loss.
                                          ️ Deviation: the published method attaches an auxiliary
                                          image decoder and applies the perturbation there. We apply
                                          the identical style operation inside the segmentation
                                          encoder and omit the reconstruction branch, because the
                                          decoder is a second network whose capacity would confound
                                          a backbone-controlled comparison. This is "MaxStyle-core",
                                          and it is labelled that way in every table.
"""
import torch
import torch.nn as nn


def _stats(x, eps=1e-6):
    mu = x.mean(dim=(2, 3), keepdim=True)
    sig = (x.var(dim=(2, 3), keepdim=True) + eps).sqrt()
    return mu, sig


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p, self.alpha = p, alpha
        self.enabled = False

    def forward(self, x):
        if not (self.training and self.enabled) or torch.rand(1).item() > self.p:
            return x
        mu, sig = _stats(x)
        xn = (x - mu) / sig
        perm = torch.randperm(x.size(0), device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        mu_m = mu * lam + mu[perm] * (1 - lam)
        sig_m = sig * lam + sig[perm] * (1 - lam)
        return xn * sig_m + mu_m


class DSU(nn.Module):
    """Domain-Shift-with-Uncertainty: sigma of the statistics is estimated across the batch."""

    def __init__(self, p=0.5, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps
        self.enabled = False

    def forward(self, x):
        if not (self.training and self.enabled) or torch.rand(1).item() > self.p:
            return x
        mu, sig = _stats(x)
        xn = (x - mu) / sig
        s_mu = (mu.var(dim=0, keepdim=True) + self.eps).sqrt()
        s_sig = (sig.var(dim=0, keepdim=True) + self.eps).sqrt()
        mu_n = mu + s_mu * torch.randn_like(mu)
        sig_n = sig + s_sig * torch.randn_like(sig)
        return xn * sig_n + mu_n


class MaxStyleCore(nn.Module):
    """Style mixing + noise whose parameters are *learned adversarially*.

    The three parameters of the published layer are kept: the mixing weight lambda and the two noise
    vectors on the scale and shift. They are held as buffers, reset each batch, and updated by the
    training loop with one gradient-ASCENT step on the segmentation loss (see `train.py`).
    """

    def __init__(self, p=0.5, noise_std=0.5):
        super().__init__()
        self.p, self.noise_std = p, noise_std
        self.enabled = False
        self.params = None          # (lam, eps_gamma, eps_beta)
        self.need_reset = False     # set by the training loop before the adversarial pass

    def _reset(self, x):
        """the parameters can only be created once the feature shape is known, i.e. inside forward."""
        b, c = x.shape[:2]
        dev = x.device
        lam = torch.rand(b, 1, 1, 1, device=dev).requires_grad_(True)
        eg = (torch.randn(b, c, 1, 1, device=dev) * self.noise_std).requires_grad_(True)
        eb = (torch.randn(b, c, 1, 1, device=dev) * self.noise_std).requires_grad_(True)
        self.params = [lam, eg, eb]
        self.perm = torch.randperm(b, device=dev)
        self.need_reset = False

    def forward(self, x):
        if not (self.training and self.enabled):
            return x
        if self.need_reset or self.params is None or self.params[0].shape[0] != x.shape[0]:
            self._reset(x)
        lam, eg, eb = self.params
        mu, sig = _stats(x)
        xn = (x - mu) / sig
        mu_m = mu * lam + mu[self.perm] * (1 - lam)
        sig_m = sig * lam + sig[self.perm] * (1 - lam)
        # noise on the style statistics, scaled by their batch spread (as in the paper)
        s_mu = mu.std(dim=0, keepdim=True) + 1e-6
        s_sig = sig.std(dim=0, keepdim=True) + 1e-6
        return xn * (sig_m + eg * s_sig) + (mu_m + eb * s_mu)


def make_style(name):
    return {'mixstyle': MixStyle, 'dsu': DSU, 'maxstyle': MaxStyleCore}[name]()
