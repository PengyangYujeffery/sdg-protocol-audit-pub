"""
Reimplementation of ADA (Huang et al., *ADA: An Adaptive Augmentation Framework for Single-Source
Domain Generalization in Medical Image Segmentation*, MICCAI 2025, LNCS 15969, pp. 45-54).

No official code was released at the time of writing (checked 2026-08-09), so this follows the
paper's equations. **Every choice the paper leaves open is listed in `METHODS_REIMPLEMENTATION.md`
and flagged here.** The point of running ADA is a specific question — does an *adaptive* method
remove the cross-benchmark reversal we measured between BigAug and RandConv? — so it must be given
its best honest shot, including a sweep over the parameters the paper does not state.

Paper equations implemented verbatim:
  (1) [P1; P2] = tanh(W·A + b), W in R^{3x4} -- control points from the image's channel means
  (2) A_{b,c} = mean over H,W of X -- adaptive average pooling
  (3) Bezier(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3, P0=(0,0), P3=(1,1)
  (4) X_enhanced = Bezier(X_norm)
  (5) [S; T] = tanh(F(A) + b) -- per-channel scale and shift
  (6) Ê = E·(1+S) + T
  (8) x_w = x_s ⊙ α where |∇L| ≥ θ, else x_s -- gradient-guided feature weakening

Underdetermined in the paper, and therefore parameters here:
  * α (the weakening factor) and θ (the gradient percentile) are never given a value.
  * The Bézier is a curve in the plane, so eq. (4) is ambiguous about how a scalar intensity maps
    through it. We use the standard treatment from the nonlinear-intensity-transform literature:
    evaluate the curve on a dense grid of t, then interpolate y against x at x = intensity. Control
    points are *not* constrained to be monotone (tanh gives [-1,1]), exactly as eq. (1) states, so
    the transform can be non-monotone; that is the paper's design, not our addition.
  * The paper's Table 4 selects the cubic Bézier and the Abs Grad weakening strategy; both
    are the defaults here.
  * The learnable modules carry no separate loss in the paper: "minimise domain disparity … mapping
    it into a unified feature space". They are therefore trained by the *segmentation* loss, jointly
    with the network — i.e. ADA is a learned per-sample normaliser, not a randomiser. This is
    the single most consequential reading in this file and it is stated openly.
"""
import torch
import torch.nn as nn


class LearnableBezierRemap(nn.Module):
    def __init__(self, c_in, n_t=256):
        super().__init__()
        self.fc = nn.Linear(c_in, 4)              # -> P1x, P1y, P2x, P2y   (eq. 1, W in R^{3x4})
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.register_buffer('t', torch.linspace(0, 1, n_t).view(1, 1, -1))

    def forward(self, x):
        b, c = x.shape[:2]
        a = x.mean(dim=(2, 3))                                        # eq. 2
        p = torch.tanh(self.fc(a))                                    # eq. 1
        p1x, p1y, p2x, p2y = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]

        t = self.t                                                    # (1,1,T)
        w0 = (1 - t) ** 3
        w1 = 3 * (1 - t) ** 2 * t
        w2 = 3 * (1 - t) * t ** 2
        w3 = t ** 3
        cx = (w1 * p1x.unsqueeze(-1) + w2 * p2x.unsqueeze(-1) + w3).squeeze(1)   # (B,T), P0x=0,P3x=1
        cy = (w1 * p1y.unsqueeze(-1) + w2 * p2y.unsqueeze(-1) + w3).squeeze(1)   # (B,T)

        # map intensity through the curve: for each pixel value v, find the t whose x is nearest v.
        # `cx` is sorted only when the control points are monotone, so we search on a sorted copy and
        # carry the permutation -- this keeps the non-monotone case well defined instead of silently
        # producing garbage.
        cx_s, order = torch.sort(cx, dim=1)
        cy_s = torch.gather(cy, 1, order)
        lo = x.reshape(b, -1).min(1, keepdim=True).values.unsqueeze(-1)
        hi = x.reshape(b, -1).max(1, keepdim=True).values.unsqueeze(-1)
        v = ((x.reshape(b, -1) - lo.squeeze(-1)) / (hi - lo + 1e-8).squeeze(-1)).clamp(0, 1)  # eq.4 X_norm
        idx = torch.searchsorted(cx_s.contiguous(), v.contiguous()).clamp(0, cx_s.shape[1] - 1)
        y = torch.gather(cy_s, 1, idx)
        y = y.view_as(x)
        return y * (hi.view(b, 1, 1, 1) - lo.view(b, 1, 1, 1)) + lo.view(b, 1, 1, 1)


class ChannelShiftControl(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.fc = nn.Linear(c_in, 2 * c_in)                            # eq. 5
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.c = c_in

    def forward(self, e):
        a = e.mean(dim=(2, 3))
        st = torch.tanh(self.fc(a))
        s, t = st[:, :self.c], st[:, self.c:]
        s = s.view(-1, self.c, 1, 1); t = t.view(-1, self.c, 1, 1)
        return e * (1 + s) + t                                         # eq. 6


class ADAInput(nn.Module):
    """the two input-side operators, applied in the paper's order: Bezier remap then channel shift."""

    def __init__(self, c_in, use_bezier=True, use_shift=True):
        super().__init__()
        self.bez = LearnableBezierRemap(c_in) if use_bezier else None
        self.shift = ChannelShiftControl(c_in) if use_shift else None

    def forward(self, x):
        if self.bez is not None:
            x = self.bez(x)
        if self.shift is not None:
            x = self.shift(x)
        return x


def weaken(feat, grad, pct=67.0, alpha=0.5, strategy='abs'):
    """eq. 8 — scale down the features whose |dL/dfeat| is above the `pct` percentile.

    `strategy` follows the paper's Table 4: 'abs' (best), 'pos', 'neg'. `pct` and `alpha` are NOT
    given in the paper; they are swept, and the value used is recorded in every result JSON.
    """
    g = {'abs': grad.abs(), 'pos': grad.clamp(min=0), 'neg': (-grad).clamp(min=0)}[strategy]
    flat = g.reshape(g.shape[0], -1)
    thr = torch.quantile(flat.float(), pct / 100.0, dim=1).view(-1, 1, 1, 1)
    return torch.where(g >= thr, feat * alpha, feat)
