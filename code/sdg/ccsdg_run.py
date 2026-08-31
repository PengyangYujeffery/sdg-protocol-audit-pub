#!/usr/bin/env python
"""
C²SDG (Hu, Liao & Xia, MICCAI 2023) run under OUR evaluation protocol — but NOT under our
backbone-controlled protocol, because it cannot be.

READ THIS BEFORE PUTTING ANY NUMBER FROM HERE IN A TABLE.

Every other method in this study shares one backbone, one schedule, one loss and one selection rule,
so a difference between methods is a difference between methods. C²SDG cannot join that arm: its
official implementation ships its own network *and* its own training loop. Concretely, from the
authors' source (github.com/ShishuaiHu/CCSDG, read 2026-08-17):

  * `UNetCCSDG` adds a learnable `channel_prompt` (`nn.Parameter(torch.randn(2, 64, 1, 1))`) that
    splits the 64 first-layer channels into a content and a style group, plus a `Projector` head.
  * Training uses two optimizers — one for the segmentation network, one for the projector and
    the channel prompt — stepped alternately.
  * The prompt is trained on a content loss (symmetric L1 between the projected content features
    of three views) minus a style loss (the same quantity on the style features, negated).
  * The three views are the image, an FDA (Fourier-domain) variant, and a GLA variant, i.e.
    it builds on SLAug's global location-scale augmentation.
  * The segmentation loss is BCE only, taken over three separate forward/backward passes per
    batch (FDA view, GLA view, original) — not our Dice+BCE, and not one step per batch.

So the decision recorded in METHODS_REIMPLEMENTATION.md is option 1: run it **as the authors built
it, and report it as a separate, explicitly non-backbone-controlled** entry. It is excluded from
the main effect table, the oracle ceilings and the rank figure, all of which are defined over a fixed
backbone; putting it there would silently redefine the axis.

What we *do* impose, because these are evaluation choices rather than method choices, and holding
them fixed is the entire point of the study:
  * our source/target split and our case-level source-validation split;
  * our model-selection rule (best source-val, never the target);
  * our metric, computed by the same code that scores every other method.

Only RIGA+ is run. C²SDG is a fundus method, RIGA+ is its own benchmark, and its dataloader and mask
convention are RIGA-specific. Its mask encoding (`>0 -> 1`, `==128 -> 2`) is identical to the one
this project derived independently in PROTOCOL_v1 §2.2, so no relabelling is needed.

    python ccsdg_run.py --source BinRushed --seed 0 --out .../outputs/sdg_ccsdg
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

CCSDG = os.environ.get('CCSDG_SRC', os.path.join(SCRATCH, '..', 'ccsdg_src'))
sys.path.insert(0, CCSDG)
sys.path.insert(0, _HERE)

import data as D                                    # our loaders and domain lists
from ccsdg.models.unet_ccsdg import UNetCCSDG, Projector
from ccsdg.utils.fourier import FDA_img_to_hfi
from ccsdg.datasets.utils.slaug import LocationScaleAugmentation


def dice_np(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2.0 * (p & g).sum() / s)


def views(x_np, lsa):
    """The three views the method trains on: original, FDA (high-frequency inject), GLA.

    FDA takes the WHOLE BATCH. The authors' FDA_img_to_hfi is batch code: it ffts over axes
    (-2, -1), but low_freq_mutate_np_hfi unpacks `_, _, h, w = a_src.shape` and zeroes
    `a_src[:, :, h1:h2, w1:w2]`, both of which require exactly four dimensions. Calling it per
    image passed (C, H, W) and raised
        ValueError: not enough values to unpack (expected 4, got 3)
    which is what killed E12b on 2026-08-21 and the E16 gate on 2026-08-25. The per-image loop was
    our deviation, not theirs; the operation is independent per (n, c) slice, so passing the batch
    is equivalent as well as correct. GLA stays per image -- that is how SLAug's
    LocationScaleAugmentation is written.
    """
    fda = FDA_img_to_hfi(x_np, L=0.01)
    gla = np.stack([lsa.Global_Location_Scale_Augmentation(x_np[i].copy()) for i in range(len(x_np))])
    return fda.astype(np.float32), gla.astype(np.float32)


@torch.no_grad()
def predict(model, X, bs, device, tau):
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        out.append((torch.sigmoid(model(x, tau=tau)) > 0.5).cpu().numpy())
    return np.concatenate(out)


def score(model, dom, bs, device, tau):
    """Our metric, our convention: disc and cup Dice per image, never averaged together."""
    X, Y, _ = D.riga_domain(dom, 'r1')
    mn, sd = X.mean(axis=(2, 3), keepdims=True), X.std(axis=(2, 3), keepdims=True)
    P = predict(model, ((X - mn) / (sd + 1e-6)).astype(np.float32), bs, device, tau)
    per = {}
    for i in range(len(P)):
        per[str(i)] = {'disc': {'dice': round(dice_np(P[i, 0].astype(bool), Y[i, 0].astype(bool)), 4)},
                       'cup': {'dice': round(dice_np(P[i, 1].astype(bool), Y[i, 1].astype(bool)), 4)}}
    dv = np.array([v['disc']['dice'] for v in per.values()])
    cv = np.array([v['cup']['dice'] for v in per.values()])
    return {'n': len(per), 'disc_mean': round(float(dv.mean()), 4),
            'disc_std': round(float(dv.std()), 4), 'cup_mean': round(float(cv.mean()), 4),
            'cup_std': round(float(cv.std()), 4), 'per_image': per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=8000)
    ap.add_argument('--val_every', type=int, default=500)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda'

    # ---- our split, our source-validation rule ------------------------------------------------
    # `riga_source_split` is the project's own splitter: the benchmark's official train/test lists
    # become train/source-val, and no target domain is touched. D.load_riga already scales to [0,1],
    # so the location-scale augmentation runs with vrange=(0,1), not (0,255).
    Xtr_raw, Ytr, Xva_raw, Yva, _ = D.riga_source_split(a.source, 'r1')
    zsc = lambda v: ((v - v.mean(axis=(2, 3), keepdims=True)) /
                     (v.std(axis=(2, 3), keepdims=True) + 1e-6)).astype(np.float32)
    Xtr, Xva = zsc(Xtr_raw), zsc(Xva_raw)
    print('source %s: %d train / %d source-val images'
          % (a.source, len(Xtr), len(Xva)), flush=True)

    # ---- the authors' network and optimisation, unchanged --------------------------------------
    model = UNetCCSDG(num_classes=2).to(device)
    projector = Projector().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.99, nesterov=True)
    opt_p = torch.optim.SGD(list(projector.parameters()) + [model.channel_prompt],
                            lr=a.lr, momentum=0.99, nesterov=True)
    crit = nn.BCEWithLogitsLoss()
    lsa = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)

    rng = np.random.RandomState(a.seed)
    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    for it in range(1, a.iters + 1):
        k = rng.choice(len(Xtr), a.bs, replace=len(Xtr) < a.bs)
        raw, xb, yb = Xtr_raw[k], Xtr[k], Ytr[k]
        fda_np, gla_np = views(raw, lsa)          # both built from the [0,1] image, then z-scored
        x = torch.from_numpy(xb).to(device)
        xf = torch.from_numpy(zsc(fda_np)).to(device)
        xg = torch.from_numpy(zsc(gla_np)).to(device)
        seg = torch.from_numpy(yb.astype(np.float32)).to(device)

        # prompt step: pull content together across views, push style apart
        opt_p.zero_grad()
        fc, fs = model.forward_first_layer(x, tau=a.tau)
        fcf, fsf = model.forward_first_layer(xf, tau=a.tau)
        fcg, fsg = model.forward_first_layer(xg, tau=a.tau)
        # .contiguous() is required by the authors' Projector, not by us. Its forward ends in
        #     x = x.view(x.size(0), -1)
        # and `view` refuses a tensor whose strides span two subspaces, which is what
        # forward_first_layer returns under this torch build:
        #     RuntimeError: view size is not compatible with input tensor's size and stride
        # Fixed on our side rather than by editing the vendored package, so the method's own source
        # stays exactly as published. `.contiguous()` copies without changing any value, which is
        # precisely what `.reshape()` would do internally, so no number is affected.
        pj = lambda t: projector(t.contiguous())
        pc, pcf, pcg = pj(fc), pj(fcf), pj(fcg)
        ps, psf, psg = pj(fs), pj(fsf), pj(fsg)
        pair = lambda u, v: F.l1_loss(u, v) + F.l1_loss(v, u)
        c_loss = pair(pc, pcf) + pair(pc, pcg) + pair(pcg, pcf)
        s_loss = -(pair(ps, psf) + pair(ps, psg) + pair(psg, psf))
        (c_loss + s_loss).backward()
        opt_p.step()

        # segmentation: one step per view, BCE on disc and cup, as in the original loop
        for xv in (xf, xg, x):
            opt.zero_grad()
            o = model(xv, tau=a.tau)
            loss = crit(o[:, 0], seg[:, 0]) + crit(o[:, 1], seg[:, 1])
            loss.backward()
            opt.step()

        if it % a.val_every == 0 or it == a.iters:
            model.eval()
            P = predict(model, Xva, a.bs, device, a.tau)
            d = float(np.mean([dice_np(P[i, 0].astype(bool), Yva[i, 0].astype(bool))
                               for i in range(len(P))]))
            c = float(np.mean([dice_np(P[i, 1].astype(bool), Yva[i, 1].astype(bool))
                               for i in range(len(P))]))
            v = (d + c) / 2
            hist.append({'iter': it, 'val': round(v, 4), 'disc': round(d, 4), 'cup': round(c, 4),
                         'loss': round(float(loss.item()), 4)})
            print('it %5d  loss %.4f  source-val %.4f (disc %.4f cup %.4f)  %.1f min'
                  % (it, loss.item(), v, d, c, (time.time() - t0) / 60), flush=True)
            if v > best:
                best = v
                best_state = {k_: t.detach().clone() for k_, t in model.state_dict().items()}
            model.train()

    model.load_state_dict(best_state); model.eval()
    res = {'config': {'bench': 'riga', 'source': a.source, 'method': 'ccsdg', 'seed': a.seed,
                      'gt': 'r1', 'backbone': 'ccsdg-own', 'protocol': 'NOT backbone-controlled',
                      'iters': a.iters, 'lr': a.lr, 'tau': a.tau, 'bs': a.bs},
           'best_source_val': round(best, 4), 'history': hist, 'per_domain': {},
           'minutes': round((time.time() - t0) / 60, 1), 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    for dom in D.RIGA_DOMAINS:
        res['per_domain'][dom] = score(model, dom, a.bs, device, a.tau)
        print('%-8s %-16s disc %.4f  cup %.4f'
              % ('SOURCE' if dom == a.source else 'target', dom,
                 res['per_domain'][dom]['disc_mean'], res['per_domain'][dom]['cup_mean']), flush=True)
    tg = [v for d_, v in res['per_domain'].items() if d_ != a.source]
    res['target_mean_disc'] = round(float(np.mean([t['disc_mean'] for t in tg])), 4)
    res['target_mean_cup'] = round(float(np.mean([t['cup_mean'] for t in tg])), 4)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, 'riga_%s_ccsdg_r1_s%d.json' % (a.source, a.seed))
    json.dump(res, open(p, 'w'), indent=1)
    print('\nTARGET disc %.4f  cup %.4f  -> %s'
          % (res['target_mean_disc'], res['target_mean_cup'], p))


if __name__ == '__main__':
    main()
