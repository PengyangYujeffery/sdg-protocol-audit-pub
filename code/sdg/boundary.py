#!/usr/bin/env python
"""
Boundary metrics (HD95, ASSD) from the saved checkpoints — no retraining.

Dice-only evaluation is the single most common reviewer complaint about segmentation benchmarks, and
it hides exactly the failure this project keeps measuring. Under shift, 26.6 % of target cases produce
essentially nothing; Dice records that as a number near zero, but a distance metric records it as
a catastrophic outlier, which is what a clinician would experience.

Convention for degenerate cases, stated because it drives the numbers. HD95 and ASSD are
undefined when either mask is empty. We use the standard benchmark convention:
  * prediction empty, ground truth non-empty -> penalty = the image diagonal (the worst possible)
  * ground truth empty, prediction non-empty -> same penalty
  * both empty -> 0 (correct silence)
The fraction of cases that hit the penalty is reported separately, because a mean HD95 that is
mostly penalty is not a distance, it is a miss rate wearing a distance's clothes.
"""
import argparse, glob, json, os
import numpy as np
import torch
from scipy import ndimage

import data as D
from unet import UNet

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))


def surface(m):
    if not m.any():
        return None
    er = ndimage.binary_erosion(m, iterations=1, border_value=0)
    return np.argwhere(m & ~er)


def hd95_assd(pred, gt, spacing=1.0, penalty=None):
    sp, sg = surface(pred), surface(gt)
    if sp is None or sg is None:
        if not pred.any() and not gt.any():
            return 0.0, 0.0, False
        return penalty, penalty, True
    dt_g = ndimage.distance_transform_edt(~gt) * spacing
    dt_p = ndimage.distance_transform_edt(~pred) * spacing
    d_pg = dt_g[tuple(sp.T)]
    d_gp = dt_p[tuple(sg.T)]
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    assd = float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))
    return hd95, assd, False


@torch.no_grad()
def predict(model, X, bs, device):
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        out.append((torch.sigmoid(model(x))[:, 0] > 0.5).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default=os.path.join(SCRATCH, 'ckpt_h2h'))
    ap.add_argument('--methods', default='erm,bigaug,randconv,mixstyle,dsu,maxstyle,ada')
    ap.add_argument('--bench', default='prostate', choices=['prostate', 'riga', 'brats'])
    ap.add_argument('--region', default='wt')
    ap.add_argument('--struct', default='cup', choices=['disc', 'cup'])
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    methods = a.methods.split(',')
    pat = {'prostate': 'prostate_*_%s_r1_s0_fg.pt', 'riga': 'riga_*_%s_r1_s0.pt',
           'brats': 'brats_*_%s_r1_s0_%s.pt' % ('%s', a.region)}[a.bench]
    ch = {'prostate': 1, 'riga': 3, 'brats': 4}[a.bench]
    oc = 2 if a.bench == 'riga' else 1
    si = 1 if (a.bench == 'riga' and a.struct == 'cup') else 0

    agg = {m: {'hd': [], 'assd': [], 'pen': []} for m in methods}
    for m in methods:
        for ck in sorted(glob.glob(os.path.join(a.ckpt_dir, pat % m))):
            st = torch.load(ck, map_location='cpu', weights_only=False)
            c = st['config']
            if c.get('backbone', 'scratch') != 'scratch':
                continue
            model = UNet(ch, oc, base=c.get('base', 32), norm=c.get('norm', 'bn')).to(device)
            model.load_state_dict(st['state_dict']); model.eval()
            src = c['source']
            doms = {'prostate': D.PROSTATE_SITES, 'riga': D.RIGA_DOMAINS,
                    'brats': D.BRATS_TARGETS}[a.bench]
            for tgt in doms:
                if tgt == src:
                    continue
                if a.bench == 'prostate':
                    X, Y, case = D.prostate_domain(tgt, 'fg')
                    G = Y[:, 0].astype(bool)
                elif a.bench == 'riga':
                    X, Y, case = D.riga_domain(tgt, 'r1')
                    G = Y[:, si].astype(bool)
                else:
                    X, Y, case = D.brats_domain(tgt, 'tumour', a.region)
                    G = Y[:, 0].astype(bool)
                if a.bench == 'riga':
                    mn, sd = X.mean(axis=(2, 3), keepdims=True), X.std(axis=(2, 3), keepdims=True)
                    X = (X - mn) / (sd + 1e-6)
                Pr = predict(model, X, 8, device)
                P = Pr[:, si].astype(bool) if Pr.ndim == 4 else Pr.astype(bool)
                diag = float(np.sqrt(X.shape[-1] ** 2 + X.shape[-2] ** 2))
                units = case.tolist() if a.bench != 'riga' else list(range(len(P)))
                for cs in (sorted(set(units)) if a.bench != 'riga' else units):
                    k = (np.array(units) == cs) if a.bench != 'riga' else np.array(units) == cs
                    h, s, pen = hd95_assd(P[k], G[k], penalty=diag)
                    agg[m]['hd'].append(h); agg[m]['assd'].append(s); agg[m]['pen'].append(int(pen))
            print('%-9s %-14s done (%d units so far)' % (m, src, len(agg[m]['hd'])), flush=True)

    L = ['# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0\n',
         'HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty '
         '(%.1f px); their share is reported separately.\n' % 543.1,
         '\n| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |',
         '|---|---|---|---|---|---|']
    for m in methods:
        if not agg[m]['hd']:
            continue
        hd = np.array(agg[m]['hd']); ass = np.array(agg[m]['assd']); pen = np.array(agg[m]['pen'])
        L.append('| %s | %.1f | %.1f | %.1f | %.1f | **%.3f** |'
                 % (m, np.median(hd), hd.mean(), np.median(ass), ass.mean(), pen.mean()))
    L.append('\nA mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the '
             'last column first.\n')
    txt = '\n'.join(L)
    print('\n' + txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
