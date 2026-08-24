#!/usr/bin/env python
"""
Phase 2 — failure analysis. Three questions, each stated as a prediction that could come out wrong.

Q1 Does RandConv fail on MRI by losing the gland, or by over-drawing it?
    Hypothesis from phase 1: RandConv randomises the intensity->label map. In fundus RGB that
    destroys nuisance (colour, illumination); in T2 MRI the prostate is *defined* by intensity
    texture, so the same operation destroys signal. If that is right the damage is a recall
    collapse (gland not found), not a precision collapse (gland found and over-drawn).
    A precision collapse would falsify it.

Q2 Is the ERM transfer matrix explained by intensity-distribution distance between sites?
    Computed directly from the preprocessed data (histogram L1 over the body region of foreground
    slices). The matrix is strongly asymmetric (BIDMC->HK 0.810 vs HK->BIDMC 0.515) and any
    symmetric distance therefore *cannot* explain all of it — the interesting quantity is how much
    is left over, and what it tracks.

Q3 What does a total failure (Dice < 0.10) actually look like? Predicted-to-true volume ratio
    separates "predicted nothing" from "predicted everywhere", and the mix differs by method.

Reads the phase-2 pass (`outputs/sdg_mech/`), which stores per-case precision/recall/volumes.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

SITES = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
PROSTATE_DIR = os.environ.get('PROSTATE_2D', os.path.join(SCRATCH, 'prostate_2d'))
METHODS = ['erm', 'bigaug', 'randconv']


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, 'prostate_*.json'))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def site_hist(site, bins):
    """intensity histogram of the body region over gland-bearing slices -- the quantity a
    style-transfer or intensity-augmentation method is implicitly trying to align."""
    z = np.load('%s/%s.npz' % (PROSTATE_DIR, site))
    X, fg = z['X'], z['fg']
    v = X[fg > 0]
    v = v[np.abs(v) > 1e-6]
    h, _ = np.histogram(v, bins=bins, density=True)
    return h / (h.sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mech_dir', default=os.path.join(OUTPUTS, 'sdg_mech'))
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    runs = load(a.mech_dir)
    if not runs:
        print('no runs in %s' % a.mech_dir); return

    L = ['# Phase-2 failure analysis — DG Prostate\n',
         '%d runs (seed 0, slices=fg, per-case precision/recall/volumes)\n' % len(runs)]

    # ---------------------------------------------------------------- Q1
    agg = defaultdict(lambda: defaultdict(list))
    for r in runs:
        src, m = r['config']['source'], r['config']['method']
        for dom, d in r['per_domain'].items():
            if dom == src:
                continue
            for v in d['per_case'].values():
                for k in ('dice', 'precision', 'recall'):
                    agg[m][k].append(v[k])
                if np.isfinite(v.get('vol_ratio', np.nan)):
                    agg[m]['vol_ratio'].append(v['vol_ratio'])

    L.append('\n## Q1 — precision vs recall on unseen sites (all 30 transfer pairs pooled)\n')
    L.append('| method | Dice | precision | recall | median pred/true volume |')
    L.append('|---|---|---|---|---|')
    for m in METHODS:
        if not agg[m]['dice']:
            continue
        L.append('| %s | %.4f | %.4f | %.4f | %.3f |'
                 % (m, np.mean(agg[m]['dice']), np.mean(agg[m]['precision']),
                    np.mean(agg[m]['recall']), np.median(agg[m]['vol_ratio'])))
    if agg['randconv']['dice'] and agg['bigaug']['dice']:
        dp = np.mean(agg['randconv']['precision']) - np.mean(agg['bigaug']['precision'])
        dr = np.mean(agg['randconv']['recall']) - np.mean(agg['bigaug']['recall'])
        L.append('\n**RandConv − BigAug: precision %+.4f, recall %+.4f.** '
                 'The hypothesis predicted the loss would sit in *recall*; %s\n'
                 % (dp, dr,
                    'it does — RandConv finds less gland rather than over-drawing it.'
                    if dr < dp - 0.02 else
                    ('it does NOT — the loss is in precision, so the "destroys signal" story is '
                     'wrong and RandConv over-segments instead.' if dp < dr - 0.02 else
                     'it does not separate them — precision and recall fall together, which is '
                     'consistent with neither mechanism and needs a different probe.')))

    # ---------------------------------------------------------------- Q3
    L.append('\n## Q3 — anatomy of a total failure (per-case Dice < 0.10)\n')
    L.append('| method | share of cases | of those, predicted ~nothing (vol<0.1×) | predicted ~everywhere (vol>3×) |')
    L.append('|---|---|---|---|')
    for m in METHODS:
        d = np.array(agg[m]['dice']); vr = np.array(agg[m]['vol_ratio'])
        if not len(d):
            continue
        bad = d < 0.10
        n = int(bad.sum())
        if n == 0:
            L.append('| %s | 0.000 | — | — |' % m); continue
        vb = vr[bad[:len(vr)]] if len(vr) == len(d) else vr
        L.append('| %s | %.3f | %.2f | %.2f |'
                 % (m, bad.mean(), float((vb < 0.1).mean()), float((vb > 3).mean())))

    # ---------------------------------------------------------------- Q2
    bins = np.linspace(-3, 6, 91)
    H = {s: site_hist(s, bins) for s in SITES}
    dist = {(s, t): float(np.abs(H[s] - H[t]).sum()) for s in SITES for t in SITES if s != t}

    erm = {}
    for r in runs:
        if r['config']['method'] != 'erm':
            continue
        src = r['config']['source']
        for dom, d in r['per_domain'].items():
            if dom != src:
                erm[(src, dom)] = d['dice_mean']

    pairs = [k for k in erm if k in dist]
    if len(pairs) >= 6:
        x = np.array([dist[k] for k in pairs]); y = np.array([erm[k] for k in pairs])
        rho = float(np.corrcoef(x, y)[0, 1])
        L.append('\n## Q2 — is transferability explained by intensity-histogram distance?\n')
        L.append('Pearson r between source↔target histogram L1 distance and ERM target Dice over '
                 '%d ordered pairs: **r = %+.3f** (r² = %.3f).\n' % (len(pairs), rho, rho ** 2))
        asym = [(s, t, erm.get((s, t)), erm.get((t, s))) for s in SITES for t in SITES
                if s < t and (s, t) in erm and (t, s) in erm]
        if asym:
            gaps = [abs(a - b) for _, _, a, b in asym]
            L.append('The distance is **symmetric by construction**, but the transfer matrix is not: '
                     'the mean |Dice(s→t) − Dice(t→s)| over %d site pairs is **%.3f** '
                     '(max %.3f). Whatever explains direction is therefore not a distance.\n'
                     % (len(asym), float(np.mean(gaps)), float(np.max(gaps))))
            L.append('| pair | s→t | t→s | gap |')
            L.append('|---|---|---|---|')
            for s, t, ab, ba in sorted(asym, key=lambda z: -abs(z[2] - z[3])):
                L.append('| %s ↔ %s | %.3f | %.3f | %.3f |' % (s, t, ab, ba, abs(ab - ba)))

    # ---------------------------------------------------------------- Q4
    # If a pairwise "domain gap" governs transfer, the 6x6 matrix should NOT be well described by
    # one number per source plus one number per target. Fit that additive model and see what is left.
    L.append('\n## Q4 — is the transfer matrix pairwise, or just "source capability + target difficulty"?\n')
    L.append('| method | R² of additive model | residual SD | source effects α (best→worst) | target effects β (easiest→hardest) |')
    L.append('|---|---|---|---|---|')
    alphas = {}
    for m in METHODS:
        mat = {}
        for r in runs:
            if r['config']['method'] != m:
                continue
            s = r['config']['source']
            for t, d in r['per_domain'].items():
                if t != s:
                    mat[(s, t)] = d['dice_mean']
        keys = [k for k in mat if k[0] in SITES and k[1] in SITES]
        if len(keys) < 20:
            continue
        idx = {s: i for i, s in enumerate(SITES)}
        A = np.zeros((len(keys), 1 + 2 * len(SITES)))
        y = np.array([mat[k] for k in keys])
        for i, (s, t) in enumerate(keys):
            A[i, 0] = 1
            A[i, 1 + idx[s]] = 1
            A[i, 1 + len(SITES) + idx[t]] = 1
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        al = {s: coef[1 + idx[s]] for s in SITES}
        be = {s: coef[1 + len(SITES) + idx[s]] for s in SITES}
        al = {s: v - np.mean(list(al.values())) for s, v in al.items()}
        be = {s: v - np.mean(list(be.values())) for s, v in be.items()}
        alphas[m] = al
        L.append('| %s | **%.3f** | %.3f | %s | %s |'
                 % (m, r2, float(np.std(y - pred)),
                    ' > '.join('%s %+.2f' % (s, al[s]) for s in sorted(al, key=al.get, reverse=True)),
                    ' > '.join('%s %+.2f' % (s, be[s]) for s in sorted(be, key=be.get, reverse=True))))

    # what does source capability track? (cheap covariates from the preprocessed summary)
    try:
        summ = json.load(open('%s/summary.json' % PROSTATE_DIR))
        if 'erm' in alphas:
            L.append('\n**What does a good source have?** α (ERM source effect) against the site '
                     'properties measured at preprocessing:\n')
            L.append('| covariate | Pearson r with α |')
            L.append('|---|---|')
            al = alphas['erm']
            av = np.array([al[s] for s in SITES])
            for name, key in [('cases', 'cases'), ('slices', 'slices'),
                              ('foreground slice fraction', 'fg_fraction_of_slices'),
                              ('mean gland area (px)', 'mean_fg_area_px')]:
                cv = np.array([summ[s][key] for s in SITES], float)
                L.append('| %s | %+.3f |' % (name, float(np.corrcoef(cv, av)[0, 1])))
    except Exception as e:                                   # summary.json is optional
        L.append('\n(site covariates unavailable: %s)' % e)

    txt = '\n'.join(L)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')
        print('\n-> %s' % a.md)


if __name__ == '__main__':
    main()
