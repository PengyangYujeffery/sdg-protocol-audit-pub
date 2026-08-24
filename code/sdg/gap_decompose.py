#!/usr/bin/env python
"""
What is the Western -> Sub-Saharan-Africa performance gap actually made of?

Everyone assumes domain shift and reaches for a method. The BraTS-Africa paper attributes the gap to
under-representation; the challenge entries attribute it to low-field scanners. **Nobody has
decomposed it.** We can, because our ET failure analysis found the failing cases are the ones with a
small and faint enhancing tumour (AUC 0.268 for volume, 0.267 for contrast) while **no label-free
image property separates them at all** (best AUC 0.382).

So the question is sharp: is the African cohort worse because the *cases are different* (more small,
faint tumours -- a case-mix effect that would also hurt Western cases) or because the *images are
different* (a genuine domain effect)?

Direct standardisation answers it, and it is the standard tool for exactly this in epidemiology:

    gap_total = Dice(Western held-out) - Dice(African)
    gap_casemix = Dice(Western held-out) - Dice(Western held-out, REWEIGHTED to Africa's case mix)
    gap_domain = Dice(Western held-out, reweighted) - Dice(African)

The Western reference is the held-out validation split recorded in each run's `val_cases`, never
the full source domain -- most of which the model trained on.

Standardisation is only honest where the two cohorts overlap on the covariates. The **common
support** is reported first; where it is poor, the reweighting is extrapolation and the decomposition
must not be trusted.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import ndimage

import data as D

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))


def covariates(domain, region, cases_wanted=None):
    """per-case (ET volume fraction, ET-vs-rim contrast) straight from the preprocessed volumes."""
    X, Y, case = D.brats_domain(domain, 'tumour', region)
    out = {}
    for c in sorted(set(case.tolist())):
        if cases_wanted is not None and c not in cases_wanted:
            continue
        k = case == c
        et = Y[k, 0] > 0.5
        if not et.any():
            continue
        t1c = X[k][:, 0]
        brain = np.abs(X[k]).sum(1) > 1e-3
        rim = ndimage.binary_dilation(et, iterations=3) & ~et
        con = (abs(t1c[et].mean() - t1c[rim].mean()) / (t1c[brain].std() + 1e-6)
               if rim.any() else 0.0)
        out[c] = (float(et.mean()), float(con))
    return out


def standardise(ref_cov, ref_dice, tgt_cov, col, nbin=4, min_ref=5):
    """Reweight the reference cohort to the target's distribution of ONE covariate.

    Rewritten 2026-08-13. The first version binned both covariates jointly on a 3x3 grid and
    produced an unusable answer on ET: parts of 225 % and -125 % of the whole. The cause was not an
    arithmetic error -- the parts sum to the total by construction -- but extrapolation: cells
    holding one or two reference cases were given large weights. Two covariates that move in
    *opposite* directions between the cohorts (African tumours are larger but fainter) make a joint
    grid especially unstable.

    The fix: standardise on one covariate at a time, require at least `min_ref` reference cases in a
    bin before that bin may contribute, and report the occupancy so the reader can see the support
    rather than trust a single number.
    """
    keys_r = sorted(ref_cov); keys_t = sorted(tgt_cov)
    R = np.array([ref_cov[k][col] for k in keys_r])
    T = np.array([tgt_cov[k][col] for k in keys_t])
    dice_r = np.array([ref_dice[k] for k in keys_r])
    edges = np.quantile(np.concatenate([R, T]), np.linspace(0, 1, nbin + 1)[1:-1])
    br, bt = np.digitize(R, edges), np.digitize(T, edges)
    num, den, covered, occ = 0.0, 0.0, 0, []
    for b in range(nbin):
        nt = int((bt == b).sum()); m = br == b; nr = int(m.sum())
        occ.append((b, nr, nt, float(dice_r[m].mean()) if nr else float('nan')))
        if nt == 0 or nr < min_ref:      # too few reference analogues -> outside usable support
            continue
        num += nt * dice_r[m].mean(); den += nt; covered += nt
    std = num / den if den else float('nan')
    return std, covered / max(len(T), 1), occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(OUTPUTS, 'sdg_brats'))
    ap.add_argument('--region', default='et')
    ap.add_argument('--target', default='africa_glioma')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    west, afr, valsets = defaultdict(list), defaultdict(list), set()
    methods = set()
    for f in sorted(glob.glob(os.path.join(a.runs, 'brats_gli2023_*_%s.json' % a.region))):
        r = json.load(open(f)); m = r['config']['method']; methods.add(m)
        vc = set(r.get('val_cases', []))
        valsets |= vc
        for k, v in r['per_domain']['gli2023']['per_case'].items():
            if k in vc:                                  # held-out Western cases only
                west[(m, k)].append(v['dice'])
        for k, v in r['per_domain'][a.target]['per_case'].items():
            afr[(m, k)].append(v['dice'])

    cov_w = covariates('gli2023', a.region, valsets)
    cov_a = covariates(a.target, a.region)

    L = ['# What is the Western → African gap made of? (%s, %s)\n' % (a.region.upper(), a.target),
         'Western reference = the **held-out** source validation cases recorded in each run '
         '(%d distinct cases across runs); never the full source domain.\n' % len(valsets)]

    # ---- covariate shift between cohorts
    W = np.array([cov_w[k] for k in sorted(cov_w)]); A = np.array([cov_a[k] for k in sorted(cov_a)])
    L.append('\n## 1. Do the cohorts differ in case mix?\n')
    L.append('| covariate | Western held-out (median, IQR) | African (median, IQR) | ratio of medians |')
    L.append('|---|---|---|---|')
    for j, nm in enumerate(['ET volume fraction', 'ET-vs-rim contrast']):
        w, t = W[:, j], A[:, j]
        L.append('| %s | %.4f (%.4f–%.4f) | %.4f (%.4f–%.4f) | **%.2f×** |'
                 % (nm, np.median(w), *np.percentile(w, [25, 75]),
                    np.median(t), *np.percentile(t, [25, 75]),
                    np.median(t) / (np.median(w) + 1e-9)))

    # ---- the decomposition, one covariate at a time
    for col, cname in ((0, 'ET volume fraction'), (1, 'ET-vs-rim contrast')):
        L.append('\n## 2.%d Decomposition, standardised on **%s**\n' % (col + 1, cname))
        L.append('| method | Western held-out | African | **total gap** | case-mix | **domain** | support |')
        L.append('|---|---|---|---|---|---|---|')
        rows, occ_show = [], None
        for m in sorted(methods):
            rd = {k[1]: np.mean(v) for k, v in west.items() if k[0] == m and k[1] in cov_w}
            ad = {k[1]: np.mean(v) for k, v in afr.items() if k[0] == m and k[1] in cov_a}
            if len(rd) < 8 or len(ad) < 8:
                continue
            w_obs = float(np.mean(list(rd.values()))); a_obs = float(np.mean(list(ad.values())))
            w_std, cs, occ = standardise({k: cov_w[k] for k in rd}, rd, {k: cov_a[k] for k in ad}, col)
            occ_show = occ_show or occ
            rows.append((m, w_obs, a_obs, w_obs - a_obs, w_obs - w_std, w_std - a_obs, cs))
            L.append('| %s | %.4f | %.4f | **%+.4f** | %+.4f | **%+.4f** | %.2f |' % rows[-1])
        if occ_show:
            L.append('\nBin occupancy (reference / target per quantile bin, and the reference Dice '
                     'in that bin) — **a bin with few reference cases cannot carry weight**:\n')
            L.append('| bin | n Western | n African | Western Dice |')
            L.append('|---|---|---|---|')
            for b, nr, nt, dv in occ_show:
                L.append('| %d | %d | %d | %s |' % (b, nr, nt, '—' if dv != dv else '%.4f' % dv))
        if rows:
            arr = np.array([r[1:] for r in rows], float)
            tot, cm, dm = arr[:, 2].mean(), arr[:, 3].mean(), arr[:, 4].mean()
            share = 100 * cm / (abs(tot) + 1e-9)
            stable = abs(cm) <= 1.2 * abs(tot) and abs(dm) <= 1.2 * abs(tot)
            L.append('\n**total %+.4f = case-mix %+.4f (%.0f %%) + domain %+.4f (%.0f %%)** — %s\n'
                     % (tot, cm, share, dm, 100 - share,
                        'stable' if stable else '🔴 **UNSTABLE: a part exceeds the whole, which means '
                                                'the reweighting is extrapolating. Do not use.**'))
            if stable:
                L.append('- reading: **%s**\n' % (
                    'majority case mix — the cohort contains more of the cases that are hard '
                    'everywhere' if share >= 60 else
                    'majority domain — matched on this covariate, African scans are still worse'
                    if share <= 25 else 'mixed; neither framing alone is adequate'))

    txt = '\n'.join(L)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
