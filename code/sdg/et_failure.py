#!/usr/bin/env python
"""
MICCAI line — data-driven failure characterisation on BraTS enhancing tumour.

Standing position after four kills: stop proposing mechanisms from the armchair. This script does
not assume one. It takes the per-case results that already exist for the ET arm and asks the data
which measurable property separates the cases where a Western-trained model fails on African scans
from the cases where it succeeds — and, separately, which property separates the cases where BigAug
fails (−0.135 overall) from those where RandConv succeeds (+0.057).

Covariates are deliberately of two kinds, because the distinction decides what a method could ever do:
  * label-dependent (tumour volume, fragmentation, ET-vs-brain contrast) — descriptive only; a
    deployed model does not have these;
  * label-free (per-modality intensity statistics, a noise proxy, an SNR proxy, brain volume) —
    these a method *could* condition on at test time, so a separating variable found here is
    actionable and one found only among label-dependent covariates is not.

Output: for each covariate, the AUC with which it separates failing from succeeding cases, and the
correlation with the per-case BigAug−RandConv difference.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import ndimage

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))


def auc(score, label):
    """AUC of `score` for predicting label==1, computed by ranks (no sklearn dependency)."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(OUTPUTS, 'sdg_brats'))
    ap.add_argument('--target', default='africa_glioma')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    import data as D

    # ---- per-case Dice for each method on the ET arm (seeds averaged)
    per = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(a.runs, 'brats_gli2023_*_et.json'))):
        r = json.load(open(f))
        m = r['config']['method']
        for k, v in r['per_domain'][a.target]['per_case'].items():
            per[m][k].append(v['dice'])
    methods = sorted(per)
    cases = sorted(set.intersection(*[set(per[m]) for m in methods]))
    print('%d methods, %d target cases' % (len(methods), len(cases)))
    dice = {m: np.array([np.mean(per[m][c]) for c in cases]) for m in methods}

    # ---- covariates straight from the preprocessed target volumes
    X, Y, case = D.brats_domain(a.target, 'tumour', 'et')
    Yall = D.brats_domain(a.target, 'tumour', 'wt')[1]
    cov = defaultdict(list)
    for c in cases:
        k = case == c
        et = Y[k, 0] > 0.5
        wt = Yall[k, 0] > 0.5
        img = X[k]                                     # (S, 4, H, W): t1c, t1n, t2f, t2w
        brain = np.abs(img).sum(1) > 1e-3
        # label-dependent
        cov['et_volume_frac'].append(float(et.mean()))
        cov['et_frac_of_wt'].append(float(et.sum() / max(wt.sum(), 1)))
        cov['et_components'].append(float(ndimage.label(et)[1]))
        t1c = img[:, 0]
        rim = ndimage.binary_dilation(et, iterations=3) & ~et
        cov['et_t1c_contrast'].append(
            float(abs(t1c[et].mean() - t1c[rim].mean()) / (t1c[brain].std() + 1e-6))
            if et.any() and rim.any() else 0.0)
        # label-free (a deployed model could compute all of these)
        for i, nm in enumerate(('t1c', 't1n', 't2f', 't2w')):
            v = img[:, i][brain]
            cov['%s_std' % nm].append(float(v.std()))
            cov['%s_kurtosis' % nm].append(float(((v - v.mean()) ** 4).mean() / (v.var() ** 2 + 1e-9)))
        cov['brain_frac'].append(float(brain.mean()))
        # high-frequency energy as a noise/resolution proxy
        lap = ndimage.laplace(img[:, 2].astype(np.float32))
        cov['t2f_hf_energy'].append(float(np.abs(lap[brain]).mean()))
        cov['n_slices'].append(float(k.sum()))
    cov = {k: np.array(v) for k, v in cov.items()}

    best = max(methods, key=lambda m: dice[m].mean())
    fail = (dice[best] < 0.5).astype(int)
    L = ['# BraTS ET — what separates the failing African cases? (%s target)\n' % a.target,
         '%d cases, %d methods. "Failure" = per-case Dice < 0.5 under the best method (%s); '
         '%d of %d cases (%.1f %%).\n' % (len(cases), len(methods), best, fail.sum(), len(fail),
                                          100 * fail.mean()),
         '\n## Per-method mean Dice on this target\n', '| method | mean | median | frac < 0.5 |',
         '|---|---|---|---|']
    for m in sorted(methods, key=lambda m: -dice[m].mean()):
        L.append('| %s | %.4f | %.4f | %.3f |'
                 % (m, dice[m].mean(), np.median(dice[m]), float((dice[m] < 0.5).mean())))

    L.append('\n## Which covariate separates failure from success?\n')
    L.append('AUC > 0.5 means a **larger** value goes with failure. Label-free covariates are the '
             'only actionable ones — a deployed model has no ground truth.\n')
    L.append('| covariate | kind | AUC(failure) | r with (RandConv − BigAug) per case |')
    L.append('|---|---|---|---|')
    lab_dep = {'et_volume_frac', 'et_frac_of_wt', 'et_components', 'et_t1c_contrast'}
    diff = dice.get('randconv', dice[best]) - dice.get('bigaug', dice[best])
    rows = []
    for k, v in cov.items():
        if np.std(v) < 1e-12:
            continue
        rows.append((k, 'label-dep' if k in lab_dep else '**label-free**',
                     auc(v, fail), float(np.corrcoef(v, diff)[0, 1])))
    for k, kind, au, rr in sorted(rows, key=lambda z: -abs(z[2] - 0.5)):
        L.append('| %s | %s | **%.3f** | %+.3f |' % (k, kind, au, rr))

    L.append('\n## Reading\n')
    top = max((r for r in rows), key=lambda z: abs(z[2] - 0.5))
    lf = [r for r in rows if 'free' in r[1]]
    top_lf = max(lf, key=lambda z: abs(z[2] - 0.5)) if lf else None
    L.append('- strongest separator overall: **%s** (AUC %.3f, %s)\n' % (top[0], top[2], top[1]))
    if top_lf:
        L.append('- strongest **label-free** separator: **%s** (AUC %.3f) — this is the only kind a '
                 'method could act on\n' % (top_lf[0], top_lf[2]))
    L.append('\n🔴 No mechanism is asserted here. A separator with AUC near 0.5 means the failing '
             'cases are not distinguished by that property, and a method conditioned on it cannot '
             'work. Anything above ~0.70 among the label-free covariates is worth a designed test.\n')
    txt = '\n'.join(L)
    print('\n' + txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
