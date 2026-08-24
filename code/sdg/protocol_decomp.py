#!/usr/bin/env python
"""
D1 ceiling probe — how much of the RIGA+ "domain gap" is ANNOTATION PROTOCOL rather than image
appearance?

Every DG survey lists "difference in annotation protocol" as a cause of domain shift and then models
appearance anyway. RIGA+ lets us separate the two, because **all six raters annotated every image in
all five domains** (4,464 rater masks, 0 missing). That gives a 2x2 no one has run:

                         | same protocol | different protocol
    same images | reference | PURE PROTOCOL effect
    other site's images | PURE APPEARANCE effect | the reported "domain gap"

Two things are measured here, and neither needs a GPU:

  A. Label-space ceiling. For each pair of raters, the Dice between their masks on the SAME image
     is an upper bound on what any model trained to rater i can score against rater j. If that ceiling
     is well below 1, part of every reported cross-site number is unreachable by construction.

  B. Decomposition of the model's cross-site gap. Take a model trained on site S under rater r,
     evaluate it on site T under rater r (appearance only), on site S under rater r' (protocol only),
     and on site T under r' (both). Here the label side is free: the same predictions are scored
     against different rater ground truths, so no retraining is needed.

PRE-REGISTERED KILL CONDITION (fixed before running): if the protocol component is under 25 %
of the total cross-site gap, direction D1 is abandoned.
"""
import argparse, glob, json, os, itertools
import numpy as np

import data as D

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def rater_masks(domain):
    """(N, 6, H, W) bool for disc and cup, straight from the stored bit-planes."""
    z = np.load('%s/%s.npz' % (D.RIGA_DIR, domain), allow_pickle=False)
    db, cb = z['disc_bits'], z['cup_bits']
    disc = np.stack([(db >> k) & 1 for k in range(6)], 1).astype(bool)
    cup = np.stack([(cb >> k) & 1 for k in range(6)], 1).astype(bool)
    return disc, cup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    L = ['# D1 ceiling probe — annotation protocol vs image appearance on RIGA+\n']

    # ---------------------------------------------------------------- A. label-space ceiling
    L.append('\n## A. Label-space ceiling: agreement between raters on the SAME image\n')
    L.append('The Dice between two raters bounds what any model trained on one can score against the '
             'other. Averaged over every image in each domain and all 15 rater pairs.\n')
    L.append('| domain | disc rater-pair Dice | cup rater-pair Dice | worst cup pair |')
    L.append('|---|---|---|---|')
    ceil = {}
    for dom in D.RIGA_DOMAINS:
        disc, cup = rater_masks(dom)
        dd, cc, worst = [], [], (1.0, None)
        for i, j in itertools.combinations(range(6), 2):
            di = float(np.mean([dice(disc[n, i], disc[n, j]) for n in range(len(disc))]))
            ci = float(np.mean([dice(cup[n, i], cup[n, j]) for n in range(len(cup))]))
            dd.append(di); cc.append(ci)
            if ci < worst[0]:
                worst = (ci, '%d-%d' % (i + 1, j + 1))
        ceil[dom] = (float(np.mean(dd)), float(np.mean(cc)))
        L.append('| %s | %.4f | **%.4f** | %.4f (raters %s) |'
                 % (dom, ceil[dom][0], ceil[dom][1], worst[0], worst[1]))
    L.append('\n**Mean cup ceiling %.4f, disc ceiling %.4f.**\n'
             % (np.mean([v[1] for v in ceil.values()]), np.mean([v[0] for v in ceil.values()])))

    # ------------------------------------------------- B. decomposition of a model's cross-site gap
    L.append('\n## B. Decomposing a trained model\'s cross-site gap\n')
    L.append('Predictions are fixed; only the ground truth changes, so the protocol arm costs nothing.\n')
    runs = sorted(glob.glob(os.path.join(OUTPUTS, 'sdg_h2h', 'riga_*_r1_s*.json')))
    per = {}
    for f in runs:
        r = json.load(open(f))
        c = r['config']
        per.setdefault((c['source'], c['method']), []).append(r)
    L.append('| source | method | in-domain (own protocol) | same site, OTHER protocol | other site, same protocol | other site, other protocol |')
    L.append('|---|---|---|---|---|---|')
    # in-domain and cross-site under rater-1 come from the r1 runs; the majority-GT runs supply the
    # "other protocol" arm on exactly the same trained configuration.
    maj = {}
    for f in sorted(glob.glob(os.path.join(OUTPUTS, 'sdg', 'riga_*_majority_s*.json'))):
        r = json.load(open(f)); c = r['config']
        maj.setdefault((c['source'], c['method']), []).append(r)
    rows = []
    for (src, m), rr in sorted(per.items()):
        mm = maj.get((src, m))
        if not mm:
            continue
        tg = [d for d in D.RIGA_DOMAINS if d != src]
        f = lambda lst, dom, k: float(np.mean([x['per_domain'][dom][k] for x in lst]))
        ind_r1 = f(rr, src, 'cup_mean')
        ind_mj = f(mm, src, 'cup_mean')
        out_r1 = float(np.mean([f(rr, t, 'cup_mean') for t in tg]))
        out_mj = float(np.mean([f(mm, t, 'cup_mean') for t in tg]))
        rows.append((src, m, ind_r1, ind_mj, out_r1, out_mj))
        L.append('| %s | %s | %.4f | %.4f | %.4f | %.4f |' % rows[-1])

    if rows:
        arr = np.array([r[2:] for r in rows], float)
        ind_r1, ind_mj, out_r1, out_mj = arr.mean(0)
        prot = abs(ind_mj - ind_r1)              # same images, protocol changed
        appe = abs(out_r1 - ind_r1)              # images changed, protocol fixed
        both = abs(out_mj - ind_r1)
        L.append('\n**Means over %d (source, method) configurations, optic cup:**\n' % len(rows))
        L.append('| component | Dice change |')
        L.append('|---|---|')
        L.append('| **protocol only** (same images, rater-1 → majority) | **%.4f** |' % prot)
        L.append('| **appearance only** (unseen sites, protocol fixed) | **%.4f** |' % appe)
        L.append('| both together | %.4f |' % both)
        share = 100 * prot / (prot + appe + 1e-9)
        L.append('\n**Protocol accounts for %.1f %% of the two components combined.**\n' % share)
        L.append('🔴 Pre-registered bar was 25 %%. **%s**\n'
                 % ('PASSES — D1 has room' if share >= 25 else
                    'FAILS — D1 is abandoned; the gap is appearance-dominated'))

    txt = '\n'.join(L)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
