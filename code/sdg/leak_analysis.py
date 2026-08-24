#!/usr/bin/env python
"""
E1 — how much does selecting the model on the target domain inflate the reported number?

The claim that target-peeking inflates results is old (DomainBed named "test-domain validation" as one
of three selection strategies and told authors to disclaim it). What is missing for medical
single-source segmentation is the magnitude. This measures it on the same runs, so nothing but
the selection rule differs:

  honest pick the iteration with the best SOURCE-validation score, report its target Dice
  leaked pick the iteration with the best TARGET Dice, report that
  last no selection at all -- report the final iteration

`leaked - honest` is the inflation a paper buys by peeking. `honest - last` is what a legitimate
selection rule is worth. Both are reported because a reader needs to know the second to judge the first.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    r = np.random.RandomState(seed)
    b = [v[r.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(OUTPUTS, 'sdg_leak'))
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    per = defaultdict(list)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, '*.json'))):
        r = json.load(open(f)); c = r['config']
        h = [x for x in r['history'] if 'target_now' in x]
        if len(h) < 4:
            continue
        sv = np.array([x['val'] for x in h])
        tg = np.array([x['target_now'] for x in h])
        honest = float(tg[int(sv.argmax())])
        leaked = float(tg.max())
        last = float(tg[-1])
        per[c['method']].append((honest, leaked, last))
        rows.append((c['source'], c['method'], c['seed'], honest, leaked, last,
                     int(h[int(sv.argmax())]['iter']), int(h[int(tg.argmax())]['iter'])))

    L = ['# E1 — the model-selection leak, measured (DG Prostate)\n',
         '%d runs. Each run is scored under three selection rules applied to its own trajectory, so '
         'the only difference between the columns is the rule.\n' % len(rows),
         '\n| rule | target Dice | 95 %% CI |', '|---|---|---|']
    allv = np.array([v for vs in per.values() for v in vs], float)
    names = ['**honest** — best source-val', '**leaked** — best target', 'no selection — last iter']
    for j, nm in enumerate(names):
        m, lo, hi = boot(allv[:, j])
        L.append('| %s | **%.4f** | [%.4f, %.4f] |' % (nm, m, lo, hi))

    infl = allv[:, 1] - allv[:, 0]
    gain = allv[:, 0] - allv[:, 2]
    mi, li, hi_ = boot(infl); mg, lg, hg = boot(gain)
    L.append('\n**Inflation from peeking at the target: %+.4f Dice [%+.4f, %+.4f].** '
             'For scale, an honest selection rule is worth %+.4f [%+.4f, %+.4f] over no selection '
             'at all.\n' % (mi, li, hi_, mg, lg, hg))
    L.append('Relative: peeking adds **%.1f %%** on top of the honestly-selected score.\n'
             % (100 * mi / allv[:, 0].mean()))

    L.append('\n## By method\n')
    L.append('| method | honest | leaked | **inflation** | inflation as %% of honest |')
    L.append('|---|---|---|---|---|')
    for m in sorted(per):
        v = np.array(per[m], float)
        d = (v[:, 1] - v[:, 0]).mean()
        L.append('| %s | %.4f | %.4f | **%+.4f** | %.1f %% |'
                 % (m, v[:, 0].mean(), v[:, 1].mean(), d, 100 * d / v[:, 0].mean()))

    it_h = np.array([r[6] for r in rows]); it_t = np.array([r[7] for r in rows])
    L.append('\n## Where the two rules stop\n')
    L.append('Source-val picks iteration %.0f on average; the target peaks at %.0f. '
             'They agree exactly in %.0f %% of runs.\n'
             % (it_h.mean(), it_t.mean(), 100 * (it_h == it_t).mean()))
    L.append('\n🔴 This does not discover that peeking inflates — DomainBed said so in 2021. It '
             'supplies the magnitude for medical single-source segmentation, which is what a reader '
             'needs in order to judge published numbers that do not state their selection rule.\n')

    txt = '\n'.join(L)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
