#!/usr/bin/env python
"""
Is there any headroom for an ADAPTIVE method at all?

After EFDR died, the question to answer before designing anything else is not "which adaptive rule?"
but "could any adaptive rule win?". The 162 head-to-head runs already on disk answer it, on CPU,
in minutes.

Three ceilings, each stronger than the last, all measured against the best *fixed* policy:

  fixed one method for everything -- what a normal paper proposes.
  oracle/source one method per source domain, chosen with knowledge of the targets. Achievable in
                principle by a method that adapts to the training set it is given.
  oracle/pair one method per (source, target) pair. NOT achievable without target knowledge; it is
                the absolute ceiling and is reported to bound the whole question.
  oracle/case one method per individual target case. The ceiling for per-sample adaptation, which
                is what EFDR and ADA both attempt.

If oracle/source is close to the best fixed method, adaptive selection has nothing to win and the
honest paper is a findings paper. If it is far above, a method has room -- and the gap between
oracle/source and oracle/case says whether that room is at the domain level or the sample level.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))

METHODS = ['erm', 'bigaug', 'randconv', 'mixstyle', 'dsu', 'maxstyle', 'ada']


def load(d, bench, gt='r1'):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, '%s_*.json' % bench))):
        r = json.load(open(f))
        if r['config'].get('gt', 'r1') != gt:
            continue
        runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(OUTPUTS, 'sdg_h2h'))
    ap.add_argument('--bench', required=True, choices=['prostate', 'riga', 'brats'])
    ap.add_argument('--struct', default='gland')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    runs = load(a.dir, a.bench)
    # per_case[(src, tgt, method)][case] = list over seeds of dice
    per_case = defaultdict(lambda: defaultdict(list))
    for r in runs:
        src, m = r['config']['source'], r['config']['method']
        for tgt, d in r['per_domain'].items():
            if tgt == src:
                continue
            items = d.get('per_case') or d.get('per_image')
            for k, v in items.items():
                val = v if a.bench in ('prostate','brats') else v[a.struct]
                per_case[(src, tgt, m)][k].append(val['dice'] if isinstance(val, dict) else val)

    pairs = sorted({(s, t) for (s, t, m) in per_case})
    srcs = sorted({s for s, t in pairs})
    have = lambda s, t, m: len(per_case[(s, t, m)]) > 0

    def pair_mean(s, t, m):
        cs = per_case[(s, t, m)]
        return float(np.mean([np.mean(v) for v in cs.values()])) if cs else np.nan

    M = {m: np.array([pair_mean(s, t, m) for (s, t) in pairs]) for m in METHODS
         if all(have(s, t, m) for (s, t) in pairs)}
    L = ['# Headroom for adaptive selection — %s (%s)\n' % (a.bench, a.struct),
         '%d transfer pairs, %d methods, seeds averaged within a pair.\n' % (len(pairs), len(M))]

    L.append('\n## Ceilings\n')
    L.append('| policy | mean target Dice | Δ vs best fixed |')
    L.append('|---|---|---|')
    fixed = {m: float(np.mean(v)) for m, v in M.items()}
    best_fixed_m = max(fixed, key=fixed.get)
    bf = fixed[best_fixed_m]
    L.append('| **best fixed** (%s) | **%.4f** | — |' % (best_fixed_m, bf))

    # oracle per source: pick the method maximising that source's mean over its targets
    tot, choice = [], {}
    for s in srcs:
        idx = [i for i, (ss, _) in enumerate(pairs) if ss == s]
        sc = {m: float(np.mean(v[idx])) for m, v in M.items()}
        b = max(sc, key=sc.get); choice[s] = (b, sc[b])
        tot.extend([M[b][i] for i in idx])
    o_src = float(np.mean(tot))
    L.append('| oracle per **source** | %.4f | %+.4f |' % (o_src, o_src - bf))

    stack = np.stack([M[m] for m in M])
    o_pair = float(np.mean(stack.max(0)))
    L.append('| oracle per **pair** (not achievable) | %.4f | %+.4f |' % (o_pair, o_pair - bf))

    cvals = []
    for (s, t) in pairs:
        keys = set.intersection(*[set(per_case[(s, t, m)]) for m in M])
        for k in keys:
            cvals.append(max(np.mean(per_case[(s, t, m)][k]) for m in M))
    o_case = float(np.mean(cvals))
    base_case = []
    for (s, t) in pairs:
        keys = set.intersection(*[set(per_case[(s, t, m)]) for m in M])
        for k in keys:
            base_case.append(np.mean(per_case[(s, t, best_fixed_m)][k]))
    L.append('| oracle per **case** (ceiling of per-sample adaptation) | %.4f | %+.4f |'
             % (o_case, o_case - float(np.mean(base_case))))

    L.append('\n## Which method would the source-level oracle pick?\n')
    L.append('| source | oracle choice | its mean | best-fixed (%s) mean |' % best_fixed_m)
    L.append('|---|---|---|---|')
    for s in srcs:
        idx = [i for i, (ss, _) in enumerate(pairs) if ss == s]
        L.append('| %s | **%s** | %.4f | %.4f |'
                 % (s, choice[s][0], choice[s][1], float(np.mean(M[best_fixed_m][idx]))))

    L.append('\n## Fixed-policy table\n')
    L.append('| method | mean over pairs |')
    L.append('|---|---|')
    for m in sorted(fixed, key=fixed.get, reverse=True):
        L.append('| %s | %.4f |' % (m, fixed[m]))

    txt = '\n'.join(L)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')
        print('\n-> %s' % a.md)


if __name__ == '__main__':
    main()
