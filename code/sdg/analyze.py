#!/usr/bin/env python
"""
Aggregate the phase-1 SDG sweep into (a) the comparison table and (b) the inputs to the phase-2
failure analysis.

Statistics rules, inherited from the HD-Cal post-mortem and binding here:
  * the headline is an effect size with a CI, never a win count -- win counts jitter +-1.5-2 with
    the seed on identical data;
  * seeds are averaged *within* a (source, target) cell first, so the unit of analysis is the
    transfer pair, not the run; a per-run CI would treat three seeds of one pair as independent
    evidence, which is the pseudo-replication that cost this project a headline p-value once already;
  * the bootstrap resamples source domains, the level at which the pairs are correlated.

It also reports what a mean Dice hides: the fraction of target cases that fail outright. A method
that lifts the mean by pulling mid-range cases up is a different thing from one that rescues
failures, and the SDG literature almost never separates them.
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))

METHODS = ['erm', 'bigaug', 'randconv', 'mixstyle', 'dsu', 'maxstyle', 'ada']


def load(out_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        with open(f) as fh:
            r = json.load(fh)
        c = r['config']
        r['_tag'] = os.path.basename(f)[:-5]
        runs.append(r)
    return runs


def _dice(x):
    """runs before 2026-08-09 18:00 stored a bare Dice float per unit; later runs store a dict with
    precision/recall/volumes as well. Read both."""
    return float(x['dice']) if isinstance(x, dict) else float(x)


def per_unit(r, dom, field='dice'):
    """the per-case / per-image values of one run on one domain."""
    d = r['per_domain'][dom]
    if r['config']['bench'] == 'prostate':
        v = list(d['per_case'].values())
        if field != 'dice':
            return {'gland': np.array([x[field] for x in v if isinstance(x, dict)])}
        return {'gland': np.array([_dice(x) for x in v])}
    v = list(d['per_image'].values())
    if field != 'dice':
        return {s: np.array([x[s][field] for x in v if isinstance(x[s], dict)])
                for s in ('disc', 'cup')}
    return {s: np.array([_dice(x[s]) for x in v]) for s in ('disc', 'cup')}


MIN_CLUSTERS = 4


def boot_ci(vals, groups, n=10000, seed=0):
    """Cluster bootstrap over `groups` (source domains). Returns (mean, lo, hi, unit).

    A cluster bootstrap with two clusters is not a confidence interval -- resampling 2 items with
    replacement has only 3 distinct outcomes, and the interval it produces is meaningless (RIGA+ has
    exactly 2 source domains, and the naive version produced absurdly tight intervals such as
    [-0.0038, -0.0027]). Below MIN_CLUSTERS we resample transfer pairs instead and say so, which
    ignores the correlation within a source but at least is an interval rather than an artefact.
    """
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    gs = np.unique(groups)
    rng = np.random.RandomState(seed)
    if len(gs) >= MIN_CLUSTERS:
        out = [np.mean(np.concatenate([vals[groups == g] for g in rng.choice(gs, len(gs), True)]))
               for _ in range(n)]
        unit = 'source-clustered'
    else:
        out = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n)]
        unit = 'pair-level (only %d sources -- NOT source-clustered)' % len(gs)
    return float(vals.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(OUTPUTS, 'sdg'))
    ap.add_argument('--bench', required=True, choices=['prostate', 'riga'])
    ap.add_argument('--gt', default='r1')
    ap.add_argument('--md', default=None, help='write the markdown report here')
    ap.add_argument('--targets', default=None,
                    help='comma list restricting the unseen domains; use it to reproduce a published '
                         'averaging set exactly (RIGA+ papers average the three MESSIDOR domains '
                         'only, not all four unseen domains)')
    a = ap.parse_args()
    keep_t = set(a.targets.split(',')) if a.targets else None

    runs = [r for r in load(a.out_dir)
            if r['config']['bench'] == a.bench and r['config'].get('gt', 'r1') == a.gt]
    if not runs:
        print('no runs found for bench=%s gt=%s in %s' % (a.bench, a.gt, a.out_dir)); return

    structs = ['gland'] if a.bench == 'prostate' else ['disc', 'cup']
    # cell[(source, target, method)][struct] -> list over seeds of (mean dice, per-case array)
    cell = defaultdict(lambda: defaultdict(list))
    seeds, sources = set(), set()
    for r in runs:
        src, m, s = r['config']['source'], r['config']['method'], r['config']['seed']
        seeds.add(s); sources.add(src)
        for dom in r['per_domain']:
            if dom == src or (keep_t is not None and dom not in keep_t):
                continue
            pu = per_unit(r, dom)
            for st in structs:
                cell[(src, dom, m)][st].append(pu[st])

    L = []
    L.append('# Phase-1 SDG re-benchmark — %s (GT=%s)\n' % (a.bench, a.gt))
    L.append('%d runs · sources %s · seeds %s · 8,000 iters · strict source-val selection\n'
             % (len(runs), ','.join(sorted(sources)), ','.join(map(str, sorted(seeds)))))

    # ---- table 1: mean target Dice per source x method
    for st in structs:
        L.append('\n## Target-mean Dice (%s), averaged over the unseen domains\n' % st)
        L.append('| source | ' + ' | '.join(METHODS) + ' | ΔBigAug | ΔRandConv |')
        L.append('|---|' + '---|' * (len(METHODS) + 2))
        for src in sorted(sources):
            tg = sorted({t for (s_, t, m_) in cell if s_ == src})
            row, means = [], {}
            for m in METHODS:
                per_seed = []
                for si in range(len(seeds)):
                    vals = [cell[(src, t, m)][st][si].mean() for t in tg
                            if len(cell[(src, t, m)][st]) > si]
                    if vals:
                        per_seed.append(np.mean(vals))
                if not per_seed:
                    row.append('—'); means[m] = np.nan; continue
                means[m] = float(np.mean(per_seed))
                row.append('%.4f ±%.4f' % (means[m], float(np.std(per_seed))))
            L.append('| %s | %s | %+.4f | %+.4f |'
                     % (src, ' | '.join(row), means['bigaug'] - means['erm'],
                        means['randconv'] - means['erm']))

        # ---- effect size vs ERM, cluster-bootstrapped over source domains
        L.append('\n**Δ vs ERM per transfer pair (seeds averaged within a pair; '
                 '95 %% CI bootstrapped over source domains):**\n')
        L.append('| method | mean Δ Dice | 95% CI | pairs | resampling unit |')
        L.append('|---|---|---|---|---|')
        for m in METHODS[1:]:
            d, g = [], []
            for (src, t, mm) in list(cell):
                # both sides must actually have runs -- `cell` is a defaultdict and the table above
                # has already touched (and so created) empty entries for methods still queued.
                if mm != m or not cell[(src, t, m)][st] or not cell[(src, t, 'erm')][st]:
                    continue
                a_ = np.mean([v.mean() for v in cell[(src, t, m)][st]])
                b_ = np.mean([v.mean() for v in cell[(src, t, 'erm')][st]])
                d.append(a_ - b_); g.append(src)
            if d:
                mu, lo, hi, unit = boot_ci(d, g)
                L.append('| %s | %+.4f | [%+.4f, %+.4f] | %d | %s |' % (m, mu, lo, hi, len(d), unit))

        # ---- what the mean hides
        L.append('\n**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method '
                 'that rescues failures from one that polishes mid-range cases:\n')
        L.append('| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |')
        L.append('|---|---|---|---|---|')
        for m in METHODS:
            chunks = [np.concatenate(v[st]) for (s_, t_, m_), v in cell.items()
                      if m_ == m and v[st]]
            if not chunks:                      # sweep still running: method not started yet
                L.append('| %s | — | — | — | — |' % m); continue
            allv = np.concatenate(chunks)
            L.append('| %s | %.4f | %.4f–%.4f | **%.3f** | %.3f |'
                     % (m, np.median(allv), np.percentile(allv, 25), np.percentile(allv, 75),
                        float((allv < 0.10).mean()), float((allv < 0.50).mean())))

        # ---- per transfer pair, ERM only: where does the baseline actually break?
        L.append('\n**ERM per-pair target Dice (%s) — the failure map phase 2 starts from:**\n' % st)
        tgts = sorted({t for (_, t, _) in cell})
        L.append('| source \\ target | ' + ' | '.join(tgts) + ' |')
        L.append('|---|' + '---|' * len(tgts))
        for src in sorted(sources):
            cells = []
            for t in tgts:
                v = cell[(src, t, 'erm')][st]
                cells.append('—' if not v else '%.3f' % np.mean([x.mean() for x in v]))
            L.append('| %s | %s |' % (src, ' | '.join(cells)))

    txt = '\n'.join(L)
    print(txt)
    if a.md:
        with open(a.md, 'w') as fh:
            fh.write(txt + '\n')
        print('\n-> %s' % a.md)


if __name__ == '__main__':
    main()
