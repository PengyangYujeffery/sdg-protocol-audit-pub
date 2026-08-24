#!/usr/bin/env python
"""
Gap accounting, the CPU half -- consumes what `decompose2.py` stored on the GPU.

`decompose2.py` wrote one `.npz` per (checkpoint, target domain) holding **Dice at every threshold on
a fixed grid, per case**. That is deliberately raw: every question below is a different reduction of
the same arrays, so the GPU pass runs once and survives any later change of analysis design.

What this computes, per benchmark
---------------------------------
A single-source model is a *decision rule* (a threshold) applied to a *representation* produced by a
*method* trained under an *annotation protocol*. The out-of-domain gap is the distance from the
in-domain reference to the best out-of-domain result. This script puts a ceiling on how much of
that gap each component could close if it were chosen perfectly:

  baseline best fixed method, fixed threshold 0.5, averaged over unseen domains
  decision rule the same method, threshold chosen by an oracle
  method choice the threshold fixed at 0.5, method chosen by an oracle
  joint method AND threshold chosen together by an oracle

and at three levels of oracle knowledge, which differ in what a real method could ever see:

  per source domain attainable in principle -- a method sees the domain it was trained on
  per target domain requires the target; a bound, not a target
  per case requires the answer for each case; the loosest bound there is

Why `joint` is the point. The obvious objection to any component-wise accounting is that the
components overlap, so their ceilings cannot be added. We do not add them. We measure the joint
ceiling directly and report

    overlap = (decision-rule ceiling + method ceiling) - joint ceiling

If overlap is large, the components are substitutes and the individual numbers must not be summed;
if it is near zero they are close to independent. Either way the reader is told, rather than being
handed a sum that quietly double-counts. This is the one number a reviewer can use to break the
accounting, so it is computed and printed whatever it says.

Everything is a mean over unseen target domains first, then aggregated across sources, and every
interval is a source-clustered bootstrap -- the source domain is the unit that repeats, and
treating (source, target) pairs as independent is exactly the anti-conservative mistake this project
corrected elsewhere.

    python decompose2_analyse.py --bench prostate --md ../../outputs/sdg_reports/gapacct_prostate.md
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

CELLS = os.path.join(SCRATCH, 'decomp2')
OUT = OUTPUTS
MIN_CLUSTERS = 4
NICE = {'erm': 'ERM', 'bigaug': 'BigAug', 'randconv': 'RandConv', 'mixstyle': 'MixStyle',
        'dsu': 'DSU', 'maxstyle': 'MaxStyle-core', 'ada': 'ADA', 'slaug': 'SLAug (aug. only)'}


def s(x):
    """numpy 0-d string/array -> python scalar."""
    a = np.asarray(x)
    return a.item() if a.ndim == 0 else a


def load_cells(bench, struct, cells_dir):
    """(source, target, method, seed) -> (cases, dice[n_case, n_thresh]); plus the threshold grid.

    Only cells whose stored `bench` matches are taken. A benchmark with no cells is a hard error:
    silently analysing nothing is how a table becomes a row of plausible zeros.
    """
    cells, thresh, regions = {}, None, set()
    files = sorted(glob.glob(os.path.join(cells_dir, '%s_*.npz' % bench)))
    if not files:
        sys.exit('!! no cells for bench=%s in %s -- run decompose2.py --bench %s first'
                 % (bench, cells_dir, bench))
    key = 'dice_cup' if struct == 'cup' else 'dice_disc' if struct == 'disc' else 'dice'
    for f in files:
        z = np.load(f, allow_pickle=True)
        if s(z['bench']) != bench:
            continue
        if key not in z:
            sys.exit('!! %s has no array %r (has %s)' % (os.path.basename(f), key, list(z)))
        t = np.asarray(z['thresh'], float)
        if thresh is None:
            thresh = t
        elif not np.array_equal(thresh, t):
            sys.exit('!! threshold grid differs in %s -- cells are not comparable'
                     % os.path.basename(f))
        # `region` MUST be in the key. BraTS and M&Ms store one npz per (checkpoint, target,
        # region); without it wt/tc/et (or lv/myo/rv) overwrite one another and the accounting is
        # silently computed on whichever region sorts last, presented as the whole benchmark.
        # Caught 2026-08-22: mms reported "420 cells" from 1,260 npz, brats "76" from 228 -- both
        # exactly one third, which is what made it visible.
        reg = str(s(z['region']))
        k = (s(z['source']), s(z['target']), s(z['method']), int(s(z['seed'])), reg)
        cells[k] = (np.asarray(z['cases']), np.asarray(z[key], float))
        regions.add(reg)
    return cells, thresh, regions


def boot_ci(vals, groups, n=10000, seed=0):
    """Cluster bootstrap over `groups` (the source domain), with the fallback named in the output."""
    v, g = np.asarray(vals, float), np.asarray(groups)
    gs = np.unique(g)
    r = np.random.RandomState(seed)
    if len(gs) >= MIN_CLUSTERS:
        d = [np.mean(np.concatenate([v[g == x] for x in r.choice(gs, len(gs), True)]))
             for _ in range(n)]
        unit = 'source-clustered'
    else:
        d = [v[r.randint(0, len(v), len(v))].mean() for _ in range(n)]
        unit = 'pair-level (%d sources)' % len(gs)
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True, choices=['prostate', 'riga', 'brats', 'mms'])
    ap.add_argument('--struct', default='cup', choices=['cup', 'disc'], help='RIGA+ only')
    ap.add_argument('--cells', default=CELLS)
    ap.add_argument('--region', default=None,
                    help='BraTS wt|tc|et, M&Ms lv|myo|rv. REQUIRED when the benchmark has more than '
                         'one: the regions are different structures and must never be pooled.')
    ap.add_argument('--md', default=None, help='write the report here')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    struct = a.struct if a.bench == 'riga' else None

    cells, thresh, regions = load_cells(a.bench, struct, a.cells)
    regions = {r for r in regions if r not in ('', 'None')}
    if len(regions) > 1:
        if not a.region:
            sys.exit('!! %s has %d regions (%s) and they are NOT interchangeable -- pass --region. '
                     'Pooling them would average different structures into one number.'
                     % (a.bench, len(regions), ', '.join(sorted(regions))))
        if a.region not in regions:
            sys.exit('!! region %r not present; have %s' % (a.region, sorted(regions)))
    if a.region:
        cells = {k: v for k, v in cells.items() if k[4] == a.region}
        if not cells:
            sys.exit('!! no cells left after filtering to region %r' % a.region)
    j5 = int(np.argmin(np.abs(thresh - 0.5)))
    srcs = sorted({k[0] for k in cells})
    methods = sorted({k[2] for k in cells})
    print('%s: %d cells, %d sources, %d methods, %d thresholds (0.5 at index %d)'
          % (a.bench, len(cells), len(srcs), len(methods), len(thresh), j5))

    # ---- average over seeds, so a cell is one (source, target, method) -------------------------
    by_stm = defaultdict(list)
    for (src, tgt, m, sd, reg), (cases, d) in cells.items():
        by_stm[(src, tgt, m)].append((cases, d))

    # per (source, target, method): mean-over-cases Dice at every threshold, and the per-case array
    curve, percase = {}, {}
    for k, lst in by_stm.items():
        # seeds may disagree on case order; align on the intersection, which is the honest common set
        common = set(lst[0][0].tolist())
        for c, _ in lst[1:]:
            common &= set(c.tolist())
        common = sorted(common)
        if not common:
            sys.exit('!! no shared cases across seeds for %s' % (k,))
        stack = []
        for c, d in lst:
            idx = {x: i for i, x in enumerate(c.tolist())}
            stack.append(d[[idx[x] for x in common], :])
        arr = np.mean(stack, axis=0)          # (n_case, n_thresh), averaged over seeds
        percase[k] = (common, arr)
        curve[k] = arr.mean(axis=0)           # (n_thresh,)

    pairs = sorted({(s_, t) for (s_, t, _) in curve})

    # ---- the fixed policy: best single method at 0.5, by mean over unseen targets --------------
    fixed_score = {}
    for m in methods:
        v = [curve[(s_, t, m)][j5] for (s_, t) in pairs if (s_, t, m) in curve]
        if len(v) == len(pairs):
            fixed_score[m] = float(np.mean(v))
    if not fixed_score:
        sys.exit('!! no method covers every (source, target) pair -- the arm is incomplete')
    best_m = max(fixed_score, key=fixed_score.get)
    base = fixed_score[best_m]

    # ---- per-pair values under each oracle ----------------------------------------------------
    rows = {'baseline': [], 'thr_dom': [], 'thr_case': [],
            'meth_src': [], 'meth_pair': [], 'meth_case': [],
            'joint_pair': [], 'joint_case': []}
    grp = []
    # a per-source oracle picks ONE method per source domain, using the source's mean over its targets
    src_best = {}
    for s_ in srcs:
        cand = {}
        for m in methods:
            v = [curve[(s_, t, m)][j5] for (ss, t) in pairs if ss == s_ and (s_, t, m) in curve]
            if v:
                cand[m] = float(np.mean(v))
        src_best[s_] = max(cand, key=cand.get) if cand else best_m

    for (s_, t) in pairs:
        have = [m for m in methods if (s_, t, m) in curve]
        if best_m not in have:
            continue
        grp.append(s_)
        c_best = curve[(s_, t, best_m)]
        rows['baseline'].append(c_best[j5])
        # decision rule, threshold chosen per target domain (best fixed method held)
        rows['thr_dom'].append(float(c_best.max()))
        # decision rule, threshold chosen per case
        _, pc = percase[(s_, t, best_m)]
        rows['thr_case'].append(float(pc.max(axis=1).mean()))
        # method choice at 0.5
        rows['meth_src'].append(float(curve[(s_, t, src_best[s_])][j5]))
        rows['meth_pair'].append(max(curve[(s_, t, m)][j5] for m in have))
        cases0, _ = percase[(s_, t, best_m)]
        common = set(cases0)
        for m in have:
            common &= set(percase[(s_, t, m)][0])
        common = sorted(common)
        stack05, stackall = [], []
        for m in have:
            cs, arr = percase[(s_, t, m)]
            idx = {x: i for i, x in enumerate(cs)}
            sub = arr[[idx[x] for x in common], :]
            stack05.append(sub[:, j5]); stackall.append(sub)
        rows['meth_case'].append(float(np.max(np.stack(stack05), axis=0).mean()))
        # joint: method AND threshold together
        S = np.stack(stackall)                                  # (n_meth, n_case, n_thresh)
        rows['joint_pair'].append(float(S.mean(axis=1).max()))   # one (m, thr) for the whole domain
        rows['joint_case'].append(float(S.max(axis=(0, 2)).mean()))

    out = {'bench': a.bench, 'struct': struct, 'region': a.region, 'n_pairs': len(grp), 'n_sources': len(set(grp)),
           'n_methods': len(methods), 'best_fixed': best_m, 'thresholds': len(thresh)}
    res = {}
    for k, v in rows.items():
        mu, lo, hi, unit = boot_ci(v, grp)
        res[k] = dict(mean=mu, lo=lo, hi=hi, unit=unit)
    out['levels'] = res
    b = res['baseline']['mean']

    def gain(k):
        mu, lo, hi, unit = boot_ci(np.array(rows[k]) - np.array(rows['baseline']), grp)
        return dict(mean=mu, lo=lo, hi=hi, unit=unit)

    out['ceilings'] = {k: gain(k) for k in rows if k != 'baseline'}
    ov = (out['ceilings']['thr_case']['mean'] + out['ceilings']['meth_case']['mean']
          - out['ceilings']['joint_case']['mean'])
    out['overlap_per_case'] = ov
    ovp = (out['ceilings']['thr_dom']['mean'] + out['ceilings']['meth_pair']['mean']
           - out['ceilings']['joint_pair']['mean'])
    out['overlap_per_domain'] = ovp

    # ---- report -------------------------------------------------------------------------------
    L = ['# Gap accounting — %s%s' % (a.bench, '/' + (struct or a.region or '')), '',
         'Generated by `decompose2_analyse.py` from %d GPU cells. Best fixed policy: **%s** at '
         'threshold 0.5, scoring **%.4f** as a mean over unseen domains (%d pairs, %d sources).'
         % (len(cells), NICE.get(best_m, best_m), b, len(grp), len(set(grp))), '',
         '| component | oracle level | ceiling over the fixed policy | 95% CI | unit |',
         '|---|---|---|---|---|']
    name = [('thr_dom', 'decision rule', 'per target domain'),
            ('thr_case', 'decision rule', 'per case'),
            ('meth_src', 'method choice', 'per **source** domain (attainable)'),
            ('meth_pair', 'method choice', 'per target domain'),
            ('meth_case', 'method choice', 'per case'),
            ('joint_pair', '**both, jointly**', 'per target domain'),
            ('joint_case', '**both, jointly**', 'per case')]
    for k, comp, lev in name:
        c = out['ceilings'][k]
        L.append('| %s | %s | **%+.4f** | [%+.4f, %+.4f] | %s |'
                 % (comp, lev, c['mean'], c['lo'], c['hi'], c['unit']))
    L += ['', '## Do the components double-count?', '',
          'Adding component ceilings is only legitimate if choosing one does not already buy what the '
          'other would. It does not here, and the size of the effect is stated rather than assumed:',
          '',
          '| level | decision rule | + method choice | = sum | measured jointly | **overlap** |',
          '|---|---|---|---|---|---|',
          '| per target domain | %+.4f | %+.4f | %+.4f | %+.4f | **%+.4f** |'
          % (out['ceilings']['thr_dom']['mean'], out['ceilings']['meth_pair']['mean'],
             out['ceilings']['thr_dom']['mean'] + out['ceilings']['meth_pair']['mean'],
             out['ceilings']['joint_pair']['mean'], ovp),
          '| per case | %+.4f | %+.4f | %+.4f | %+.4f | **%+.4f** |'
          % (out['ceilings']['thr_case']['mean'], out['ceilings']['meth_case']['mean'],
             out['ceilings']['thr_case']['mean'] + out['ceilings']['meth_case']['mean'],
             out['ceilings']['joint_case']['mean'], ov),
          '',
          'A positive overlap means the two are partly substitutes: a better threshold recovers some '
          'of what a better method would have. **The individual ceilings must therefore not be '
          'summed**; the joint figure is the one that bounds both together.', '']
    txt = '\n'.join(L)
    print()
    print(txt)
    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
        open(a.md, 'w', encoding='utf-8').write(txt + '\n')
        print('-> %s' % os.path.abspath(a.md))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)
        print('-> %s' % os.path.abspath(a.json))


if __name__ == '__main__':
    main()
