#!/usr/bin/env python
"""
Generate every table in the JBHI paper from the run JSONs, in one pass, as Markdown and LaTeX.

Scope-freeze rule 2: *no number in the paper is hand-copied.* Everything a reviewer can see is
produced here, from `outputs/*/**.json`, so a rerun regenerates the paper's numbers exactly. Each
table prints the run count it was built from, and a table whose runs are incomplete is emitted with a
loud INCOMPLETE marker rather than silently averaging over whatever happens to be on disk.

Three outputs:
  tables.md human-readable, for the repo record
  tables.tex \\begin{table} blocks, \\input by the manuscript
  numbers.tex \\newcommand macros for every number quoted in *prose*, so the running text is
               generated too and cannot drift from the tables (this is the part rule 2 kept missing)

Two analyses (boundary metrics, inter-rater agreement) are produced by scripts that need the GPU and
the raw npz volumes, so they are not recomputed here: their released Markdown reports are parsed
strictly, and a report that is missing or whose rows do not parse yields INCOMPLETE rather than a
gap in the table.

    python make_tables.py --out_dir .../outputs/paper_tables
"""
import argparse, glob, io, json, os, re
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))

OUT = OUTPUTS
METHODS = ['erm', 'bigaug', 'randconv', 'mixstyle', 'dsu', 'maxstyle', 'ada', 'slaug']
NICE = {'erm': 'ERM', 'bigaug': 'BigAug', 'randconv': 'RandConv', 'mixstyle': 'MixStyle',
        'dsu': 'DSU', 'maxstyle': 'MaxStyle-core', 'ada': 'ADA',
        # 'aug. only': SLAug's Saliency-Balancing Fusion needs the network gradient, so it is a
        # training-loop change and is excluded from this arm. The label carries that, so no table
        # can present this as the complete published method. See slaug.py.
        'slaug': 'SLAug (aug.)'}
MIN_CLUSTERS = 4
INC = '— INCOMPLETE —'

NUM = {}          # macro name -> formatted value, written to numbers.tex
WARN = []         # anything that makes a table untrustworthy, echoed at the end


def note(msg):
    WARN.append(msg)
    print('  !! ' + msg)


def num(key, value, fmt='%.4f'):
    """Record a number for numbers.tex and return it formatted, so a table and the prose agree.

    The key becomes a LaTeX control sequence, and a control sequence may contain letters only.
    A key like `hd95BigAug` silently parses as `\\nhd` followed by the text `95BigAug`, which is an
    undefined-control-sequence error at build time rather than a wrong number -- but only if someone
    compiles. The assert makes it impossible to emit one in the first place.
    """
    assert key.isalpha(), 'macro key must be letters only (LaTeX control sequence): %r' % key
    NUM[key] = fmt % value if isinstance(value, float) else str(value)
    return NUM[key]


def load(d, pat='*.json'):
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, d, pat))):
        try:
            with open(f, encoding='utf-8') as fh:
                out.append(json.load(fh))
        except Exception as e:
            note('unreadable %s: %s' % (f, e))
    return out


def boot(vals, groups, n=10000, seed=0):
    """Cluster bootstrap over `groups`; falls back to pair-level below MIN_CLUSTERS and says so."""
    v, g = np.asarray(vals, float), np.asarray(groups)
    gs = np.unique(g)
    r = np.random.RandomState(seed)
    if len(gs) >= MIN_CLUSTERS:
        d = [np.mean(np.concatenate([v[g == x] for x in r.choice(gs, len(gs), True)]))
             for _ in range(n)]
        unit = 'source-clustered'
    else:
        d = [v[r.randint(0, len(v), len(v))].mean() for _ in range(n)]
        unit = 'pair-level (%d source%s)' % (len(gs), '' if len(gs) == 1 else 's')
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), unit


def per_target(r, struct=None):
    """mean over unseen domains of one run, and the per-source key."""
    src = r['config']['source']
    vals = []
    for dom, d in r['per_domain'].items():
        if dom == src:
            continue
        vals.append(d['cup_mean'] if struct == 'cup' else
                    d['disc_mean'] if struct == 'disc' else d['dice_mean'])
    return src, float(np.mean(vals))


def delta_table(runs, struct=None, label=''):
    """Δ vs ERM per (source) with a CI; returns markdown rows and the run count."""
    by = defaultdict(list)
    for r in runs:
        src, v = per_target(r, struct)
        by[(src, r['config']['method'])].append(v)
    srcs = sorted({s for s, _ in by})
    rows = []
    for m in METHODS:
        d, g = [], []
        for s in srcs:
            if (s, m) not in by or (s, 'erm') not in by:
                continue
            d.append(np.mean(by[(s, m)]) - np.mean(by[(s, 'erm')])); g.append(s)
        if not d or m == 'erm':
            continue
        mu, lo, hi, unit = boot(d, g)
        rows.append((NICE[m], mu, lo, hi, len(d), unit))
    erm = float(np.mean([np.mean(by[(s, 'erm')]) for s in srcs if (s, 'erm') in by])) \
        if any((s, 'erm') in by for s in srcs) else float('nan')
    return rows, erm, len(runs), srcs


def expect(name, runs, n_expected):
    """Check against the target run counts. A short table is a missing run, not a result.

    Distinguishes the two directions, because during a scale-up they mean opposite things:
      fewer than target -> the arm is incomplete; do not write from this table.
      more than target -> a new arm landed and the constant here is stale; update it.
    """
    n = len(runs)
    if n < n_expected:
        note('%s has %d of %d runs (%.0f%%) — arm INCOMPLETE, do not write from this table'
             % (name, n, n_expected, 100.0 * n / max(n_expected, 1)))
        return False
    if n > n_expected:
        note('%s has %d runs, MORE than the recorded target %d — a new arm landed; update the '
             'constant in make_tables.py' % (name, n, n_expected))
    return True


def md_table(head, rows):
    L = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    L += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(L)


def tex_table(head, rows, caption, label, wide=False):
    esc = lambda s: (str(s).replace('%', '\\%').replace('_', '\\_')
                     .replace('Δ', '$\\Delta$').replace('±', '$\\pm$').replace('≈', '$\\approx$'))
    # Placement stays [t]. Do not "fix" this to [!tb] to save a page: tried 2026-08-24, recovered
    # nothing, only redistributed floats. The overflow is not a packing problem.
    #
    # `wide=True` emits table*, spanning both columns. Several of these tables carry text in their
    # cells ("SLAug (aug. only) 0.568") and cannot be squeezed into one IEEE column: the 2026-08-24
    # proof had six of them overfull by up to 199pt against a ~252pt column, printing one table over
    # its neighbour. \footnotesize and a tighter \tabcolsep are the standard IEEE remedy and are
    # applied to every table; table* is for the ones that still do not fit.
    env = 'table*' if wide else 'table'
    L = ['\\begin{%s}[t]' % env, '\\centering',
         '\\caption{%s}' % caption, '\\label{%s}' % label,
         # scriptsize + 2pt gutters is what makes eight of the ten tables fit one IEEE column.
         # Measured 2026-08-24: footnotesize/6pt left six tables overfull by 56-199pt (one printed
         # over its neighbour in the proof); scriptsize/3pt cut that to 14-30pt; 2pt clears it.
         # The two gap-accounting tables carry seven columns and stay table*.
         '\\scriptsize', '\\setlength{\\tabcolsep}{2pt}',
         '\\renewcommand{\\arraystretch}{1.05}',
         '\\begin{tabular}{%s}' % ('l' + 'r' * (len(head) - 1)), '\\hline',
         ' & '.join(esc(h) for h in head) + ' \\\\', '\\hline']
    L += [' & '.join(esc(c) for c in r) + ' \\\\' for r in rows]
    L += ['\\hline', '\\end{tabular}', '\\end{%s}' % env]
    return '\n'.join(L)


# --------------------------------------------------------------------------- report parsing
def parse_md_rows(path, ncols):
    """Strictly parse the data rows of the first Markdown table with `ncols` columns.

    A row counts as data only when every cell after the first parses as a number. Anything
    looser swallows the header — `median HD95` contains a digit — and a header row silently becomes
    a row of NaN. Returns a list of cell-lists, or None; a report whose format drifted must surface
    as INCOMPLETE, never as a silently wrong row.
    """
    if not os.path.exists(path):
        note('report missing: %s' % path)
        return None
    rows = []
    for ln in open(path, encoding='utf-8'):
        ln = ln.strip()
        if not ln.startswith('|'):
            continue
        cells = [c.strip().strip('*').strip() for c in ln.strip('|').split('|')]
        if len(cells) != ncols:
            continue
        if set(''.join(cells)) <= set('-: '):          # the |---|---| separator
            continue
        vals = [re.match(r'^[-+]?[0-9]*\.?[0-9]+', c) for c in cells[1:]]
        if all(vals):
            rows.append([cells[0]] + [v.group(0) for v in vals])
    if not rows:
        note('no parseable %d-column rows in %s' % (ncols, path))
        return None
    return rows


def f(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')


# --------------------------------------------------------------------------- oracle ceilings
def per_case_map(runs, struct=None):
    """(src, tgt, method) -> {unit: [dice over seeds]} — the input every ceiling needs."""
    pc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        src, m = r['config']['source'], r['config']['method']
        for tgt, d in r['per_domain'].items():
            if tgt == src:
                continue
            items = d.get('per_case') or d.get('per_image')
            if not items:
                continue
            for k, v in items.items():
                val = v[struct] if struct else v
                pc[(src, tgt, m)][k].append(val['dice'] if isinstance(val, dict) else val)
    return pc


def ceilings(runs, struct=None):
    """best fixed / oracle-per-source / oracle-per-pair / oracle-per-case, as in headroom.py."""
    pc = per_case_map(runs, struct)
    pairs = sorted({(s, t) for (s, t, _) in pc})
    if not pairs:
        return None
    srcs = sorted({s for s, _ in pairs})
    have = lambda m: all(len(pc[(s, t, m)]) > 0 for (s, t) in pairs)
    pair_mean = lambda s, t, m: float(np.mean([np.mean(v) for v in pc[(s, t, m)].values()]))
    M = {m: np.array([pair_mean(s, t, m) for (s, t) in pairs]) for m in METHODS if have(m)}
    if len(M) < 2:
        return None
    fixed = {m: float(np.mean(v)) for m, v in M.items()}
    best = max(fixed, key=fixed.get)
    bf = fixed[best]

    tot, choice = [], {}
    for s in srcs:
        idx = [i for i, (ss, _) in enumerate(pairs) if ss == s]
        sc = {m: float(np.mean(v[idx])) for m, v in M.items()}
        b = max(sc, key=sc.get); choice[s] = b
        tot.extend([M[b][i] for i in idx])
    o_src = float(np.mean(tot))
    o_pair = float(np.mean(np.stack([M[m] for m in M]).max(0)))

    o_case, base_case = [], []
    for (s, t) in pairs:
        keys = set.intersection(*[set(pc[(s, t, m)]) for m in M])
        for k in keys:
            o_case.append(max(np.mean(pc[(s, t, m)][k]) for m in M))
            base_case.append(np.mean(pc[(s, t, best)][k]))
    return dict(best=best, fixed=bf, o_src=o_src, o_pair=o_pair,
                o_case=float(np.mean(o_case)) - float(np.mean(base_case)),
                n_pairs=len(pairs), n_methods=len(M), agree=sum(choice[s] == best for s in srcs),
                n_srcs=len(srcs))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--out_root', default=None, help='override the outputs/ root (for local tests)')
    ap.add_argument('--keep_knobs_table', action='store_true',
                    help='emit the protocol-knob summary table into tables.tex. OFF by default: '
                         'Fig. 0 (make_fig0.py) carries the same content with the entry points '
                         'shown, and the paper should not state its thesis twice.')
    ap.add_argument('--keep_indomain_table', action='store_true',
                    help='emit the in-domain inflation table into tables.tex. OFF by default: every '
                         'one of its numbers is already a prose macro, so it duplicated the text.')
    ap.add_argument('--keep_freeze_table', action='store_true',
                    help='emit T11 (freeze / representation preservation) into tables.tex as well as '
                         'tables.md. OFF by default: the result is non-significant and prostate-only, '
                         'and the manuscript does not reference it. See the note in main().')
    ap.add_argument('--split_oracle', action='store_true',
                    help='write T6 (oracle ceilings) to tables_oracle.tex instead of tables.tex. '
                         'OFF by default -- see the note in main(). Only needed if work7 is ever '
                         'split into two papers again.')
    a = ap.parse_args()
    if a.out_root:
        OUT = a.out_root
    if not a.out_dir:
        a.out_dir = os.path.join(OUT, 'paper_tables')
    RPT = os.path.join(OUT, 'sdg_reports')
    os.makedirs(a.out_dir, exist_ok=True)
    # Two tables are opt-in. T6 (oracle ceilings) goes to tables.tex by default; --split_oracle
    # diverts it to tables_oracle.tex. T11 (freeze) goes to tables.md only -- its effect is
    # non-significant and prostate-only; --keep_freeze_table puts it in the paper.
    # Both choices live here, not in tables.tex, which is generated and must never be hand-edited.
    md, tex, tex_oracle = [], [], []

    # ---------------------------------------------------------------- study scale, counted
    #
    # The abstract and the contributions quote the size of the study. Typing those by hand is how
    # a paper ends up claiming four benchmarks while its tables show three -- the mirror image of the
    # error this paper is about. They are counted from the run records instead, so the prose cannot
    # outrun the evidence.
    # This list defines TotalRuns: the fp32 deterministic testbed the abstract describes, not "the
    # runs that reached a table" -- so sdg_freeze stays in although its table was cut, and
    # sdg_h2h_recheck (reproducibility repeats of cells already counted) stays out, along with the
    # fp16 exploratory, probe and smoke arms. Adding an arm here changes a headline number.
    allruns, benches, methods, backbones = [], set(), set(), set()
    for d in ('sdg_h2h', 'sdg_brats', 'sdg_backbone', 'sdg_slice', 'sdg_leak', 'sdg_mms',
              'sdg_freeze', 'sdg_africa'):
        for r in load(d):
            c = r.get('config', {})
            allruns.append(r)
            benches.add(c.get('bench')); methods.add(c.get('method'))
            backbones.add(c.get('backbone') or 'scratch')
    benches.discard(None); methods.discard(None)
    num('Benchmarks', len(benches), '%d')
    num('Methods', len(methods), '%d')
    num('Backbones', len(backbones), '%d')
    num('TotalRuns', len(allruns), '%d')
    md.append('\nStudy scale, counted from the run records: **%d benchmarks, %d methods, '
              '%d backbones, %d runs** (%s).\n'
              % (len(benches), len(methods), len(backbones), len(allruns),
                 ', '.join(sorted(x for x in benches))))

    # ---------------------------------------------------------------- T1 main comparison
    md.append('# Paper tables — generated, never hand-copied\n')
    md.append('\n## T1. Method effects vs ERM, scratch 2D U-Net\n')
    h2h_p = load('sdg_h2h', 'prostate_*_fg.json')
    h2h_r = load('sdg_h2h', 'riga_*.json')
    expect('sdg_h2h', load('sdg_h2h'), 418)   # 168 base +63 E6a +84 E6b +70 E6c +33 E9b slaug
    cells = [('DG Prostate (gland)', h2h_p, None),
             ('RIGA+ (disc)', [r for r in h2h_r if r['config'].get('gt') == 'r1'], 'disc'),
             ('RIGA+ (cup)', [r for r in h2h_r if r['config'].get('gt') == 'r1'], 'cup'),
             ('BraTS WT', load('sdg_brats', 'brats_*_wt.json'), None),
             ('BraTS TC', load('sdg_brats', 'brats_*_tc.json'), None),
             ('BraTS ET', load('sdg_brats', 'brats_*_et.json'), None),
             # M&Ms cardiac, added 2026-08-22. The arm ran whole on MeluXina (nothing of it exists
             # on Sonic), so its runs are on a different torch build and no cross-cluster
             # bit-reproducibility is claimed for it -- the paper must say so. Three structures,
             # five centres as source, 315 runs.
             ('M\&Ms (LV cavity)', load('sdg_mms', 'mms_*_lv.json'), None),
             ('M\&Ms (LV myocardium)', load('sdg_mms', 'mms_*_myo.json'), None),
             ('M\&Ms (RV cavity)', load('sdg_mms', 'mms_*_rv.json'), None)]
    expect('sdg_mms', load('sdg_mms'), 315)   # 5 centres x 3 regions x 7 methods x 3 seeds
    expect('sdg_brats', load('sdg_brats'), 114)  # 105 + the 9 SLAug runs from E15 (2026-08-21)  # 63 base + 42 E6d
    rows = []
    for name, runs, st in cells:
        if not runs:
            rows.append((name, INC, '', '', '', '', '')); continue
        rr, erm, n, srcs = delta_table(runs, st)
        got = {x[0]: x for x in rr}
        key = re.sub(r'[^A-Za-z]', '', name)
        line = [name, num('erm' + key, erm)]
        for m in ['BigAug', 'RandConv', 'MaxStyle-core', 'ADA']:
            line.append(num('d' + re.sub(r'[^A-Za-z]', '', m) + key, got[m][1], '%+.4f')
                        if m in got else '—')
        # The bootstrap unit is part of the result, not a footnote: an interval over 2 clusters is
        # not an interval. Carrying it in the table stops the Methods text and the numbers drifting
        # apart when a benchmark gains source domains, which is exactly what happened to RIGA+.
        #
        # It is abbreviated to a one-character flag, expanded once in the caption. Spelled out
        # ("228 runs / 6 src, source-clustered") this single column made the table 199pt wider than
        # an IEEE column and it printed over its neighbour. The unit is still per-row, and prose
        # reads it from \nciUnit* rather than from this cell.
        unit = rr[0][5] if rr else 'n/a'
        flag = 'c' if unit.startswith('source-clustered') else 'p'
        line.append('%d / %d$^{%s}$' % (n, len(srcs), flag))
        rows.append(tuple(line))
        num('ciUnit' + key, unit.split(' ')[0], '%s')
    # Feature-space family (MixStyle, DSU, MaxStyle-core) across EVERY benchmark, so the prose can
    # say "nowhere exceeds" without anyone typing a benchmark count that goes stale.
    featvals = []
    for name, runs, st in cells:
        if not runs:
            continue
        rr, _, _, _ = delta_table(runs, st)
        for nm, mu, _, _, _, _ in rr:
            if nm in ('MixStyle', 'DSU', 'MaxStyle-core'):
                featvals.append(mu)
    if featvals:
        num('featMax', max(featvals), '%+.4f')
        num('featCells', len(featvals), '%d')
    head = ['benchmark', 'ERM', 'ΔBigAug', 'ΔRandConv', 'ΔMaxStyle', 'ΔADA', 'n']
    md.append(md_table(head, rows))
    tex.append(tex_table(head, rows,
                         'Method effect relative to ERM on each benchmark, from-scratch 2D U-Net, '
                         'three seeds, strict source-validation model selection. The last column is '
                         'runs / source domains; $^{c}$ marks a source-clustered bootstrap and '
                         '$^{p}$ a pair-level one, used where there are too few clusters for the '
                         'former.', 'tab:main'))

    # ---------------------------------------------------------------- T2 the leak
    md.append('\n\n## T2. Model-selection leak (DG Prostate)\n')
    lk = load('sdg_leak')
    expect('sdg_leak', lk, 54)
    if lk:
        acc = []
        for r in lk:
            h = [x for x in r['history'] if 'target_now' in x]
            if len(h) < 4:
                note('run without target tracking in sdg_leak: %s' % r['config'].get('source'))
                continue
            sv = np.array([x['val'] for x in h]); tg = np.array([x['target_now'] for x in h])
            acc.append((r['config']['method'], float(tg[int(sv.argmax())]), float(tg.max()),
                        float(tg[-1])))
        A = np.array([x[1:] for x in acc], float)
        rows = [('honest (source-val)', num('leakHonest', A[:, 0].mean()), '—'),
                ('leaked (target-val)', num('leakLeaked', A[:, 1].mean()),
                 num('leakDelta', (A[:, 1] - A[:, 0]).mean(), '%+.4f')),
                ('no selection (last)', num('leakLast', A[:, 2].mean()),
                 num('leakLastDelta', (A[:, 2] - A[:, 0]).mean(), '%+.4f'))]
        # The table column is (last - honest); the prose asks the opposite question ("what is an
        # honest rule worth over no selection at all?"), so emit that direction as its own macro
        # rather than letting a sentence quote a number with the wrong sign.
        num('leakHonestOverLast', (A[:, 0] - A[:, 2]).mean(), '%+.4f')
        pct = 100 * (A[:, 1] - A[:, 0]).mean() / A[:, 0].mean()
        num('leakPct', pct, '%.1f')
        num('leakRuns', len(acc), '%d')
        md.append(md_table(['selection rule', 'target Dice', 'vs honest'], rows))
        md.append('\n\n%d runs. Inflation from peeking: **%s Dice (%s %%)**.\n'
                  % (len(acc), NUM['leakDelta'], NUM['leakPct']))
        # per-method, because the leak widening the gaps between methods is the sharper claim
        pm = []
        for m in ['erm', 'bigaug', 'randconv']:
            k = [i for i, x in enumerate(acc) if x[0] == m]
            if not k:
                continue
            d = (A[k, 1] - A[k, 0]).mean()
            pm.append((NICE[m], '%.4f' % A[k, 0].mean(), '%+.4f' % d,
                       num('leakPct' + NICE[m], 100 * d / A[k, 0].mean(), '%.1f') + ' %'))
        md.append('\n' + md_table(['method', 'honest', 'inflation', 'relative'], pm))
        tex.append(tex_table(['selection rule', 'target Dice', 'vs honest'], rows,
                             'Effect of the model-selection rule, measured on the same runs: the '
                             'three rules are applied to one trajectory, so nothing but the rule '
                             'differs.', 'tab:leak'))
    else:
        md.append(INC)

    # ---------------------------------------------------------------- T3 backbone
    md.append('\n\n## T3. Backbone arm (DG Prostate)\n')
    bb = load('sdg_backbone', 'prostate_*resnet34.json')
    # 186 was the ImageNet-ResNet34 arm alone. E10 added the DINOv2 arm (4 jobs, 12-16 h each,
    # completed 2026-08-18/21) and E5c rebuilt the RIGA+ backbone arm, taking the directory to 312.
    # The constant is updated rather than the assertion removed: an unexplained count is exactly what
    # this check exists to catch, and it did catch this one.
    expect('sdg_backbone', load('sdg_backbone'), 312)
    if bb and h2h_p:
        r1, e1, n1, _ = delta_table(h2h_p)
        r2, e2, n2, _ = delta_table(bb)
        g1 = {x[0]: x for x in r1}; g2 = {x[0]: x for x in r2}
        rows = [(NICE[m], '%+.4f' % g1[NICE[m]][1] if NICE[m] in g1 else '—',
                 '%+.4f' % g2[NICE[m]][1] if NICE[m] in g2 else '—')
                for m in METHODS if m != 'erm']
        rows.append(('*ERM absolute*', num('ermScratch', e1), num('ermPretrained', e2)))
        num('backboneErmGain', e2 - e1, '%+.4f')
        # spread of the method effects under each backbone — the compression is the claim
        sp = lambda g: (min(v[1] for v in g.values()), max(v[1] for v in g.values()))
        if g1 and g2:
            lo1, hi1 = sp(g1); lo2, hi2 = sp(g2)
            num('spreadScratchLo', lo1); num('spreadScratchHi', hi1)
            num('spreadPreLo', lo2); num('spreadPreHi', hi2)
            md.append('Method-effect spread: scratch [%.4f, %.4f] -> pretrained [%.4f, %.4f].\n'
                      % (lo1, hi1, lo2, hi2))
        md.append(md_table(['method', 'Δ scratch U-Net', 'Δ ImageNet ResNet-34'], rows))
        tex.append(tex_table(['method', 'Δ scratch', 'Δ pretrained'], rows,
                             'Method effects under two backbones. The ranking is not preserved and '
                             'the spread compresses.', 'tab:backbone'))
    else:
        md.append(INC)

    # ---------------------------------------------------------------- T4 slice policy (E4)
    md.append('\n\n## T4. Slice policy — the DG gain depends on a convention nobody states\n')
    sl = load('sdg_slice', 'prostate_*.json')
    expect('sdg_slice', sl, 54)
    if sl and h2h_p:
        by_pol = defaultdict(list)
        for r in sl + h2h_p:
            by_pol[r['config'].get('slices', 'all')].append(r)
        rows, gains = [], {}
        for pol in ['fg', 'all']:
            runs = by_pol.get(pol, [])
            if not runs:
                rows.append((pol, INC, '', '', '')); continue
            rr, erm, n, srcs = delta_table(runs)
            got = {x[0]: x for x in rr}
            gains[pol] = got
            rows.append((pol, num('erm' + pol.capitalize(), erm),
                         num('dBigAug' + pol.capitalize(), got['BigAug'][1], '%+.4f')
                         if 'BigAug' in got else '—',
                         num('dRandConv' + pol.capitalize(), got['RandConv'][1], '%+.4f')
                         if 'RandConv' in got else '—',
                         '%d runs / %d src' % (n, len(srcs))))
        if 'fg' in gains and 'all' in gains and 'BigAug' in gains['fg'] \
                and 'BigAug' in gains['all']:
            gf, ga = gains['fg']['BigAug'][1], gains['all']['BigAug'][1]
            num('sliceGainFg', gf, '%+.4f'); num('sliceGainAll', ga, '%+.4f')
            num('sliceGainShift', 100 * (gf - ga) / abs(gf), '%.0f')
            md.append("BigAug's measured DG gain moves %s %% between the two policies "
                      '(%+.4f under `fg`, %+.4f under `all`), with the methods and the data held '
                      'fixed.\n' % (NUM['sliceGainShift'], gf, ga))
        md.append(md_table(['slice policy', 'ERM', 'ΔBigAug', 'ΔRandConv', 'n'], rows))
        tex.append(tex_table(['slice policy', 'ERM', 'ΔBigAug', 'ΔRandConv', 'n'], rows,
                             'The training/evaluation slice policy, held consistent across train, '
                             'validation and target, changes the size of the measured DG gain.',
                             'tab:slice'))
    else:
        md.append(INC)

    # ---------------------------------------------------------------- T5 boundary metrics (E2)
    md.append('\n\n## T5. The ranking depends on the metric (boundary metrics, from checkpoints)\n')
    brows, ok = [], True
    for tag, fn in [('DG Prostate', 'boundary_prostate.md'), ('RIGA+ cup', 'boundary_riga_cup.md'),
                    ('RIGA+ disc', 'boundary_riga_disc.md'), ('BraTS WT', 'boundary_brats_wt.md'),
                    ('BraTS TC', 'boundary_brats_tc.md'), ('BraTS ET', 'boundary_brats_et.md')]:
        rr = parse_md_rows(os.path.join(RPT, fn), 6)
        if rr is None:
            brows.append((tag, INC, '', '', '')); ok = False; continue
        d = {c[0].lower(): [f(x) for x in c[1:]] for c in rr}
        rank = sorted(d, key=lambda m: d[m][0])           # by median HD95, lower is better
        # A ranking claim needs at least two methods to rank, and the paper's claim is about the
        # ranking of the *seven*. A report covering fewer is not a weaker version of this table --
        # it cannot support the claim at all, so it is INCOMPLETE rather than a short row.
        if len(rank) < 2:
            note('%s covers only %d method(s) (%s) — cannot rank; T5 row INCOMPLETE'
                 % (fn, len(rank), ', '.join(rank)))
            brows.append((tag, INC, '(%d method)' % len(rank), '', '')); ok = False; continue
        if len(rank) < len(METHODS):
            note('%s covers %d of %d methods (%s) — T5 row is PARTIAL, and the metric-ranking claim '
                 'may not be made for this benchmark' % (fn, len(rank), len(METHODS), ', '.join(rank)))
        top2 = rank[:2]
        gap = abs(d[top2[0]][0] - d[top2[1]][0])
        worst_pen = max(d, key=lambda m: d[m][4])
        mark = '' if len(rank) == len(METHODS) else ' (%d/%d methods)' % (len(rank), len(METHODS))
        brows.append((tag + mark, '%s %.1f' % (NICE.get(top2[0], top2[0]), d[top2[0]][0]),
                      '%s %.1f' % (NICE.get(top2[1], top2[1]), d[top2[1]][0]),
                      '%.1f' % gap,
                      '%s %.3f' % (NICE.get(worst_pen, worst_pen), d[worst_pen][4])))
        if tag == 'DG Prostate':
            for m in ('bigaug', 'randconv', 'ada'):
                if m in d:
                    nm = NICE[m].replace('-', '')
                    num('medianHD' + nm, d[m][0], '%.1f')       # letters only -- see num()
                    num('assdMean' + nm, d[m][3], '%.1f')
                    num('degen' + nm, d[m][4], '%.3f')
    head = ['benchmark', 'best (med. HD95)', 'second', 'gap px', 'worst degen.']
    md.append(md_table(head, brows))
    md.append('\nParsed from the released boundary reports; Dice ranks the same methods differently '
              '(see T1).%s\n' % ('' if ok else '  **' + INC + '**'))
    tex.append(tex_table(head, brows,
                         'Boundary metrics on every benchmark. Methods that Dice separates widely '
                         'are level on HD95, and the degenerate-prediction share is not visible in '
                         'Dice at all.', 'tab:boundary'))

    # ---------------------------------------------------------------- T6 oracle ceilings
    md.append('\n\n## T6. Oracle ceilings for adaptive selection\n')
    rows = []
    for tag, runs, st in [('DG Prostate', h2h_p, None),
                          ('RIGA+ cup', [r for r in h2h_r if r['config'].get('gt') == 'r1'], 'cup'),
                          ('BraTS ET', load('sdg_brats', 'brats_*_et.json'), None)]:
        c = ceilings(runs, st) if runs else None
        if not c:
            rows.append((tag, INC, '', '', '')); continue
        k = re.sub(r'[^A-Za-z]', '', tag)
        rows.append((tag, '%s %s' % (NICE[c['best']], num('fixed' + k, c['fixed'])),
                     num('oracleSrc' + k, c['o_src'] - c['fixed'], '%+.4f'),
                     num('oraclePair' + k, c['o_pair'] - c['fixed'], '%+.4f'),
                     num('oracleCase' + k, c['o_case'], '%+.4f')))
        num('oracleAgree' + k, '%d/%d' % (c['agree'], c['n_srcs']), '%s')
    head = ['benchmark', 'best fixed', 'Δ oracle per source', 'Δ oracle per pair', 'Δ oracle per case']
    md.append(md_table(head, rows))
    md.append('\nThe per-source oracle is the ceiling of any method that adapts to the training '
              'domain it is handed; the per-pair and per-case oracles require target knowledge and '
              'are reported to bound the question rather than as achievable targets.\n')
    # Destination decided by --split_oracle; default is tables.tex. See the note at the top of main().
    (tex_oracle if a.split_oracle else tex).append(tex_table(head, rows,
                         'Ceilings for adaptive augmentation-policy selection, measured on the '
                         'head-to-head runs. An always-correct domain-level selector gains almost '
                         'nothing over the single best fixed policy.', 'tab:oracle'))

    # ---------------------------------------------------------------- T7 rater convention
    #
    # The two conventions do NOT live in the same output directory. rater-1 runs are the fp32
    # deterministic head-to-head (`sdg_h2h`); the majority-vote runs are phase-1 (`sdg`, fp16 AMP,
    # cudnn.benchmark, three methods only). They are reported as two labelled arms and are never
    # averaged together — mixing precisions inside one cell would be exactly the kind of unstated
    # protocol choice this paper is about.
    md.append('\n\n## T7. Annotation convention (RIGA+ optic cup)\n')
    maj = [r for r in load('sdg', 'riga_*.json') if r['config'].get('gt') == 'majority']
    r1 = [r for r in h2h_r if r['config'].get('gt') == 'r1']
    shared = sorted({r['config']['method'] for r in maj} & {r['config']['method'] for r in r1})
    if r1 and maj and shared:
        rows = []
        for tag, runs, prec in [('rater 1', r1, 'fp32 deterministic'),
                                ('majority vote', maj, 'fp16 AMP (phase 1)')]:
            sub = [r for r in runs if r['config']['method'] in shared]
            rr, erm, n, srcs = delta_table(sub, 'cup')
            got = {x[0]: x for x in rr}
            k = 'R' if tag == 'rater 1' else 'Maj'
            rows.append((tag, num('cupErm' + k, erm),
                         num('cupBigAug' + k, got['BigAug'][1], '%+.4f') if 'BigAug' in got else '—',
                         num('cupRandConv' + k, got['RandConv'][1], '%+.4f')
                         if 'RandConv' in got else '—', '%d runs, %s' % (n, prec)))
        head = ['GT convention', 'ERM cup Dice', 'ΔBigAug', 'ΔRandConv', 'n / precision']
        md.append(md_table(head, rows))
        md.append('\nMethods restricted to those run under both conventions (%s). The two arms differ '
                  'in precision and are labelled, never pooled.\n' % ', '.join(shared))
        tex.append(tex_table(head, rows,
                             'The optic-cup ground-truth convention, a choice papers rarely state, '
                             'changes both the absolute score and the measured method effect. The '
                             'two arms differ in numerical precision and are reported separately.',
                             'tab:rater'))
    else:
        md.append(INC + '  (need RIGA+ runs under both gt conventions)')
        note('T7 incomplete: r1=%d, majority=%d, shared methods=%s' % (len(r1), len(maj), shared))

    # ---------------------------------------------------------------- T7b saturation vs raters
    md.append('\n\n## T7b. RIGA+ is scored past its own annotation noise floor\n')
    pr = parse_md_rows(os.path.join(RPT, 'protocol_decomp.md'), 4)
    if pr and r1:
        disc_c = np.mean([f(c[1]) for c in pr if not np.isnan(f(c[1]))])
        cup_c = np.mean([f(c[2]) for c in pr if not np.isnan(f(c[2]))])
        num('interRaterDisc', float(disc_c)); num('interRaterCup', float(cup_c))
        best = {}
        for st in ('cup', 'disc'):
            by = defaultdict(list)
            for r in r1:
                s, v = per_target(r, st)
                by[r['config']['method']].append(v)
            b = max(by, key=lambda m: np.mean(by[m]))
            best[st] = (NICE[b], float(np.mean(by[b])))
            num('best' + st.capitalize(), best[st][1])
        rows = [('optic cup', '%s %.4f' % best['cup'], '%.4f' % cup_c,
                 num('satCup', best['cup'][1] - cup_c, '%+.4f')),
                ('optic disc', '%s %.4f' % best['disc'], '%.4f' % disc_c,
                 num('satDisc', best['disc'][1] - disc_c, '%+.4f'))]
        head = ['structure', 'best method (unseen domains)', 'inter-rater Dice', 'difference']
        md.append(md_table(head, rows))
        md.append('\nInter-rater Dice is averaged over all 15 rater pairs and all five domains '
                  '(parsed from `protocol_decomp.md`). A method scoring above it is being ranked '
                  'inside the annotation noise.\n')
        tex.append(tex_table(head, rows,
                             'Best measured method against the agreement between the six human '
                             'raters on the same images. On the optic cup the benchmark is scored '
                             'past its own annotation noise floor.', 'tab:saturation'))
    else:
        md.append(INC + '  (protocol_decomp.md not parseable)')

    # ---------------------------------------------------------------- T8 nnU-Net
    md.append('\n\n## T8. Strong-baseline control: nnU-Net v2\n')
    # The nnU-Net arm's headline lives in its released report, not in a JSON beside the run records
    # (the raw dict is on scratch and scratch auto-deletes). Parse the report's own summary line.
    rp = os.path.join(RPT, 'nnunet_prostate.md')
    m = None
    if os.path.exists(rp):
        m = re.search(r'over all (\d+) transfer pairs:\s*\**\s*([0-9.]+)',
                      open(rp, encoding='utf-8').read())
    if m:
        num('nnunetPairs', int(m.group(1)), '%d')
        num('nnunetMean', float(m.group(2)))
        md.append('nnU-Net v2 (`nnUNetTrainer_250epochs`, 2d, no TTA) over %s transfer pairs: '
                  '**%s** target-mean Dice, scored with our per-case metric, against our '
                  'U-Net+BigAug. The budget differs by roughly 15x and that must travel with the '
                  'number.\n' % (NUM['nnunetPairs'], NUM['nnunetMean']))
    else:
        md.append(INC + '  (could not parse the summary line from %s)' % rp)
        note('T8 incomplete: nnunet_prostate.md missing or its summary line did not parse')

    # ---------------------------------------------------------------- T8b in-domain reference
    #
    # `train.py` scores the source domain over ALL of its cases, training cases included, so the
    # stored source-domain number is a TRAINING-SET score and overstates the in-domain reference --
    # and therefore overstates the generalization gap. The runs record `val_cases`, so the held-out
    # figure is recoverable from the same JSONs without retraining. Those cases were used for model
    # selection, so they are held out from *fitting* but not from *selection*; the table says so.
    md.append('\n\n## T8b. In-domain reference: all source cases vs held-out only\n')
    rows = []
    for m in ('erm', 'bigaug'):
        alls, held = [], []
        for r in [x for x in h2h_p if x['config']['method'] == m]:
            src = r['config']['source']
            pc = r['per_domain'][src].get('per_case') or {}
            va = {str(x) for x in (r.get('val_cases') or [])}
            if not pc:
                continue
            alls.append(float(np.mean([v['dice'] for v in pc.values()])))
            h = [v['dice'] for k, v in pc.items() if str(k) in va]
            if h:
                held.append(float(np.mean(h)))
        if alls and held:
            k = NICE[m]
            rows.append((k, num('indomAll' + k, float(np.mean(alls))),
                         num('indomHeld' + k, float(np.mean(held))),
                         num('indomInflation' + k, float(np.mean(alls)) - float(np.mean(held)),
                             '%+.4f'), '%d runs' % len(alls)))
    if rows:
        head = ['method', 'all source cases', 'held-out only', 'inflation', 'n']
        md.append(md_table(head, rows))
        if 'indomHeldBigAug' in NUM and 'fixedDGProstate' in NUM:
            num('dgGapHonest', f(NUM['indomHeldBigAug']) - f(NUM['fixedDGProstate']))
            num('dgGapInflated', f(NUM['indomAllBigAug']) - f(NUM['fixedDGProstate']))
        md.append('\nScoring the source domain over **all** its cases includes the training cases and '
                  'inflates the in-domain reference, which in turn inflates the apparent '
                  'generalization gap. Held-out cases were used for model selection, so they are held '
                  'out from fitting but not from selection.\n')
        # md only. All eight of this table's numbers are already macros (nindomAll*, nindomHeld*,
        # nindomInflation*, ndgGap*) and the Discussion states every one of them in prose, so the
        # table was pure duplication occupying a float. Dropping it buys space toward the 8-page free
        # tier without losing a single value. --keep_indomain_table restores it.
        if a.keep_indomain_table:
            tex.append(tex_table(head, rows,
                                 'The in-domain reference computed over all source cases (training '
                                 'included) and over held-out cases only. The difference propagates '
                                 'directly into any stated generalization gap.', 'tab:indomain'))
    else:
        md.append(INC)
        note('T8b: no per-case data or no val_cases recorded')

    # ---------------------------------------------------------------- T9 the knob summary
    md.append('\n\n## T9. The protocol knobs, next to the largest method effect\n')
    big = NUM.get('dBigAugDGProstategland')
    knobs = [('model-selection leak (peeking at the target)', NUM.get('leakDelta'),
              'T2, %s runs' % NUM.get('leakRuns', '?')),
             ('slice policy fg vs all', ('%+.4f' % (f(NUM['sliceGainFg']) - f(NUM['sliceGainAll'])))
              if 'sliceGainFg' in NUM and 'sliceGainAll' in NUM else None, 'T4'),
             ('backbone: ImageNet vs scratch (ERM)', NUM.get('backboneErmGain'), 'T3'),
             ('annotation convention (cup, BigAug effect)',
              ('%+.4f' % (f(NUM['cupBigAugR']) - f(NUM['cupBigAugMaj'])))
              if 'cupBigAugR' in NUM and 'cupBigAugMaj' in NUM else None, 'T7'),
             ('evaluation metric (Dice vs HD95 ranking)', 'rank change', 'T5')]
    rows = [(k, v if v else INC, s) for k, v, s in knobs]
    # The reference quantity is the gap BETWEEN COMPETING METHODS, not the ERM-to-best-method gap.
    # A paper's claim is "method A beats method B", so what a protocol choice has to be compared with
    # is |A - B|. Juxtaposing a knob against BigAug-over-ERM (+0.21) is the wrong denominator and
    # makes every knob look small; it also contradicts the abstract, which says "comparable to, or
    # larger than, the difference between the methods being compared".
    b, r = NUM.get('dBigAugDGProstategland'), NUM.get('dRandConvDGProstategland')
    if b and r:
        num('methodGap', abs(f(b) - f(r)))
        rows.append(('*gap between the top two methods (BigAug vs RandConv)*', NUM['methodGap'], 'T1'))
    rows.append(('*ERM to best method, for scale*', big if big else INC, 'T1'))
    md.append(md_table(['protocol choice', 'effect on the reported result', 'source'], rows))
    md.append('\nThe quantity a protocol choice must be compared against is the **gap between the '
              'methods being compared**, since that is what a paper claims. The ERM-to-best-method '
              'distance is listed only for scale.\n')
    # md only. This is the paper's thesis, and as of 2026-08-22 it is carried by Fig. 0
    # (`make_fig0.py`, label `fig:knobs`) instead: same content, but the pipeline shows *where* each
    # choice enters and the bars show the magnitudes against the method gap on one axis, which a
    # column of numbers cannot. Keeping both would duplicate the paper's central claim and cost a
    # float in a paper aimed at the 8-page free tier. --keep_knobs_table restores the table.
    if a.keep_knobs_table:
        tex.append(tex_table(['protocol choice', 'effect', 'source'], rows,
                             'Each protocol choice, none of which is routinely reported, against the '
                             'gap between the two leading methods --- the quantity a comparison paper '
                             'actually claims. The ERM-to-best-method distance is given for scale.',
                             'tab:knobs'))

    # ---------------------------------------------------------------- T10 freeze arm (secondary)
    md.append('\n\n## T10. Representation preservation (secondary result)\n')
    # The freeze-0 baseline is NOT a missing experiment. `train.py` only appends `_fz<N>` when
    # freeze != 0, so the unfrozen arm of these very cells is already in `sdg_backbone`. Matching on
    # (source, seed) reuses the same runs instead of retraining them, which is both free and the
    # correct comparison. Cells are matched exactly; an unmatched cell is dropped and reported.
    fz = load('sdg_freeze')
    if fz:
        base = {}
        for r in load('sdg_backbone', 'prostate_*_resnet34.json'):
            c = r['config']
            if int(c.get('freeze') or 0) == 0:
                _, v = per_target(r)
                base[(c['source'], c['method'], c['seed'])] = v
        by, missing = defaultdict(list), 0
        for r in fz:
            c = r['config']
            k = (c['source'], c['method'], c['seed'])
            if k not in base:
                missing += 1; continue
            _, v = per_target(r)
            by[int(c.get('freeze') or 0)].append((c['source'], v, base[k]))
        if missing:
            note('T10: %d freeze run(s) had no matching freeze-0 baseline in sdg_backbone' % missing)
        rows = []
        b0 = [x[2] for v in by.values() for x in v]
        if b0:
            fzsrc = sorted({x[0] for v in by.values() for x in v})
            rows.append(('freeze 0 (full fine-tuning)', num('freezeBase', float(np.mean(
                [base[k] for k in base if k[0] in fzsrc and k[1] == 'erm']))), '—', ''))
        word = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three'}   # macro keys are letters only, see num()
        for lv in sorted(by):
            srcs = [x[0] for x in by[lv]]
            d = [x[1] - x[2] for x in by[lv]]
            mu, lo, hi, unit = boot(d, srcs)
            w = word.get(lv, 'Lv')
            rows.append(('freeze %d' % lv,
                         num('freezeLv' + w, float(np.mean([x[1] for x in by[lv]]))),
                         num('freezeDelta' + w, mu, '%+.4f'),
                         '[%+.4f, %+.4f] %d runs / %d src' % (lo, hi, len(d), len(set(srcs)))))
        head = ['arm', 'target Dice', 'Δ vs full fine-tuning', '95 % CI']
        md.append(md_table(head, rows))
        md.append('\nPretrained ResNet-34, ERM, prostate, matched cell-for-cell against the '
                  'unfrozen runs already in `sdg_backbone` (no retraining). Freeze 1 = stem, '
                  'freeze 2 = stem + layer1.\n')
        # md only by default -- see the note at the top of main(). --keep_freeze_table puts it back.
        if a.keep_freeze_table:
            tex.append(tex_table(head, rows,
                                 'Freezing early pretrained layers under single-source fine-tuning, '
                                 'against the unfrozen baseline on the same cells.', 'tab:freeze'))
    else:
        md.append(INC + '  (sdg_freeze empty)')
        note('sdg_freeze empty')


    # ------------------------------------------------- failure-mode shares, computed not recalled
    #
    # The claim "the empty-prediction rate is about one percent while a quarter of target cases fall
    # below Dice 0.10" was carried in the prose as two typed numbers whose only source was the
    # phase-2 mechanism record -- an arm run at fp16 with cudnn.benchmark, i.e. NOT bit-reproducible,
    # and tiered "numbers superseded" in the project's own document map. The per-case Dice is stored
    # in every run JSON, so the shares are computed here from the bit-reproducible head-to-head runs
    # instead, per method, and the prose quotes macros.
    md.append('\n\n## T-fail. How the models fail, per method (DG Prostate, unseen domains)\n')
    frows = []
    for m in METHODS:
        below, empty, tot = 0, 0, 0
        for r in h2h_p:
            if r['config']['method'] != m:
                continue
            src = r['config']['source']
            for dom, d in r['per_domain'].items():
                if dom == src:
                    continue
                pc = d.get('per_case') or d.get('per_image') or {}
                for v in pc.values():
                    dv = v.get('dice') if isinstance(v, dict) else None
                    if dv is None:
                        continue
                    tot += 1
                    if dv < 0.10:
                        below += 1
                    if dv <= 1e-6:   # Dice EXACTLY zero -- not the same as an empty prediction
                        empty += 1
        if not tot:
            continue
        k = re.sub(r'[^A-Za-z]', '', m)
        frows.append((NICE[m], num('failBelow' + k, 100.0 * below / tot, '%.1f'),
                      num('failZero' + k, 100.0 * empty / tot, '%.1f'), tot))
    if frows:
        md.append(md_table(['method', 'cases below Dice 0.10 (%)', 'Dice exactly 0 (%)',
                            'target cases'], frows))
        md.append('\nThe two are far apart, and that difference is the finding: the model is not '
                  'silent. Note the third quantity this is NOT: the boundary reports give the '
                  'EMPTY-prediction share separately, and it is far smaller, so most zero-Dice '
                  'cases are predictions that missed entirely rather than absent ones. From '
                  'the fp32 deterministic head-to-head runs.\n')

    # ------------------------------------------------ Western/African gap decomposition, parsed
    #
    # `gap_decompose.py` writes these on the GPU; they are parsed here rather than recomputed, like
    # the boundary reports. Two numbers in the Limitations were typed by hand and one of them was
    # wrong: the prose said the TC gap is "roughly 93 % domain", while the report gives 105 % when
    # standardised on ET volume fraction and 85 % on ET-vs-rim contrast. Neither is 93. Parsing the
    # summary line means the prose quotes whichever figures the report actually carries.
    for region in ('tc', 'et'):
        gd = os.path.join(RPT, 'gap_decomp_%s.md' % region)
        if not os.path.exists(gd):
            note('gap decomposition report missing: %s' % gd)
            continue
        txt = io.open(gd, encoding='utf-8').read()
        # "total +0.1211 = case-mix -0.0056 (-5 %) + domain +0.1267 (105 %) -- stable"
        hits = re.findall(r'total\s+([-+][\d.]+)\s*=\s*case-mix\s+([-+][\d.]+)\s*\(\s*([-+]?\d+)\s*%\)'
                          r'\s*\+\s*domain\s+([-+][\d.]+)\s*\(\s*([-+]?\d+)\s*%\)\*\*\s*(?:---|—)\s*(\S+)',
                          txt)
        if len(hits) != 2:
            note('gap_decomp_%s.md: expected 2 summary lines, parsed %d — format drifted'
                 % (region, len(hits)))
            continue
        doms = sorted(int(h[4]) for h in hits)
        R = region.upper()
        num('gdDomainLo' + R, doms[0], '%d')
        num('gdDomainHi' + R, doms[1], '%d')
        num('gdStable' + R, 'stable' if all(h[5].lower().startswith('stable') for h in hits)
            else 'UNSTABLE', '%s')
        # bin occupancy of the contrast-standardised table: the support problem, in numbers
        blocks = txt.split('ET-vs-rim contrast')
        if len(blocks) > 1:
            rows = re.findall(r'^\|\s*(\d)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', blocks[-1], re.M)
            if rows:
                num('gdBinZeroRef' + R, int(rows[0][1]), '%d')
                num('gdBinZeroTgt' + R, int(rows[0][2]), '%d')
                num('gdTgtTotal' + R, sum(int(r[2]) for r in rows), '%d')

    # ---------------------------------------------------------------- T-gap: gap accounting
    #
    # Produced by decompose2_analyse.py from the GPU pass (decompose2.py). Read from its JSON rather
    # than recomputed, for the same reason the boundary reports are parsed rather than recomputed:
    # the arrays live on scratch and need a GPU to make. A missing benchmark is INCOMPLETE, never a
    # silently absent row.
    md.append('\n\n## T-gap. What each component of the gap could buy, if chosen perfectly\n')
    grows, gmissing = [], []
    # One row per (benchmark, region). BraTS and M&Ms have three structures each and they are
    # NOT interchangeable -- an accounting that pooled them would average different anatomy into one
    # number. `decompose2_analyse.py` refuses to run without --region on those two, and this list
    # mirrors T1 so the two tables can be read side by side.
    for tag, bench in [('DG Prostate', 'prostate'), ('RIGA+ cup', 'riga'),
                       ('BraTS WT', 'brats_wt'), ('BraTS TC', 'brats_tc'), ('BraTS ET', 'brats_et'),
                       ('M\&Ms LV', 'mms_lv'), ('M\&Ms myo', 'mms_myo'),
                       ('M\&Ms RV', 'mms_rv')]:
        gp = os.path.join(RPT, 'gapacct_%s.json' % bench)
        if not os.path.exists(gp):
            gmissing.append(bench); continue
        try:
            G = json.load(open(gp, encoding='utf-8'))
        except Exception as e:
            note('unreadable %s: %s' % (gp, e)); gmissing.append(bench); continue
        k = re.sub(r'[^A-Za-z]', '', tag)
        c = G['ceilings']
        grows.append((tag, NICE.get(G['best_fixed'], G['best_fixed']),
                      num('gapMethSrc' + k, c['meth_src']['mean'], '%+.4f'),
                      num('gapThrDom' + k, c['thr_dom']['mean'], '%+.4f'),
                      num('gapJointDom' + k, c['joint_pair']['mean'], '%+.4f'),
                      num('gapJointCase' + k, c['joint_case']['mean'], '%+.4f'),
                      num('gapOverlap' + k, G['overlap_per_case'], '%+.4f')))
        num('gapBase' + k, G['levels']['baseline']['mean'])
        num('gapUnit' + k, G['ceilings']['meth_src']['unit'], '%s')
    if gmissing:
        note('gap accounting missing for %s — run decompose2_analyse.py' % ', '.join(gmissing))
    # Aggregates, so the prose can make a statement over ALL rows without anyone typing "all three"
    # or "no benchmark exceeds" by hand and having it silently go stale when a row is added.
    if grows:
        srcvals = [float(r[2]) for r in grows]
        ovvals = [float(r[6]) for r in grows]
        num('gapMethSrcMax', max(srcvals), '%+.4f')
        num('gapMethSrcRows', len(grows), '%d')
        num('gapOverlapMin', min(ovvals), '%+.4f')
        num('gapOverlapMax', max(ovvals), '%+.4f')
    if grows:
        head = ['benchmark', 'best fixed', 'method, per source', 'threshold, per domain',
                'both, per domain', 'both, per case', 'overlap']
        md.append(md_table(head, grows))
        md.append('\nThe only column a deployed method can reach is **method, per source** — it is '
                  'the one that needs no knowledge of the target. The rest require the target domain '
                  'or the answer for each case, and are reported to bound the question. **Overlap** '
                  'is (threshold + method) minus the joint ceiling: it is what the two components '
                  'would double-count if their ceilings were added, and it is why they are not.\n')
        tex.append(tex_table(head, grows,
                             'Ceilings on each component of the out-of-domain gap. Only the '
                             'per-source column is attainable without target knowledge; the joint '
                             'columns bound the decision rule and the method choice together, so the '
                             'components are never summed.', 'tab:gapacct', wide=True))

    # ---------------------------------------------------------------- T-africa: transfer asymmetry
    #
    # Both directions between the Western (gli2023) and Sub-Saharan African (africa_glioma)
    # cohorts. The reverse direction is the point: training on the African cohort and testing on the
    # Western one is a measurement we could not find published, and the asymmetry between the two is
    # an equity-relevant fact that is measured rather than asserted.
    #
    # The paper must NOT read this as anatomy. BraTS-Africa's difficulty is attributed in the
    # source literature to acquisition — 1.5 T vs 3 T, motion, lower resolution, non-standardised
    # protocols. "African brains differ" is not an available claim.
    md.append('\n\n## T-africa. Both directions between the Western and Sub-Saharan African cohorts\n')
    af = load('sdg_africa')
    expect('sdg_africa', af, 27)   # 3 regions x 3 methods x 3 seeds, BraTS-Africa as SOURCE
    we = load('sdg_brats')
    arows = []
    if not af:
        note('sdg_africa is empty — the reverse-direction arm has not run')
    else:
        def mean_of(runs, src, tgt, region, method):
            v = [r['per_domain'][tgt]['dice_mean'] for r in runs
                 if r['config']['source'] == src and str(r['config'].get('region')) == region
                 and r['config']['method'] == method and tgt in r['per_domain']]
            return float(np.mean(v)) if v else None
        af_methods = sorted({r['config']['method'] for r in af})
        for region in ['wt', 'tc', 'et']:
            for m in af_methods:
                w2a = mean_of(we, 'gli2023', 'africa_glioma', region, m)
                a2w = mean_of(af, 'africa_glioma', 'gli2023', region, m)
                a_in = mean_of(af, 'africa_glioma', 'africa_glioma', region, m)
                w_in = mean_of(we, 'gli2023', 'gli2023', region, m)
                if w2a is None or a2w is None:
                    continue
                k = re.sub(r'[^A-Za-z]', '', region.upper() + m)
                arows.append((region.upper(), NICE.get(m, m),
                              num('afInW' + k, w_in) if w_in is not None else INC,
                              num('afWtoA' + k, w2a),
                              num('afInA' + k, a_in) if a_in is not None else INC,
                              num('afAtoW' + k, a2w),
                              num('afAsym' + k, abs(w2a - a2w), '%+.4f')))
                # the in-domain difference is the point: if it is positive the African cohort is not
                # the harder dataset, and the difficulty is in the transfer rather than in the data
                if w_in is not None and a_in is not None:
                    num('afInDelta' + k, a_in - w_in, '%+.4f')
    if arows:
        head = ['region', 'method', 'Western in-domain', 'Western to African',
                'African in-domain', 'African to Western', '|asymmetry|']
        md.append(md_table(head, arows))
        md.append('\nThe two directions are not mirror images. Attribution is **acquisition**, not '
                  'anatomy: the source literature ascribes the African cohort\'s difficulty to field '
                  'strength, motion, resolution and non-standardised protocols.\n')
        tex.append(tex_table(head, arows,
                             'Transfer in both directions between the Western (BraTS-GLI) and '
                             'Sub-Saharan African (BraTS-Africa) glioma cohorts, with each '
                             'cohort\'s own in-domain reference. The asymmetry is measured, not '
                             'assumed; its cause is acquisition, not anatomy.', 'tab:africa', wide=True))
        num('afRows', len(arows), '%d')

    # ---------------------------------------------------------------- write
    txt = '\n'.join(md)
    if WARN:
        txt += '\n\n## /!\\ Warnings raised while generating\n\n' + \
               '\n'.join('- %s' % w for w in WARN) + '\n'
    # encoding is explicit everywhere: the warning block is the one place a locale-dependent
    # default would crash, i.e. exactly when something is already wrong.
    open(os.path.join(a.out_dir, 'tables.md'), 'w', encoding='utf-8').write(txt + '\n')
    open(os.path.join(a.out_dir, 'tables.tex'), 'w', encoding='utf-8').write('\n\n'.join(tex) + '\n')
    if a.split_oracle:
        # Only written when the split is explicitly requested; see the note in main().
        open(os.path.join(a.out_dir, 'tables_oracle.tex'), 'w', encoding='utf-8').write(
            '%% T6 oracle ceilings -- written here only because --split_oracle was passed.\n'
            '%% If two work7 papers are ever under review at once, the one that does NOT own this\n'
            '%% result must not \\input this file.\n'
            + '\n\n'.join(tex_oracle) + '\n')
    with open(os.path.join(a.out_dir, 'numbers.tex'), 'w', encoding='utf-8') as fh:
        fh.write('%% generated by make_tables.py -- every number quoted in the prose\n')
        for k in sorted(NUM):
            fh.write('\\newcommand{\\n%s}{%s}\n' % (k, NUM[k].replace('%', '\\%')))
    print(txt)
    print('\n-> %s/{tables.md,tables.tex,numbers.tex%s}  (%d prose macros, %d warnings)'
          % (a.out_dir, ',tables_oracle.tex' if a.split_oracle else '', len(NUM), len(WARN)))
    print('   tables.tex: %d tables; oracle ceilings %s'
          % (len(tex), 'SPLIT OUT to tables_oracle.tex' if a.split_oracle else 'INCLUDED (default)'))
    if WARN:
        print('\n!! %d WARNING(S) — do not write from these tables until they are cleared' % len(WARN))


if __name__ == '__main__':
    main()
