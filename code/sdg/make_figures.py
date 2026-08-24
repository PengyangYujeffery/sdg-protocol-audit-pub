#!/usr/bin/env python
"""
Generate the paper's figures from the run JSONs — same rule as the tables: nothing hand-drawn.

    python make_figures.py --out_dir .../outputs/paper_figures

This module imports `make_tables` rather than reimplementing the loaders and the ceiling
computation. A figure that disagrees with its own table is the worst kind of error in a paper about
measurement error, and the only way to rule it out is to run one piece of code.

F1 Rank instability — each method's rank across benchmark / backbone / metric. The claim of the
    paper is that the *identity of the winner* moves; crossing lines are that claim.
F2 Oracle ceilings for adaptive selection — the novel negative result, shown as the distance
    between the best fixed policy and progressively more powerful oracles.

Both degrade gracefully: a panel whose runs are missing is drawn as an explicit "INCOMPLETE" box
rather than silently omitted, so a figure can never quietly under-report.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE figure conventions, applied 2026-08-22 after a figure QA pass.
# * fonttype 42 keeps the text as *text* in the PDF -- selectable, searchable, and editable by a
# copy editor. The default (Type 3) bakes glyphs into the drawing commands, and some publishers
# reject it outright.
# * svg.fonttype 'none' does the same for the SVG, so a vector tool can still edit the labels.
# * A portable sans stack, because Arial is not present on every machine that will rebuild this.
# Nothing here changes a single plotted value; it changes only how the text is stored.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt

# Colour-blind-safe, and paired with distinct markers/dashes so the figures survive grayscale print.
STYLE = {
    'erm':      ('#666666', 'o', (0, (1, 1))),
    'bigaug':   ('#0072B2', 's', 'solid'),
    'randconv': ('#D55E00', '^', 'solid'),
    'mixstyle': ('#009E73', 'v', (0, (4, 2))),
    'dsu':      ('#CC79A7', 'D', (0, (4, 2))),
    'maxstyle': ('#56B4E9', 'P', (0, (3, 1, 1, 1))),
    'ada':      ('#E69F00', 'X', (0, (3, 1, 1, 1))),
}
NICE = mt.NICE


def incomplete(ax, msg):
    ax.text(0.5, 0.5, 'INCOMPLETE\n%s' % msg, ha='center', va='center', fontsize=8,
            color='#B00020', transform=ax.transAxes,
            bbox=dict(boxstyle='round', fc='#FFF0F0', ec='#B00020'))
    ax.set_xticks([]); ax.set_yticks([])


def mean_by_method(runs, struct=None):
    """method -> mean target score over all its runs (higher is better)."""
    by = {}
    for r in runs:
        _, v = mt.per_target(r, struct)
        by.setdefault(r['config']['method'], []).append(v)
    return {m: float(np.mean(v)) for m, v in by.items()}


def ranks_from_scores(sc, higher_better=True):
    """method -> rank, 1 = best."""
    order = sorted(sc, key=lambda m: -sc[m] if higher_better else sc[m])
    return {m: i + 1 for i, m in enumerate(order)}


def fig1(out_dir):
    """Rank instability across benchmark, backbone and evaluation metric."""
    cols = []

    h2h_p = mt.load('sdg_h2h', 'prostate_*_fg.json')
    h2h_r = [r for r in mt.load('sdg_h2h', 'riga_*.json') if r['config'].get('gt') == 'r1']
    cols.append(('Prostate\n(Dice)', mean_by_method(h2h_p), True))
    cols.append(('RIGA+ cup\n(Dice)', mean_by_method(h2h_r, 'cup'), True))
    for reg in ('wt', 'tc', 'et'):
        rr = mt.load('sdg_brats', 'brats_*_%s.json' % reg)
        cols.append(('BraTS %s\n(Dice)' % reg.upper(), mean_by_method(rr) if rr else {}, True))
    bb = mt.load('sdg_backbone', 'prostate_*resnet34.json')
    bb = [r for r in bb if int(r['config'].get('pretrained', 1)) == 1
          and int(r['config'].get('freeze') or 0) == 0]
    cols.append(('Prostate\n(ImageNet)', mean_by_method(bb) if bb else {}, True))

    # metric axis: lower HD95 is better, parsed from the released boundary report
    rr = mt.parse_md_rows(os.path.join(mt.OUT, 'sdg_reports', 'boundary_prostate.md'), 6)
    hd = {c[0].lower(): mt.f(c[1]) for c in rr} if rr else {}
    cols.append(('Prostate\n(HD95)', hd, False))

    cols = [(n, s, hb) for n, s, hb in cols if s]
    if len(cols) < 2:
        return None
    R = [ranks_from_scores(s, hb) for _, s, hb in cols]
    methods = [m for m in mt.METHODS if all(m in r for r in R)]

    fig, ax = plt.subplots(figsize=(7.16, 3.1))
    x = np.arange(len(cols))
    for m in methods:
        c, mk, ls = STYLE[m]
        y = [r[m] for r in R]
        ax.plot(x, y, marker=mk, color=c, linestyle=ls, linewidth=1.6, markersize=5,
                label=NICE[m], zorder=3 if m in ('bigaug', 'randconv') else 2)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in cols], fontsize=7.5)
    ax.set_yticks(range(1, len(methods) + 1))
    ax.set_ylabel('rank (1 = best)', fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.6, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, 1.28), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    p = os.path.join(out_dir, 'fig1_rank_instability.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p, len(cols), len(methods)


def fig2(out_dir):
    """Oracle ceilings: what perfect knowledge of the best policy would buy."""
    panels = [('DG Prostate', mt.load('sdg_h2h', 'prostate_*_fg.json'), None),
              ('RIGA+ cup', [r for r in mt.load('sdg_h2h', 'riga_*.json')
                             if r['config'].get('gt') == 'r1'], 'cup'),
              ('BraTS ET', mt.load('sdg_brats', 'brats_*_et.json'), None)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    # Short labels: 'oracle\nper source' and 'oracle\nper pair' collide at this figure width.
    labels = ['best\nfixed', 'oracle\nsource', 'oracle\npair', 'oracle\ncase']
    any_ok = False
    for ax, (tag, runs, st) in zip(axes, panels):
        c = mt.ceilings(runs, st) if runs else None
        if not c:
            incomplete(ax, tag); ax.set_title(tag, fontsize=8.5); continue
        any_ok = True
        vals = [c['fixed'], c['fixed'] + (c['o_src'] - c['fixed']),
                c['fixed'] + (c['o_pair'] - c['fixed']), c['fixed'] + c['o_case']]
        cols = ['#0072B2', '#7FB3D5', '#BDBDBD', '#BDBDBD']
        ax.bar(range(4), vals, color=cols, edgecolor='#333333', linewidth=0.6, zorder=3)
        ax.axhline(c['fixed'], color='#0072B2', linewidth=0.9, linestyle=(0, (3, 2)), zorder=4)
        for i, v in enumerate(vals[1:], 1):
            d = v - c['fixed']
            ax.annotate('%+.4f' % d, (i, v), textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=7.0,
                        color='#B00020' if abs(d) < 0.01 else '#333333')
        lo = min(vals) - 0.06
        ax.set_ylim(max(0, lo), max(vals) + 0.06)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.0)
        ax.set_title('%s  (best fixed: %s)' % (tag, NICE[c['best']]), fontsize=8)
        ax.grid(axis='y', color='#EEEEEE', linewidth=0.6, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.8)
    axes[0].set_ylabel('target Dice', fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, 'fig2_oracle_ceilings.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return (p, any_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--out_root', default=None)
    a = ap.parse_args()
    if a.out_root:
        mt.OUT = a.out_root
    out_dir = a.out_dir or os.path.join(mt.OUT, 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    r1 = fig1(out_dir)
    if r1:
        print('F1 -> %s  (%d columns, %d methods)' % r1)
    else:
        print('F1 -> INCOMPLETE: fewer than two comparable columns')
    p2, ok = fig2(out_dir)
    print('F2 -> %s%s' % (p2, '' if ok else '  (all panels INCOMPLETE)'))
    if mt.WARN:
        print('\n!! warnings raised while loading:')
        for w in mt.WARN:
            print('   - %s' % w)


if __name__ == '__main__':
    main()
