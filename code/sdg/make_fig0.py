#!/usr/bin/env python
"""
Figure 0 -- the protocol pipeline and what each unstated choice is worth.

This is the paper's thesis in one image: where the five protocol choices enter a single-source DG
experiment, and how big each one is against the quantity a comparison paper actually claims -- the
gap between the two leading methods.

NO NUMBER IS TYPED INTO THIS FILE. Every value is read from `manuscript/numbers.tex`, which
`make_tables.py` emits from the run records. The hand-drawn draft this replaces
(`figures/fig0_protocol_draft.svg`, 2026-08-17) had four typed numbers and three of them were stale
within four days -- exactly the failure mode the paper is about.

Two outputs, deliberately:
  fig0_protocol.pdf -- for \\includegraphics; vector, fonttype 42 so the text stays editable
  fig0_protocol.svg -- for hand finishing in a vector tool

Hybrid workflow: the LAYOUT is a judgement about what a reader needs to see and may be refined by
hand in the SVG; the NUMBERS must always come back through this script. If the SVG is edited, keep
the edit log next to it and re-stamp after any regeneration of numbers.tex.

IEEE figure conventions applied: two-column width 7.16 in / 181.9 mm; final-size type 7-9 pt; no top
or right spines; colour paired with position and label so the figure survives greyscale printing.
"""
import argparse, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# --- IEEE house style -------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,      # keep text as text, not outlines
    'ps.fonttype': 42,
    'svg.fonttype': 'none',  # editable text in the SVG
})
C_MAIN = '#0B5CAD'   # the pipeline itself
C_LIGHT = '#8EC3F5'
C_LEAK = '#B64040'   # the one path that must not exist
C_COST = '#B7791F'
C_NEUT = '#6B7280'
C_GRID = '#D7DEE8'
C_TEXT = '#172033'
W2COL = 7.16         # in


def read_numbers(path):
    """numbers.tex -> {macro: value}. Fails loudly rather than substituting a guess."""
    src = open(path, encoding='utf-8').read()
    d = {}
    for m in re.finditer(r'\\newcommand\{\\n(\w+)\}\{(.*?)\}\s*$', src, re.M):
        d[m.group(1)] = m.group(2)
    return d


def f(d, k):
    if k not in d:
        raise SystemExit('!! %s is not in numbers.tex -- regenerate it with make_tables.py' % k)
    return float(d[k].replace('+', '').replace('\\%', '').strip())


def box(ax, x, y, w, h, label, sub=None, fc='white', ec=C_MAIN, lw=1.0, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.010',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha='center', va='center',
                fontsize=fs, color=C_TEXT, zorder=3)
        ax.text(x + w / 2, y + h * 0.26, sub, ha='center', va='center',
                fontsize=fs - 0.9, color=C_NEUT, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fs, color=C_TEXT, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=C_MAIN, lw=1.0, style='-|>', ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                                 lw=lw, color=color, zorder=2, linestyle=ls,
                                 connectionstyle='arc3,rad=%.2f' % rad))


def badge(ax, x, y, n, color=C_COST, r=0.0130, fs=6.6):
    ax.add_patch(Circle((x, y), r, fc=color, ec='none', zorder=5))
    ax.text(x, y, str(n), ha='center', va='center', fontsize=fs, color='white',
            fontweight='bold', zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numbers', default=None, help='path to numbers.tex')
    ap.add_argument('--out_dir', default=None)
    a = ap.parse_args()
    # Defaults follow make_tables.py, which writes numbers.tex to outputs/paper_tables/ when run
    # without arguments. That keeps the chain self-contained in a clone, where manuscript/ is absent.
    here = os.path.dirname(os.path.abspath(__file__))
    outputs = os.environ.get('SDG_OUTPUTS', os.path.join(here, '..', '..', 'outputs'))
    nums = a.numbers or os.path.join(outputs, 'paper_tables', 'numbers.tex')
    out = a.out_dir or os.path.join(outputs, 'paper_figures')
    os.makedirs(out, exist_ok=True)
    d = read_numbers(nums)

    gap = f(d, 'methodGap')
    knobs = [
        ('backbone pretraining',      f(d, 'backboneErmGain'), '3', 'ImageNet vs scratch, ERM'),
        ('slice-inclusion policy',    f(d, 'sliceGainFg') - f(d, 'sliceGainAll'), '1',
         'fg vs all, on the DG gain'),
        ('model-selection rule',      f(d, 'leakDelta'), '4', 'selecting on the target domain'),
        ('annotation convention',     f(d, 'cupBigAugR') - f(d, 'cupBigAugMaj'), '2',
         "flips the sign of BigAug's cup effect"),
    ]
    knobs.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=(W2COL, 3.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # ---------------- upper half: the pipeline -------------------------------------------------
    yb, hb = 0.735, 0.115
    box(ax, 0.015, yb, 0.135, hb, 'source domain', 'one site')
    box(ax, 0.190, yb + 0.068, 0.135, 0.062, 'training cases', fs=7.0)
    box(ax, 0.190, yb - 0.075, 0.135, 0.062, 'held-out cases', fs=7.0)
    box(ax, 0.370, yb, 0.170, hb, 'train', 'fixed backbone, 8k iters')
    box(ax, 0.585, yb, 0.150, hb, 'checkpoint', 'selection')
    box(ax, 0.780, yb, 0.200, hb, 'score', 'unseen domains', ec=C_NEUT)

    arrow(ax, 0.150, yb + hb / 2, 0.188, yb + 0.099)
    arrow(ax, 0.150, yb + hb / 2, 0.188, yb - 0.044)
    arrow(ax, 0.325, yb + 0.099, 0.368, yb + hb * 0.72)
    arrow(ax, 0.325, yb - 0.044, 0.368, yb + hb * 0.28)
    arrow(ax, 0.540, yb + hb / 2, 0.583, yb + hb / 2)
    arrow(ax, 0.735, yb + hb / 2, 0.778, yb + hb / 2)

    # the leak: information flowing back from the target into selection. It must not exist.
    arrow(ax, 0.880, yb - 0.012, 0.660, yb - 0.012, color=C_LEAK, lw=1.1, ls=(0, (3, 2)), rad=-0.28)
    ax.text(0.770, yb - 0.088, 'selecting on the target', ha='center', va='center',
            fontsize=6.6, color=C_LEAK, style='italic')

    badge(ax, 0.190, yb + 0.130, 1); badge(ax, 0.370, yb + hb + 0.014, 3)
    badge(ax, 0.585, yb + hb + 0.014, 4); badge(ax, 0.780, yb + hb + 0.014, 5)
    badge(ax, 0.257, yb - 0.089, 2)
    ax.text(0.997, yb + hb + 0.020, 'each numbered choice is measured below',
            ha='right', va='center', fontsize=6.4, color=C_NEUT, style='italic')

    # ---------------- lower half: what each is worth -------------------------------------------
    x0, x1 = 0.255, 0.905
    ymax = 0.545
    span = max(max(k[1] for k in knobs), gap) * 1.16
    sx = lambda v: x0 + (x1 - x0) * v / span
    rowh = 0.083
    for i, (name, val, num, note) in enumerate(knobs):
        y = ymax - 0.085 - i * rowh
        ax.add_patch(FancyBboxPatch((x0, y - 0.021), sx(val) - x0, 0.042,
                                    boxstyle='square,pad=0', fc=C_MAIN if val >= gap else C_LIGHT,
                                    ec='#0B3D74', lw=0.5, zorder=3))
        badge(ax, x0 - 0.024, y, num, r=0.0118, fs=6.2)
        ax.text(x0 - 0.044, y, name, ha='right', va='center', fontsize=7.2, color=C_TEXT)
        ax.text(sx(val) + 0.010, y, '%+.4f' % val, ha='left', va='center',
                fontsize=7.0, color=C_TEXT)
        ax.text(x1 + 0.008, y, note, ha='left', va='center', fontsize=6.0, color=C_NEUT)

    # the reference line: the gap a comparison paper actually claims
    xg = sx(gap)
    ax.plot([xg, xg], [ymax - 0.085 - (len(knobs) - 1) * rowh - 0.040, ymax + 0.010],
            color=C_LEAK, lw=1.1, ls=(0, (4, 2)), zorder=6)
    ax.text(xg, ymax + 0.020, 'gap between the two leading methods  %.4f' % gap,
            ha='center', va='bottom', fontsize=6.6, color=C_LEAK)

    ylast = ymax - 0.085 - (len(knobs) - 1) * rowh
    y5 = ylast - rowh
    badge(ax, x0 - 0.024, y5, '5', r=0.0118, fs=6.2)
    ax.text(x0 - 0.044, y5, 'evaluation metric', ha='right', va='center', fontsize=7.2, color=C_TEXT)
    ax.text(x0 + 0.004, y5, 'not a magnitude: overlap vs boundary changes the winner on BraTS',
            ha='left', va='center', fontsize=6.4, color=C_NEUT, style='italic')

    ax.plot([x0, x0], [y5 - 0.030, ymax + 0.010], color=C_GRID, lw=0.8, zorder=1)
    ax.text(0.5, 0.045,
            'Five choices that papers rarely state. Each is measured on our own runs and compared '
            'against the gap between the two leading\nmethods, because that is the comparison a '
            'paper claims. Bars at or beyond the dashed line can account for a reported result on '
            'their own.',
            ha='center', va='center', fontsize=6.4, color=C_TEXT)

    for ext in ('pdf', 'svg'):
        p = os.path.join(out, 'fig0_protocol.' + ext)
        fig.savefig(p, bbox_inches='tight', pad_inches=0.01)
        print('-> %s' % os.path.abspath(p))
    plt.close(fig)
    print('   values stamped from %s' % os.path.abspath(nums))
    for name, val, num, _ in knobs:
        print('   %s  %s  %+.4f' % (num, name, val))
    print('   5  evaluation metric  (qualitative)')
    print('   reference: method gap %.4f' % gap)


if __name__ == '__main__':
    main()
