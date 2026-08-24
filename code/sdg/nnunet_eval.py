#!/usr/bin/env python
"""
Score nnU-Net's cross-site predictions with OUR metric, so the comparison with our own runs is
like-for-like: per-case Dice on the gland, on the same cases, with the same foreground-slice policy.

The one thing that cannot be equalised is the training budget. nnU-Net ran 250 epochs x 250
iterations at batch 15 (~940k samples) against our 8,000 x 8 (~64k) — roughly 15x. That is stated
rather than hidden, because it means a win for nnU-Net is a win for "strong baseline, properly
configured, trained longer", not for the architecture alone.
"""
import argparse, glob, json, os
import numpy as np
import nibabel as nib

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

SITES = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2.0 * (p & g).sum() / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(SCRATCH, 'nnunet'))
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    res = {}
    for src in SITES:
        for tgt in SITES:
            if src == tgt:
                continue
            pd = os.path.join(a.root, 'pred', '%s_to_%s' % (src, tgt))
            gd = os.path.join(a.root, 'infer_gt', tgt)
            if not os.path.isdir(pd):
                continue
            ds = []
            for f in sorted(glob.glob(os.path.join(pd, '*.nii.gz'))):
                cid = os.path.basename(f)[:-7]
                gf = os.path.join(gd, cid + '.nii.gz')
                if not os.path.exists(gf):
                    continue
                p = np.asarray(nib.load(f).dataobj) > 0
                g = np.asarray(nib.load(gf).dataobj) > 0
                # our protocol scores gland-bearing slices only (slices='fg')
                keep = g.reshape(-1, g.shape[-1]).any(0)
                if keep.any():
                    ds.append(dice(p[..., keep], g[..., keep]))
            if ds:
                res[(src, tgt)] = (float(np.mean(ds)), float(np.std(ds)), len(ds))
                print('%-6s -> %-6s  n=%3d  Dice %.4f +- %.4f'
                      % (src, tgt, len(ds), res[(src, tgt)][0], res[(src, tgt)][1]), flush=True)

    L = ['# nnU-Net v2 on the single-source DG Prostate protocol\n',
         '2d configuration, fold 0, `nnUNetTrainer_250epochs`, no TTA. Scored with our metric: '
         'per-case Dice over gland-bearing slices.\n',
         '\n| source \\\\ target | ' + ' | '.join(SITES) + ' | **mean** |',
         '|---|' + '---|' * (len(SITES) + 1)]
    means = {}
    for src in SITES:
        row, vals = [], []
        for tgt in SITES:
            if src == tgt:
                row.append('—'); continue
            v = res.get((src, tgt))
            row.append('%.3f' % v[0] if v else '·')
            if v:
                vals.append(v[0])
        means[src] = float(np.mean(vals)) if vals else float('nan')
        L.append('| %s | %s | **%.4f** |' % (src, ' | '.join(row), means[src]))
    overall = float(np.mean([v for v in means.values() if v == v]))
    L.append('\n**nnU-Net target-mean Dice over all 30 transfer pairs: %.4f**\n' % overall)

    # our own numbers, from the head-to-head (scratch U-Net) and the backbone arm
    L.append('\n## Against our own backbones, same protocol, same 30 pairs\n')
    L.append('| model | target-mean Dice | training samples seen |')
    L.append('|---|---|---|')
    L.append('| our 2D U-Net, ERM | 0.3951 | 64 k |')
    L.append('| our 2D U-Net, BigAug (best of 7) | 0.6054 | 64 k |')
    L.append('| our ResNet-34 ImageNet, BigAug | see `backbone_prostate.md` | 64 k |')
    L.append('| **nnU-Net v2, 250 epochs** | **%.4f** | **~940 k** |' % overall)
    L.append('\n🔴 The budgets are not equal (~15x) and the comparison must always say so.\n')

    json.dump({'%s_to_%s' % k: v for k, v in res.items()},
              open(os.path.join(a.root, 'nnunet_dice.json'), 'w'), indent=1)
    txt = '\n'.join(L)
    print('\n' + txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
