#!/usr/bin/env python
"""
C-1 verdict: does single-source fine-tuning destroy the domain robustness that ImageNet pretraining
supplied?

PRE-REGISTERED KILL CONDITION (fixed 2026-08-10 before any result was seen): if freezing does NOT
raise target Dice above full fine-tuning, direction C-1 has no headroom and is abandoned.

The prediction was: freezing gives LOWER source-val (less capacity to fit the source) but HIGHER
target Dice (the pretrained representation survives). Both halves are reported.
"""
import os
import glob, json
from collections import defaultdict
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))

BACK = os.path.join(OUTPUTS, 'sdg_backbone')
FREEZE = os.path.join(OUTPUTS, 'sdg_freeze')

rows = defaultdict(list)
for d in (BACK, FREEZE):
    for f in sorted(glob.glob(d + '/prostate_*resnet34*.json')):
        r = json.load(open(f))
        c = r['config']
        if c['method'] != 'erm' or not c.get('pretrained', 1):
            continue
        z = c.get('freeze', 0)
        tg = [v['dice_mean'] for k, v in r['per_domain'].items() if k != c['source']]
        rows[(c['source'], z)].append((r['best_source_val'], float(np.mean(tg))))

srcs = sorted({s for s, _ in rows})
print('%-8s %22s %22s %22s' % ('source', 'freeze0 src-val/target', 'freeze1', 'freeze2'))
for s in srcs:
    line = '%-8s' % s
    for z in (0, 1, 2):
        v = rows.get((s, z))
        line += '  %20s' % ('—' if not v else '%.4f / %.4f' % tuple(np.array(v).mean(0)))
    print(line)

print()
common = [s for s in srcs if all(rows.get((s, z)) for z in (0, 1, 2))]
print('sources with all three arms: %s' % (', '.join(common) or 'none'))
base = {}
for z in (0, 1, 2):
    v = [x for s in common for x in rows[(s, z)]]
    if v:
        m = np.array(v).mean(0)
        base[z] = m
        print('freeze=%d  n=%2d  source-val %.4f   TARGET %.4f' % (z, len(v), m[0], m[1]))

if 0 in base:
    print()
    for z in (1, 2):
        if z in base:
            d = base[z][1] - base[0][1]
            print('freeze%d - freeze0 on TARGET: %+.4f  (source-val %+.4f)'
                  % (z, d, base[z][0] - base[0][0]))
    best = max((base[z][1], z) for z in base)
    print()
    print('VERDICT: %s' % ('freezing HELPS on targets -> C-1 has headroom' if best[1] != 0 else
                           'freezing does NOT beat full fine-tuning -> C-1 KILLED by its '
                           'pre-registered condition'))
