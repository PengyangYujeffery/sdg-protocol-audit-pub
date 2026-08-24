#!/usr/bin/env python
"""
Gap accounting, all four benchmarks — the GPU half. (MICCAI 2027 Track B)

`decompose.py` answered "how much of the prostate gap is a decision rule?" on one benchmark, for one
method at a time, and printed a table. Track B needs the same question answered on four
benchmarks, across seven methods, and it needs one thing the old script cannot give: the joint
oracle over (method x threshold), which is what defends the accounting against the reviewer objection
that its components double-count.

So this script does not compute the accounting. It computes and stores the raw material for it:

    for every (checkpoint, target domain, case): Dice at every threshold on a fixed grid.

Everything downstream -- per-domain threshold, per-case oracle threshold, oracle over methods, the
joint oracle, the in-domain reference, source-clustered CIs -- is pure CPU post-processing over these
arrays (`decompose2_analyse.py`). That split matters here specifically: the Sonic `csgpu` queue is
contended and a job waits days, so the GPU pass must be run once and survive every later change of
analysis design.

Design notes that are not obvious:

* Resumable and per-cell. One `.npz` per (checkpoint, target). A crash, a timeout or a single bad
  checkpoint costs that cell, not the sweep. Re-running skips what exists.
* Exact Dice at 1e-3 probability resolution. Thresholding a (N_pix,) map once per threshold is
  O(T*N_pix); instead probabilities are quantised to a 1001-bin grid and reverse-cumulated, which is
  O(N_pix) and exact for any threshold that lands on the grid. Every threshold on the default grid
  does. This is the only approximation in the file and it is bounded by the float16 sigmoid output
  the network already produces.
* `--smoke` is a correctness gate, not a demo. It recomputes Dice at 0.5 and compares it against
  the `per_case` values stored in the original run JSON. If the model rebuild, the normalisation or
  the evaluation unit were wrong, this disagrees. Nothing should be trusted until it passes.
* The model is rebuilt from `config` inside the checkpoint, never from the file name, because the
  file name does not carry `base`, `norm`, `pretrained` or `freeze`.

Usage:
    python decompose2.py --bench prostate --smoke # gate, one cell, minutes
    python decompose2.py --bench prostate # full sweep for that benchmark
    python decompose2.py --bench riga --gt r1
    python decompose2.py --bench brats --region et
    python decompose2.py --bench mms --region lv
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

import data as D
from unet import UNet

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

# ---------------------------------------------------------------------------- constants
# 0.02 .. 0.98.  Finer than decompose.py's 0.05 grid: the old run put the domain-level optimum at the
# grid edge (t=0.05) in 20 of 30 prostate pairs, which means the grid, not the data, may have set it.
THRESH = np.round(np.arange(0.02, 0.99, 0.02), 4)
QBINS = 1000                                   # probability quantisation, 1e-3

CKPT_DIRS = [os.path.join(SCRATCH, 'ckpt_h2h'),
             os.path.join(SCRATCH, 'ckpt_brats'),
             os.path.join(SCRATCH, 'ckpt_bb'),
             os.path.join(SCRATCH, 'ckpt')]
RUN_DIRS = [os.path.join(OUTPUTS, 'sdg_h2h'),
            os.path.join(OUTPUTS, 'sdg_brats'),
            os.path.join(OUTPUTS, 'sdg_backbone'),
            os.path.join(OUTPUTS, 'sdg')]

IN_OUT = {'prostate': (1, 1, False), 'brats': (4, 1, False),
          'mms': (1, 1, False), 'riga': (3, 2, True)}


# ---------------------------------------------------------------------------- model rebuild
def build_model(cfg, device):
    """Mirror of train.py lines 235-258. Any drift between the two is a silent correctness bug, so
    `--smoke` exists to catch it."""
    bench = cfg['bench']
    in_ch, out_ch, in_norm = IN_OUT[bench]
    method, backbone = cfg['method'], cfg.get('backbone', 'scratch')

    from styles import make_style
    from train import STYLE_METHODS
    style_cls = (lambda: make_style(method)) if method in STYLE_METHODS else None

    if backbone == 'scratch':
        model = UNet(in_ch, out_ch, base=cfg.get('base', 32), norm=cfg.get('norm', 'bn'),
                     style=style_cls)
    elif backbone.startswith('dinov2'):
        from unet_dino import DinoUNet
        model = DinoUNet(in_ch, out_ch, name='facebook/%s' % backbone, style=style_cls,
                         pretrained=bool(cfg.get('pretrained', 1)))
    else:
        from unet_pre import PretrainedUNet
        model = PretrainedUNet(in_ch, out_ch, arch=backbone, style=style_cls,
                               pretrained=bool(cfg.get('pretrained', 1)))
    model = model.to(device)
    if getattr(model, 'styles', None) is not None:
        for s in model.styles:
            s.enabled = True                   # as at training time; style modules stay active

    ada_in = None
    if method == 'ada':
        from ada import ADAInput

        ada_in = ADAInput(in_ch, not cfg.get('ada_no_bezier', 0),
                          not cfg.get('ada_no_shift', 0)).to(device)
    return model, ada_in, in_norm, out_ch


@torch.no_grad()
def probs(model, ada_in, X, bs, device, in_norm):
    """Sigmoid probabilities, float16. fp32 forward, AMP off -- the stored runs that these are
    checked against are `--deterministic 1` fp32, and AMP would move the fourth decimal."""
    model.eval()
    if ada_in is not None:
        ada_in.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(np.asarray(X[i:i + bs], np.float32)).to(device)
        if in_norm:
            m = x.mean(dim=(2, 3), keepdim=True); s = x.std(dim=(2, 3), keepdim=True)
            x = (x - m) / (s + 1e-6)
        if ada_in is not None:
            x = ada_in(x)
        out.append(torch.sigmoid(model(x)).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


# ---------------------------------------------------------------------------- Dice vs threshold
def dice_curve(P, G):
    """Dice at every threshold in THRESH, for one case.

    P (float prob map), G (bool). Quantise P to 1/QBINS, bincount, reverse-cumulate:
        |{P > t}| and |{P > t} & G| are both tail sums of a histogram, so one O(N) pass gives the
    whole curve. Returns (T,) float32.
    """
    q = np.clip((np.asarray(P, np.float32) * QBINS + 0.5).astype(np.int32), 0, QBINS).ravel()
    g = np.asarray(G, bool).ravel()
    h_all = np.bincount(q, minlength=QBINS + 1).astype(np.int64)
    h_pos = np.bincount(q[g], minlength=QBINS + 1).astype(np.int64)
    tail_all = np.cumsum(h_all[::-1])[::-1]     # tail_all[k] = |{q >= k}|
    tail_pos = np.cumsum(h_pos[::-1])[::-1]
    gsum = int(g.sum())

    # strictly greater: the first retained bin is floor(t*QBINS)+1
    k = np.floor(THRESH * QBINS).astype(np.int64) + 1
    k = np.clip(k, 0, QBINS)
    pred = tail_all[k]
    tp = tail_pos[k]
    den = pred + gsum
    out = np.where(den == 0, 1.0, 2.0 * tp / np.maximum(den, 1))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------- enumeration
def find_ckpts(bench, ckpt_dirs):
    seen, out = set(), []
    for d in ckpt_dirs:
        for f in sorted(glob.glob(os.path.join(d, '%s_*.pt' % bench))):
            b = os.path.basename(f)
            if b in seen:                       # ckpt_h2h wins over the older ckpt/ copies
                continue
            seen.add(b); out.append(f)
    return out


def targets_for(bench, source):
    if bench == 'prostate':
        return [s for s in D.PROSTATE_SITES if s != source]
    if bench == 'riga':
        return [d for d in D.RIGA_DOMAINS if d != source]
    if bench == 'mms':
        return [c for c in D.MMS_CENTRES if c != source]
    # BRATS_TARGETS holds only the two African domains, so the original expression silently drops
    # gli2023 whenever the source IS African -- i.e. the Africa -> Western direction, which is the
    # measurement the under-served arm is for. Including the source domain makes it symmetric; for
    # source='gli2023' this is a no-op and every existing cell is unchanged.
    return [d for d in ([D.BRATS_SOURCE] + D.BRATS_TARGETS) if d != source]


def load_domain(bench, dom, cfg):
    """The evaluation set for one domain, under the same convention the run used."""
    if bench == 'prostate':
        return D.prostate_domain(dom, cfg.get('slices', 'fg'))
    if bench == 'riga':
        return D.riga_domain(dom, cfg.get('gt', 'r1'))
    if bench == 'mms':
        return D.mms_domain(dom, cfg.get('region', 'lv'), cfg.get('slices', 'fg'))
    return D.brats_domain(dom, cfg.get('brats_slices', 'tumour'), cfg.get('region', 'wt'))


def cell_name(ck, tgt):
    return '%s__%s.npz' % (os.path.basename(ck)[:-3], tgt)


# ---------------------------------------------------------------------------- one cell
def run_cell(model, ada_in, in_norm, out_ch, bench, cfg, tgt, bs, device):
    X, Y, unit = load_domain(bench, tgt, cfg)
    P = probs(model, ada_in, X, bs, device, in_norm)
    G = np.asarray(Y, bool)

    if bench == 'riga':
        # the unit is the image, and disc and cup are reported separately, never pooled
        keys = ['%04d|%s' % (i, nm) for i, nm in enumerate(unit.tolist())]
        d_disc = np.stack([dice_curve(P[i, 0], G[i, 0]) for i in range(len(keys))])
        d_cup = np.stack([dice_curve(P[i, 1], G[i, 1]) for i in range(len(keys))])
        return {'cases': np.array(keys), 'dice_disc': d_disc, 'dice_cup': d_cup}

    cases = sorted(set(unit.tolist()))
    d = np.stack([dice_curve(P[unit == c][:, 0], G[unit == c][:, 0]) for c in cases])
    return {'cases': np.array(cases), 'dice': d}


# ---------------------------------------------------------------------------- smoke gate
def smoke(ck, res, bench, cfg, tgt, run_dirs):
    """Recomputed Dice at 0.5 must reproduce the stored run JSON. This is the gate that says the
    model rebuild, the normalisation, the evaluation unit and the threshold indexing are all right."""
    tag = os.path.basename(ck)[:-3]
    js = None
    for d in run_dirs:
        p = os.path.join(d, tag + '.json')
        if os.path.exists(p):
            js = json.load(open(p)); break
    if js is None or tgt not in js.get('per_domain', {}):
        return 'NO REFERENCE RUN JSON for %s / %s -- gate could not run' % (tag, tgt)

    j5 = int(np.argmin(np.abs(THRESH - 0.5)))
    # The per-unit record is NOT under the same key on every benchmark: prostate and BraTS write
    # `per_case`, RIGA+ writes `per_image` (its unit is an image, not a case).  Assuming `per_case`
    # cost job 723640 -- the gate passed prostate and BraTS, then died on RIGA with a bare KeyError,
    # which left the four dependent d2 jobs permanently `DependencyNeverSatisfied`.
    # The inner shape is already handled correctly below: RIGA is {'cup': {'dice': ..}, 'disc': ...},
    # the others are {'dice': ..} (verified against the stored JSONs, 2026-08-21).
    pd_ = js['per_domain'][tgt]
    per = pd_.get('per_case') or pd_.get('per_image')
    if not isinstance(per, dict) or not per:
        return ('NO PER-UNIT RECORD for %s / %s -- keys present: %s -- gate could not run'
                % (tag, tgt, sorted(pd_)))
    if bench == 'riga':
        mine = {c: float(res['dice_cup'][i, j5]) for i, c in enumerate(res['cases'])}
        theirs = {k: v['cup']['dice'] for k, v in per.items()}
        what = 'cup'
    else:
        mine = {c: float(res['dice'][i, j5]) for i, c in enumerate(res['cases'])}
        theirs = {k: v['dice'] for k, v in per.items()}
        what = 'dice'

    common = sorted(set(mine) & set(theirs))
    if not common:
        return 'KEY MISMATCH: mine=%s theirs=%s' % (list(mine)[:2], list(theirs)[:2])
    diff = np.array([abs(mine[c] - theirs[c]) for c in common])
    # the stored values are rounded to 4 dp, so agreement to 1e-3 is the tightest honest bar
    ok = diff.max() < 1e-3
    return '%s  %s  n=%d  max|Δ|=%.5f  mean|Δ|=%.5f  (%s @0.5 vs stored run)' % (
        'GATE PASS' if ok else '🔴 GATE FAIL', tag, len(common), diff.max(), diff.mean(), what)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True, choices=['prostate', 'riga', 'brats', 'mms'])
    ap.add_argument('--out', default=os.path.join(SCRATCH, 'decomp2'))
    ap.add_argument('--ckpt_dirs', nargs='*', default=CKPT_DIRS)
    ap.add_argument('--run_dirs', nargs='*', default=RUN_DIRS)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--region', default=None, help='brats wt|tc|et, mms lv|myo|rv; None = all found')
    ap.add_argument('--backbone', default='scratch', help='"any" to include the resnet/dino arms')
    ap.add_argument('--smoke', action='store_true', help='one cell + the correctness gate, then stop')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    cks = find_ckpts(a.bench, a.ckpt_dirs)
    print('[%s] %d checkpoints found across %d dirs' % (a.bench, len(cks), len(a.ckpt_dirs)),
          flush=True)
    if not cks:
        sys.exit('no checkpoints for bench=%s' % a.bench)

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    for ck in cks:
        try:
            st = torch.load(ck, map_location='cpu', weights_only=False)
        except Exception as e:                                   # a corrupt cell must not stop the sweep
            print('  🔴 unreadable %s: %s' % (os.path.basename(ck), e), flush=True); n_fail += 1; continue
        cfg = st['config']
        if a.backbone != 'any' and cfg.get('backbone', 'scratch') != a.backbone:
            n_skip += 1; continue
        if a.region and cfg.get('region') != a.region:
            n_skip += 1; continue

        src = cfg['source']
        tgts = targets_for(a.bench, src)
        model = ada_in = None
        for tgt in tgts:
            outp = os.path.join(a.out, cell_name(ck, tgt))
            if os.path.exists(outp) and not a.smoke:
                n_skip += 1; continue
            if model is None:                                   # build lazily: a fully-cached ckpt costs nothing
                model, ada_in, in_norm, out_ch = build_model(cfg, device)
                model.load_state_dict(st['state_dict'])
                if ada_in is not None and st.get('ada') is not None:
                    ada_in.load_state_dict(st['ada'])
            try:
                res = run_cell(model, ada_in, in_norm, out_ch, a.bench, cfg, tgt, a.bs, device)
            except Exception as e:
                print('  🔴 %s -> %s failed: %s' % (os.path.basename(ck), tgt, e), flush=True)
                n_fail += 1; continue
            np.savez_compressed(outp, thresh=THRESH, bench=a.bench, source=src, target=tgt,
                                method=cfg['method'], seed=cfg.get('seed', -1),
                                backbone=cfg.get('backbone', 'scratch'),
                                region=str(cfg.get('region', '')), gt=str(cfg.get('gt', '')),
                                slices=str(cfg.get('slices', cfg.get('brats_slices', ''))), **res)
            n_done += 1
            if a.smoke:
                print(smoke(ck, res, a.bench, cfg, tgt, a.run_dirs), flush=True)
                print('smoke complete -- %d cell written to %s' % (n_done, outp), flush=True)
                return
            if n_done % 10 == 0:
                print('  %4d cells  %6.1f min' % (n_done, (time.time() - t0) / 60), flush=True)
        del model, ada_in
        torch.cuda.empty_cache() if device == 'cuda' else None

    print('[%s] done: %d written, %d skipped, %d failed, %.1f min'
          % (a.bench, n_done, n_skip, n_fail, (time.time() - t0) / 60), flush=True)
    if n_fail:
        sys.exit('🔴 %d cells failed -- do not analyse this sweep until they are explained' % n_fail)


if __name__ == '__main__':
    main()
