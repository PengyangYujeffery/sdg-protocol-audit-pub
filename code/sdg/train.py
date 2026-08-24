#!/usr/bin/env python
"""
Single-source domain generalization: train on ONE domain, evaluate on every other domain.

Protocol decisions, all deliberate and all recorded in PROTOCOL.md:
  * model selection uses the source validation split only -- never a target domain;
  * training length is fixed in iterations, not epochs, because the source domains differ in
    size by 3x (prostate) and 2x (RIGA+); epochs would silently give the big source more updates;
  * prostate Dice is computed per case, under one slice policy applied to train/val/target alike;
  * RIGA+ Dice is per image, reported separately for optic disc and cup.

Methods share the backbone, the schedule, the loss and the selection rule; only the augmentation /
adaptation differs. Input-space: erm, bigaug, randconv. Feature-space: mixstyle, dsu, maxstyle
(adversarial inner step). Learned input adaptation: ada (MICCAI 2025, reimplemented -- see ada.py).

Reproducibility: `--deterministic` fixes cuDNN algorithm selection and enables PyTorch's
deterministic kernels, so a rerun with the same seed is bit-identical. `--amp 0` runs in fp32; the
head-to-head comparison uses fp32 for every method because the adversarial inner step of MaxStyle and
ADA differentiates through the forward pass, and fp16 gradients there are not trustworthy.
Everything needed to reproduce a run is written into the output JSON.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

import data as D
from unet import UNet
from aug import apply_policy
from styles import make_style
from ada import ADAInput, weaken
from efdr import efdr as efdr_apply
from aug import spatial as spatial_aug

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('SDG_OUTPUTS', os.path.join(_HERE, '..', '..', 'outputs'))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

INPUT_METHODS = ['erm', 'bigaug', 'randconv', 'slaug']
STYLE_METHODS = ['mixstyle', 'dsu', 'maxstyle']
EFDR_METHODS = ['efdr', 'efdr_bigaug', 'efdr_randconv']
# `slaug_sbf` is kept OUT of INPUT_METHODS on purpose: Saliency-Balancing Fusion needs the network
# gradient, so it modifies the training loop rather than the augmentation policy. Grouping it with
# the input-space methods would quietly break the one property that makes this comparison readable
# (same backbone, same schedule, same loss -- only the augmentation differs).
LOOP_METHODS = ['slaug_sbf']
ALL_METHODS = INPUT_METHODS + STYLE_METHODS + ['ada'] + EFDR_METHODS + LOOP_METHODS


def dice_np(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2.0 * (p & g).sum() / s)


def unit_metrics(p, g):
    """Dice plus the decomposition phase 2 needs. A Dice of 0.5 caused by predicting twice the gland
    is a different failure from a Dice of 0.5 caused by finding half of it, and only precision vs
    recall tells them apart."""
    tp = float((p & g).sum()); pv = float(p.sum()); gv = float(g.sum())
    return {'dice': dice_np(p, g),
            'precision': 1.0 if pv == 0 else tp / pv,
            'recall': 1.0 if gv == 0 else tp / gv,
            'pred_vox': pv, 'gt_vox': gv,
            'vol_ratio': float('nan') if gv == 0 else pv / gv}


def make_batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n - bs + 1, bs):
        yield idx[i:i + bs]


def seg_loss(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    inter = (p * y).sum((2, 3)); den = p.sum((2, 3)) + y.sum((2, 3))
    return bce + 1 - ((2 * inter + 1e-5) / (den + 1e-5)).mean()


@torch.no_grad()
def predict(model, ada_in, X, bs, device, in_norm, amp):
    model.eval()
    if ada_in is not None:
        ada_in.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        if in_norm:
            m = x.mean(dim=(2, 3), keepdim=True); s = x.std(dim=(2, 3), keepdim=True)
            x = (x - m) / (s + 1e-6)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if ada_in is not None:
                x = ada_in(x)                      # the learned adapter is part of the model at test
            out.append((torch.sigmoid(model(x)) > 0.5).cpu().numpy())
    return np.concatenate(out)


def eval_brats(model, ada_in, dom, bs, device, in_norm, amp, slices='tumour', region='wt'):
    X, Y, case = D.brats_domain(dom, slices, region)
    P = predict(model, ada_in, X, bs, device, in_norm, amp)[:, 0].astype(bool)
    G = Y[:, 0].astype(bool)
    return {c: unit_metrics(P[case == c], G[case == c]) for c in sorted(set(case.tolist()))}


def eval_prostate(model, ada_in, site, bs, device, in_norm, amp, slices='all'):
    X, Y, case = D.prostate_domain(site, slices)
    P = predict(model, ada_in, X, bs, device, in_norm, amp)[:, 0].astype(bool)
    G = Y[:, 0].astype(bool)
    return {c: unit_metrics(P[case == c], G[case == c]) for c in sorted(set(case.tolist()))}


def eval_mms(model, ada_in, centre, bs, device, in_norm, amp, region, slices='fg'):
    X, Y, case = D.mms_domain(centre, region, slices)
    P = predict(model, ada_in, X, bs, device, in_norm, amp)[:, 0].astype(bool)
    G = Y[:, 0].astype(bool)
    # the unit is the subject-FRAME (ED or ES): one cardiac phase of one heart is the thing a
    # clinician sees segmented, and the two phases of one patient are not independent samples.
    return {c: unit_metrics(P[case == c], G[case == c]) for c in sorted(set(case.tolist()))}


def eval_riga(model, ada_in, dom, bs, device, in_norm, amp, gt):
    X, Y, name = D.riga_domain(dom, gt)
    P = predict(model, ada_in, X, bs, device, in_norm, amp).astype(bool)
    G = Y.astype(bool)
    # index-prefixed key: RIGA+ repeats file names across sub-folders, so a bare name collapses
    # 195 BinRushed images onto 51 entries.
    return {'%04d|%s' % (i, nm): {'disc': unit_metrics(P[i, 0], G[i, 0]),
                                  'cup': unit_metrics(P[i, 1], G[i, 1])}
            for i, nm in enumerate(name.tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True, choices=['prostate', 'riga', 'brats', 'mms'])
    ap.add_argument('--region', default='wt',
                    choices=['wt', 'tc', 'et', 'lv', 'myo', 'rv'],
                    help='BraTS: whole tumour / tumour core / enhancing tumour. '
                         'M&Ms: LV cavity / myocardium / RV cavity. Reported separately, never '
                         'pooled, for the same reason in both cases -- the headline region is the '
                         'easy one and pooling hides the region that carries the clinical decision')
    ap.add_argument('--brats_slices', default='tumour', choices=['tumour', 'all'])
    ap.add_argument('--source', required=True)
    ap.add_argument('--method', required=True, choices=ALL_METHODS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--norm', default='bn', choices=['bn', 'in'])
    ap.add_argument('--backbone', default='scratch',
                    choices=['scratch', 'resnet34', 'resnet50', 'dinov2-base', 'dinov2-large'])
    ap.add_argument('--freeze', type=int, default=0, choices=[0, 1, 2],
                    help='0 = fine-tune everything; 1 = freeze stem+layer1; 2 = freeze the whole '
                         'encoder. Measures whether single-source fine-tuning destroys the domain '
                         'robustness that pretraining supplied')
    ap.add_argument('--pretrained', type=int, default=1,
                    help='ImageNet weights for the resnet backbones; 0 gives the same architecture '
                         'trained from scratch, which separates ARCHITECTURE from PRETRAINING')
    ap.add_argument('--gt', default='r1', choices=['r1', 'majority'])
    ap.add_argument('--slices', default='all', choices=['fg', 'all'])
    ap.add_argument('--val_every', type=int, default=500)
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--deterministic', type=int, default=0)
    ap.add_argument('--tag_suffix', default='')
    # -- MaxStyle
    ap.add_argument('--adv_lr', type=float, default=0.1, help='step of the adversarial inner update')
    # -- EFDR (ours): one hyperparameter, selected on source-val only
    ap.add_argument('--tau', type=float, default=0.7,
                    help='evidence floor: admit a perturbation only if it keeps at least this '
                         'fraction of the original foreground/background separability')
    ap.add_argument('--efdr_band', type=int, default=12,
                    help='radius in pixels of the background band around the structure')
    # -- ADA (values the paper does not state; swept, and recorded in the output JSON)
    ap.add_argument('--ada_static', default='noise_contrast',
                    choices=['erm', 'noise_contrast', 'bigaug'],
                    help="the 'Diversify Input' stage; the paper says it follows CCSDG's settings "
                         "(Gaussian noise, contrast adjustment), which is `noise_contrast`")
    ap.add_argument('--ada_pct', type=float, default=67.0)
    ap.add_argument('--ada_alpha', type=float, default=0.5)
    ap.add_argument('--ada_grad', default='abs', choices=['abs', 'pos', 'neg'])
    ap.add_argument('--ada_no_bezier', action='store_true')
    ap.add_argument('--ada_no_shift', action='store_true')
    ap.add_argument('--ada_no_weaken', action='store_true')
    ap.add_argument('--out', default=os.path.join(OUTPUTS, 'sdg'))
    ap.add_argument('--ckpt_dir', default=os.path.join(SCRATCH, 'ckpt'))
    a = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0)
    # Sonic's V100 nodes (sm_70) ship a cuDNN that cannot execute ANY convolution under
    # torch 2.6.0+cu126 -- "FIND was unable to find an engine". Detect it and fall back rather than
    # crash mid-sweep; the run is then slow but correct, and the JSON records which.
    cudnn_ok = True
    try:
        F.conv2d(torch.randn(2, 1, 8, 8, device=device), torch.randn(4, 1, 3, 3, device=device))
        torch.cuda.synchronize()
    except RuntimeError as e:
        cudnn_ok = False
        torch.backends.cudnn.enabled = False
        print('WARNING cuDNN unusable on %s -> native convs. %s' % (gpu, str(e)[:70]), flush=True)

    rng_np = np.random.RandomState(a.seed)
    rng_t = torch.Generator(device=device); rng_t.manual_seed(a.seed + 12345)

    if a.bench == 'brats':
        Xtr, Ytr, Xva, Yva, case_va, val_cases, tr_cases = D.brats_source_split(
            a.source, a.seed, slices=a.brats_slices, region=a.region)
        targets = [d for d in D.BRATS_TARGETS if d != a.source]
        in_ch, out_ch, in_norm = 4, 1, False        # 4 modalities; z-scored in the brain mask
    elif a.bench == 'prostate':
        Xtr, Ytr, Xva, Yva, case_va, val_cases, tr_cases = D.prostate_source_split(
            a.source, a.seed, slices=a.slices)
        targets = [s for s in D.PROSTATE_SITES if s != a.source]
        in_ch, out_ch, in_norm = 1, 1, False        # already per-case z-scored at preprocessing
    elif a.bench == 'mms':
        # Split unit is the SUBJECT, not the subject-frame: a patient's ED and ES frames are the
        # same heart, so letting them straddle train/val would be a leak of exactly the kind this
        # study measures. `mms_source_split` enforces that.
        Xtr, Ytr, Xva, Yva, case_va = D.mms_source_split(
            a.source, a.seed, region=a.region, slices=a.slices)
        val_cases = sorted(set(case_va.tolist()))
        tr_cases = []
        targets = [c for c in D.MMS_CENTRES if c != a.source]
        in_ch, out_ch, in_norm = 1, 1, False        # per-case normalised at preprocessing
    else:
        Xtr, Ytr, Xva, Yva, name_va = D.riga_source_split(a.source, a.gt)
        targets = [d for d in D.RIGA_DOMAINS if d != a.source]
        in_ch, out_ch, in_norm = 3, 2, True         # per-image z-score, as in the published protocol
        val_cases, tr_cases = name_va.tolist(), []

    style_cls = (lambda: make_style(a.method)) if a.method in STYLE_METHODS else None
    if a.backbone == 'scratch':
        model = UNet(in_ch, out_ch, base=a.base, norm=a.norm, style=style_cls).to(device)
    elif a.backbone.startswith('dinov2'):
        # Third backbone. The decoder differs from the ResNet arm by necessity (a ViT has no
        # feature pyramid), so this arm compares backbone+decoder, not the encoder alone --
        # see unet_dino.py for every deviation.
        from unet_dino import DinoUNet
        model = DinoUNet(in_ch, out_ch, name='facebook/%s' % a.backbone, style=style_cls,
                         pretrained=bool(a.pretrained)).to(device)
        if a.freeze:
            model.freeze_encoder(a.freeze)
    else:
        from unet_pre import PretrainedUNet
        model = PretrainedUNet(in_ch, out_ch, arch=a.backbone, style=style_cls,
                               pretrained=bool(a.pretrained)).to(device)
        if a.freeze:
            model.freeze_encoder(a.freeze)
    if model.styles is not None:
        for s in model.styles:
            s.enabled = True
    ada_in = None
    if a.method == 'ada':
        ada_in = ADAInput(in_ch, not a.ada_no_bezier, not a.ada_no_shift).to(device)

    params = ([p for p in model.parameters() if p.requires_grad]
              + (list(ada_in.parameters()) if ada_in else []))
    print('trainable tensors: %d of %d' % (len(params), len(list(model.parameters()))), flush=True)
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    amp = bool(a.amp)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    it, best, best_state, best_ada, hist = 0, -1.0, None, None, []
    efdr_stats = {}
    t0 = time.time()
    model.train()
    while it < a.iters:
        for b in make_batches(len(Xtr_t), a.bs, rng_np):
            if it >= a.iters:
                break
            x = Xtr_t[b].to(device, non_blocking=True); y = Ytr_t[b].to(device, non_blocking=True)

            # ---- input-space stage (every method gets the same spatial augmentation)
            if a.method in INPUT_METHODS:
                x, y = apply_policy(a.method, x, y, rng_t)
            elif a.method in EFDR_METHODS:
                fam = {'efdr': 'both', 'efdr_bigaug': 'bigaug',
                       'efdr_randconv': 'randconv'}[a.method]
                x, y = spatial_aug(x, y, rng_t)
                fam_i = ('randconv' if torch.rand(1, generator=rng_t, device=device).item() < 0.5
                         else 'bigaug') if fam == 'both' else fam
                x, strength = efdr_apply(x, y, rng_t, tau=a.tau, family=fam_i,
                                         band=a.efdr_band, return_strength=True)
                # diagnostic: does the floor actually throttle RandConv on MRI and pass it on RGB?
                st = efdr_stats.setdefault(fam_i, [0.0, 0])
                st[0] += float(strength.sum()); st[1] += int(strength.numel())
            elif a.method in STYLE_METHODS:
                x, y = apply_policy('erm', x, y, rng_t)
            else:                                              # ada: paper's "Diversify Input"
                x, y = apply_policy(a.ada_static if a.ada_static != 'noise_contrast' else 'erm',
                                    x, y, rng_t)
                if a.ada_static == 'noise_contrast':
                    from aug import noise_contrast

                    x = noise_contrast(x, rng_t)
            if in_norm:
                m = x.mean(dim=(2, 3), keepdim=True); s = x.std(dim=(2, 3), keepdim=True)
                x = (x - m) / (s + 1e-6)

            for g in opt.param_groups:
                g['lr'] = a.lr * (1 - it / a.iters) ** 0.9

            # ---- MaxStyle: one gradient-ASCENT step on the style parameters before the model step
            if a.method == 'maxstyle':
                # pass 1 creates fresh style parameters and measures dL/d(style); pass 2 (below)
                # trains the network against the style that *maximises* the loss.
                for s_ in model.styles:
                    s_.need_reset = True
                loss_adv = seg_loss(model(x), y)
                sp = [p_ for s_ in model.styles for p_ in s_.params]
                gs = torch.autograd.grad(loss_adv, sp, allow_unused=True)
                with torch.no_grad():
                    for p_, g_ in zip(sp, gs):
                        if g_ is not None:
                            p_.add_(a.adv_lr * g_.sign())          # ascent: maximise the loss
                for p_ in sp:
                    p_.requires_grad_(False)
                opt.zero_grad(set_to_none=True)

            # ---- forward / backward
            if a.method == 'ada':
                x = ada_in(x)
                logit, feat = model(x, return_feat=True)
                if a.ada_no_weaken:
                    loss = seg_loss(logit, y)
                else:
                    l1 = seg_loss(logit, y)
                    g_feat = torch.autograd.grad(l1, feat, retain_graph=True)[0]
                    feat_w = weaken(feat, g_feat.detach(), a.ada_pct, a.ada_alpha, a.ada_grad)
                    loss = seg_loss(model(None, feat_override=feat_w), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 12)
                opt.step()
            else:
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    loss = seg_loss(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 12)
                scaler.step(opt); scaler.update()
            it += 1

            if it % a.val_every == 0 or it == a.iters:
                if model.styles is not None:
                    for s_ in model.styles:
                        s_.enabled = False           # style perturbation is a training-time operation
                P = predict(model, ada_in, Xva, a.bs, device, in_norm, amp).astype(bool)
                G = Yva.astype(bool)
                if a.bench in ('prostate', 'brats', 'mms'):
                    v = float(np.mean([dice_np(P[case_va == c, 0], G[case_va == c, 0])
                                       for c in sorted(set(case_va.tolist()))]))
                    parts = {'prostate': 'gland', 'brats': a.region, 'mms': a.region}[a.bench]
                    parts = {parts: round(v, 4)}
                else:
                    d = float(np.mean([dice_np(P[i, 0], G[i, 0]) for i in range(len(P))]))
                    c = float(np.mean([dice_np(P[i, 1], G[i, 1]) for i in range(len(P))]))
                    v, parts = (d + c) / 2, {'disc': round(d, 4), 'cup': round(c, 4)}
                hist.append({'iter': it, 'val': round(v, 4), **parts,
                             'loss': round(float(loss.item()), 4)})
                print('it %5d  loss %.4f  source-val %.4f  %s  (%.1f min)'
                      % (it, loss.item(), v, parts, (time.time() - t0) / 60), flush=True)
                if v > best:
                    best = v
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                    best_ada = ({k: t.detach().clone() for k, t in ada_in.state_dict().items()}
                                if ada_in else None)
                model.train()
                if model.styles is not None:
                    for s_ in model.styles:
                        s_.enabled = True

    model.load_state_dict(best_state)
    if ada_in is not None:
        ada_in.load_state_dict(best_ada)
    if model.styles is not None:
        for s_ in model.styles:
            s_.enabled = False

    res = {'config': vars(a), 'best_source_val': round(best, 4), 'history': hist,
           'train_slices': int(len(Xtr)), 'val_units': len(val_cases), 'train_cases': tr_cases,
           'val_cases': val_cases, 'minutes': round((time.time() - t0) / 60, 1),
           'torch': torch.__version__, 'gpu': gpu, 'cudnn': cudnn_ok,
           'efdr_mean_strength': {k: round(v[0] / max(v[1], 1), 4) for k, v in efdr_stats.items()},
           'per_domain': {}}
    if efdr_stats:
        print('EFDR mean admitted strength: %s' % res['efdr_mean_strength'], flush=True)

    for dom in ([a.source] + targets):
        if a.bench in ('prostate', 'brats', 'mms'):
            per = (eval_brats(model, ada_in, dom, a.bs, device, in_norm, amp,
                              a.brats_slices, a.region) if a.bench == 'brats'
                   else eval_mms(model, ada_in, dom, a.bs, device, in_norm, amp, a.region,
                                 a.slices) if a.bench == 'mms'
                   else eval_prostate(model, ada_in, dom, a.bs, device, in_norm, amp, a.slices))
            vals = np.array([v['dice'] for v in per.values()])
            res['per_domain'][dom] = {
                'n': len(per), 'dice_mean': round(float(vals.mean()), 4),
                'dice_std': round(float(vals.std()), 4),
                'precision_mean': round(float(np.mean([v['precision'] for v in per.values()])), 4),
                'recall_mean': round(float(np.mean([v['recall'] for v in per.values()])), 4),
                'per_case': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per.items()}}
            print('%-8s %-16s n=%3d  Dice %.4f +- %.4f'
                  % ('SOURCE' if dom == a.source else 'target', dom, len(per),
                     vals.mean(), vals.std()), flush=True)
        else:
            per = eval_riga(model, ada_in, dom, a.bs, device, in_norm, amp, a.gt)
            dv = np.array([v['disc']['dice'] for v in per.values()])
            cv = np.array([v['cup']['dice'] for v in per.values()])
            res['per_domain'][dom] = {
                'n': len(per), 'disc_mean': round(float(dv.mean()), 4),
                'disc_std': round(float(dv.std()), 4), 'cup_mean': round(float(cv.mean()), 4),
                'cup_std': round(float(cv.std()), 4),
                'per_image': {k: {s: {kk: round(vv, 4) for kk, vv in m.items()}
                                  for s, m in v.items()} for k, v in per.items()}}
            print('%-8s %-16s n=%3d  disc %.4f +- %.4f   cup %.4f +- %.4f'
                  % ('SOURCE' if dom == a.source else 'target', dom, len(per),
                     dv.mean(), dv.std(), cv.mean(), cv.std()), flush=True)

    tg = [res['per_domain'][d] for d in targets]
    if a.bench in ('prostate', 'brats', 'mms'):
        res['target_mean_dice'] = round(float(np.mean([t['dice_mean'] for t in tg])), 4)
        print('\nTARGET MEAN Dice = %.4f' % res['target_mean_dice'])
    else:
        res['target_mean_disc'] = round(float(np.mean([t['disc_mean'] for t in tg])), 4)
        res['target_mean_cup'] = round(float(np.mean([t['cup_mean'] for t in tg])), 4)
        print('\nTARGET MEAN  disc = %.4f   cup = %.4f'
              % (res['target_mean_disc'], res['target_mean_cup']))

    os.makedirs(a.out, exist_ok=True)
    tag = '%s_%s_%s_%s_s%d' % (a.bench, a.source, a.method, a.gt, a.seed)
    if a.bench == 'prostate' and a.slices != 'all':
        tag += '_' + a.slices
    if a.bench in ('brats', 'mms'):
        tag += '_' + a.region
    if a.backbone != 'scratch':
        tag += '_%s%s' % (a.backbone, '' if a.pretrained else 'scr')
        if a.freeze:
            tag += '_fz%d' % a.freeze
    tag += a.tag_suffix
    if a.ckpt_dir:
        os.makedirs(a.ckpt_dir, exist_ok=True)
        torch.save({'state_dict': best_state, 'ada': best_ada if ada_in else None,
                    'config': vars(a), 'best_source_val': best},
                   os.path.join(a.ckpt_dir, tag + '.pt'))
        res['ckpt'] = os.path.join(a.ckpt_dir, tag + '.pt')
    with open(os.path.join(a.out, tag + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('-> %s' % os.path.join(a.out, tag + '.json'))


if __name__ == '__main__':
    main()
