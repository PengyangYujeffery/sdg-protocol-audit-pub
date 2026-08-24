"""
Data access for the two 2D SDG benchmarks. One rule governs everything here:

    model selection never sees a target domain.

Every SDG paper says it trains on one source; many then pick the best epoch on the target. That
single leak is enough to explain a large part of why published baseline numbers do not agree. Here
the source domain is split into train/val and the val split is the *only* thing selection may look at.

Splits are deterministic given the seed and are made at the unit that can leak:
  * prostate -- by case. Slices of one patient must never straddle train and val.
  * RIGA+ -- by the benchmark's own train/test CSV lists (156/39 for BinRushed etc.), so the
                 source-val set is the one the dataset authors defined.
"""
import os
import numpy as np

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

# overridable so the same code runs on another cluster, where the scratch layout differs
PROSTATE_DIR = os.environ.get('PROSTATE_2D', os.path.join(SCRATCH, 'prostate_2d'))
RIGA_DIR = os.environ.get('RIGA_2D', os.path.join(SCRATCH, 'riga_2d'))
BRATS_DIR = os.environ.get('BRATS_2D', os.path.join(SCRATCH, 'brats_2d'))
MMS_DIR = os.environ.get('MMS_2D', os.path.join(SCRATCH, 'mms_2d'))
PROSTATE_SITES = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
# M&Ms: domain = CENTRE, not vendor. The two are near-collinear (only Philips spans two centres),
# and centre is the finer partition, giving five domains -- enough for a source-clustered bootstrap.
# The collinearity means no vendor-vs-site claim is available from this dataset. See prep_mms.py.
MMS_CENTRES = ['centre1', 'centre2', 'centre3', 'centre4', 'centre5']
MMS_REGIONS = {'lv': (1,), 'myo': (2,), 'rv': (3,)}   # reported separately, never pooled
RIGA_DOMAINS = ['BinRushed', 'Magrabia', 'MESSIDOR_Base1', 'MESSIDOR_Base2', 'MESSIDOR_Base3']
RIGA_SOURCES = ['BinRushed', 'Magrabia']          # the published single-source protocol
RIGA_TARGETS = ['MESSIDOR_Base1', 'MESSIDOR_Base2', 'MESSIDOR_Base3']


# ---------------------------------------------------------------- prostate
def load_prostate(site):
    """-> X (N,H,W) float32 already per-case normalised, Y (N,H,W) uint8 binary whole gland,
    case (N,) str, fg (N,) uint8."""
    z = np.load('%s/%s.npz' % (PROSTATE_DIR, site), allow_pickle=False)
    return z['X'], z['Y'], z['case'], z['fg']


def prostate_source_split(site, seed, val_frac=0.25, slices='all'):
    """Case-level train/val split of one source site.

    `slices` sets train and validation policy together -- a model trained only on gland-bearing
    slices has never been shown an empty slice, so scoring it on whole volumes measures a mismatch
    we created rather than a property of the method. Whichever policy is chosen, it is applied at
    training, at validation and at target evaluation alike.
    """
    X, Y, case, fg = load_prostate(site)
    cases = np.array(sorted(set(case.tolist())))
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)
    n_val = max(2, int(round(val_frac * len(cases))))
    val_cases = set(cases[:n_val].tolist())
    is_val = np.array([c in val_cases for c in case])
    keep = (fg > 0) if slices == 'fg' else np.ones(len(X), bool)
    tr = ~is_val & keep
    va = is_val & keep
    return (X[tr][:, None], Y[tr][:, None].astype(np.float32),
            X[va][:, None], Y[va][:, None].astype(np.float32), case[va],
            sorted(val_cases), sorted(set(case[~is_val].tolist())))


def prostate_domain(site, slices='all'):
    """whole site, for target-domain evaluation: X, Y, case.

    `slices='fg'` keeps only slices whose ground truth contains gland -- the convention the
    SAML/FedDG-line papers evaluate under, and the only one comparable with their published Dice.
    `slices='all'` scores the whole volume, which is the deployment-realistic setting and punishes
    false positives on empty slices. The two are NOT interchangeable and the difference is measured
    rather than assumed; see PROTOCOL §3.
    """
    X, Y, case, fg = load_prostate(site)
    if slices == 'fg':
        keep = fg > 0
        X, Y, case = X[keep], Y[keep], case[keep]
    return X[:, None], Y[:, None].astype(np.float32), case


# ---------------------------------------------------------------- M&Ms (fourth benchmark)
def load_mms(centre):
    """-> X (N,1,H,W) float32 per-case normalised, Y (N,1,H,W) uint8 raw labels {0,1,2,3},
    case (N,) '<subject>_t<frame>', fg (N,) bool. Ground truth exists only at the ED and ES
    frames; every other cardiac phase was dropped at preprocessing."""
    z = np.load('%s/%s.npz' % (MMS_DIR, centre), allow_pickle=False)
    return z['X'], z['Y'], z['case'], z['fg']


def _mms_y(Y, region):
    out = np.zeros(Y.shape, np.float32)
    for v in MMS_REGIONS[region]:
        out[Y == v] = 1.0
    return out


def mms_source_split(centre, seed, region='lv', val_frac=0.25, slices='fg'):
    """Case-level split of one centre. The split unit is the subject, not the subject-frame:
    a patient's ED and ES frames are the same heart and must never straddle train and validation."""
    X, Y, case, fg = load_mms(centre)
    subj = np.array([c.rsplit('_t', 1)[0] for c in case])
    if slices == 'fg':
        keep = fg > 0
        X, Y, case, subj = X[keep], Y[keep], case[keep], subj[keep]
    su = np.array(sorted(set(subj.tolist())))
    rng = np.random.RandomState(seed)
    va_s = set(su[rng.permutation(len(su))[:max(1, int(round(val_frac * len(su))))]].tolist())
    va = np.array([s in va_s for s in subj])
    Yb = _mms_y(Y, region)
    return X[~va], Yb[~va], X[va], Yb[va], case[va]


def mms_domain(centre, region='lv', slices='fg'):
    """whole centre, for target-domain evaluation: X, Y, case."""
    X, Y, case, fg = load_mms(centre)
    if slices == 'fg':
        keep = fg > 0
        X, Y, case = X[keep], Y[keep], case[keep]
    return X, _mms_y(Y, region), case


# ---------------------------------------------------------------- RIGA+
def _riga_gt(z, gt):
    """Ground-truth convention. `r1` = rater 1 only, which is what the published RIGA+ SDG numbers
    use (C2SDG, MICCAI 2023). `majority` = >=3 of 6 raters. They are NOT interchangeable: measured at
    preprocessing time, r1-vs-majority Dice is 0.969 for the disc but only ~0.90 for the cup, i.e.
    the annotation convention moves the cup score by more than the gap between published methods."""
    db, cb = z['disc_bits'], z['cup_bits']
    if gt == 'r1':
        d, c = (db & 1) > 0, (cb & 1) > 0
    elif gt == 'majority':
        pc = np.unpackbits(db[..., None], axis=-1, count=6, bitorder='little').sum(-1)
        pu = np.unpackbits(cb[..., None], axis=-1, count=6, bitorder='little').sum(-1)
        d, c = pc >= 3, pu >= 3
    else:
        raise ValueError(gt)
    return np.stack([d, c], 1).astype(np.float32)   # channel 0 = disc, 1 = cup (nested inside disc)


def load_riga(domain, gt='r1'):
    """-> X (N,3,H,W) float32 in [0,1], Y (N,2,H,W) float32, name (N,), split (N,)."""
    z = np.load('%s/%s.npz' % (RIGA_DIR, domain), allow_pickle=False)
    X = z['X'].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return X, _riga_gt(z, gt), z['name'], z['split']


def riga_source_split(domain, gt='r1'):
    """the benchmark's own train/test lists become train/val -- no target domain is touched."""
    X, Y, name, split = load_riga(domain, gt)
    tr, va = split == 'train', split == 'test'
    return X[tr], Y[tr], X[va], Y[va], name[va]


def riga_domain(domain, gt='r1'):
    X, Y, name, _ = load_riga(domain, gt)
    return X, Y, name


# ---------------------------------------------------------------- BraTS (population axis)
BRATS_SOURCE = 'gli2023'                          # Western, PRE-OPERATIVE adult glioma
BRATS_TARGETS = ['africa_glioma', 'africa_other']  # never pooled: 95 glioma vs 51 other neoplasms
BRATS_REGIONS = {'wt': (1, 2, 3), 'tc': (1, 3), 'et': (3,)}


def load_brats(name):
    """memory-mapped, because these arrays are several GB and every run would otherwise pay the
    decompression cost. X is (N,4,H,W) float16; Y keeps the raw BraTS label values."""
    p = lambda s: os.path.join(BRATS_DIR, '%s_%s.npy' % (name, s))
    return (np.load(p('X'), mmap_mode='r'), np.load(p('Y'), mmap_mode='r'),
            np.load(p('case')), np.load(p('tumour')))


def _brats_y(Y, region):
    lab = BRATS_REGIONS[region]
    out = np.zeros(Y.shape, np.float32)
    for v in lab:
        out[Y == v] = 1.0
    return out[:, None]


def brats_source_split(name, seed, val_frac=0.2, slices='tumour', region='wt'):
    """patient-level split. BraTS case ids carry a timepoint suffix and one patient can appear many
    times; `prep_brats.py` already keeps one timepoint per patient, and the split is by case id."""
    X, Y, case, tum = load_brats(name)
    cases = np.array(sorted(set(case.tolist())))
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)
    n_val = max(2, int(round(val_frac * len(cases))))
    val_cases = set(cases[:n_val].tolist())
    is_val = np.array([c in val_cases for c in case])
    keep = (tum > 0) if slices == 'tumour' else np.ones(len(case), bool)
    tr, va = np.where(~is_val & keep)[0], np.where(is_val & keep)[0]
    return (np.asarray(X[tr], np.float32), _brats_y(np.asarray(Y[tr]), region),
            np.asarray(X[va], np.float32), _brats_y(np.asarray(Y[va]), region), case[va],
            sorted(val_cases), sorted(set(case[~is_val].tolist())))


def brats_domain(name, slices='tumour', region='wt'):
    X, Y, case, tum = load_brats(name)
    idx = np.where(tum > 0)[0] if slices == 'tumour' else np.arange(len(case))
    return np.asarray(X[idx], np.float32), _brats_y(np.asarray(Y[idx]), region), case[idx]
