#!/usr/bin/env python
"""
Convert DG Prostate into nnU-Net v2 raw format — one dataset per SOURCE site.

Why nnU-Net matters here: every conclusion in this project so far was measured on a from-scratch 2D
U-Net, and the most predictable reviewer objection is that the findings are artefacts of a weak
baseline. nnU-Net is the de-facto strong baseline in medical segmentation and wins challenges without
manual tuning, so it is the right instrument to test that objection.

Two questions it answers, and only the second needs custom work:
  Q1 How far below a properly configured nnU-Net is our backbone on the SAME single-source task?
  Q2 Do the augmentation effects survive on that backbone? nnU-Net ships official trainer variants
      (`nnUNetTrainerNoDA`, the default, `nnUNetTrainerDA5`) that span an augmentation-strength axis,
      so this comes almost free and uses *their* implementation rather than ours.

Layout produced (one dataset per source site, so nnU-Net trains on exactly the data our protocol
allows it to see):

    nnUNet_raw/Dataset5NN_ProstSITE/
        imagesTr/<case>_0000.nii.gz labelsTr/<case>.nii.gz dataset.json
    infer/<SITE>/ <case>_0000.nii.gz (every other site, for prediction)
    infer_gt/<SITE>/ <case>.nii.gz

Slices are stacked back into a pseudo-volume per case so nnU-Net's own 2D pipeline does the
slicing; spacing is set to 1 so its resampling is a no-op on data that is already resampled. The
foreground-slice policy is NOT applied here — nnU-Net chooses its own sampling, and forcing ours
would be exactly the train/eval mismatch that cost us a sweep on 2026-08-09.
"""
import json, os, argparse
import numpy as np
import nibabel as nib

import data as D

# Roots. Override with SDG_OUTPUTS / SDG_SCRATCH; the defaults assume this file sits in
# code/sdg/ of the released repository, with the run records mirrored under outputs/.
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('SDG_SCRATCH', os.path.join(_HERE, 'scratch'))

IDS = {'BIDMC': 51, 'BMC': 52, 'HK': 53, 'I2CVB': 54, 'RUNMC': 55, 'UCL': 56}


def write_case(path, arr, dtype):
    img = nib.Nifti1Image(np.transpose(arr, (1, 2, 0)).astype(dtype), np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(SCRATCH, 'nnunet'))
    a = ap.parse_args()
    raw = os.path.join(a.root, 'nnUNet_raw')

    # every site written once as an inference set, reused by every source's model
    for site in D.PROSTATE_SITES:
        X, Y, case, fg = D.load_prostate(site)
        di = os.path.join(a.root, 'infer', site); dg = os.path.join(a.root, 'infer_gt', site)
        os.makedirs(di, exist_ok=True); os.makedirs(dg, exist_ok=True)
        for cs in sorted(set(case.tolist())):
            k = case == cs
            write_case(os.path.join(di, '%s_0000.nii.gz' % cs), X[k], np.float32)
            write_case(os.path.join(dg, '%s.nii.gz' % cs), Y[k], np.uint8)
        print('infer set %-7s %3d cases' % (site, len(set(case.tolist()))), flush=True)

    for site, num in IDS.items():
        X, Y, case, fg = D.load_prostate(site)
        d = os.path.join(raw, 'Dataset%03d_Prost%s' % (num, site))
        os.makedirs(os.path.join(d, 'imagesTr'), exist_ok=True)
        os.makedirs(os.path.join(d, 'labelsTr'), exist_ok=True)
        cases = sorted(set(case.tolist()))
        for cs in cases:
            k = case == cs
            write_case(os.path.join(d, 'imagesTr', '%s_0000.nii.gz' % cs), X[k], np.float32)
            write_case(os.path.join(d, 'labelsTr', '%s.nii.gz' % cs), Y[k], np.uint8)
        json.dump({'channel_names': {'0': 'T2'},
                   'labels': {'background': 0, 'gland': 1},
                   'numTraining': len(cases),
                   'file_ending': '.nii.gz',
                   'description': 'DG Prostate, single source site %s, whole gland' % site},
                  open(os.path.join(d, 'dataset.json'), 'w'), indent=2)
        print('Dataset%03d_Prost%-7s %3d training cases -> %s' % (num, site, len(cases), d), flush=True)


if __name__ == '__main__':
    main()
