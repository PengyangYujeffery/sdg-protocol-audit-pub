# BraTS ET — what separates the failing African cases? (africa_other target)

51 cases, 4 methods. "Failure" = per-case Dice < 0.5 under the best method (randconv); 10 of 51 cases (19.6 %).


## Per-method mean Dice on this target

| method | mean | median | frac < 0.5 |
|---|---|---|---|
| randconv | 0.6935 | 0.7666 | 0.196 |
| erm | 0.6907 | 0.7880 | 0.176 |
| ada | 0.6203 | 0.6667 | 0.275 |
| bigaug | 0.5824 | 0.6828 | 0.314 |

## Which covariate separates failure from success?

AUC > 0.5 means a **larger** value goes with failure. Label-free covariates are the only actionable ones — a deployed model has no ground truth.

| covariate | kind | AUC(failure) | r with (RandConv − BigAug) per case |
|---|---|---|---|
| et_t1c_contrast | label-dep | **0.129** | +0.108 |
| et_frac_of_wt | label-dep | **0.234** | +0.190 |
| et_volume_frac | label-dep | **0.239** | +0.255 |
| t1n_std | **label-free** | **0.707** | -0.083 |
| et_components | label-dep | **0.356** | +0.222 |
| n_slices | **label-free** | **0.598** | +0.214 |
| t2f_std | **label-free** | **0.422** | -0.217 |
| t2f_hf_energy | **label-free** | **0.578** | +0.010 |
| t1c_std | **label-free** | **0.429** | -0.014 |
| t2w_kurtosis | **label-free** | **0.568** | +0.126 |
| brain_frac | **label-free** | **0.544** | +0.151 |
| t1c_kurtosis | **label-free** | **0.463** | +0.199 |
| t2w_std | **label-free** | **0.476** | +0.072 |
| t1n_kurtosis | **label-free** | **0.515** | +0.209 |
| t2f_kurtosis | **label-free** | **0.510** | +0.071 |

## Reading

- strongest separator overall: **et_t1c_contrast** (AUC 0.129, label-dep)

- strongest **label-free** separator: **t1n_std** (AUC 0.707) — this is the only kind a method could act on


🔴 No mechanism is asserted here. A separator with AUC near 0.5 means the failing cases are not distinguished by that property, and a method conditioned on it cannot work. Anything above ~0.70 among the label-free covariates is worth a designed test.

