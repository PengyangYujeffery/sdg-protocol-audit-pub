# BraTS ET — what separates the failing African cases? (africa_glioma target)

95 cases, 4 methods. "Failure" = per-case Dice < 0.5 under the best method (randconv); 10 of 95 cases (10.5 %).


## Per-method mean Dice on this target

| method | mean | median | frac < 0.5 |
|---|---|---|---|
| randconv | 0.6998 | 0.7497 | 0.105 |
| erm | 0.6429 | 0.6886 | 0.211 |
| ada | 0.5727 | 0.6327 | 0.326 |
| bigaug | 0.5077 | 0.5730 | 0.442 |

## Which covariate separates failure from success?

AUC > 0.5 means a **larger** value goes with failure. Label-free covariates are the only actionable ones — a deployed model has no ground truth.

| covariate | kind | AUC(failure) | r with (RandConv − BigAug) per case |
|---|---|---|---|
| et_t1c_contrast | label-dep | **0.267** | +0.070 |
| et_volume_frac | label-dep | **0.268** | -0.043 |
| et_frac_of_wt | label-dep | **0.319** | -0.041 |
| t2w_kurtosis | **label-free** | **0.382** | -0.046 |
| t2f_kurtosis | **label-free** | **0.582** | +0.102 |
| t1c_std | **label-free** | **0.424** | -0.078 |
| brain_frac | **label-free** | **0.567** | +0.002 |
| t1c_kurtosis | **label-free** | **0.434** | +0.058 |
| t2f_hf_energy | **label-free** | **0.564** | +0.129 |
| t2w_std | **label-free** | **0.456** | +0.104 |
| n_slices | **label-free** | **0.466** | +0.109 |
| t1n_kurtosis | **label-free** | **0.471** | +0.020 |
| t2f_std | **label-free** | **0.475** | -0.282 |
| et_components | label-dep | **0.524** | +0.046 |
| t1n_std | **label-free** | **0.508** | -0.020 |

## Reading

- strongest separator overall: **et_t1c_contrast** (AUC 0.267, label-dep)

- strongest **label-free** separator: **t2w_kurtosis** (AUC 0.382) — this is the only kind a method could act on


🔴 No mechanism is asserted here. A separator with AUC near 0.5 means the failing cases are not distinguished by that property, and a method conditioned on it cannot work. Anything above ~0.70 among the label-free covariates is worth a designed test.

