# Phase-2 failure analysis — DG Prostate

18 runs (seed 0, slices=fg, per-case precision/recall/volumes)


## Q1 — precision vs recall on unseen sites (all 30 transfer pairs pooled)

| method | Dice | precision | recall | median pred/true volume |
|---|---|---|---|---|
| erm | 0.3887 | 0.6151 | 0.3483 | 0.479 |
| bigaug | 0.6379 | 0.7663 | 0.6047 | 0.824 |
| randconv | 0.5334 | 0.8277 | 0.4473 | 0.506 |

**RandConv − BigAug: precision +0.0614, recall -0.1574.** The hypothesis predicted the loss would sit in *recall*; it does — RandConv finds less gland rather than over-drawing it.


## Q3 — anatomy of a total failure (per-case Dice < 0.10)

| method | share of cases | of those, predicted ~nothing (vol<0.1×) | predicted ~everywhere (vol>3×) |
|---|---|---|---|
| erm | 0.247 | 0.57 | 0.01 |
| bigaug | 0.036 | 0.52 | 0.00 |
| randconv | 0.102 | 0.76 | 0.00 |

## Q2 — is transferability explained by intensity-histogram distance?

Pearson r between source↔target histogram L1 distance and ERM target Dice over 30 ordered pairs: **r = +0.041** (r² = 0.002).

The distance is **symmetric by construction**, but the transfer matrix is not: the mean |Dice(s→t) − Dice(t→s)| over 15 site pairs is **0.174** (max 0.435). Whatever explains direction is therefore not a distance.

| pair | s→t | t→s | gap |
|---|---|---|---|
| HK ↔ UCL | 0.124 | 0.559 | 0.435 |
| BIDMC ↔ UCL | 0.093 | 0.408 | 0.315 |
| BMC ↔ HK | 0.482 | 0.168 | 0.314 |
| BIDMC ↔ HK | 0.807 | 0.541 | 0.265 |
| HK ↔ I2CVB | 0.302 | 0.526 | 0.224 |
| HK ↔ RUNMC | 0.625 | 0.794 | 0.170 |
| BMC ↔ RUNMC | 0.599 | 0.432 | 0.166 |
| RUNMC ↔ UCL | 0.579 | 0.729 | 0.150 |
| BIDMC ↔ RUNMC | 0.251 | 0.393 | 0.143 |
| BIDMC ↔ BMC | 0.122 | 0.254 | 0.132 |
| BMC ↔ UCL | 0.629 | 0.750 | 0.121 |
| I2CVB ↔ RUNMC | 0.314 | 0.398 | 0.084 |
| BMC ↔ I2CVB | 0.177 | 0.110 | 0.067 |
| BIDMC ↔ I2CVB | 0.266 | 0.244 | 0.021 |
| I2CVB ↔ UCL | 0.061 | 0.066 | 0.006 |

## Q4 — is the transfer matrix pairwise, or just "source capability + target difficulty"?

| method | R² of additive model | residual SD | source effects α (best→worst) | target effects β (easiest→hardest) |
|---|---|---|---|---|
| erm | **0.577** | 0.148 | RUNMC +0.15 > UCL +0.09 > BMC +0.02 > HK +0.01 > BIDMC -0.09 > I2CVB -0.18 | HK +0.24 > RUNMC +0.14 > BIDMC -0.04 > BMC -0.07 > UCL -0.08 > I2CVB -0.19 |
| bigaug | **0.722** | 0.097 | RUNMC +0.18 > BMC +0.10 > UCL +0.07 > HK -0.05 > I2CVB -0.14 > BIDMC -0.15 | RUNMC +0.11 > HK +0.10 > BMC +0.06 > UCL +0.05 > I2CVB -0.11 > BIDMC -0.21 |
| randconv | **0.802** | 0.097 | RUNMC +0.20 > BMC +0.16 > UCL +0.13 > BIDMC -0.05 > HK -0.12 > I2CVB -0.32 | HK +0.12 > RUNMC +0.11 > BMC -0.01 > UCL -0.02 > BIDMC -0.05 > I2CVB -0.14 |

**What does a good source have?** α (ERM source effect) against the site properties measured at preprocessing:

| covariate | Pearson r with α |
|---|---|
| cases | +0.373 |
| slices | -0.567 |
| foreground slice fraction | +0.769 |
| mean gland area (px) | -0.102 |
