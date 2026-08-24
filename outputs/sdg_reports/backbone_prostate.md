# Phase-1 SDG re-benchmark — prostate (GT=r1)

144 runs · sources BIDMC,BMC,HK,I2CVB,RUNMC,UCL · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (gland), averaged over the unseen domains

| source | erm | bigaug | randconv | mixstyle | dsu | maxstyle | ada | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|---|---|---|---|
| BIDMC | 0.6437 ±0.0468 | 0.6977 ±0.0236 | 0.6832 ±0.0067 | 0.6973 ±0.0305 | 0.7140 ±0.0327 | 0.7266 ±0.0409 | 0.7386 ±0.0248 | +0.0540 | +0.0396 |
| BMC | 0.6870 ±0.0099 | 0.7480 ±0.0508 | 0.7965 ±0.0208 | 0.6867 ±0.0052 | 0.6620 ±0.0231 | 0.7430 ±0.0175 | 0.7105 ±0.0411 | +0.0610 | +0.1094 |
| HK | 0.4721 ±0.1102 | 0.6750 ±0.0630 | 0.6361 ±0.0270 | 0.5715 ±0.0551 | 0.5791 ±0.0106 | 0.6051 ±0.0278 | 0.6187 ±0.0319 | +0.2029 | +0.1640 |
| I2CVB | 0.2044 ±0.0330 | 0.5304 ±0.0096 | 0.5206 ±0.1016 | 0.2860 ±0.1353 | 0.2288 ±0.0475 | 0.3836 ±0.0626 | 0.3312 ±0.0293 | +0.3260 | +0.3162 |
| RUNMC | 0.6720 ±0.0492 | 0.8104 ±0.0257 | 0.8160 ±0.0288 | 0.7617 ±0.0239 | 0.7238 ±0.0294 | 0.7948 ±0.0198 | 0.7951 ±0.0356 | +0.1385 | +0.1440 |
| UCL | 0.6500 ±0.0353 | 0.7241 ±0.0098 | 0.7300 ±0.0184 | 0.7226 ±0.0364 | 0.7032 ±0.0250 | 0.7258 ±0.0218 | 0.7366 ±0.0037 | +0.0741 | +0.0799 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs | resampling unit |
|---|---|---|---|---|
| bigaug | +0.1372 | [+0.0747, +0.2239] | 30 | source-clustered |
| randconv | +0.1469 | [+0.0843, +0.2215] | 30 | source-clustered |
| mixstyle | +0.0708 | [+0.0387, +0.0988] | 30 | source-clustered |
| dsu | +0.0517 | [+0.0148, +0.0853] | 30 | source-clustered |
| maxstyle | +0.1130 | [+0.0789, +0.1485] | 30 | source-clustered |
| ada | +0.1050 | [+0.0677, +0.1377] | 30 | source-clustered |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.5922 | 0.2790–0.7683 | **0.151** | 0.405 |
| bigaug | 0.7689 | 0.5975–0.8486 | **0.034** | 0.179 |
| randconv | 0.7714 | 0.6174–0.8509 | **0.019** | 0.156 |
| mixstyle | 0.7132 | 0.5026–0.8241 | **0.079** | 0.248 |
| dsu | 0.7078 | 0.4649–0.8193 | **0.094** | 0.269 |
| maxstyle | 0.7519 | 0.5775–0.8431 | **0.036** | 0.194 |
| ada | 0.7566 | 0.5583–0.8440 | **0.054** | 0.210 |

**ERM per-pair target Dice (gland) — the failure map phase 2 starts from:**

| source \ target | BIDMC | BMC | HK | I2CVB | RUNMC | UCL |
|---|---|---|---|---|---|---|
| BIDMC | — | 0.613 | 0.862 | 0.487 | 0.693 | 0.564 |
| BMC | 0.510 | — | 0.707 | 0.603 | 0.805 | 0.810 |
| HK | 0.633 | 0.311 | — | 0.412 | 0.560 | 0.407 |
| I2CVB | 0.099 | 0.064 | 0.350 | — | 0.403 | 0.123 |
| RUNMC | 0.316 | 0.695 | 0.764 | 0.676 | — | 0.787 |
| UCL | 0.298 | 0.730 | 0.740 | 0.674 | 0.809 | — |
