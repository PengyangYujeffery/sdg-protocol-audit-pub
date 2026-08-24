# Phase-1 SDG re-benchmark — prostate (GT=r1)

126 runs · sources BIDMC,BMC,HK,I2CVB,RUNMC,UCL · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (gland), averaged over the unseen domains

| source | erm | bigaug | randconv | mixstyle | dsu | maxstyle | ada | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|---|---|---|---|
| BIDMC | 0.3819 ±0.0498 | 0.4763 ±0.0996 | 0.4282 ±0.0600 | 0.3575 ±0.0697 | 0.3819 ±0.0280 | 0.4431 ±0.0641 | 0.4659 ±0.0340 | +0.0945 | +0.0464 |
| BMC | 0.4499 ±0.0412 | 0.6457 ±0.0411 | 0.6964 ±0.0250 | 0.4538 ±0.0343 | 0.4504 ±0.0675 | 0.4874 ±0.0495 | 0.4196 ±0.0173 | +0.1958 | +0.2465 |
| HK | 0.3362 ±0.0186 | 0.6002 ±0.0219 | 0.4259 ±0.0242 | 0.3210 ±0.0261 | 0.3482 ±0.0537 | 0.3995 ±0.0064 | 0.4520 ±0.0232 | +0.2639 | +0.0897 |
| I2CVB | 0.2414 ±0.0288 | 0.4677 ±0.0214 | 0.2159 ±0.0407 | 0.2980 ±0.0458 | 0.2635 ±0.0540 | 0.3587 ±0.0254 | 0.2990 ±0.0136 | +0.2263 | -0.0255 |
| RUNMC | 0.5180 ±0.0197 | 0.7635 ±0.0079 | 0.7168 ±0.0093 | 0.5547 ±0.0254 | 0.5668 ±0.0347 | 0.5883 ±0.0093 | 0.5455 ±0.0391 | +0.2455 | +0.1988 |
| UCL | 0.4430 ±0.0338 | 0.6791 ±0.0065 | 0.6370 ±0.0497 | 0.4383 ±0.0258 | 0.4488 ±0.0772 | 0.4862 ±0.0603 | 0.4937 ±0.0619 | +0.2361 | +0.1940 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs | resampling unit |
|---|---|---|---|---|
| bigaug | +0.2104 | [+0.1616, +0.2469] | 30 | source-clustered |
| randconv | +0.1250 | [+0.0477, +0.1964] | 30 | source-clustered |
| mixstyle | +0.0088 | [-0.0119, +0.0331] | 30 | source-clustered |
| dsu | +0.0148 | [+0.0032, +0.0301] | 30 | source-clustered |
| maxstyle | +0.0655 | [+0.0477, +0.0881] | 30 | source-clustered |
| ada | +0.0509 | [+0.0132, +0.0852] | 30 | source-clustered |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.3971 | 0.0998–0.6538 | **0.251** | 0.587 |
| bigaug | 0.7067 | 0.4870–0.8243 | **0.058** | 0.261 |
| randconv | 0.5639 | 0.2921–0.7614 | **0.116** | 0.428 |
| mixstyle | 0.4154 | 0.1080–0.6666 | **0.237** | 0.575 |
| dsu | 0.4223 | 0.1121–0.6733 | **0.238** | 0.570 |
| maxstyle | 0.5064 | 0.2056–0.7085 | **0.157** | 0.496 |
| ada | 0.5063 | 0.1622–0.7183 | **0.204** | 0.489 |

**ERM per-pair target Dice (gland) — the failure map phase 2 starts from:**

| source \ target | BIDMC | BMC | HK | I2CVB | RUNMC | UCL |
|---|---|---|---|---|---|---|
| BIDMC | — | 0.231 | 0.824 | 0.419 | 0.324 | 0.111 |
| BMC | 0.237 | — | 0.567 | 0.145 | 0.616 | 0.684 |
| HK | 0.448 | 0.176 | — | 0.254 | 0.613 | 0.190 |
| I2CVB | 0.235 | 0.065 | 0.514 | — | 0.331 | 0.062 |
| RUNMC | 0.423 | 0.453 | 0.811 | 0.274 | — | 0.629 |
| UCL | 0.232 | 0.714 | 0.410 | 0.162 | 0.697 | — |
