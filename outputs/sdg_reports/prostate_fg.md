# Phase-1 SDG re-benchmark — prostate (GT=r1)

54 runs · sources BIDMC,BMC,HK,I2CVB,RUNMC,UCL · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (gland), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BIDMC | 0.3521 ±0.0303 | 0.5585 ±0.0179 | 0.4476 ±0.0194 | +0.2064 | +0.0955 |
| BMC | 0.4261 ±0.0265 | 0.6665 ±0.0273 | 0.6795 ±0.0097 | +0.2404 | +0.2534 |
| HK | 0.3248 ±0.0220 | 0.5754 ±0.0289 | 0.4432 ±0.0524 | +0.2506 | +0.1184 |
| I2CVB | 0.2512 ±0.0023 | 0.5223 ±0.0165 | 0.2219 ±0.0335 | +0.2711 | -0.0293 |
| RUNMC | 0.5315 ±0.0091 | 0.7607 ±0.0074 | 0.6935 ±0.0053 | +0.2292 | +0.1620 |
| UCL | 0.4652 ±0.0392 | 0.6877 ±0.0209 | 0.6517 ±0.0372 | +0.2225 | +0.1865 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | +0.2367 | [+0.2210, +0.2540] | 30 |
| randconv | +0.1311 | [+0.0591, +0.1966] | 30 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.3952 | 0.0862–0.6487 | **0.266** | 0.587 |
| bigaug | 0.7184 | 0.5352–0.8259 | **0.043** | 0.222 |
| randconv | 0.5837 | 0.3199–0.7601 | **0.111** | 0.414 |

**ERM per-pair target Dice (gland) — the failure map phase 2 starts from:**

| source \ target | BIDMC | BMC | HK | I2CVB | RUNMC | UCL |
|---|---|---|---|---|---|---|
| BIDMC | — | 0.192 | 0.810 | 0.366 | 0.305 | 0.087 |
| BMC | 0.247 | — | 0.512 | 0.072 | 0.616 | 0.684 |
| HK | 0.515 | 0.164 | — | 0.244 | 0.555 | 0.147 |
| I2CVB | 0.252 | 0.080 | 0.506 | — | 0.356 | 0.062 |
| RUNMC | 0.403 | 0.522 | 0.782 | 0.286 | — | 0.665 |
| UCL | 0.275 | 0.723 | 0.514 | 0.123 | 0.692 | — |
