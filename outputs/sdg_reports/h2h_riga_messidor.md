# Phase-1 SDG re-benchmark — riga (GT=r1)

42 runs · sources BinRushed,Magrabia · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (disc), averaged over the unseen domains

| source | erm | bigaug | randconv | mixstyle | dsu | maxstyle | ada | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|---|---|---|---|
| BinRushed | 0.8317 ±0.0070 | 0.9102 ±0.0153 | 0.9518 ±0.0005 | 0.8447 ±0.0173 | 0.8473 ±0.0107 | 0.8771 ±0.0045 | 0.9067 ±0.0084 | +0.0784 | +0.1201 |
| Magrabia | 0.9295 ±0.0028 | 0.9382 ±0.0023 | 0.9480 ±0.0017 | 0.9256 ±0.0075 | 0.9299 ±0.0056 | 0.9264 ±0.0056 | 0.9340 ±0.0087 | +0.0087 | +0.0184 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs | resampling unit |
|---|---|---|---|---|
| bigaug | +0.0436 | [+0.0108, +0.0823] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| randconv | +0.0693 | [+0.0204, +0.1294] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| mixstyle | +0.0045 | [-0.0042, +0.0126] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| dsu | +0.0080 | [+0.0014, +0.0147] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| maxstyle | +0.0211 | [-0.0000, +0.0456] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| ada | +0.0397 | [+0.0056, +0.0782] | 6 | pair-level (only 2 sources -- NOT source-clustered) |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.9503 | 0.9137–0.9646 | **0.020** | 0.062 |
| bigaug | 0.9520 | 0.9291–0.9643 | **0.000** | 0.014 |
| randconv | 0.9576 | 0.9404–0.9685 | **0.000** | 0.000 |
| mixstyle | 0.9498 | 0.9155–0.9637 | **0.012** | 0.055 |
| dsu | 0.9493 | 0.9152–0.9641 | **0.013** | 0.048 |
| maxstyle | 0.9514 | 0.9232–0.9640 | **0.008** | 0.041 |
| ada | 0.9508 | 0.9238–0.9641 | **0.001** | 0.018 |

**ERM per-pair target Dice (disc) — the failure map phase 2 starts from:**

| source \ target | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 |
|---|---|---|---|
| BinRushed | 0.901 | 0.748 | 0.846 |
| Magrabia | 0.943 | 0.906 | 0.940 |

## Target-mean Dice (cup), averaged over the unseen domains

| source | erm | bigaug | randconv | mixstyle | dsu | maxstyle | ada | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|---|---|---|---|
| BinRushed | 0.7429 ±0.0070 | 0.7940 ±0.0115 | 0.8686 ±0.0034 | 0.7391 ±0.0086 | 0.7558 ±0.0164 | 0.7907 ±0.0035 | 0.8134 ±0.0092 | +0.0511 | +0.1257 |
| Magrabia | 0.8389 ±0.0038 | 0.8213 ±0.0033 | 0.8712 ±0.0062 | 0.8362 ±0.0089 | 0.8375 ±0.0063 | 0.8447 ±0.0068 | 0.8584 ±0.0069 | -0.0176 | +0.0323 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs | resampling unit |
|---|---|---|---|---|
| bigaug | +0.0167 | [-0.0181, +0.0606] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| randconv | +0.0790 | [+0.0323, +0.1383] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| mixstyle | -0.0032 | [-0.0092, +0.0019] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| dsu | +0.0057 | [-0.0015, +0.0130] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| maxstyle | +0.0268 | [+0.0085, +0.0478] | 6 | pair-level (only 2 sources -- NOT source-clustered) |
| ada | +0.0449 | [+0.0175, +0.0767] | 6 | pair-level (only 2 sources -- NOT source-clustered) |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.8596 | 0.7853–0.9018 | **0.040** | 0.077 |
| bigaug | 0.8395 | 0.7699–0.8841 | **0.004** | 0.032 |
| randconv | 0.8877 | 0.8419–0.9183 | **0.000** | 0.001 |
| mixstyle | 0.8575 | 0.7790–0.9012 | **0.039** | 0.079 |
| dsu | 0.8568 | 0.7879–0.9014 | **0.031** | 0.073 |
| maxstyle | 0.8668 | 0.8073–0.9082 | **0.025** | 0.049 |
| ada | 0.8679 | 0.8082–0.9074 | **0.005** | 0.023 |

**ERM per-pair target Dice (cup) — the failure map phase 2 starts from:**

| source \ target | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 |
|---|---|---|---|
| BinRushed | 0.810 | 0.650 | 0.769 |
| Magrabia | 0.842 | 0.815 | 0.860 |
