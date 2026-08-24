# Phase-1 SDG re-benchmark — riga (GT=r1)

18 runs · sources BinRushed,Magrabia · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (disc), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BinRushed | 0.8537 ±0.0174 | 0.9173 ±0.0083 | 0.9512 ±0.0010 | +0.0637 | +0.0975 |
| Magrabia | 0.9300 ±0.0031 | 0.9410 ±0.0010 | 0.9495 ±0.0030 | +0.0110 | +0.0195 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | +0.0374 | [+0.0110, +0.0637] | 6 |
| randconv | +0.0585 | [+0.0195, +0.0975] | 6 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.9517 | 0.9204–0.9646 | **0.011** | 0.051 |
| bigaug | 0.9528 | 0.9307–0.9648 | **0.001** | 0.011 |
| randconv | 0.9579 | 0.9409–0.9684 | **0.000** | 0.000 |

**ERM per-pair target Dice (disc) — the failure map phase 2 starts from:**

| source \ target | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 |
|---|---|---|---|
| BinRushed | 0.912 | 0.780 | 0.869 |
| Magrabia | 0.944 | 0.904 | 0.942 |

## Target-mean Dice (cup), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BinRushed | 0.7468 ±0.0271 | 0.7988 ±0.0089 | 0.8674 ±0.0028 | +0.0520 | +0.1205 |
| Magrabia | 0.8407 ±0.0108 | 0.8239 ±0.0007 | 0.8721 ±0.0065 | -0.0168 | +0.0314 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | +0.0176 | [-0.0168, +0.0520] | 6 |
| randconv | +0.0760 | [+0.0314, +0.1205] | 6 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.8609 | 0.7921–0.9034 | **0.039** | 0.078 |
| bigaug | 0.8384 | 0.7758–0.8860 | **0.004** | 0.026 |
| randconv | 0.8865 | 0.8405–0.9185 | **0.000** | 0.001 |

**ERM per-pair target Dice (cup) — the failure map phase 2 starts from:**

| source \ target | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 |
|---|---|---|---|
| BinRushed | 0.813 | 0.653 | 0.775 |
| Magrabia | 0.846 | 0.813 | 0.863 |
