# Phase-1 SDG re-benchmark — riga (GT=r1)

18 runs · sources BinRushed,Magrabia · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (disc), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BinRushed | 0.8549 ±0.0158 | 0.9133 ±0.0071 | 0.9466 ±0.0010 | +0.0584 | +0.0918 |
| Magrabia | 0.9217 ±0.0038 | 0.9332 ±0.0018 | 0.9432 ±0.0031 | +0.0115 | +0.0216 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | +0.0350 | [+0.0115, +0.0584] | 8 |
| randconv | +0.0567 | [+0.0216, +0.0918] | 8 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.9498 | 0.9130–0.9643 | **0.012** | 0.047 |
| bigaug | 0.9509 | 0.9252–0.9641 | **0.003** | 0.011 |
| randconv | 0.9566 | 0.9374–0.9679 | **0.001** | 0.001 |

**ERM per-pair target Dice (disc) — the failure map phase 2 starts from:**

| source \ target | BinRushed | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 | Magrabia |
|---|---|---|---|---|---|
| BinRushed | — | 0.912 | 0.780 | 0.869 | 0.858 |
| Magrabia | 0.897 | 0.944 | 0.904 | 0.942 | — |

## Target-mean Dice (cup), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BinRushed | 0.7345 ±0.0237 | 0.7827 ±0.0078 | 0.8541 ±0.0026 | +0.0481 | +0.1195 |
| Magrabia | 0.8228 ±0.0106 | 0.8074 ±0.0019 | 0.8562 ±0.0054 | -0.0154 | +0.0334 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | +0.0164 | [-0.0154, +0.0481] | 8 |
| randconv | +0.0764 | [+0.0334, +0.1195] | 8 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.8533 | 0.7650–0.9004 | **0.037** | 0.085 |
| bigaug | 0.8305 | 0.7563–0.8813 | **0.005** | 0.042 |
| randconv | 0.8792 | 0.8213–0.9152 | **0.001** | 0.006 |

**ERM per-pair target Dice (cup) — the failure map phase 2 starts from:**

| source \ target | BinRushed | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 | Magrabia |
|---|---|---|---|---|---|
| BinRushed | — | 0.813 | 0.653 | 0.775 | 0.698 |
| Magrabia | 0.769 | 0.846 | 0.813 | 0.863 | — |
