# Phase-1 SDG re-benchmark — riga (GT=majority)

18 runs · sources BinRushed,Magrabia · seeds 0,1,2 · 8,000 iters · strict source-val selection


## Target-mean Dice (disc), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BinRushed | 0.8698 ±0.0382 | 0.9291 ±0.0055 | 0.9622 ±0.0010 | +0.0593 | +0.0923 |
| Magrabia | 0.9382 ±0.0038 | 0.9441 ±0.0065 | 0.9579 ±0.0043 | +0.0059 | +0.0196 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | +0.0326 | [+0.0059, +0.0593] | 6 |
| randconv | +0.0560 | [+0.0196, +0.0923] | 6 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.9654 | 0.9363–0.9755 | **0.014** | 0.050 |
| bigaug | 0.9617 | 0.9398–0.9725 | **0.000** | 0.012 |
| randconv | 0.9677 | 0.9539–0.9765 | **0.000** | 0.000 |

**ERM per-pair target Dice (disc) — the failure map phase 2 starts from:**

| source \ target | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 |
|---|---|---|---|
| BinRushed | 0.928 | 0.796 | 0.885 |
| Magrabia | 0.953 | 0.909 | 0.953 |

## Target-mean Dice (cup), averaged over the unseen domains

| source | erm | bigaug | randconv | ΔBigAug | ΔRandConv |
|---|---|---|---|---|---|
| BinRushed | 0.7854 ±0.0340 | 0.8137 ±0.0116 | 0.8973 ±0.0026 | +0.0283 | +0.1119 |
| Magrabia | 0.8606 ±0.0047 | 0.8257 ±0.0043 | 0.8982 ±0.0016 | -0.0349 | +0.0376 |

**Δ vs ERM per transfer pair (seeds averaged within a pair; 95 %% CI bootstrapped over source domains):**

| method | mean Δ Dice | 95% CI | pairs |
|---|---|---|---|
| bigaug | -0.0033 | [-0.0349, +0.0283] | 6 |
| randconv | +0.0747 | [+0.0376, +0.1119] | 6 |

**Distribution of per-case Dice on unseen domains** — a mean cannot tell a method that rescues failures from one that polishes mid-range cases:

| method | median | IQR | frac < 0.10 (total failure) | frac < 0.50 |
|---|---|---|---|---|
| erm | 0.8878 | 0.8238–0.9217 | **0.034** | 0.062 |
| bigaug | 0.8528 | 0.7822–0.8935 | **0.003** | 0.019 |
| randconv | 0.9104 | 0.8754–0.9319 | **0.000** | 0.000 |

**ERM per-pair target Dice (cup) — the failure map phase 2 starts from:**

| source \ target | MESSIDOR_Base1 | MESSIDOR_Base2 | MESSIDOR_Base3 |
|---|---|---|---|
| BinRushed | 0.844 | 0.697 | 0.816 |
| Magrabia | 0.877 | 0.836 | 0.869 |
