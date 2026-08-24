# What is the Western → African gap made of? (TC, africa_glioma)

Western reference = the **held-out** source validation cases recorded in each run (144 distinct cases across runs); never the full source domain.


## 1. Do the cohorts differ in case mix?

| covariate | Western held-out (median, IQR) | African (median, IQR) | ratio of medians |
|---|---|---|---|
| ET volume fraction | 0.0081 (0.0039–0.0140) | 0.0121 (0.0045–0.0217) | **1.50×** |
| ET-vs-rim contrast | 1.1539 (0.6078–1.6854) | 0.8020 (0.5388–1.1437) | **0.69×** |

## 2.1 Decomposition, standardised on **ET volume fraction**

| method | Western held-out | African | **total gap** | case-mix | **domain** | support |
|---|---|---|---|---|---|---|
| ada | 0.8232 | 0.6893 | **+0.1339** | -0.0081 | **+0.1420** | 1.00 |
| bigaug | 0.7682 | 0.6313 | **+0.1369** | -0.0001 | **+0.1371** | 1.00 |
| dsu | 0.8067 | 0.6791 | **+0.1276** | -0.0058 | **+0.1334** | 1.00 |
| erm | 0.8064 | 0.6789 | **+0.1275** | -0.0027 | **+0.1302** | 1.00 |
| maxstyle | 0.8501 | 0.7487 | **+0.1014** | -0.0066 | **+0.1080** | 1.00 |
| mixstyle | 0.8121 | 0.6990 | **+0.1131** | -0.0059 | **+0.1190** | 1.00 |
| randconv | 0.7980 | 0.6909 | **+0.1071** | -0.0099 | **+0.1171** | 1.00 |

Bin occupancy (reference / target per quantile bin, and the reference Dice in that bin) — **a bin with few reference cases cannot carry weight**:

| bin | n Western | n African | Western Dice |
|---|---|---|---|
| 0 | 39 | 21 | 0.6745 |
| 1 | 38 | 21 | 0.8658 |
| 2 | 41 | 19 | 0.8946 |
| 3 | 26 | 34 | 0.8713 |

**total +0.1211 = case-mix -0.0056 (-5 %) + domain +0.1267 (105 %)** — stable

- reading: **majority domain — matched on this covariate, African scans are still worse**


## 2.2 Decomposition, standardised on **ET-vs-rim contrast**

| method | Western held-out | African | **total gap** | case-mix | **domain** | support |
|---|---|---|---|---|---|---|
| ada | 0.8232 | 0.6893 | **+0.1339** | +0.0192 | **+0.1146** | 1.00 |
| bigaug | 0.7682 | 0.6313 | **+0.1369** | +0.0247 | **+0.1122** | 1.00 |
| dsu | 0.8067 | 0.6791 | **+0.1276** | +0.0162 | **+0.1115** | 1.00 |
| erm | 0.8064 | 0.6789 | **+0.1275** | +0.0211 | **+0.1064** | 1.00 |
| maxstyle | 0.8501 | 0.7487 | **+0.1014** | +0.0166 | **+0.0849** | 1.00 |
| mixstyle | 0.8121 | 0.6990 | **+0.1131** | +0.0212 | **+0.0919** | 1.00 |
| randconv | 0.7980 | 0.6909 | **+0.1071** | +0.0120 | **+0.0951** | 1.00 |

Bin occupancy (reference / target per quantile bin, and the reference Dice in that bin) — **a bin with few reference cases cannot carry weight**:

| bin | n Western | n African | Western Dice |
|---|---|---|---|
| 0 | 33 | 27 | 0.7587 |
| 1 | 28 | 31 | 0.7812 |
| 2 | 38 | 22 | 0.8354 |
| 3 | 45 | 15 | 0.8863 |

**total +0.1211 = case-mix +0.0187 (15 %) + domain +0.1024 (85 %)** — stable

- reading: **majority domain — matched on this covariate, African scans are still worse**

