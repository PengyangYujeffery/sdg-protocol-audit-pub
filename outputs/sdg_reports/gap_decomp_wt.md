# What is the Western → African gap made of? (WT, africa_glioma)

Western reference = the **held-out** source validation cases recorded in each run (144 distinct cases across runs); never the full source domain.


## 1. Do the cohorts differ in case mix?

| covariate | Western held-out (median, IQR) | African (median, IQR) | ratio of medians |
|---|---|---|---|
| ET volume fraction | 0.0241 (0.0180–0.0344) | 0.0370 (0.0228–0.0479) | **1.53×** |
| ET-vs-rim contrast | 0.3110 (0.1278–0.5639) | 0.3202 (0.1871–0.6241) | **1.03×** |

## 2.1 Decomposition, standardised on **ET volume fraction**

| method | Western held-out | African | **total gap** | case-mix | **domain** | support |
|---|---|---|---|---|---|---|
| ada | 0.9002 | 0.8658 | **+0.0344** | -0.0084 | **+0.0428** | 1.00 |
| bigaug | 0.9015 | 0.8751 | **+0.0265** | -0.0096 | **+0.0360** | 1.00 |
| dsu | 0.9107 | 0.8852 | **+0.0255** | -0.0073 | **+0.0328** | 1.00 |
| erm | 0.9101 | 0.8880 | **+0.0221** | -0.0054 | **+0.0275** | 1.00 |
| maxstyle | 0.9097 | 0.8822 | **+0.0275** | -0.0084 | **+0.0359** | 1.00 |
| mixstyle | 0.9077 | 0.8843 | **+0.0234** | -0.0046 | **+0.0279** | 1.00 |
| randconv | 0.8925 | 0.8713 | **+0.0211** | -0.0055 | **+0.0266** | 1.00 |

Bin occupancy (reference / target per quantile bin, and the reference Dice in that bin) — **a bin with few reference cases cannot carry weight**:

| bin | n Western | n African | Western Dice |
|---|---|---|---|
| 0 | 41 | 19 | 0.8643 |
| 1 | 45 | 14 | 0.9037 |
| 2 | 40 | 20 | 0.9225 |
| 3 | 18 | 42 | 0.9237 |

**total +0.0258 = case-mix -0.0070 (-27 %) + domain +0.0328 (127 %)** — 🔴 **UNSTABLE: a part exceeds the whole, which means the reweighting is extrapolating. Do not use.**


## 2.2 Decomposition, standardised on **ET-vs-rim contrast**

| method | Western held-out | African | **total gap** | case-mix | **domain** | support |
|---|---|---|---|---|---|---|
| ada | 0.9002 | 0.8658 | **+0.0344** | -0.0004 | **+0.0348** | 1.00 |
| bigaug | 0.9015 | 0.8751 | **+0.0265** | -0.0002 | **+0.0267** | 1.00 |
| dsu | 0.9107 | 0.8852 | **+0.0255** | -0.0003 | **+0.0258** | 1.00 |
| erm | 0.9101 | 0.8880 | **+0.0221** | -0.0003 | **+0.0224** | 1.00 |
| maxstyle | 0.9097 | 0.8822 | **+0.0275** | -0.0004 | **+0.0279** | 1.00 |
| mixstyle | 0.9077 | 0.8843 | **+0.0234** | +0.0001 | **+0.0232** | 1.00 |
| randconv | 0.8925 | 0.8713 | **+0.0211** | -0.0001 | **+0.0212** | 1.00 |

Bin occupancy (reference / target per quantile bin, and the reference Dice in that bin) — **a bin with few reference cases cannot carry weight**:

| bin | n Western | n African | Western Dice |
|---|---|---|---|
| 0 | 41 | 19 | 0.9000 |
| 1 | 33 | 26 | 0.8950 |
| 2 | 37 | 23 | 0.8957 |
| 3 | 33 | 27 | 0.9106 |

**total +0.0258 = case-mix -0.0002 (-1 %) + domain +0.0260 (101 %)** — stable

- reading: **majority domain — matched on this covariate, African scans are still worse**

