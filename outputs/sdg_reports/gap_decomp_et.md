# What is the Western → African gap made of? (ET, africa_glioma)

Western reference = the **held-out** source validation cases recorded in each run (144 distinct cases across runs); never the full source domain.


## 1. Do the cohorts differ in case mix?

| covariate | Western held-out (median, IQR) | African (median, IQR) | ratio of medians |
|---|---|---|---|
| ET volume fraction | 0.0050 (0.0024–0.0086) | 0.0072 (0.0035–0.0134) | **1.44×** |
| ET-vs-rim contrast | 2.0748 (1.6044–2.4937) | 1.2006 (0.9676–1.5988) | **0.58×** |

## 2.1 Decomposition, standardised on **ET volume fraction**

| method | Western held-out | African | **total gap** | case-mix | **domain** | support |
|---|---|---|---|---|---|---|
| ada | 0.6792 | 0.5727 | **+0.1065** | -0.0267 | **+0.1332** | 1.00 |
| bigaug | 0.6792 | 0.5077 | **+0.1715** | -0.0313 | **+0.2028** | 1.00 |
| dsu | 0.7382 | 0.6613 | **+0.0769** | -0.0291 | **+0.1061** | 1.00 |
| erm | 0.7260 | 0.6429 | **+0.0832** | -0.0328 | **+0.1159** | 1.00 |
| mixstyle | 0.7229 | 0.6410 | **+0.0818** | -0.0291 | **+0.1109** | 1.00 |
| randconv | 0.7890 | 0.6998 | **+0.0892** | -0.0224 | **+0.1116** | 1.00 |

Bin occupancy (reference / target per quantile bin, and the reference Dice in that bin) — **a bin with few reference cases cannot carry weight**:

| bin | n Western | n African | Western Dice |
|---|---|---|---|
| 0 | 41 | 18 | 0.5229 |
| 1 | 39 | 20 | 0.6749 |
| 2 | 39 | 20 | 0.7997 |
| 3 | 23 | 37 | 0.7609 |

**total +0.1015 = case-mix -0.0286 (-28 %) + domain +0.1301 (128 %)** — 🔴 **UNSTABLE: a part exceeds the whole, which means the reweighting is extrapolating. Do not use.**


## 2.2 Decomposition, standardised on **ET-vs-rim contrast**

| method | Western held-out | African | **total gap** | case-mix | **domain** | support |
|---|---|---|---|---|---|---|
| ada | 0.6792 | 0.5727 | **+0.1065** | +0.2112 | **-0.1046** | 1.00 |
| bigaug | 0.6792 | 0.5077 | **+0.1715** | +0.2318 | **-0.0602** | 1.00 |
| dsu | 0.7382 | 0.6613 | **+0.0769** | +0.2161 | **-0.1392** | 1.00 |
| erm | 0.7260 | 0.6429 | **+0.0832** | +0.2175 | **-0.1343** | 1.00 |
| mixstyle | 0.7229 | 0.6410 | **+0.0818** | +0.2119 | **-0.1301** | 1.00 |
| randconv | 0.7890 | 0.6998 | **+0.0892** | +0.1735 | **-0.0843** | 1.00 |

Bin occupancy (reference / target per quantile bin, and the reference Dice in that bin) — **a bin with few reference cases cannot carry weight**:

| bin | n Western | n African | Western Dice |
|---|---|---|---|
| 0 | 14 | 45 | 0.2862 |
| 1 | 29 | 30 | 0.5541 |
| 2 | 44 | 15 | 0.7293 |
| 3 | 55 | 5 | 0.8052 |

**total +0.1015 = case-mix +0.2103 (207 %) + domain -0.1088 (-107 %)** — 🔴 **UNSTABLE: a part exceeds the whole, which means the reweighting is extrapolating. Do not use.**

