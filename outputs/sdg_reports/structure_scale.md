# Does augmentation harm scale with the target structure? — 2026-08-12

48 source images per cell, 12 transform draws per family, no training.


| cell | size (area frac) | contrast (Fisher) | fragments | BigAug degr. | RandConv degr. | **BigAug Δ** | **RandConv Δ** |
|---|---|---|---|---|---|---|---|
| prostate / gland | 0.0342 | 0.612 | 1.0 | +0.035 | +0.087 | **+0.2104** | **+0.1250** |
| riga / disc | 0.0678 | 1.927 | 1.0 | +0.030 | -0.009 | **+0.0436** | **+0.0693** |
| riga / cup | 0.0138 | 0.962 | 1.0 | +0.020 | +0.056 | **+0.0167** | **+0.0790** |
| brats / wt | 0.0262 | 2.474 | 2.4 | +0.063 | +0.079 | **-0.0129** | **-0.0167** |
| brats / tc | 0.0100 | 1.631 | 1.0 | +0.167 | +0.208 | **-0.0476** | **+0.0120** |
| brats / et | 0.0060 | 1.620 | 1.4 | +0.207 | +0.293 | **-0.1352** | **+0.0569** |

## Descriptive correlations (n = 6 cells — NOT inference)

| predictor | r with BigAug Δ | r with RandConv Δ |
|---|---|---|
| structure size | +0.516 | +0.263 |
| contrast | -0.559 | -0.856 |
| log fragments | -0.338 | -0.700 |
| contrast degradation by that family | -0.747 | -0.223 |

## Pre-registered test

In each cell, the family that degrades the structure's contrast **more** should be the family with the **lower** Dice effect. Cells where that holds: **3 of 6**.

🔴 Bar was 5 of 6. **NOT SUPPORTED — the structure-scale mechanism is rejected, as EFDR was**

