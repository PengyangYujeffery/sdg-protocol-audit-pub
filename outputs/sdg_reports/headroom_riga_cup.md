# Headroom for adaptive selection — riga (cup)

8 transfer pairs, 7 methods, seeds averaged within a pair.


## Ceilings

| policy | mean target Dice | Δ vs best fixed |
|---|---|---|
| **best fixed** (randconv) | **0.8524** | — |
| oracle per **source** | 0.8524 | +0.0000 |
| oracle per **pair** (not achievable) | 0.8524 | +0.0000 |
| oracle per **case** (ceiling of per-sample adaptation) | 0.8748 | +0.0226 |

## Which method would the source-level oracle pick?

| source | oracle choice | its mean | best-fixed (randconv) mean |
|---|---|---|---|
| BinRushed | **randconv** | 0.8528 | 0.8528 |
| Magrabia | **randconv** | 0.8519 | 0.8519 |

## Fixed-policy table

| method | mean over pairs |
|---|---|
| randconv | 0.8524 |
| ada | 0.8145 |
| maxstyle | 0.8001 |
| bigaug | 0.7928 |
| dsu | 0.7801 |
| mixstyle | 0.7769 |
| erm | 0.7763 |
