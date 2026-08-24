# Headroom for adaptive selection — prostate (gland)

30 transfer pairs, 7 methods, seeds averaged within a pair.


## Ceilings

| policy | mean target Dice | Δ vs best fixed |
|---|---|---|
| **best fixed** (bigaug) | **0.6054** | — |
| oracle per **source** | 0.6139 | +0.0084 |
| oracle per **pair** (not achievable) | 0.6523 | +0.0468 |
| oracle per **case** (ceiling of per-sample adaptation) | 0.6729 | +0.0500 |

## Which method would the source-level oracle pick?

| source | oracle choice | its mean | best-fixed (bigaug) mean |
|---|---|---|---|
| BIDMC | **bigaug** | 0.4763 | 0.4763 |
| BMC | **randconv** | 0.6964 | 0.6457 |
| HK | **bigaug** | 0.6002 | 0.6002 |
| I2CVB | **bigaug** | 0.4677 | 0.4677 |
| RUNMC | **bigaug** | 0.7635 | 0.7635 |
| UCL | **bigaug** | 0.6791 | 0.6791 |

## Fixed-policy table

| method | mean over pairs |
|---|---|
| bigaug | 0.6054 |
| randconv | 0.5200 |
| maxstyle | 0.4606 |
| ada | 0.4459 |
| dsu | 0.4099 |
| mixstyle | 0.4039 |
| erm | 0.3951 |
