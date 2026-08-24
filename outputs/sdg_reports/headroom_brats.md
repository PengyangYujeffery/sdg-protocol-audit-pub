# Headroom for adaptive selection — brats (wt)

2 transfer pairs, 7 methods, seeds averaged within a pair.


## Ceilings

| policy | mean target Dice | Δ vs best fixed |
|---|---|---|
| **best fixed** (erm) | **0.8428** | — |
| oracle per **source** | 0.8428 | +0.0000 |
| oracle per **pair** (not achievable) | 0.8440 | +0.0012 |
| oracle per **case** (ceiling of per-sample adaptation) | 0.8788 | +0.0224 |

## Which method would the source-level oracle pick?

| source | oracle choice | its mean | best-fixed (erm) mean |
|---|---|---|---|
| gli2023 | **erm** | 0.8428 | 0.8428 |

## Fixed-policy table

| method | mean over pairs |
|---|---|
| erm | 0.8428 |
| dsu | 0.8426 |
| mixstyle | 0.8421 |
| maxstyle | 0.8384 |
| bigaug | 0.8298 |
| ada | 0.8219 |
| randconv | 0.8202 |
