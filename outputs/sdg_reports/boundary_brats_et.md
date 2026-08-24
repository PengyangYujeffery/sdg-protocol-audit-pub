# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0

HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty (543.1 px); their share is reported separately.


| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |
|---|---|---|---|---|---|
| erm | 13.6 | 25.8 | 1.9 | 10.5 | **0.014** |
| bigaug | 13.3 | 42.7 | 2.8 | 30.8 | **0.075** |
| randconv | 27.1 | 32.7 | 2.7 | 9.6 | **0.014** |
| mixstyle | 11.6 | 24.3 | 1.9 | 10.3 | **0.014** |
| dsu | 14.8 | 29.5 | 2.0 | 12.5 | **0.021** |
| maxstyle | 14.5 | 28.5 | 2.6 | 12.8 | **0.021** |
| ada | 12.2 | 33.2 | 2.8 | 17.7 | **0.034** |
| slaug | 339.4 | 334.8 | 339.4 | 334.8 | **0.986** |

A mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the last column first.

