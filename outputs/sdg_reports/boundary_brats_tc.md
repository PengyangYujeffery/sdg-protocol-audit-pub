# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0

HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty (543.1 px); their share is reported separately.


| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |
|---|---|---|---|---|---|
| erm | 20.3 | 27.3 | 3.2 | 7.1 | **0.007** |
| bigaug | 27.2 | 34.6 | 3.7 | 9.5 | **0.007** |
| randconv | 29.2 | 33.2 | 4.7 | 9.3 | **0.007** |
| mixstyle | 17.9 | 25.8 | 3.3 | 10.6 | **0.014** |
| dsu | 28.8 | 34.0 | 4.2 | 8.7 | **0.007** |
| maxstyle | 21.0 | 26.7 | 3.0 | 7.3 | **0.007** |
| ada | 58.1 | 59.7 | 9.8 | 14.3 | **0.007** |
| slaug | 339.4 | 207.5 | 339.4 | 199.7 | **0.568** |

A mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the last column first.

