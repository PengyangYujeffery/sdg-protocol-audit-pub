# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0

HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty (543.1 px); their share is reported separately.


| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |
|---|---|---|---|---|---|
| erm | 9.6 | 17.8 | 1.4 | 2.6 | **0.000** |
| bigaug | 13.4 | 19.5 | 1.8 | 3.1 | **0.000** |
| randconv | 18.7 | 22.7 | 2.1 | 3.8 | **0.000** |
| mixstyle | 12.9 | 17.7 | 1.6 | 2.8 | **0.000** |
| dsu | 12.8 | 18.0 | 1.6 | 2.9 | **0.000** |
| maxstyle | 12.7 | 18.2 | 1.6 | 3.0 | **0.000** |
| ada | 24.5 | 29.3 | 3.5 | 8.6 | **0.007** |
| slaug | 27.7 | 32.0 | 5.0 | 10.4 | **0.007** |

A mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the last column first.

