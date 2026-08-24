# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0

HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty (543.1 px); their share is reported separately.


| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |
|---|---|---|---|---|---|
| erm | 71.6 | 77.9 | 11.4 | 24.7 | **0.010** |
| bigaug | 20.1 | 34.2 | 3.3 | 6.3 | **0.000** |
| randconv | 21.4 | 33.9 | 3.5 | 8.6 | **0.003** |
| mixstyle | 68.1 | 72.8 | 10.7 | 22.6 | **0.005** |
| dsu | 63.8 | 73.0 | 10.9 | 22.4 | **0.007** |
| maxstyle | 60.7 | 65.2 | 8.5 | 16.3 | **0.002** |
| ada | 44.0 | 71.0 | 6.4 | 30.4 | **0.036** |
| slaug | 41.2 | 55.1 | 6.5 | 23.8 | **0.024** |

A mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the last column first.

