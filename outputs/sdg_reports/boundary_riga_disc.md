# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0

HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty (543.1 px); their share is reported separately.


| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |
|---|---|---|---|---|---|
| erm | 2.0 | 17.1 | 0.2 | 3.2 | **0.001** |
| bigaug | 1.4 | 17.6 | 0.2 | 2.5 | **0.000** |
| randconv | 1.0 | 10.7 | 0.1 | 2.1 | **0.001** |
| mixstyle | 1.4 | 18.6 | 0.1 | 5.1 | **0.004** |
| dsu | 2.0 | 17.0 | 0.2 | 3.8 | **0.002** |
| maxstyle | 1.4 | 16.7 | 0.1 | 3.3 | **0.002** |
| ada | 5.0 | 66.4 | 0.8 | 42.1 | **0.049** |
| slaug | 1.0 | 13.8 | 0.1 | 1.8 | **0.000** |

A mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the last column first.

