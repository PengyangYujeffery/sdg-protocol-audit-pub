# Boundary metrics on unseen domains — DG Prostate, scratch U-Net, seed 0

HD95 and ASSD in pixels. Degenerate cases take the image-diagonal penalty (543.1 px); their share is reported separately.


| method | median HD95 | mean HD95 | median ASSD | mean ASSD | **degenerate share** |
|---|---|---|---|---|---|
| erm | 39.6 | 52.9 | 13.0 | 17.5 | **0.001** |
| bigaug | 40.2 | 55.4 | 13.3 | 17.5 | **0.000** |
| randconv | 38.8 | 48.0 | 12.8 | 16.3 | **0.001** |
| mixstyle | 39.2 | 53.9 | 13.0 | 19.1 | **0.004** |
| dsu | 39.6 | 52.6 | 13.1 | 17.8 | **0.002** |
| maxstyle | 39.5 | 52.8 | 13.1 | 17.6 | **0.002** |
| ada | 40.5 | 90.2 | 13.8 | 52.4 | **0.049** |
| slaug | 40.3 | 52.1 | 13.5 | 16.9 | **0.000** |

A mean HD95 dominated by the penalty is a **miss rate**, not a distance. Read the last column first.

