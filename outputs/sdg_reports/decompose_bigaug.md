# Error decomposition — DG Prostate, method = bigaug

30 transfer pairs. In-domain reference = the same site used as a SOURCE, same method.


| quantity | Dice | share of the in-domain gap |
|---|---|---|
| out-of-domain @ threshold 0.5 | 0.6243 | — |
| + one **tuned threshold per target domain** | 0.6538 | **10.1 %** |
| + a **per-case oracle threshold** (upper bound of any calibration fix) | 0.6598 | **12.2 %** |
| **in-domain** (model trained on that site) | 0.9168 | 100 % |

**Total in-domain gap: 0.2926 Dice.** Threshold alone recovers 0.0296 (10.1 %); even the per-case oracle threshold recovers 0.0356 (12.2 %). The remaining **0.2570 (87.8 %)** needs a different representation, not a different decision rule.


| source | target | @0.5 | best-domain thr | per-case oracle | in-domain |
|---|---|---|---|---|---|
| BIDMC | BMC | 0.5398 | 0.6175 (t=0.05) | 0.6175 | 0.9226 |
| BIDMC | HK | 0.6750 | 0.7094 (t=0.95) | 0.7126 | 0.9114 |
| BIDMC | I2CVB | 0.4130 | 0.5022 (t=0.05) | 0.5025 | 0.9218 |
| BIDMC | RUNMC | 0.4997 | 0.5567 (t=0.05) | 0.5599 | 0.9273 |
| BIDMC | UCL | 0.4496 | 0.5007 (t=0.05) | 0.5007 | 0.9206 |
| BMC | BIDMC | 0.6048 | 0.6126 (t=0.10) | 0.6289 | 0.8973 |
| BMC | HK | 0.7701 | 0.7793 (t=0.90) | 0.7864 | 0.9114 |
| BMC | I2CVB | 0.5433 | 0.5963 (t=0.05) | 0.5987 | 0.9218 |
| BMC | RUNMC | 0.8134 | 0.8317 (t=0.95) | 0.8327 | 0.9273 |
| BMC | UCL | 0.8111 | 0.8112 (t=0.60) | 0.8279 | 0.9206 |
| HK | BIDMC | 0.6260 | 0.6353 (t=0.05) | 0.6426 | 0.8973 |
| HK | BMC | 0.5974 | 0.6364 (t=0.05) | 0.6364 | 0.9226 |
| HK | I2CVB | 0.2419 | 0.2896 (t=0.05) | 0.2896 | 0.9218 |
| HK | RUNMC | 0.6593 | 0.6933 (t=0.05) | 0.6935 | 0.9273 |
| HK | UCL | 0.6270 | 0.6697 (t=0.05) | 0.6697 | 0.9206 |
| I2CVB | BIDMC | 0.1873 | 0.2248 (t=0.05) | 0.2248 | 0.8973 |
| I2CVB | BMC | 0.5721 | 0.6157 (t=0.05) | 0.6194 | 0.9226 |
| I2CVB | HK | 0.5460 | 0.5958 (t=0.05) | 0.5969 | 0.9114 |
| I2CVB | RUNMC | 0.6923 | 0.6924 (t=0.40) | 0.7099 | 0.9273 |
| I2CVB | UCL | 0.5422 | 0.6155 (t=0.05) | 0.6236 | 0.9206 |
| RUNMC | BIDMC | 0.5490 | 0.5727 (t=0.05) | 0.5843 | 0.8973 |
| RUNMC | BMC | 0.8351 | 0.8525 (t=0.05) | 0.8555 | 0.9226 |
| RUNMC | HK | 0.8467 | 0.8469 (t=0.60) | 0.8538 | 0.9114 |
| RUNMC | I2CVB | 0.8066 | 0.8104 (t=0.90) | 0.8274 | 0.9218 |
| RUNMC | UCL | 0.8630 | 0.8633 (t=0.60) | 0.8697 | 0.9206 |
| UCL | BIDMC | 0.2467 | 0.2924 (t=0.05) | 0.2924 | 0.8973 |
| UCL | BMC | 0.8010 | 0.8160 (t=0.05) | 0.8212 | 0.9226 |
| UCL | HK | 0.8287 | 0.8296 (t=0.25) | 0.8379 | 0.9114 |
| UCL | I2CVB | 0.7265 | 0.7271 (t=0.30) | 0.7448 | 0.9218 |
| UCL | RUNMC | 0.8138 | 0.8181 (t=0.10) | 0.8339 | 0.9273 |
