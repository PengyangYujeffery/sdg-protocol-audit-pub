# Error decomposition — DG Prostate, method = erm

30 transfer pairs. In-domain reference = the same site used as a SOURCE, same method.


| quantity | Dice | share of the in-domain gap |
|---|---|---|
| out-of-domain @ threshold 0.5 | 0.3936 | — |
| + one **tuned threshold per target domain** | 0.4117 | **3.4 %** |
| + a **per-case oracle threshold** (upper bound of any calibration fix) | 0.4138 | **3.8 %** |
| **in-domain** (model trained on that site) | 0.9326 | 100 % |

**Total in-domain gap: 0.5390 Dice.** Threshold alone recovers 0.0181 (3.4 %); even the per-case oracle threshold recovers 0.0203 (3.8 %). The remaining **0.5188 (96.2 %)** needs a different representation, not a different decision rule.


| source | target | @0.5 | best-domain thr | per-case oracle | in-domain |
|---|---|---|---|---|---|
| BIDMC | BMC | 0.1217 | 0.1505 (t=0.05) | 0.1505 | 0.9348 |
| BIDMC | HK | 0.8069 | 0.8072 (t=0.30) | 0.8136 | 0.9445 |
| BIDMC | I2CVB | 0.2654 | 0.3057 (t=0.05) | 0.3075 | 0.9258 |
| BIDMC | RUNMC | 0.2508 | 0.2878 (t=0.05) | 0.2878 | 0.9462 |
| BIDMC | UCL | 0.0930 | 0.1181 (t=0.05) | 0.1181 | 0.9361 |
| BMC | BIDMC | 0.2540 | 0.2675 (t=0.05) | 0.2675 | 0.9081 |
| BMC | HK | 0.4826 | 0.4831 (t=0.30) | 0.4971 | 0.9445 |
| BMC | I2CVB | 0.1776 | 0.2215 (t=0.05) | 0.2215 | 0.9258 |
| BMC | RUNMC | 0.5989 | 0.6041 (t=0.95) | 0.6136 | 0.9462 |
| BMC | UCL | 0.6291 | 0.6358 (t=0.95) | 0.6410 | 0.9361 |
| HK | BIDMC | 0.5414 | 0.5415 (t=0.90) | 0.5490 | 0.9081 |
| HK | BMC | 0.1684 | 0.1787 (t=0.05) | 0.1787 | 0.9348 |
| HK | I2CVB | 0.3018 | 0.3151 (t=0.05) | 0.3153 | 0.9258 |
| HK | RUNMC | 0.6248 | 0.6379 (t=0.05) | 0.6379 | 0.9462 |
| HK | UCL | 0.1241 | 0.1322 (t=0.05) | 0.1322 | 0.9361 |
| I2CVB | BIDMC | 0.2445 | 0.2598 (t=0.05) | 0.2642 | 0.9081 |
| I2CVB | BMC | 0.1101 | 0.1261 (t=0.05) | 0.1264 | 0.9348 |
| I2CVB | HK | 0.5263 | 0.5447 (t=0.05) | 0.5482 | 0.9445 |
| I2CVB | RUNMC | 0.3142 | 0.3315 (t=0.05) | 0.3362 | 0.9462 |
| I2CVB | UCL | 0.0607 | 0.0645 (t=0.05) | 0.0645 | 0.9361 |
| RUNMC | BIDMC | 0.3933 | 0.4174 (t=0.05) | 0.4174 | 0.9081 |
| RUNMC | BMC | 0.4324 | 0.4730 (t=0.05) | 0.4730 | 0.9348 |
| RUNMC | HK | 0.7945 | 0.8153 (t=0.05) | 0.8154 | 0.9445 |
| RUNMC | I2CVB | 0.3981 | 0.4343 (t=0.05) | 0.4354 | 0.9258 |
| RUNMC | UCL | 0.5793 | 0.6117 (t=0.05) | 0.6117 | 0.9361 |
| UCL | BIDMC | 0.4076 | 0.4239 (t=0.05) | 0.4240 | 0.9081 |
| UCL | BMC | 0.7503 | 0.7630 (t=0.05) | 0.7630 | 0.9348 |
| UCL | HK | 0.5592 | 0.5822 (t=0.05) | 0.5830 | 0.9445 |
| UCL | I2CVB | 0.0664 | 0.0756 (t=0.05) | 0.0756 | 0.9258 |
| UCL | RUNMC | 0.7292 | 0.7412 (t=0.05) | 0.7447 | 0.9462 |
