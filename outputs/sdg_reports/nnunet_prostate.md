# nnU-Net v2 on the single-source DG Prostate protocol

2d configuration, fold 0, `nnUNetTrainer_250epochs`, no TTA. Scored with our metric: per-case Dice over gland-bearing slices.


| source \\ target | BIDMC | BMC | HK | I2CVB | RUNMC | UCL | **mean** |
|---|---|---|---|---|---|---|---|
| BIDMC | — | 0.455 | 0.732 | 0.068 | 0.412 | 0.258 | **0.3851** |
| BMC | 0.238 | — | 0.194 | 0.151 | 0.667 | 0.765 | **0.4030** |
| HK | 0.489 | 0.365 | — | 0.516 | 0.766 | 0.431 | **0.5132** |
| I2CVB | 0.004 | 0.036 | 0.069 | — | 0.174 | 0.039 | **0.0643** |
| RUNMC | 0.471 | 0.746 | 0.787 | 0.570 | — | 0.830 | **0.6810** |
| UCL | 0.360 | 0.815 | 0.349 | 0.544 | 0.764 | — | **0.5662** |

**nnU-Net target-mean Dice over all 30 transfer pairs: 0.4355**


## Against our own backbones, same protocol, same 30 pairs

| model | target-mean Dice | training samples seen |
|---|---|---|
| our 2D U-Net, ERM | 0.3951 | 64 k |
| our 2D U-Net, BigAug (best of 7) | 0.6054 | 64 k |
| our ResNet-34 ImageNet, BigAug | see `backbone_prostate.md` | 64 k |
| **nnU-Net v2, 250 epochs** | **0.4355** | **~940 k** |

🔴 The budgets are not equal (~15x) and the comparison must always say so.

