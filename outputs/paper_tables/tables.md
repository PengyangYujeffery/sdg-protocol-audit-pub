
Study scale, counted from the run records: **4 benchmarks, 8 methods, 3 backbones, 1318 runs** (brats, mms, prostate, riga).

# Paper tables — generated, never hand-copied


## T1. Method effects vs ERM, scratch 2D U-Net

| benchmark | ERM | ΔBigAug | ΔRandConv | ΔMaxStyle | ΔADA | n |
|---|---|---|---|---|---|---|
| DG Prostate (gland) | 0.3948 | +0.2202 | +0.1270 | +0.0750 | +0.0441 | 228 / 6$^{c}$ |
| RIGA+ (disc) | 0.9048 | +0.0240 | +0.0352 | +0.0110 | +0.0211 | 190 / 5$^{c}$ |
| RIGA+ (cup) | 0.7932 | +0.0002 | +0.0498 | +0.0173 | +0.0327 | 190 / 5$^{c}$ |
| BraTS WT | 0.8408 | -0.0119 | -0.0249 | -0.0019 | -0.0240 | 38 / 1$^{p}$ |
| BraTS TC | 0.7037 | -0.0741 | -0.0246 | +0.0419 | +0.0001 | 38 / 1$^{p}$ |
| BraTS ET | 0.6637 | -0.0995 | +0.0283 | +0.0202 | -0.0927 | 38 / 1$^{p}$ |
| M\&Ms (LV cavity) | 0.8169 | -0.0226 | +0.0415 | +0.0146 | +0.0074 | 105 / 5$^{c}$ |
| M\&Ms (LV myocardium) | 0.7470 | -0.0157 | +0.0411 | +0.0169 | +0.0108 | 105 / 5$^{c}$ |
| M\&Ms (RV cavity) | 0.6821 | -0.0361 | +0.0789 | +0.0273 | +0.0331 | 105 / 5$^{c}$ |


## T2. Model-selection leak (DG Prostate)

| selection rule | target Dice | vs honest |
|---|---|---|
| honest (source-val) | 0.5068 | — |
| leaked (target-val) | 0.5445 | +0.0376 |
| no selection (last) | 0.4992 | -0.0077 |


54 runs. Inflation from peeking: **+0.0376 Dice (7.4 %)**.


| method | honest | inflation | relative |
|---|---|---|---|
| ERM | 0.3951 | +0.0225 | 5.7 % |
| BigAug | 0.6054 | +0.0474 | 7.8 % |
| RandConv | 0.5200 | +0.0430 | 8.3 % |


## T3. Backbone arm (DG Prostate)

Method-effect spread: scratch [0.0115, 0.2202] -> pretrained [0.0334, 0.1315].

| method | Δ scratch U-Net | Δ ImageNet ResNet-34 |
|---|---|---|
| BigAug | +0.2202 | +0.1315 |
| RandConv | +0.1270 | +0.1286 |
| MixStyle | +0.0142 | +0.0525 |
| DSU | +0.0115 | +0.0334 |
| MaxStyle-core | +0.0750 | +0.0947 |
| ADA | +0.0441 | +0.0867 |
| SLAug (aug.) | +0.0585 | — |
| *ERM absolute* | 0.3948 | 0.5685 |


## T4. Slice policy — the DG gain depends on a convention nobody states

BigAug's measured DG gain moves 21 % between the two policies (+0.2202 under `fg`, +0.1740 under `all`), with the methods and the data held fixed.

| slice policy | ERM | ΔBigAug | ΔRandConv | n |
|---|---|---|---|---|
| fg | 0.3948 | +0.2202 | +0.1270 | 228 runs / 6 src |
| all | 0.3238 | +0.1740 | +0.1063 | 54 runs / 6 src |


## T5. The ranking depends on the metric (boundary metrics, from checkpoints)

| benchmark | best (med. HD95) | second | gap px | worst degen. |
|---|---|---|---|---|
| DG Prostate | BigAug 20.1 | RandConv 21.4 | 1.3 | ADA 0.036 |
| RIGA+ cup | RandConv 38.8 | MixStyle 39.2 | 0.4 | ADA 0.049 |
| RIGA+ disc | RandConv 1.0 | SLAug (aug.) 1.0 | 0.0 | ADA 0.049 |
| BraTS WT | ERM 9.6 | MaxStyle-core 12.7 | 3.1 | ADA 0.007 |
| BraTS TC | MixStyle 17.9 | ERM 20.3 | 2.4 | SLAug (aug.) 0.568 |
| BraTS ET | MixStyle 11.6 | ADA 12.2 | 0.6 | SLAug (aug.) 0.986 |

Parsed from the released boundary reports; Dice ranks the same methods differently (see T1).



## T6. Oracle ceilings for adaptive selection

| benchmark | best fixed | Δ oracle per source | Δ oracle per pair | Δ oracle per case |
|---|---|---|---|---|
| DG Prostate | BigAug 0.6150 | +0.0053 | +0.0513 | +0.0609 |
| RIGA+ cup | RandConv 0.8430 | +0.0016 | +0.0035 | +0.0319 |
| BraTS ET | RandConv 0.6920 | +0.0000 | +0.0044 | +0.0589 |

The per-source oracle is the ceiling of any method that adapts to the training domain it is handed; the per-pair and per-case oracles require target knowledge and are reported to bound the question rather than as achievable targets.



## T7. Annotation convention (RIGA+ optic cup)

| GT convention | ERM cup Dice | ΔBigAug | ΔRandConv | n / precision |
|---|---|---|---|---|
| rater 1 | 0.7932 | +0.0002 | +0.0498 | 75 runs, fp32 deterministic |
| majority vote | 0.8108 | -0.0070 | +0.0707 | 18 runs, fp16 AMP (phase 1) |

Methods restricted to those run under both conventions (bigaug, erm, randconv). The two arms differ in precision and are labelled, never pooled.



## T7b. RIGA+ is scored past its own annotation noise floor

| structure | best method (unseen domains) | inter-rater Dice | difference |
|---|---|---|---|
| optic cup | RandConv 0.8430 | 0.8003 | +0.0427 |
| optic disc | RandConv 0.9400 | 0.9556 | -0.0156 |

Inter-rater Dice is averaged over all 15 rater pairs and all five domains (parsed from `protocol_decomp.md`). A method scoring above it is being ranked inside the annotation noise.



## T8. Strong-baseline control: nnU-Net v2

nnU-Net v2 (`nnUNetTrainer_250epochs`, 2d, no TTA) over 30 transfer pairs: **0.4355** target-mean Dice, scored with our per-case metric, against our U-Net+BigAug. The budget differs by roughly 15x and that must travel with the number.



## T8b. In-domain reference: all source cases vs held-out only

| method | all source cases | held-out only | inflation | n |
|---|---|---|---|---|
| ERM | 0.9353 | 0.8535 | +0.0818 | 30 runs |
| BigAug | 0.9177 | 0.8825 | +0.0352 | 30 runs |

Scoring the source domain over **all** its cases includes the training cases and inflates the in-domain reference, which in turn inflates the apparent generalization gap. Held-out cases were used for model selection, so they are held out from fitting but not from selection.



## T9. The protocol knobs, next to the largest method effect

| protocol choice | effect on the reported result | source |
|---|---|---|
| model-selection leak (peeking at the target) | +0.0376 | T2, 54 runs |
| slice policy fg vs all | +0.0462 | T4 |
| backbone: ImageNet vs scratch (ERM) | +0.1737 | T3 |
| annotation convention (cup, BigAug effect) | +0.0072 | T7 |
| evaluation metric (Dice vs HD95 ranking) | rank change | T5 |
| *gap between the top two methods (BigAug vs RandConv)* | 0.0932 | T1 |
| *ERM to best method, for scale* | +0.2202 | T1 |

The quantity a protocol choice must be compared against is the **gap between the methods being compared**, since that is what a paper claims. The ERM-to-best-method distance is listed only for scale.



## T10. Representation preservation (secondary result)

| arm | target Dice | Δ vs full fine-tuning | 95 % CI |
|---|---|---|---|
| freeze 0 (full fine-tuning) | 0.5293 | — |  |
| freeze 1 | 0.5495 | +0.0202 | [-0.0246, +0.0576] 12 runs / 4 src |
| freeze 2 | 0.4922 | -0.0371 | [-0.1238, +0.0841] 12 runs / 4 src |

Pretrained ResNet-34, ERM, prostate, matched cell-for-cell against the unfrozen runs already in `sdg_backbone` (no retraining). Freeze 1 = stem, freeze 2 = stem + layer1.



## T-fail. How the models fail, per method (DG Prostate, unseen domains)

| method | cases below Dice 0.10 (%) | Dice exactly 0 (%) | target cases |
|---|---|---|---|
| ERM | 24.4 | 7.2 | 2900 |
| BigAug | 4.7 | 0.4 | 2900 |
| RandConv | 11.0 | 1.6 | 2900 |
| MixStyle | 23.9 | 7.3 | 2900 |
| DSU | 23.4 | 7.3 | 2900 |
| MaxStyle-core | 15.7 | 4.1 | 2900 |
| ADA | 21.0 | 7.7 | 2900 |
| SLAug (aug.) | 23.9 | 5.6 | 1740 |

The two are far apart, and that difference is the finding: the model is not silent. Note the third quantity this is NOT: the boundary reports give the EMPTY-prediction share separately, and it is far smaller, so most zero-Dice cases are predictions that missed entirely rather than absent ones. From the fp32 deterministic head-to-head runs.



## T-gap. What each component of the gap could buy, if chosen perfectly

| benchmark | best fixed | method, per source | threshold, per domain | both, per domain | both, per case | overlap |
|---|---|---|---|---|---|---|
| DG Prostate | BigAug | +0.0053 | +0.0408 | +0.0821 | +0.1030 | +0.0138 |
| RIGA+ cup | RandConv | +0.0014 | +0.0099 | +0.0122 | +0.0502 | +0.0096 |
| BraTS WT | DSU | +0.0000 | +0.0005 | +0.0011 | +0.0306 | +0.0021 |
| BraTS TC | MaxStyle-core | +0.0000 | +0.0054 | +0.0054 | +0.0587 | -0.0012 |
| BraTS ET | RandConv | +0.0000 | +0.0051 | +0.0094 | +0.0934 | +0.0085 |
| M\&Ms LV | RandConv | +0.0020 | +0.0084 | +0.0148 | +0.0406 | +0.0040 |
| M\&Ms myo | RandConv | +0.0008 | +0.0050 | +0.0084 | +0.0355 | +0.0034 |
| M\&Ms RV | RandConv | +0.0033 | +0.0114 | +0.0154 | +0.0498 | +0.0057 |

The only column a deployed method can reach is **method, per source** — it is the one that needs no knowledge of the target. The rest require the target domain or the answer for each case, and are reported to bound the question. **Overlap** is (threshold + method) minus the joint ceiling: it is what the two components would double-count if their ceilings were added, and it is why they are not.



## T-africa. Both directions between the Western and Sub-Saharan African cohorts

| region | method | Western in-domain | Western to African | African in-domain | African to Western | |asymmetry| |
|---|---|---|---|---|---|---|
| WT | BigAug | 0.9041 | 0.8744 | 0.9199 | 0.8730 | +0.0014 |
| WT | ERM | 0.9144 | 0.8861 | 0.9273 | 0.8840 | +0.0021 |
| WT | RandConv | 0.8928 | 0.8673 | 0.9082 | 0.8458 | +0.0215 |
| TC | BigAug | 0.7501 | 0.6145 | 0.8196 | 0.7223 | +0.1078 |
| TC | ERM | 0.8279 | 0.7038 | 0.8622 | 0.7808 | +0.0770 |
| TC | RandConv | 0.7880 | 0.6859 | 0.7666 | 0.6719 | +0.0140 |
| ET | BigAug | 0.6862 | 0.5287 | 0.7148 | 0.6310 | +0.1023 |
| ET | ERM | 0.7206 | 0.6429 | 0.7104 | 0.6322 | +0.0106 |
| ET | RandConv | 0.7647 | 0.6972 | 0.7694 | 0.6903 | +0.0069 |

The two directions are not mirror images. Attribution is **acquisition**, not anatomy: the source literature ascribes the African cohort's difficulty to field strength, motion, resolution and non-standardised protocols.

