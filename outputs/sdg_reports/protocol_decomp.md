# D1 ceiling probe — annotation protocol vs image appearance on RIGA+


## A. Label-space ceiling: agreement between raters on the SAME image

The Dice between two raters bounds what any model trained on one can score against the other. Averaged over every image in each domain and all 15 rater pairs.

| domain | disc rater-pair Dice | cup rater-pair Dice | worst cup pair |
|---|---|---|---|
| BinRushed | 0.9568 | **0.7557** | 0.6743 (raters 4-6) |
| Magrabia | 0.9558 | **0.7984** | 0.7085 (raters 4-6) |
| MESSIDOR_Base1 | 0.9568 | **0.8132** | 0.7588 (raters 3-4) |
| MESSIDOR_Base2 | 0.9529 | **0.8132** | 0.7376 (raters 4-6) |
| MESSIDOR_Base3 | 0.9558 | **0.8210** | 0.7379 (raters 4-5) |

**Mean cup ceiling 0.8003, disc ceiling 0.9556.**


## B. Decomposing a trained model's cross-site gap

Predictions are fixed; only the ground truth changes, so the protocol arm costs nothing.

| source | method | in-domain (own protocol) | same site, OTHER protocol | other site, same protocol | other site, other protocol |
|---|---|---|---|---|---|
| BinRushed | bigaug | 0.8512 | 0.8627 | 0.7788 | 0.8005 |
| BinRushed | erm | 0.9194 | 0.9058 | 0.7307 | 0.7741 |
| BinRushed | randconv | 0.8943 | 0.9022 | 0.8529 | 0.8813 |
| Magrabia | bigaug | 0.8787 | 0.8777 | 0.8067 | 0.8070 |
| Magrabia | erm | 0.9425 | 0.9592 | 0.8220 | 0.8475 |
| Magrabia | randconv | 0.9076 | 0.9275 | 0.8519 | 0.8817 |

**Means over 6 (source, method) configurations, optic cup:**

| component | Dice change |
|---|---|
| **protocol only** (same images, rater-1 → majority) | **0.0069** |
| **appearance only** (unseen sites, protocol fixed) | **0.0918** |
| both together | 0.0669 |

**Protocol accounts for 7.0 % of the two components combined.**

🔴 Pre-registered bar was 25 %. **FAILS — D1 is abandoned; the gap is appearance-dominated**

