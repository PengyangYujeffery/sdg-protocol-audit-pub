# E1 — the model-selection leak, measured (DG Prostate)

54 runs. Each run is scored under three selection rules applied to its own trajectory, so the only difference between the columns is the rule.


| rule | target Dice | 95 %% CI |
|---|---|---|
| **honest** — best source-val | **0.5068** | [0.4632, 0.5495] |
| **leaked** — best target | **0.5445** | [0.5016, 0.5859] |
| no selection — last iter | **0.4992** | [0.4545, 0.5428] |

**Inflation from peeking at the target: +0.0376 Dice [+0.0287, +0.0475].** For scale, an honest selection rule is worth +0.0077 [-0.0013, +0.0176] over no selection at all.

Relative: peeking adds **7.4 %** on top of the honestly-selected score.


## By method

| method | honest | leaked | **inflation** | inflation as %% of honest |
|---|---|---|---|---|
| bigaug | 0.6054 | 0.6528 | **+0.0474** | 7.8 % |
| erm | 0.3951 | 0.4176 | **+0.0225** | 5.7 % |
| randconv | 0.5200 | 0.5631 | **+0.0430** | 8.3 % |

## Where the two rules stop

Source-val picks iteration 6120 on average; the target peaks at 5287. They agree exactly in 13 % of runs.


🔴 This does not discover that peeking inflates — DomainBed said so in 2021. It supplies the magnitude for medical single-source segmentation, which is what a reader needs in order to judge published numbers that do not state their selection rule.

