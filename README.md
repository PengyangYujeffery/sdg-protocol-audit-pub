# sdg-protocol-audit

Testbed and run records for a protocol audit of single-source domain generalization (SDG) in medical
image segmentation.

**1,318 runs** across **4 benchmarks** (DG Prostate, RIGA+, BraTS, M&Ms), **8 methods** and
**3 backbones**, all at fp32 with deterministic kernels. Every table, macro and figure in the paper
is generated from the run records in this repository — none is transcribed by hand.

## Reproduce the paper's numbers

No arguments, no configuration, no data download:

```bash
python code/sdg/make_tables.py     # -> outputs/paper_tables/{numbers.tex,tables.tex,tables.md}
python code/sdg/make_fig0.py       # -> outputs/paper_figures/
python code/sdg/make_figures.py
```

Needs Python 3.8+, `numpy`, and `matplotlib` for the figures. `make_tables.py` prints a warning for
any arm whose run count is short of its target and refuses to endorse the tables; a clean run reports
`0 warnings`.

## What is here

```
code/sdg/     the testbed: training, data pipeline, the eight method re-implementations,
              the analysis scripts, and the table/figure generators
outputs/      1,479 run records (one JSON per run) and the analysis reports they are read from
```

Each run JSON holds the full configuration, the per-domain and per-case Dice, the training history
and the environment (torch and cuDNN versions).

The two counts differ on purpose. **1,318** is the fp32 deterministic testbed the paper reports and
is what `\nTotalRuns` resolves to. The extra records are the earlier mixed-precision arm (used only
where the paper labels it as such), the repeat runs behind the bit-reproducibility check, and the
BraTS-Africa reverse-direction arm — all shipped so the record is complete, none of them pooled into
a headline number.

## What is *not* here, and why

**The images.** All four corpora are third-party and are obtained from their original distributors:
DG Prostate (NCI-ISBI 2013, I2CVB, PROMISE12, in the six-site split of Liu et al.), RIGA+, BraTS and
BraTS-Africa, and M&Ms. Several require registration and a data-use agreement, so none is
redistributed here.

The consequence is worth stating plainly:

* **regenerating every number, table and figure** needs only this repository;
* **retraining** additionally needs the preprocessed 2D corpora, which are derived from the datasets
  above and are not ours to redistribute. `code/sdg/data.py` reads them from `SDG_SCRATCH`.

## Paths

Two environment variables, both optional:

| variable | default | holds |
|---|---|---|
| `SDG_OUTPUTS` | `./outputs` | run records and reports |
| `SDG_SCRATCH` | `code/sdg/scratch` | preprocessed data and checkpoints |

## Reproducibility

Runs are bit-reproducible **within an arm** — verified by an identical checksum on a repeat run. The
cardiac (M&Ms) arm was executed in full on a second cluster with a different CUDA build, so each arm
is kept whole on the machine that produced it and no comparison in the paper spans the two.

## Licence

| what | licence |
|---|---|
| the software in `code/` | **MIT** |
| the run records and reports in `outputs/` | **CC BY 4.0** |
| the underlying imaging datasets | not ours, not redistributed, governed by their own terms |

## Citation

The accompanying paper is under review; citation details will be added here on acceptance.

## A one-line check that you reproduced it

`outputs/paper_tables/` is committed. After running the three commands above, `git status` should
report no changes — the regenerated tables are byte-identical to the ones the paper was built from.
