# Script overview

This repository contains Python scripts for the PCHIP / MGI / FBRM onset-analysis workflow.

## Primary analysis and figure generators

- `xy-pchip.py` creates PCHIP-interpolated curve data from MGI/RGB temperature data.
- `bw-pchip-onsets-t1t2.py` detects MGI/PCHIP onset temperatures and writes MGI result figures.
- `fbrm-onsets.py` detects FBRM Total Counts onset temperatures and writes FBRM result figures. See `fbrm-onsets.md`.
- `make-mgi-fbrm-combo.py` creates combined MGI/FBRM comparison figures from already generated MGI/PCHIP and FBRM outputs.

## Workflow runners

- `run-onset-workflow.py` orchestrates the full PCHIP / MGI / FBRM workflow.
- `run-bw-pchip-workflow.py` runs the MGI/BW PCHIP workflow for a single analysis directory or a discovered batch of analysis directories.

## Gallery generation

- `make-onset-gallery.py` creates static offline HTML review galleries for MGI/PCHIP, FBRM, combo, or combined onset outputs.

The gallery is a review and quality-control output. It does not calculate onsets, redraw analysis figures, modify onset results, or feed back into the computational onset-detection workflow.

### Required directory layout for gallery discovery

The unified gallery generator discovers active workflow outputs from the normal experiment-tree layout. The result directories must be under an experiment directory, not directly at the gallery root.

Required MGI/PCHIP layout:

```text
<gallery-root>/
└── <experiment>/
    └── tl/
        └── roi*/
            └── rgb/
                └── analysis/
                    └── pchip/
                        ├── *.png
                        ├── *.csv
                        ├── *.json
                        └── *.txt
```

Optional FBRM and combo outputs follow the same analysis-root convention:

```text
<gallery-root>/
└── <experiment>/
    └── tl/
        └── roi*/
            └── rgb/
                └── analysis/
                    ├── fbrm_onsets/
                    └── combo/
```

A compact single-analysis directory such as this is valid for running the PCHIP workflow itself:

```text
rgb-tr.csv
rgb-tr-sg.csv
pchip/
```

but it is not sufficient for `make-onset-gallery.py` discovery. For gallery generation, place the analysis directory under an experiment tree such as:

```text
<experiment>/tl/roi1/rgb/analysis/
```

This requirement avoids special-case gallery logic and prevents the gallery builder from accidentally collecting legacy, root-level, or unrelated result directories.

For gallery usage examples, image-only export packages, and screenshot documentation, see `gallery.md`.

## Shared helpers

- `onset_config.py` handles human-editable per-experiment onset-count configuration.
- `onset_view_limits.py` provides display-only axis-limit handling for onset figures.

## Preprocessing and control scripts

- `tr2csv.py` converts temperature-log spreadsheet data to CSV.
- `tr2rgb.py` aligns temperature data with RGB/MGI time-lapse data.
- `fbrm2csv.py` converts FBRM/iC FBRM export data to timestamped FBRM Total Counts CSV.
- `align_fbrm_tr.py` aligns FBRM Total Counts with the experiment temperature axis.
- `tr-bw-sg.py` creates the optional Savitzky-Golay control input.
- `clean-onset-outputs.py` safely archives or removes generated onset-workflow outputs.

## Example helper scripts

- `examples/scripts/make-synthetic-rgb-tr.py` creates the synthetic RGB-TR demonstration input used by the compact example.

## Legacy scripts

`scripts/legacy/` contains older gallery scripts retained for traceability with earlier workflow diagrams and archived analysis workflows. New gallery generation uses `scripts/make-onset-gallery.py`.

