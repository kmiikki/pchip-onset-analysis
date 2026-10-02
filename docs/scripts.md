# Script overview

This repository contains Python scripts for the PCHIP / MGI / FBRM onset-analysis workflow.

## Primary analysis and figure generators

See [MGI/PCHIP onset analysis](mgi-onsets.md) for the full detector method and parameter reference.

- `xy-pchip.py` creates PCHIP-interpolated curve data from MGI/RGB temperature data.
- `bw-pchip-onsets-t1t2.py` detects MGI/PCHIP onset temperatures and writes MGI result figures.
- `fbrm-onsets.py` detects FBRM Total Counts onset temperatures and writes FBRM result figures. See [FBRM method](fbrm-onsets.md).
- `make-mgi-fbrm-combo.py` creates combined figures/data exports using saved MGI/PCHIP and onset results plus aligned source measurements. It performs its own existing FBRM curve preparation/smoothing; it is not the saved-results-only publication renderer.

## Workflow runners

- `run-onset-workflow.py` orchestrates PCHIP, MGI, FBRM, combo and gallery stages. `--project-root` selects repository code under `scripts/`; `--data-root` selects inputs/outputs (defaults to the project root).
- `run-bw-pchip-workflow.py` runs the MGI/BW PCHIP workflow for a single analysis directory or a discovered batch of analysis directories. Use explicit `--xy-pchip` and `--onset-script` paths to this clone; bare defaults may resolve external PATH tools.

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

For gallery usage examples, image-only export packages, and screenshot documentation, see [gallery documentation](gallery.md).

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


## Saved-result publication tools

- `render-publication-figures.py`: explicit saved MGI curves/onsets, verified saved-parameter FBRM replay, and saved combo exports.
- `saved_fbrm_replay.py`: validates saved boundary, smoothing and witness metadata; does not import or execute the detector.
- `render-supplementary-series.py`: deterministic discovery or explicit saved plans; shown/hidden onset presentation variants.
- `render-supplementary-gallery.py`: offline thumbnails and structured-manifest comparison with client-side filters.

See [commands and data contracts](publication-rendering.md).

All 19 top-level Python files in `scripts/` are classified above. Image acquisition
and ROI extraction utilities mentioned in the broader pipeline are upstream tools,
not additional top-level scripts shipped here. Legacy gallery implementations are
kept only under `scripts/legacy/`.
