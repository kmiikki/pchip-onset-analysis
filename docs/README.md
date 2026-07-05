# Documentation

This directory contains supplementary documentation and figures used by the repository README.

## Pipeline documentation

- `pipeline.md` describes the repository-level PCHIP / MGI / FBRM analysis workflow.
- `pipeline.mmd` contains the standalone Mermaid source for the pipeline diagram embedded in `pipeline.md`.

The pipeline documentation describes the final computational workflow and uses repository-level Mermaid documentation rather than manuscript figures.

## Script and tool documentation

- `scripts.md` summarizes the repository scripts and their roles.
- `fbrm-onsets.md` describes FBRM Total Counts onset detection, inputs, parameters, and output files.
- `xy-pchip.md` documents the `xy-pchip.py` command-line tool.
- `xy-pchip-algorithm.md` describes the robust binning and PCHIP interpolation algorithm used by `xy-pchip.py`.

## Figures

The `figures/` directory contains selected images copied from the synthetic example output.

```text
figures/raw-pchip-diagnostic.png
figures/sg-pchip-diagnostic.png
figures/gallery-screenshot.png
figures/bw-raw-vs-temp-pchip.png
```

Figure roles:

- `raw-pchip-diagnostic.png` shows an example diagnostic MGI/PCHIP onset plot from the raw MGI branch.
- `sg-pchip-diagnostic.png` shows the corresponding Savitzky-Golay control branch diagnostic plot.
- `gallery-screenshot.png` shows the static HTML review-gallery layout generated from analysis outputs.
- `bw-raw-vs-temp-pchip.png` shows a basic raw MGI/BW signal with the PCHIP-interpolated curve.

These images are overview figures for the main `README.md`. They are documentation examples, not computational inputs.

The synthetic example, including CSV files and PCHIP output plots, is available under:

```text
examples/synthetic-rgb-tr/
```
