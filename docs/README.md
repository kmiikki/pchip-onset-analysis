# Documentation

## Core methods

- [MGI/PCHIP onset analysis](mgi-onsets.md): primary implementation reference, CLI and reproducibility requirements.
- [FBRM onset analysis](fbrm-onsets.md): independent sequential detector and preparation contract.
- [PCHIP construction](xy-pchip-algorithm.md): robust binning and interpolation.
- [PCHIP CLI](xy-pchip.md): input selection, parameters and output naming.
- [Current method changes](current-method.md): concise qualification and replay update summary.

## Workflow

- [Pipeline](pipeline.md) and [Mermaid source](pipeline.mmd): scientific branches and downstream outputs.
- [Script inventory](scripts.md): active tools, preprocessing, helpers and legacy classification.
- [Synthetic example](../examples/synthetic-rgb-tr/README.md): reproducible public MGI example.

## Rendering and review

- [Offline analysis galleries](gallery.md): discovery, synthetic screenshot and portable ZIP export.
- [Publication and supplementary rendering](publication-rendering.md): saved-result contracts and shown/hidden variants.

## Repository and release policy

- [Publication policy](publication-policy.md): public/private boundaries and immutable references.
- [Integration review](integration-review.md): transferred/excluded components and remaining source-help follow-up.

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
