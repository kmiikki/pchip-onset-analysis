# FBRM onset analysis

Sequential FBRM Total Counts onset detection using measured/interpolated temperature.

The tool is intended for FBRM Total Counts data aligned with measured/interpolated `Tr (°C)`, typically in a file named `ts-fbrm-tr.csv`.


## Repository context

This document describes the FBRM Total Counts onset detector included as `scripts/fbrm-onsets.py`.

The current synthetic example in `examples/synthetic-rgb-tr/` demonstrates the MGI/PCHIP branch only. It does not include synthetic FBRM input data or generated FBRM outputs.

## Important notes

* `Tr_regular` is not used.
* Final FBRM onset temperatures are reported on the measured/interpolated `Tr (°C)` axis.
* The script writes standardized CSV, JSON, text report, and diagnostic PNG outputs.
* FBRM results should be checked visually, especially when the curve shape is ambiguous.

## Method overview

The script detects FBRM onsets sequentially from the start of the cooling period. It works on the original input order and does not sort the data by temperature.

The basic idea is:

1. Determine the start of the cooling region.
2. Estimate an initial clear-solution baseline from the early valid FBRM signal.
3. Smooth the FBRM Total Counts signal for robust onset detection.
4. Search for candidate points where the signal leaves the previous baseline or trend.
5. Test whether each candidate is physically meaningful:

   * the signal must rise above the local baseline/trend,
   * the rise must be sustained,
   * a small local bump is not enough,
   * a rejected T1-like candidate does not consume the T1 slot; the search continues to later candidates.
6. Accept at most the requested number of onsets.
7. Write accepted onsets, all candidates, rejection reasons, parameters, reports, and plots.

The detector is designed for FBRM Total Counts curves where the first crystallization-related change may be small or gradual, while later changes may be steeper. It does not force the requested number of onsets: the output may contain zero, one, or more accepted onsets depending on the data and parameters.

## Basic use

Run inside a directory containing `ts-fbrm-tr.csv`:

```bash
python3 scripts/fbrm-onsets.py
```

The default output directory is:

```text
fbrm_onsets/
```

## Outputs

```text
fbrm_onsets/fbrm-onsets.csv
fbrm_onsets/fbrm-onset-candidates.csv
fbrm_onsets/fbrm-onset-params.json
fbrm_onsets/fbrm-onset-report.txt
fbrm_onsets/fbrm-onset-publication.png
fbrm_onsets/fbrm-onset-diagnostic.png
fbrm_onsets/fbrm-onset-candidates.png
```

## Output files

### `fbrm-onsets.csv`

Accepted onset points. Each accepted onset includes its label, temperature, Total Counts value, acceptance mode, and related diagnostic values.

### `fbrm-onset-candidates.csv`

All detected candidate points, including accepted and rejected candidates. Rejected candidates include rejection reasons. This file is useful for visual quality control and troubleshooting.

### `fbrm-onset-params.json`

Parameters and metadata used in the run. This file is intended to make the analysis reproducible.

### `fbrm-onset-report.txt`

Human-readable text summary of the run.

### PNG plots

The script writes three plots:

```text
fbrm-onset-publication.png
fbrm-onset-diagnostic.png
fbrm-onset-candidates.png
```

The publication plot shows the raw and smoothed Total Counts signal with accepted onsets. The diagnostic plot includes additional derivative panels. The candidates plot shows accepted and rejected candidate points.

## Options

By default, the script reports at most two accepted FBRM onsets:

```bash
python3 scripts/fbrm-onsets.py
```

More onsets can be requested explicitly:

```bash
python3 scripts/fbrm-onsets.py --onsets 3
```

This does not force three onsets. It reports 0..N accepted onsets, where N is the requested maximum.

## Input columns

The default input file is:

```text
ts-fbrm-tr.csv
```

Expected default columns:

```text
Tr (°C)
Total Counts
```

The column names can be changed with command-line options if needed.


## Repository version notes

The repository version of `scripts/fbrm-onsets.py` is the current v14.6 workflow version.

Compared with earlier standalone FBRM onset versions, this repository version also supports:

- `--analysis-dir` for locating the analysis directory and `onset-config.ini`,
- onset-count configuration through the shared onset configuration helper,
- optional repository-style publication plot variants with `--publication-plots`,
- grouped or individual reference-onset display for MGI/PCHIP comparison overlays,
- shared view-limit arguments and `view-limits.json` output for display-only axis-limit handling.

These options do not change the basic FBRM onset model: the detector preserves measurement order, uses measured/interpolated `Tr (°C)`, does not use `Tr_regular`, and reports zero to the requested maximum number of valid onsets.

## Current bounded precursor qualification

Before taking the first valid event, a weak event can be disqualified by a
later event within `--min-separation-C`. Both the level shift and sustained
shift must strictly exceed `--nearby-stronger-factor` times the larger of the
early metric and its corresponding threshold. The default factor is 4;
0 disables this qualification for historical comparison. A distant stronger
event cannot suppress an earlier event, and one strong metric is insufficient.
The first surviving event retains the existing sequential selection rule.
Candidate exports identify the rejecting candidate and cooling-progress gap.
This changes selection qualification, not cooling preparation or SG smoothing.
