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

Run from the repository root, using a placeholder input path:

```bash
python3 scripts/fbrm-onsets.py --csv data/example/analysis/ts-fbrm-tr.csv
```

The default output directory is relative to the working directory (set
`--output-dir` explicitly to place it beside a selected input):

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

The default script writes three plots; `--publication-plots` additionally writes
eight `fbrm-onset-publication-{clean,main,onsets,onsets-clean}-{color,mono}.png` variants
and shared `view-limits.json` records display bounds. The default plots are:

```text
fbrm-onset-publication.png
fbrm-onset-diagnostic.png
fbrm-onset-candidates.png
```

The publication plot shows the raw and smoothed Total Counts signal with accepted onsets. The diagnostic plot includes additional derivative panels. The candidates plot shows accepted and rejected candidate points.

## Options

When `--onsets` is omitted, `[FBRM] onsets` in the inferred or explicit
`--analysis-dir` configuration supplies the maximum (initial default 2).
`--onsets 0` requests no accepted FBRM onsets, unlike the MGI zero-count mode.

```bash
python3 scripts/fbrm-onsets.py --csv data/example/analysis/ts-fbrm-tr.csv
```

More onsets can be requested explicitly:

```bash
python3 scripts/fbrm-onsets.py --csv data/example/analysis/ts-fbrm-tr.csv --onsets 3
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

The current source header and saved method identifiers still use v14.6
(`fbrm_total_counts_sequential_v14_6`), including the bounded precursor update.
Use the Git revision and saved parameters, not this label alone, to identify the implementation.

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

## Preparation and parameter contract

`--input` (alias `--csv`) defaults to `ts-fbrm-tr.csv`; `--xcol` and `--ycol`
can override detected columns. `--order input` preserves row order by default;
`--order time` explicitly sorts by a supplied/detected time column, and
`--reverse-input` explicitly reverses loading order. Neither sorts by temperature.
Cooling start defaults to `--cooling-start auto`; `max-temp`, `row` and `none`
are explicit alternatives. These options are scientific preparation choices.

The first raw Total Counts sample reaching `--max-search-count 10000`
(alias `--max-valid-count`) ends the search prefix, excluding the crossing sample
and everything after it. Zero disables this cutoff. It is an analysis/search
reliability limit, not a universal display maximum or independently established
instrument-validity boundary. Default SG uses `--savgol-window 101` and
`--savgol-polyorder 3`, with existing short-sequence adjustment. The analysis
prefix and outside-search context are smoothed separately; no SG kernel crosses
the cutoff. `--plot-full-data` may display outside-search context without
extending the onset search. Diagnostic derivatives are computed per sample.

For bounded precursor qualification, the later cooling-progress gap must be
positive and at most `--min-separation-C` (default 1.0 °C). The factor must be
finite and greater than 1, or exactly 0 to disable. See the rule above and
[current changes](current-method.md#fbrm-remains-a-separate-sequential-method).

Use `python3 scripts/fbrm-onsets.py --help` for the complete CLI. Display-only
limits and optional reference-onset overlays do not alter candidate selection.
For rendering existing results without detection, see
[saved-result FBRM replay](publication-rendering.md#fbrm-and-combo-contracts).
