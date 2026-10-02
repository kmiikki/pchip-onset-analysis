# Synthetic RGB-TR Example

This directory contains a synthetic demonstration dataset for the PCHIP onset workflow.

The example is included so the workflow can be tested and inspected without using unpublished experimental data.

## Files

Initial input:

```text
rgb-tr.csv
```

Generated SG-preprocessing output:

```text
rgb-tr-sg.csv
rgb-tr-sg.png
```

Generated PCHIP workflow output:

```text
bw-pchip-workflow-summary.csv
bw-pchip-workflow-report.txt
pchip/
```

## Synthetic data

The synthetic data represent a cooling experiment from approximately 60 °C to 25 °C.

The signal is intentionally flat before T1, representing a transparent solution before the first crystallization-related optical change. The RGB/MGI increase starts only after T1.

The example contains two synthetic optical transitions:

```text
T1: first visible optical change
T2: stronger crystallization-related transition
```

The data are synthetic demonstration data. They are not experimental results and must not be used for scientific conclusions.

## Regenerate the synthetic input file

```bash
python3 ../scripts/make-synthetic-rgb-tr.py
```

This writes:

```text
rgb-tr.csv
```

## Run the workflow

From this directory:

```bash
python3 ../../scripts/tr-bw-sg.py -quiet
python3 ../../scripts/run-bw-pchip-workflow.py --execute \
  --xy-pchip ../../scripts/xy-pchip.py \
  --onset-script ../../scripts/bw-pchip-onsets-t1t2.py
```

Explicit script paths keep both workflow stages within this clone instead of
resolving similarly named tools from `PATH`. Scientific parameters use the
normal defaults and the existing `pchip/onset-config.ini`.

The refreshed bends CSVs and `view-limits.json` include the current valley
reporting provenance. Accepted synthetic T1/T2 results are unchanged. Committed
logs and reports use repository-relative paths and a neutral `python3` interpreter
name; their command working directories are relative to the repository root.

## Static gallery

A local review gallery can be generated from this example after copying the
`pchip/` output into the experiment-tree layout required by
`make-onset-gallery.py`.

`make-onset-gallery.py` discovers active workflow outputs from the normal
experiment-tree layout. A compact single-analysis directory is valid for running
the PCHIP workflow itself:

```text
rgb-tr.csv
rgb-tr-sg.csv
pchip/
```

but that compact layout is not sufficient for gallery discovery.

For gallery generation, the analysis directory must be placed under an
experiment tree:

```text
<gallery-root>/
└── <experiment>/
    └── tl/
        └── roi*/
            └── rgb/
                └── analysis/
                    └── pchip/
```

For this synthetic example, a gallery-ready layout can be:

```text
<gallery-root>/
└── synthetic-rgb-tr/
    └── tl/
        └── roi1/
            └── rgb/
                └── analysis/
                    ├── rgb-tr.csv
                    ├── rgb-tr-sg.csv
                    ├── rgb-tr-sg.png
                    └── pchip/
```

This requirement keeps the gallery generator generic: it uses the same discovery
logic for synthetic examples and full experiment projects, and it avoids
accidentally collecting legacy, root-level, or unrelated result directories.

See `../../docs/gallery.md` for gallery commands, data-link galleries,
image-only export packages, and image-plus-data export packages.

## Important result files

Primary summary:

```text
bw-pchip-workflow-summary.csv
bw-pchip-workflow-report.txt
pchip/bw-pchip-bend-summary.csv
```

Raw MGI PCHIP results:

```text
pchip/bw-raw-vs-temp-pchip.csv
pchip/bw-raw-vs-temp-pchip-bends.csv
pchip/bw-raw-vs-temp-pchip-diagnostic-bends.png
```

SG-smoothed MGI PCHIP results:

```text
pchip/bw-sg-vs-temp-pchip.csv
pchip/bw-sg-vs-temp-pchip-bends.csv
pchip/bw-sg-vs-temp-pchip-diagnostic-bends.png
```

## Note about `rgb-tr-sg.png`

`rgb-tr-sg.png` is a quick-look plot produced by the Savitzky-Golay preprocessing script. The red dashed bend markers in this legacy plot are not the final PCHIP T1/T2 onset results.

Use the PCHIP workflow outputs in the `pchip/` directory for onset interpretation.
