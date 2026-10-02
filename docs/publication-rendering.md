# Saved-result publication and supplementary rendering

All commands below refer only to the repository synthetic example or placeholders.
Run from the repository root with the same Python environment as the analysis.
OpenCV is needed for RGB/grayscale and true 1-bit PNG encoding.

## Explicit MGI sources

```bash
python3 scripts/render-publication-figures.py mgi \
  --curve examples/synthetic-rgb-tr/pchip/bw-raw-vs-temp-pchip.csv \
  --bends examples/synthetic-rgb-tr/pchip/bw-raw-vs-temp-pchip-bends.csv \
  --branch raw --points show \
  --measurements examples/synthetic-rgb-tr/rgb-tr.csv --measurement-ycol BW \
  --acquisition-extent --view full --mode gray --onsets markers \
  --name synthetic-mgi --output-dir generated/synthetic-publication
```

Select a saved SG curve with `--branch sg`. If its signal column is named
`BW_smooth`, specify `--ycol BW_smooth`; some saved exports use `BW`.
The filename alone does not determine column names. No PCHIP or detector runs.
`--onsets none` removes markers and the values block; `text` keeps only values;
`markers` includes the MGI marker layer. `--show-onset-labels` is optional and
never enabled by the supplementary batch. `--mgi-linewidth` affects only the
standalone curve. Labels use collision-aware placement with small leader lines.

## FBRM and combo contracts

FBRM requires `--data`, `--onsets-file`, `--candidates-file`, `--params-file`:
the original aligned CSV and saved production outputs. `--points show` displays
original samples. `--onsets text` includes the existing FBRM markers and values;
`none` hides both. Full means the complete saved **analysis prefix**, not the
excluded post-cutoff cooling context. The shared `saved_fbrm_replay.py` extracts only the two production smoothing
helpers, never importing the detector module or running candidate selection.
Saved boundary, SG-window and raw/smooth candidate/onset witnesses must validate;
a mismatch fails rather than rendering an unverified curve. See
[replay validation](current-method.md#combo-and-saved-result-replay).

Combo uses `--data` pointing to the existing combo export. It plots no raw
points and no vertical onset markers. Optional saved onset files supply only
text. There is no standalone signal preparation in the publication renderer.

## Style and output safety

Default single-panel width is 83.5 mm; a two-panel canvas can use 167 mm.
Typography is 8 pt, linewidth scale 0.50, no grid, `Tr [°C]`, `tight_layout`.
Temperature ticks depend on span: up to 2.5/6/15/40 degrees use 0.5/1/2/5-degree
steps, then 10 degrees up to a 100-degree span and an automatic locator beyond.
Cooling direction is retained. MGI has no redundant legend; combo retains one.

- `gray`: 600 dpi, black analysis curve, gray points; combo black solid / gray solid.
- `bw`: true 1-bit 1200 dpi; combo black solid / black dashed (optional dotted).
- `color`: RGB 300 dpi; combo blue / orange solid. Standalone MGI curve stays
  black with blue measurement points at alpha 0.3.

`--view limited --x-range LOWER UPPER` changes only display. Independent
`--mgi-y-lower/upper` and `--fbrm-y-lower/upper` leave missing bounds automatic;
`--*-y-range` specifies both bounds and cannot be combined with the independent
syntax. Combo also supports panel-specific bounds. Read `--help` for the full CLI.

Default outputs are under `generated/`. Explicit view outputs are allowed only
under repository `data/.../analysis/.../views/<view-id>/`; symlink escapes are
rejected. FBRM replay outputs stay under `generated/`. Existing names are refused
unless `-f/--force` explicitly replaces that PNG/manifest pair. No scientific
source file may be replaced. Manifests retain source hashes, options, accepted
onsets, actual axis limits, mode, DPI, style, renderer hash and Git commit.
Treat manifests from private input as private data.

## Supplementary planning and variants

Discovery expects `<data-root>/<date>_ex_<id>/.../rgb/analysis/` and includes all
available ROIs and RAW/SG branches. The compact public example intentionally
has no experiment-tree wrapper; the synthetic tests build that layout in a
temporary directory. Supply a suitable tree or use an explicit saved plan.

```bash
python3 scripts/render-supplementary-series.py --data-root data \
  --dry-run --output-dir generated/synthetic-plan
python3 scripts/render-supplementary-series.py \
  --plan generated/synthetic-plan/plan.json --run \
  --presentation-variant with-onsets --output-dir generated/synthetic-shown
python3 scripts/render-supplementary-series.py \
  --plan generated/synthetic-plan/plan.json --run \
  --presentation-variant no-onsets --output-dir generated/synthetic-hidden
```

The latter commands require a plan from your synthetic tree; they do not
create or discover input measurements. Full views are always planned. Limited
views come only from saved view metadata or matching renderer manifests; no
new limits are inferred. Archived views may be retained as alternatives.
COMBO is BW; standalone MGI/FBRM are gray with points when available. RAW points
remain RAW even beside an SG-derived MGI curve. No separate onset labels are
added. The generated supplementary COMBO commands use `--onsets none` in both variants,
so these pairs already have no onset layer. This does not remove the standalone
combo renderer's optional saved-value text capability.

The batch writes `plan.json`, PNG/manifest pairs and a render log. It does not
write a consolidated production manifest automatically. For gallery building,
provide JSON with a `jobs` list retaining plan metadata and adding:
`job_index` (stable across variants), `status: "OK"`, `output_path` (relative to
the manifest directory under `figures/`), `output_sha256`, and `onsets_variant`
(`shown` or `hidden`). The end-to-end synthetic test demonstrates this schema.
Do not confuse an unexecuted plan with a successful production manifest.

```bash
python3 scripts/render-supplementary-gallery.py \
  generated/synthetic-hidden/manifest.json \
  --additional-manifest generated/synthetic-shown/manifest.json
```

The gallery verifies PNG hashes, makes thumbnails and links original images.
No external libraries or server are needed in the browser. Experiment, ROI,
type, branch, view and onset-visibility filters are local JavaScript. Keep both
production directories together when copying/zipping a comparison. Full images
stay independent; thumbnails are not publication replacements.

Only synthetic/public inputs belong in public examples. Plans, renderer manifests,
render logs and galleries from private data can expose paths and accepted values,
even when the PNG hides onset annotations. Keep these artifacts private under the
[publication policy](publication-policy.md). Presentation variants do not anonymize
research data.
