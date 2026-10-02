# Current method and parameter contract

This supplements the existing [pipeline](pipeline.md), [PCHIP algorithm](xy-pchip-algorithm.md)
and [FBRM method](fbrm-onsets.md). No experimental onset values are reported here.
The CLI help and saved parameter records are the executable parameter reference.

## Inputs and preprocessing

MGI input is a temperature/signal CSV, conventionally `rgb-tr.csv` with
`Tr (°C)` and `BW`. The optional control input `rgb-tr-sg.csv` uses
`BW_smooth`. `tr-bw-sg.py` performs the existing Savitzky–Golay processing;
RAW and SG must remain distinct branches. The workflow SG template explicitly
selects `BW_smooth`. Do not silently use `BW` for that branch.

`xy-pchip.py` preserves its public implementation: numeric x/y filtering,
sorting, robust binning and duplicate-x handling precede a dense PCHIP grid.
Defaults are 200 bins and 2000 dense points. PCHIP interpolates the prepared
samples; it is not a substitute for SG smoothing. Reversing the plotted
cooling axis changes display direction, not CSV values or interpolation.
See the existing algorithm document for the complete binning rules.

## MGI derivative anchors and reporting

`bw-pchip-onsets-t1t2.py` loads the saved interpolated curve, computes first
and second derivatives, and identifies local peaks in the absolute first
derivative. Height, prominence, edge and separation qualifications determine
candidate anchors. Rough backtracking, baseline departure and segmented local
line intersections provide report candidates. The hybrid reporting mode
reconciles departure/intersection evidence according to its existing settings.
Production selection remains first-valid in cooling order after qualification;
requesting two onsets does not guarantee two accepted candidates.

The current valley rule separates geometry from final report qualification:

1. `--interpeak-rise-frac` (0.03) determines the rough valley departure used
   for eligibility and fitting geometry.
2. `--interpeak-report-rise-frac` (0.10) determines forced and explicit
   valley report points. It must not move the rough fitting windows.
3. After the derivative report cap, `--forced-valley-report-floor` (0.05)
   checks only reports with forced-valley precedence. If the derivative is
   below this fraction of the same anchor's absolute derivative, the scan
   advances toward that anchor until the floor is reached.
4. Ordinary intersection/departure reports do not acquire this floor.
   Anchors are not re-ranked by the floor.

Both new fractions must be finite and between 0 and 1. A zero floor disables
that final qualification. The former reporting configuration can be compared
explicitly using report-rise 0.03 and floor 0; this does not restore saved
historical results automatically.

Bends and candidate CSVs now include reporting provenance: the geometry
fraction, report fraction, forced precedence, geometry temperature, qualified
report temperature, post-cap temperature and whether the floor applied.
`view-limits.json` also records the reporting rule and configured fractions.
Consumers must allow these additional columns.

Fractional derivative qualifications are scale-independent; fitting widths,
separations and gaps are temperature-scale parameters. Absolute baseline
amplitude thresholds (such as `--departure-min-abs`) depend on signal units.
They are not automatically transferable to a different measurement method.

## FBRM remains a separate sequential method

`fbrm-onsets.py` reads aligned measured-temperature/Total Counts samples in
measurement order, selects cooling start, and identifies the high-count
search cutoff. SG uses the configured production window/order (defaults
101 and 3), adjusted only by the existing short-sequence rules. Prefix and
outside context are smoothed separately so no kernel crosses the cutoff.
The 10000-count default is an analysis/search reliability threshold, not a
universal display maximum. Non-monotonic temperature can produce a visible
loop; sorting by temperature would change the measurement geometry.

Local baseline/trend departure and sustained-rise qualifications produce
candidate events. The bounded precursor rule requires **both** later shift
metrics to exceed factor 4 times the corresponding early metric/threshold
maximum, strictly, within the existing minimum-separation window. Candidate
order and metrics are retained; first-valid selection skips qualified weak
precursors. Factor 0 disables this qualification; other factors must be finite
and greater than 1. See the FBRM document for the unchanged preparation model.
There is no production FBRM-PCHIP detector in this public update.

## COMBO and saved-result replay

The existing combo generator combines saved MGI information with the existing
FBRM preparation path; it is unchanged in this update. It exports named series
in `mgi-fbrm-combo-data.csv`. The publication renderer selects `MGI/curve` and
`FBRM/smoothed_curve_main_panel` from that export without reconstructing or
resmoothing the combo signal. Preserve sample order and repeated temperatures.

Standalone FBRM publication rendering uses saved input, onset, candidate and
parameter files. The shared replay helper extracts only the two pure SG
helpers from production source; it never imports the detector module or
runs selection. It checks saved input order, cooling row/index, cutoff
boundary, used SG window and every saved candidate/onset raw and smooth
witness (relative tolerance 1e-9, absolute tolerance 1e-7). Mismatches fail.
Only the saved analysis prefix is displayed. This is a witness-based replay
contract, not a claim that a historical dense smooth curve was exported.

Full/limited and y-bound controls are presentation-only. Accepted values
always come from saved outputs. Shown/hidden variants change only onset
annotations; scientific arrays are not recomputed for a limited view.
