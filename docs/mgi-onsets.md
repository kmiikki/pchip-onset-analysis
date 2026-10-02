# MGI/PCHIP onset analysis

## Purpose and scope

This is the implementation reference for [bw-pchip-onsets-t1t2.py](../scripts/bw-pchip-onsets-t1t2.py), inspected at repository revision `1aa3c50`. Function names below identify the executable sources of each rule. This describes the current implementation, including diagnostic exceptions; it does not propose parameter tuning or establish physical correctness for a particular measurement.

MGI means Mean Gray Intensity, represented by the historical `BW` signal column. The detector analyzes an already saved temperature–signal PCHIP curve. It does not extract image intensities, align timestamps, perform Savitzky–Golay (SG) filtering, or construct a new PCHIP interpolator. RAW and SG are separate input branches, not competing candidates in one analysis. FBRM detection is a separate method.

Related context: [current method](current-method.md), [pipeline](pipeline.md), [PCHIP CLI](xy-pchip.md), and [PCHIP construction](xy-pchip-algorithm.md).

## Conceptual summary

The method starts from a saved MGI-versus-temperature PCHIP curve and identifies transition anchors as peaks in the absolute first derivative. For each anchor it backtracks toward higher temperature to locate rough onset geometry. Baseline departure and segmented-line intersection provide report candidates in the default hybrid mode; when the inter-peak conditions hold, separate valley geometry and reporting rules determine the applicable valley report.

After report construction, candidate rejection and significance qualification distinguish eligible reports from diagnostic alternatives. With a positive requested onset count, the default first-valid selection accepts separated reports in cooling order and names them T1, T2, and so on. The detailed algorithm below specifies the precedence, fallbacks and diagnostic exceptions; requesting a count does not guarantee that many accepted onsets.

## Inputs

Supply one or more positional PCHIP CSV paths. With none, `autodetect_inputs()` looks in the current directory for `bw-raw-vs-temp-pchip.csv` and `bw-sg-vs-temp-pchip.csv`, using whichever exist; neither existing is an error.

| Branch | Upstream measurement signal | Saved detector input | Expected signal column |
|---|---|---|---|
| RAW | `rgb-tr.csv` / `BW` | `bw-raw-vs-temp-pchip.csv` | `BW` |
| SG control | `rgb-tr-sg.csv` / `BW_smooth` | `bw-sg-vs-temp-pchip.csv` | `BW_smooth` |

Temperature is conventionally `Tr (°C)`. The detector defaults to the **first and second CSV columns**, not hard-coded column names. `--xcol` and `--ycol` select names explicitly. A single explicit `--ycol` applies to every input in that invocation; use separate commands if branch column names differ. `infer_method()` labels a file `sg_pchip` if its name contains `sg` or its selected y-column contains `smooth`, otherwise `raw_pchip` if its name contains `raw`, otherwise `pchip`. This label does not replace column selection.

`read_xy()` requires at least two columns, coerces selected values to numbers, removes nonfinite x/y pairs, and requires at least ten remaining rows **before** duplicate collapse. It sorts temperature ascending and averages y at duplicate temperatures. It does not separately recheck the ten-row condition after collapse. Indices in results refer to this prepared curve, not measurement CSV row numbers.

`--outdir` defaults to each curve's parent. Bends and optional candidate files use that input's stem. `--analysis-dir` controls onset-count configuration lookup. Original measurement CSVs are optional plot inputs through `--raw-csv` or inferred neighboring files; they are not inputs to `detect_bends()`.

## Overall algorithm

For each saved curve, `analyze_one()` and `detect_bends()` execute:

1. Read, clean and sort x/y; compute numerical first and second derivatives.
2. Find permissive absolute-derivative peak anchors and retain their prominences.
3. For each anchor, estimate rough derivative backtracking; optionally replace rough geometry with an earlier inter-peak valley departure and record forced-report eligibility.
4. For intersection/departure/hybrid modes, fit local segments and estimate both intersection and baseline departure.
5. Give an eligible forced-valley report precedence; otherwise apply the requested reporting mode.
6. Apply the internal derivative report cap, then the forced-only derivative floor, to these peak-anchored candidates.
7. Create candidate records; append separate explicit valley-rise candidates. These appended candidates do **not** pass through step 6.
8. Reject excessive report–anchor gaps; compute the final significance reference from remaining candidate records and reject weak final peaks.
9. Assign diagnostic ranks. For a positive requested count, traverse first-valid cooling order (or optional strength order), reject near-duplicates, and enforce the count limit.
10. Sort selected reports in descending temperature and assign T1, T2, etc.; write CSVs, plots and view metadata.

The special `--bends 0` path differs from step 9: see [T1/T2 selection](#t1t2-selection).

## PCHIP curve and derivatives

Write prepared samples as $(T_i,I_i)$ with increasing $T_i$. `compute_derivatives()` uses `np.gradient(y, x)` followed by `np.gradient(dy_dx, x)`, not analytic derivatives of the original PCHIP polynomial:

$$
D_i=\mathrm{gradient}(I,T)_i,\qquad
A_i=|D_i|,\qquad
Q_i=\mathrm{gradient}(D,T)_i.
$$

$D$ has signal-units/°C; $Q$ has signal-units/°C². Absolute slope permits anchors of either derivative sign. There is no additional sign test requiring increasing intensity during cooling. The second derivative is stored at report points; it is not an anchor, qualification, or selection criterion.

Cooling order is decreasing temperature, hence decreasing prepared-array index. Backtracking toward the earlier, warmer side increases index. Reversing the plot x-axis is independent of this numerical ordering. It cannot reverse the algorithm's selection order.

## Candidate anchor detection

`detect_peak_anchors()` defines $A_{\max}=\max_i A_i$ before edge masking. It zeros nonfinite samples and, normally, the first/last $n_e=\mathrm{round}(n f_e)$ samples, with default $f_e=0.02$. Masking applies only if $n_e>0$ and $2n_e<n$.

`scipy.signal.find_peaks` receives:

$$
h_{\min}=f_h A_{\max},\qquad
p_{\min}=f_p A_{\max},\qquad
n_{\mathrm{sep}}=\max(1,\mathrm{round}(\Delta T_c/\delta T)),
$$

where $\delta T$ is the median absolute nonzero grid spacing, $f_h=f_p=0.05$, and $\Delta T_c=0.10$ °C. Height and prominence are separate conditions. `distance` is a sample-index distance derived from median spacing, not an exact temperature-distance check on irregular grids.

Anchors are ranked by descending `(prominence, peak height)`. If the requested count $N>0$, only the first $\max(20N,50)$ anchors are retained; with zero, this cap is disabled. Thus even the requested count can affect the search on a curve with very many peaks. Anchor-search separation and final report separation are different parameters.

## Candidate qualification

### Anchor detection and qualification

Anchor qualification is the preceding height/prominence/edge/distance search. It determines which derivative peaks enter report construction, not which reports become T1/T2.

### Report construction

Rough backtracking, local fits, reporting-mode selection and the applicable valley/cap/floor rules construct reports before the filtering below. These steps are detailed in the following sections and in [Overall algorithm](#overall-algorithm). Report construction is not itself final acceptance.

### Candidate rejection: report–anchor gap

After ordinary and explicit valley candidates exist, a positive `--max-report-peak-gap-C` rejects candidates satisfying

$$
|T_{\mathrm{report}}-T_{\mathrm{anchor}}|>G,\qquad G=3.0\;\mathrm{°C}.
$$

The status is `rejected_report_peak_gap`; zero disables this check. The code tests finite gaps before rejecting.

### Significance qualification

Next, `compute_final_peak_reference()` sorts positive finite anchor heights from **candidate records not already rejected**. These are not deduplicated by anchor: an explicit valley and an ordinary candidate can contribute the same height. For the two largest values $H_1,H_2$:

$$
R=\min(H_1,cH_2),\qquad c=5.0.
$$

With one eligible positive value, $R=H_1$; with none, $R=0$. A nonpositive dominance factor disables capping and uses $H_1$. A positive final fraction rejects $H<f_{\mathrm{final}}R$, default $f_{\mathrm{final}}=0.15$, with status `rejected_weak_final_peak`. Equality passes. Zero disables this filter. This reference is not necessarily the global maximum of the sampled derivative.

### Final selection

The later selection loop handles `rejected_duplicate_transition`, `not_selected_after_limit` and `accepted`. Candidate CSV `status` and `reason` distinguish these stages; a candidate's existence is not evidence of acceptance.

## Rough backtracking and valley geometry

`backtrack_onset_index()` walks from an anchor toward higher temperature while $A_i\ge f_b A_p$, default $f_b=0.10$, and returns the preceding index (bounded by the anchor). `backtrack_derivative_departure_index()` supplements this with a local derivative-background estimate.

Its internal defaults are a 0.4 °C gap, 3.0 °C background width (converted to at least one and five samples), and

$$
\theta_D=\mathrm{median}(A_{\mathrm{bg}})
 +\max(6\sigma_{\mathrm{bg}},0.03A_p,1.0).
$$

Here $\sigma=1.4826\mathrm{median}|z-\mathrm{median}(z)|$, falling back to standard deviation if MAD is zero. A short background window falls back to all higher-index samples; fewer than five finite samples gives zero background/noise. The absolute `1.0` is in signal-units/°C. The returned index is the larger of this departure index and the simple fractional-backtrack index: this adjustment can move earlier/warmer, not later than that fallback.

For an anchor $p$, `interpeak_valley_departure_index()` considers the nearest higher-temperature retained anchor $q$. It requires a positive gap at most 2.0 °C, at least three index steps between anchors, an interior derivative minimum $v$, finite values, $A_p>A_v$, and

$$
\frac{\min(A_p,A_q)-A_v}{\min(A_p,A_q)}\ge0.20.
$$

From $v$ toward $p$ in cooling order it returns the **first single sample** satisfying

$$
A_i\ge A_v+f(A_p-A_v).
$$

Despite the helper's comment saying “sustained rise”, there is no multi-sample confirmation in this valley scan. The default geometry fraction is `--interpeak-rise-frac 0.03`.

If this geometry index is larger than the existing rough index, it replaces rough geometry and enables a separate forced report calculation. Geometry drives fitting windows and forced eligibility; it is not automatically the final report. It can still become the final report through fallback or the derivative cap.

## Baseline departure

`baseline_departure_for_peak()` uses the pre-transition fitted line $L_b(T)=m_bT+b_b$, reusing the segmented fit when available. Its window is $[T_r+g_b,T_r+g_b+w_b]$, where $T_r$ is rough temperature, $g_b=0.04$ °C and $w_b=1.0$ °C.

Residual noise uses the same MAD-to-sigma rule (standard-deviation fallback), or zero for fewer than three finite residuals. The signal threshold is

$$
\theta_I=\max(a_{\min},k\sigma_I),\qquad a_{\min}=0.5,\quad k=5.0.
$$

Scanning from the index nearest $\min(T_{b,\min},T_r+g_b)$ toward the anchor, it requires three consecutive samples, each satisfying

$$
|I_j-L_b(T_j)|\ge\theta_I.
$$

It reports the highest-temperature sample of the qualifying three-sample window, not its last confirmation sample. Missing fit or no qualifying window returns no departure. `--departure-min-abs` has signal units; `--departure-k` is dimensionless.

## Segmented line intersection

`fit_line()` uses ordinary least-squares `np.polyfit(...,1)` with at least `--min-fit-points` finite points and nonzero temperature span. The baseline window is as above. The transition window is

$$
[\max(T_{\min},T_p-0.35w_t),\ \max(T_p+0.05w_t,T_r-g_t)],
$$

with $w_t=0.80$ °C and $g_t=0.02$ °C. If reversed/empty, it falls back to $[\max(T_{\min},T_p-0.5w_t),\min(T_{\max},T_p+0.5w_t)]$.

For baseline and transition lines, `line_intersection_x()` computes

$$
T_\cap=\frac{b_t-b_b}{m_b-m_t}.
$$

It rejects a nonfinite denominator, $|m_b-m_t|<10^{-12}$, nonfinite intersection, missing fit, or intersection outside

$$
[\min(T_p,T_r)-0.20w_t,\ T_r+0.50w_b].
$$

An intersection is a continuous temperature estimate; departure is a sampled baseline-deviation event. They are not interchangeable. Report derivatives/indices use the nearest saved-grid index, while report intensity uses linear `np.interp` on the saved curve, clamping its evaluation temperature to the saved domain. No new PCHIP fit occurs here.

## Hybrid reporting

Forced-valley precedence is evaluated before all `--method` branches, including `peak` and `backtrack`. Without precedence:

| Mode | Initial report |
|---|---|
| `peak` | Anchor temperature |
| `backtrack` | Rough temperature |
| `departure` | Departure, otherwise rough with `used_fallback=True` |
| `intersection` | Valid intersection, otherwise rough with fallback |
| `hybrid` (default) | Both missing: rough/fallback. One available: that estimate. Both available: departure if their absolute temperature difference exceeds `--hybrid-disagree-C` (0.35 °C), otherwise intersection. |

All these peak-anchored initial reports then undergo the internal cap: if finite $A_p>0$ and $A_{\mathrm{report}}/A_p>0.20$, reset report temperature/index to rough geometry. This includes `peak` mode; its name does not guarantee an unchanged peak-temperature output. The cap is a replacement rule, not a clamp or a proof that the replacement ratio is at most 0.20. It does not itself set `used_fallback`.

## Valley reporting and forced-valley qualification

The geometry helper is called again with `--interpeak-report-rise-frac` (0.10) only when the geometry departure moved rough earlier. This separate crossing becomes the forced report and overrides the requested reporting mode. Fits still use the 0.03 geometry, not the 0.10 crossing.

After the cap, `qualify_forced_valley_report()` acts only when `forced_valley_precedence=True` and floor $f_F>0$. If the current derivative is below $f_FA_p$, it scans toward the **same anchor**, inclusive, taking the first finite sample satisfying

$$
A_{\mathrm{report}}\ge f_F A_p,\qquad f_F=0.05.
$$

It does not re-rank anchors, recompute fits or impose a floor on ordinary intersection/departure reports. Zero disables this final qualification. The cap is not rerun after the floor. The implementation validates both report fraction and floor as finite values in $[0,1]$. The older geometry fraction has only a nonnegative CLI check, not the same finite/unit-interval validation.

A second path, `build_interpeak_valley_candidates()`, appends **independent explicit valley-rise candidates** for adjacent retained anchors in cooling order. It uses the report fraction, the same 2.0 °C gap and 0.20 valley-drop requirements, and additionally requires both neighbor heights at least `--interpeak-min-neighbor-peak-frac` (0.20) times the global sampled derivative maximum. The report must be at least 0.05 °C from its lower-temperature anchor. These constants are internal, not CLI options.

Explicit candidates carry `report_mode=<method>+valley-rise`, no fitted lines, and `used_fallback=True`. Their rough fields equal their report point; the separate geometry provenance field records the geometry-fraction crossing. They are added after the cap/floor loop and have no forced precedence. The neighbor-height rule applies to this explicit path, **not** to the forced peak-anchored helper. Both paths still encounter gap/significance/selection rules afterward.

## T1/T2 selection

For positive `--bends N`, default `first-valid` sorts by final report temperature descending, skips already rejected candidates, then rejects any report closer than `--min-separation-C` (0.25 °C) to an already accepted report:

$$
|T_c-T_a|<\Delta T_{\min}.
$$

Equality is allowed. Duplicate checking occurs before the count-limit check. Other eligible candidates beyond $N$ receive `not_selected_after_limit`. There is no separate enforced uniqueness of anchor identity. `strongest` instead traverses descending prominence then anchor height; it is a diagnostic alternative, not the default.

Selected records are finally sorted in cooling order and assigned ranks 1, 2, etc. These are T1/T2 labels, not the strongest/second-strongest peaks. Requesting two allows zero or one accepted onset; no missing onset is fabricated. With zero accepted, bends CSV contains a header, plots contain no accepted markers, and stdout says none detected. With one accepted, only T1 is present.

> **Implementation caveat — `--bends 0`**
>
> The current implementation marks **every generated candidate** accepted in cooling order, overwriting earlier rejection status/reason and bypassing the final duplicate filter. Do not interpret this as “all valid candidates” or “all candidates surviving quality filters”, despite the CLI wording.
>
> Positive-count production selection retains the qualification and duplicate checks described above. This caveat documents the current zero-count behavior; it does not change it.

`--bends` omitted reads `[MGI] onsets` from `onset-config.ini` (initial default 2). An explicit nonnegative count is a one-run override; disagreement prints a warning, without updating the configured decision. Configuration resolution runs before CSV analysis and can create a missing config or back up/reset a malformed config according to [onset_config.py](../scripts/onset_config.py). This script is an analysis writer, not a read-only renderer.

## Provenance and saved outputs

Bends CSV stores input basename, branch label, requested report mode, final rank, temperature/intensity, rough and anchor values, derivatives, prominence, prepared-curve indices, fit coefficients/counts and `used_fallback`. Candidate CSV additionally stores candidate ID, strength/cooling ranks, acceptance, final rank, status/reason and derivative ratio. IDs describe construction order, not T-rank. Most numerical result/fit fields use six significant digits; indices and saved input are important for more precise reconstruction.

Both CSVs append these `ValleyReportProvenance` fields:

| Field | Meaning |
|---|---|
| `interpeak_rise_frac` | Rough geometry fraction |
| `interpeak_report_rise_frac` | Forced/explicit report fraction |
| `forced_valley_report_floor` | Configured forced-only floor |
| `forced_valley_precedence` | Whether the peak-anchored forced report overrode method selection |
| `valley_geometry_temperature_C` | Geometry crossing when found, even if it did not move rough |
| `valley_qualified_report_temperature_C` | Initial report-fraction crossing, before cap/floor; not necessarily final |
| `post_derivative_cap_temperature_C` | Peak-anchored report immediately after cap, before floor |
| `forced_valley_floor_applied` | Whether the floor actually changed report index |

Absent temperatures are blank in CSV. Explicit valley candidates leave forced precedence/floor-applied false and post-cap temperature unset. Bends preserve selected candidate provenance, but their `report_mode` is the requested method, so use candidate CSV for the explicit `+valley-rise` distinction.

`view-limits.json` records view bounds, display-only semantics, primary axis and the three reporting fractions under `onset_reporting`, with rule `separate_valley_report_with_forced_only_floor`. It is not a complete detector-parameter manifest. Preserve the saved curve, full CLI/config, code revision and numerical environment for reproducibility; the combined summary omits the new valley fields. CSV/summary paths can reflect caller-supplied local paths and need review before public sharing.

## Reproducibility requirements

**The saved detector input curve is part of the scientific provenance.** Final onset temperatures, plots or summary rows alone do not identify all samples, geometry and choices used to obtain them. Preserve the following information together:

| Required record | What it establishes |
|---|---|
| Exact input PCHIP CSV(s) | The numerical samples analyzed, including grid spacing and signal values; retain each branch separately. |
| RAW/SG identity and selected x/y columns | Whether the input represents RAW `BW` or SG `BW_smooth`, and which temperature column was actually selected. A filename-derived branch label alone is insufficient. |
| Complete detector command and effective parameters | Reporting/selection modes and every scientific threshold, width and count, including defaults used in that run. |
| Relevant `onset-config.ini` state and explicit count override | The requested onset count when configuration supplies it, and whether `--bends` overrode that state. Preserve the effective state, including any configuration creation/repair. |
| Code revision / Git commit and any local modifications | The implementation and internal constants used; CLI values alone do not capture internal rules. |
| Numerical environment | Python, NumPy and SciPy versions and relevant build/environment details sufficient to identify the numerical behavior used for gradients, peak finding and fitting. |
| Bends and, where decision provenance is needed, candidate CSVs | Accepted reports plus rejected alternatives, ranks, indices, fit information and valley reporting provenance. Candidate export must be requested explicitly. |
| Source preprocessing choices | RAW versus SG, and the SG settings used upstream when applicable. Repeating the upstream pipeline additionally requires its source input data. |
| PCHIP construction settings | Selected columns, binning and dense-grid parameters and the upstream implementation that produced the saved detector input. These are needed to reconstruct that input from source. |
| `view-limits.json` and figure settings | Presentation bounds and the saved reporting-fraction metadata, kept separate from the complete scientific parameter record. |

**`view-limits.json` alone is not a full detector manifest and cannot reproduce the analysis.** It contains presentation/reporting metadata, not all scientific parameters, configuration state or input samples. Repeating detection from the exact saved curve is distinct from regenerating that curve through preprocessing and PCHIP construction. Preserve both levels of provenance when claiming end-to-end reproduction. The existing CSV precision and summary omissions described above still apply.

## Output files

Let `<stem>` be the input CSV stem. `analyze_one()`, `save_requested_plots()` and the CSV writers produce:

| Output | Condition |
|---|---|
| `<stem>-bends.csv` | Always, including header-only result |
| `<stem>-candidates.csv` | `--candidate-diagnostics` or `--plot-mode candidates` |
| `<stem>-bends.png` | Smooth plot |
| `<stem>-raw-bends.png` | Raw-only plot |
| `<stem>-raw-smooth-bends.png` | Measurement points plus curve |
| `<stem>-diagnostic-main-bends.png` | Main diagnostic without derivative subplot |
| `<stem>-diagnostic-bends.png` | Diagnostic with derivative subplot |
| `<stem>-candidate-diagnostics.png` | Candidates requested |
| `<stem>-publication-{clean,main,onsets,onsets-clean}-{color,mono}.png` | Eight variants with `--publication-plots` |
| `view-limits.json` | Each analyzed output directory |
| `bw-pchip-bend-summary.csv` | More than one input, configurable `--summary` |

Default plot set writes smooth, raw-smooth, diagnostic-main and diagnostic. `all` adds raw; candidate diagnostics remain separately requested. `single` uses `--plot-mode`. The script prints a report to stdout; it does not itself create the runner's workflow TXT report or workflow log, nor a complete parameter JSON. It does not write PCHIP curve CSVs. Config creation/repair is the additional possible INI side effect noted above.

Limited views route to `<base-output>/views/<view-id>/` unless `--view-output-dir` overrides it; bends/candidates are still computed from the full input. A relative summary path is rooted under `--outdir` if supplied, otherwise the working directory. These analysis-time publication variants are distinct from the locked saved-result publication renderer and do not imply its DPI/encoding policy.

## CLI reference by purpose

Defaults below are from `parse_args()` and shared `add_view_limit_arguments()`. **S** affects scientific input, geometry, qualification or selection; **P** affects presentation only; **O** controls files/diagnostics/configuration. Count is dimensionless; “signal” means the chosen y-column units. Flags default false unless stated.

### Input, output and onset count

| Argument | Default | Units | Purpose / effect |
|---|---|---|---|
| `csv_files` | Auto-discovery | Paths | Saved curves; S |
| `--xcol`, `--ycol` | First, second column | Names | Numerical column selection; S |
| `--bends` | Config, initially 2 | Count | Maximum positive-count selection; zero diagnostic exception; S |
| `--analysis-dir` | Inferred | Path | Onset config location, can affect count; S/O |
| `--outdir` | Input parent | Path | Output base; also config inference; O, potentially S through config |
| `--summary` | `bw-pchip-bend-summary.csv` | Path | Combined multi-input summary; O |
| `--raw-csv` | Inferred neighboring source | Path | Plot points/acquisition extent only; P |
| `-h`, `--help` | Exit with help | — | CLI usage; O |

### Anchor detection and final qualification

| Argument | Default | Units | Purpose / effect |
|---|---|---|---|
| `--candidate-min-separation-C` | 0.10 | °C | Anchor sample-distance conversion; S |
| `--min-height-frac` | 0.05 | Fraction | Global derivative peak height threshold; S |
| `--prominence-frac` | 0.05 | Fraction | Global derivative prominence threshold; S |
| `--edge-frac` | 0.02 | Fraction of samples | Edge exclusion; S |
| `--min-final-peak-height-frac` | 0.15 | Fraction | Final reference threshold, zero disables; S |
| `--final-peak-dominance-factor` | 5.0 | Ratio | Reference cap, nonpositive disables in helper; S |
| `--max-report-peak-gap-C` | 3.0 | °C | Reject distant reports, zero disables; S |
| `--min-separation-C` | 0.25 | °C | Positive-count report duplicate filter; S |
| `--select-mode` | `first-valid` | Category | `first-valid` or `strongest`; S |

### Backtracking and valley reporting

| Argument | Default | Units | Purpose / effect |
|---|---|---|---|
| `--method` | `hybrid` | Category | `hybrid`, `departure`, `intersection`, `backtrack`, `peak`; S |
| `--backtrack-frac` | 0.10 | Fraction | Simple derivative fallback, requires 0 < value ≤ 1; S |
| `--no-interpeak-valley-onsets` | False | Flag | Disable both valley paths; S |
| `--interpeak-rise-frac` | 0.03 | Fraction | Rough geometry, CLI rejects negative values; S |
| `--interpeak-report-rise-frac` | 0.10 | Fraction | Final valley crossing, finite [0,1]; S |
| `--forced-valley-report-floor` | 0.05 | Fraction | Forced-only post-cap floor, finite [0,1]; S |
| `--interpeak-min-neighbor-peak-frac` | 0.20 | Fraction | Explicit valley neighbor strength, nonnegative CLI check; S |

### Departure and intersection

| Argument | Default | Units | Purpose / effect |
|---|---|---|---|
| `--pre-width-C` | 1.0 | °C | Baseline window width; S |
| `--pre-gap-C` | 0.04 | °C | Rough-to-baseline gap; S |
| `--trans-width-C` | 0.80 | °C | Transition fitting geometry; S |
| `--trans-gap-C` | 0.02 | °C | Rough-to-transition gap; S |
| `--min-fit-points` | 8 | Samples | Minimum per fit, CLI requires ≥2; S |
| `--departure-k` | 5.0 | Ratio | Residual-noise multiplier; S |
| `--departure-min-abs` | 0.5 | Signal | Minimum baseline departure; S |
| `--hybrid-disagree-C` | 0.35 | °C | Departure/intersection disagreement threshold; S |

### Plotting, views and diagnostics

| Argument | Default | Units | Purpose / effect |
|---|---|---|---|
| `--no-reverse-x` | False | Flag | Disable cooling-axis display reversal; P |
| `--title` | `PCHIP Bend Point Analysis` | Text | Base title; P |
| `--no-title` / `--show-title` | Titles hidden | Flags | Toggle supported plot titles; P |
| `--bw` | False | Flag | Grayscale/BW styling, not true 1-bit guarantee; P |
| `--dpi` | 300 | Dots/inch | Plot resolution; P |
| `--plot-mode` | `smooth` | Category | Single-mode choice: smooth/raw/raw-smooth/diagnostic-main/diagnostic/candidates; P/O |
| `--plot-set` | `default` | Category | default/single/all; P/O |
| `--candidate-diagnostics` | False | Flag | Extra candidate CSV and plot; O/P |
| `--same-aspect-diagnostics` | False | Flag | Main-plot canvas aspect for diagnostics; P |
| `--publication-plots` | False | Flag | Eight additional publication variants; P/O |
| `--show-diagnostics` | False | Flag | Helper geometry/fit annotations; P |
| `--verbose` | False | Flag | Parsed, but no subsequent `args.verbose` use in current script; no additional behavior established |
| `--x-range` | Unset | Two °C bounds | Display range; P |
| `--x-lower`, `--x-upper` | Unset each | °C | Independent display bounds; P |
| `--mgi-y-range` | Unset | Two signal bounds | Primary MGI display limits; P |
| `--mgi-y-lower`, `--mgi-y-upper` | Unset each | Signal | Independent MGI display bounds; P |
| `--fbrm-y-range` | Unset | Two count bounds | Shared parser option, no FBRM axis in this detector; metadata/routing only |
| `--fbrm-y-lower`, `--fbrm-y-upper` | Unset each | Counts | Shared parser options; no MGI numerical effect |
| `--view-output-dir` | Automatic view directory | Path | Limited-output destination; O |

Range pairs cannot be mixed with their corresponding independent bounds. The shared parser rejects lower > upper but does not normalize reversed pairs or reject equal bounds there. Finite bounds are required by later view-ID formatting; this is not uniform early validation. Use finite increasing bounds. A missing side remains automatic. MGI manual y-limits do not constrain derivative subplots. Other legacy scientific parameters do not all have comprehensive finite/range validation; do not interpret an accepted CLI value as scientific validation.

## Parameter units and transferability

Fractions, ratios and MAD multipliers are dimensionless; their individual formulas are invariant to positive multiplicative signal scaling, but the entire algorithm is **not** scale-invariant. Temperature gaps/widths/separations are in °C, edge fraction is a sample fraction, and confirmation/fit counts are in samples.

Amplitude-dependent terms include `--departure-min-abs` (signal), the internal derivative-background minimum 1.0 (signal/°C), and the near-parallel slope tolerance $10^{-12}$ (signal/°C). These must not silently be interpreted in particle-count units or another signal scale. Peak prominence and height are derivative quantities, though their configured thresholds are relative. The code supplies no automatic cross-signal calibration or uncertainty estimate.

## RAW vs SG branch

RAW uses `BW`; SG control uses upstream `BW_smooth`. Each receives its own PCHIP construction in `xy-pchip.py`, then independent execution of the same detector. SG filtering precedes PCHIP; neither the SG measurement CSV nor its quick-look bend markers substitute for the saved SG PCHIP result. The detector's filename-derived branch label alone cannot detect a wrongly selected signal column. Preserve explicit workflow column choices and both saved curves.

## Presentation-only settings

Axis bounds, axis reversal, colors, fixed plotting linewidths, titles, legends, helper lines, point visibility and tick layout do not enter `detect_bends()`. Display bounds do not crop the numerical curve before detection. The detector nevertheless **runs analysis** whenever invoked, including a limited-view invocation; it is not saved-results-only replay.

For styling existing accepted results without analysis, use [render-publication-figures.py](../scripts/render-publication-figures.py). Its full/limited controls, onset shown/hidden options and linewidth settings are a separate CLI; do not assume those switches exist in this detector. `--raw-csv` can change plotted acquisition extent/points without changing the analyzed PCHIP. Input x/y selection, preprocessing, and grid construction are scientific choices, never presentation settings.

## Known limitations / interpretation

- A positive requested onset count is a maximum; no guarantee of two valid reports exists.
- `--bends 0` bypasses rejection in current code and needs separate interpretation.
- Absolute derivative anchors do not prove a particular physical event or intensity-change direction.
- Grid density affects derivative samples, nearest-index reporting, distance conversion and three-sample confirmation. Nonuniform grids are supported numerically but use median spacing for search distances.
- Saved PCHIP preparation and SG settings can change results; SG is not equivalent to PCHIP.
- A continuous intersection may have a sampled derivative evaluated at a nearby temperature. CSV rounding also limits directly reported precision.
- The forced floor is local to one anchor and provenance; it does not globally enforce derivative floors or ceilings on every final report.
- Candidate diagnostics are necessary to distinguish rejected alternatives from accepted ranks. Summary and view metadata alone do not encode a complete replay contract.

## Relationship to other scripts

| Script | Role relative to this detector |
|---|---|
| [xy-pchip.py](../scripts/xy-pchip.py) | Constructs and saves the prepared PCHIP grid; separate scientific stage |
| [tr-bw-sg.py](../scripts/tr-bw-sg.py) | Creates SG control measurement signal and quick-look plot upstream |
| [run-bw-pchip-workflow.py](../scripts/run-bw-pchip-workflow.py) | Orchestrates RAW/SG PCHIP and detector for an analysis directory; creates workflow summaries/logs |
| [run-onset-workflow.py](../scripts/run-onset-workflow.py) | Broader code-root/data-root orchestration; preserves RAW BW and SG BW_smooth choices |
| [render-publication-figures.py](../scripts/render-publication-figures.py) | Renders saved MGI curves and accepted bends; does not rerun this detector |

Documentation maintenance observations are tracked in the [MGI reference review follow-up](integration-review.md#documentation-follow-up-identified-during-mgi-reference-review).
