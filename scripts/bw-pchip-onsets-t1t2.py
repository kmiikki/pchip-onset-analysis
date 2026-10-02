#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bw-pchip-onsets-t1t2.py
------------------------

PCHIP onset analysis for BW vs. temperature curves.

Scientific purpose
==================
This script estimates physically meaningful onset temperatures from BW vs.
temperature curves produced by the PCHIP workflow. The main production target is
usually two onsets:

    T1 = first physically significant onset during cooling
    T2 = second physically significant onset during cooling

The important methodological rule is that ``--bends 2`` means "report at most
the first two valid physical onsets". It does not mean "force two points" and it
does not mean "return the two strongest transitions".

Typical workflow
================
The script is normally run in an ``analysis/pchip`` directory containing one or
both of these files:

    bw-raw-vs-temp-pchip.csv
    bw-sg-vs-temp-pchip.csv

These files are usually generated from:

    rgb-tr.csv / rgb-tr-sg.csv
        -> xy-pchip.py
        -> bw-raw-vs-temp-pchip.csv / bw-sg-vs-temp-pchip.csv
        -> bw-pchip-onsets-t1t2.py
        -> onset temperatures and diagnostic plots

The raw-PCHIP result is intended to be the primary method. The SG-PCHIP result is
primarily a control/diagnostic method. Agreement between raw-PCHIP and SG-PCHIP
is strong evidence that the reported onset is robust.

Coordinate convention
=====================
Input data are sorted internally by increasing temperature. Cooling experiments,
however, are interpreted from high temperature to low temperature. Therefore:

    higher-temperature side  = larger array index
    lower-temperature side   = smaller array index
    cooling order            = descending temperature

Most comments in the numerical code explicitly refer to this convention because
getting the direction wrong is the easiest way to move an onset to the wrong side
of a transition.

Method overview
===============
For each PCHIP curve the script does the following:

1. Compute first and second derivatives with respect to temperature.
2. Detect many derivative-peak anchors from ``|dY/dT|``.
3. For each anchor, estimate a candidate onset using a hybrid of:
   - derivative-background departure/backtracking,
   - baseline-departure detection, and
   - local segmented-line intersection.
4. Add optional inter-peak valley-rise candidates for split/multistage events.
5. Reject clearly nonlocal or too-weak final candidates.
6. Select the first valid candidates in cooling order by default.

Candidate diagnostics are intentionally more permissive than final selection.
This is deliberate: the candidate plot answers "what did the algorithm see?",
whereas the result plot reports only the accepted physical onsets.

Mathematical methods
====================
The main numerical methods are documented in their individual function
docstrings. The central ideas are:

- ``np.gradient`` estimates ``dY/dT`` and ``d2Y/dT2`` on the PCHIP grid.
- ``scipy.signal.find_peaks`` finds derivative-peak anchors from ``|dY/dT|``.
- local linear least-squares fits model the pre-transition baseline and the
  transition segment.
- the intersection of those local lines is used only when it is local and
  numerically valid.
- robust derivative-background thresholds use median/MAD estimates so that one
  steep transition does not define all onset thresholds.

Outputs
=======
For each input CSV:

    <input-stem>-bends.csv
    <input-stem>-bends.png

With ``--candidate-diagnostics``:

    <input-stem>-candidates.csv
    <input-stem>-candidate-diagnostics.png

If multiple input files are analyzed, a combined summary is also written:

    bw-pchip-bend-summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from onset_config import ensure_onset_config, infer_analysis_dir_from_paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from onset_view_limits import (
    NONE_LIMITS,
    ViewLimits,
    add_view_limit_arguments,
    apply_signal_y_view_limits,
    apply_temperature_view_limits,
    parse_view_limits_from_args,
    resolve_output_dir_for_limits,
    write_view_limits_json,
)
from matplotlib.ticker import MultipleLocator, NullLocator

try:
    from scipy.signal import find_peaks
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[error] scipy is required for peak detection. "
        "Install scipy or run this in the lab314 environment."
    ) from exc



# ---------------------------------------------------------------------------
# Default file names
# ---------------------------------------------------------------------------
DEFAULT_INPUTS = (
    "bw-raw-vs-temp-pchip.csv",
    "bw-sg-vs-temp-pchip.csv",
)



# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitLine:
    """Least-squares line segment ``y = slope*x + intercept``.

    The ``x_min`` and ``x_max`` fields record the actual fit window. They are
    used when plotting diagnostic line segments so that the visualized line is
    only drawn over the data range that was used for the fit.
    """
    slope: float
    intercept: float
    n_points: int
    x_min: float
    x_max: float


@dataclass(frozen=True)
class ValleyReportProvenance:
    """Keep report qualification separate from the geometry used by the fits."""
    interpeak_rise_frac: float = 0.03
    interpeak_report_rise_frac: float = 0.10
    forced_valley_report_floor: float = 0.05
    forced_valley_precedence: bool = False
    valley_geometry_temperature_C: float | None = None
    valley_qualified_report_temperature_C: float | None = None
    post_derivative_cap_temperature_C: float | None = None
    forced_valley_floor_applied: bool = False


@dataclass(frozen=True)
class BendPoint:
    """Final accepted bend/onset point written to the result CSV.

    ``temperature_C`` is the reported onset temperature. ``rough_onset_*`` is
    the derivative/backtracking estimate, while ``peak_*`` is the derivative
    peak that anchored this onset. Keeping all three positions is important for
    diagnostics: a valid onset should remain local relative to its peak anchor.
    """
    rank: int
    temperature_C: float
    value: float

    rough_onset_temperature_C: float
    rough_onset_value: float

    peak_temperature_C: float
    peak_value: float

    derivative_at_bend: float
    derivative_at_peak: float
    abs_derivative_at_bend: float
    abs_derivative_at_peak: float
    second_derivative_at_bend: float
    prominence: float

    method: str
    report_mode: str
    input_csv: str

    bend_point_index: int
    rough_onset_point_index: int
    peak_point_index: int

    pre_line: FitLine | None
    transition_line: FitLine | None
    used_fallback: bool
    valley_report: ValleyReportProvenance | None = None


@dataclass
class CandidatePoint:
    """Candidate onset before final rank selection.

    Candidate records deliberately keep accepted and rejected points together.
    This makes the candidate CSV and diagnostic plot explain both the final
    result and the rejected alternatives. Final reporting is controlled by the
    ``accepted``, ``final_rank``, ``status`` and ``reason`` fields.
    """
    candidate_id: int
    temperature_C: float
    value: float

    rough_onset_temperature_C: float
    rough_onset_value: float

    peak_temperature_C: float
    peak_value: float

    derivative_at_report: float
    derivative_at_peak: float
    abs_derivative_at_report: float
    abs_derivative_at_peak: float
    second_derivative_at_report: float
    prominence: float
    report_derivative_fraction: float

    method: str
    report_mode: str
    input_csv: str
    selection_mode: str

    report_point_index: int
    rough_onset_point_index: int
    peak_point_index: int

    pre_line: FitLine | None
    transition_line: FitLine | None
    used_fallback: bool

    candidate_rank_by_strength: int = 0
    candidate_rank_by_cooling_order: int = 0
    accepted: bool = False
    final_rank: int | None = None
    status: str = "candidate"
    reason: str = ""
    valley_report: ValleyReportProvenance | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis result for one input CSV.

    The object carries both numerical results and paths to the files written for
    this input. Plotting and CSV writing are kept outside the detection logic so
    that the mathematical part stays easier to inspect and test.
    """
    input_csv: Path
    method: str
    xcol: str
    ycol: str
    x: np.ndarray
    y: np.ndarray
    dy_dx: np.ndarray
    d2y_dx2: np.ndarray
    bends: list[BendPoint]
    candidates: list[CandidatePoint]
    out_csv: Path
    out_png: Path
    out_candidates_csv: Path | None
    out_candidates_png: Path | None



# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect bend points from BW vs. temperature PCHIP curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help=(
            "Input PCHIP CSV file(s). If omitted, the script looks for "
            "bw-raw-vs-temp-pchip.csv and bw-sg-vs-temp-pchip.csv in the current directory."
        ),
    )
    parser.add_argument("--xcol", type=str, default=None,
                        help="Temperature column name. Default: first CSV column.")
    parser.add_argument("--ycol", type=str, default=None,
                        help="BW column name. Default: second CSV column.")

    parser.add_argument("--bends", type=int, default=None,
                        help="Maximum number of bend points to report per file. Use 0 for all accepted peaks.")
    parser.add_argument(
        "--select-mode",
        choices=["first-valid", "strongest"],
        default="first-valid",
        help=(
            "Candidate selection mode. first-valid reports the first physically "
            "valid onsets in cooling order; strongest reports the strongest "
            "transitions for diagnostics."
        ),
    )
    parser.add_argument(
        "--candidate-min-separation-C",
        type=float,
        default=0.10,
        help=(
            "Minimum temperature separation between derivative-peak anchors during "
            "candidate search. This should be small enough that nearby T1/T2 "
            "peaks are not suppressed before candidate selection."
        ),
    )
    parser.add_argument(
        "--min-separation-C",
        type=float,
        default=0.25,
        help=(
            "Minimum temperature separation between final reported onset "
            "temperatures. This duplicate filter is applied after candidate "
            "detection. The default removes near-duplicate report points from "
            "the same transition while still allowing close real T1/T2 stages; "
            "derivative anchor search is controlled separately by "
            "--candidate-min-separation-C."
        ),
    )
    parser.add_argument("--min-height-frac", type=float, default=0.05,
                        help=("Minimum |dY/dT| peak height as a fraction of maximum |dY/dT|. "
                              "Lower default helps detect smaller bend points when another transition is very steep."))
    parser.add_argument("--prominence-frac", type=float, default=0.05,
                        help="Minimum peak prominence as a fraction of maximum |dY/dT|.")
    parser.add_argument(
        "--min-final-peak-height-frac",
        type=float,
        default=0.15,
        help=(
            "Minimum derivative-peak height for candidates that may be selected "
            "as final reported onsets, as a fraction of a robust/capped "
            "reference |dY/dT| peak. Candidate search remains more permissive "
            "so the diagnostic plot can show weak candidates, but first-valid "
            "selection does not let a weak intermediate shoulder consume the "
            "next T-rank before a later stronger transition. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--final-peak-dominance-factor",
        type=float,
        default=5.0,
        help=(
            "Cap the final peak-height reference to this factor times the "
            "second strongest candidate peak. This prevents one dominant "
            "near-vertical transition from suppressing earlier valid onsets. "
            "Use 0 to disable capping and use the global maximum directly."
        ),
    )
    parser.add_argument("--edge-frac", type=float, default=0.02,
                        help="Ignore this fraction of points at both ends to avoid edge artifacts.")

    parser.add_argument("--method", choices=["hybrid", "departure", "intersection", "backtrack", "peak"],
                        default="hybrid",
                        help=(
                            "Bend reporting method: hybrid = use baseline-departure when it "
                            "disagrees strongly with segmented-line intersection, departure = "
                            "baseline-departure onset, intersection = local segmented-line "
                            "intersection, backtrack = derivative threshold onset, peak = "
                            "derivative maximum."
                        ))
    parser.add_argument("--backtrack-frac", type=float, default=0.10,
                        help=(
                            "Rough onset threshold. Backtrack from derivative peak to the "
                            "higher-temperature side until |dY/dT| falls below "
                            "backtrack_frac * peak(|dY/dT|)."
                        ))
    parser.add_argument(
        "--no-interpeak-valley-onsets",
        action="store_true",
        help=(
            "Disable inter-peak valley-departure onset adjustment. By default, "
            "when a later derivative peak starts from a clear valley between two "
            "peaks, the onset is allowed to move back to the point where |dY/dT| "
            "starts rising from that valley. This helps split close T1/T2 stages."
        ),
    )
    parser.add_argument(
        "--interpeak-rise-frac",
        type=float,
        default=0.03,
        help=(
            "Valley rise fraction for rough geometry, fitting windows and forced "
            "eligibility; final reports use --interpeak-report-rise-frac."
        ),
    )
    parser.add_argument(
        "--interpeak-report-rise-frac", type=float, default=0.10,
        help="Final forced/explicit valley report fraction (0..1); does not change rough geometry.",
    )
    parser.add_argument(
        "--forced-valley-report-floor", type=float, default=0.05,
        help="Anchor-relative derivative floor for forced-valley reports only (0..1; 0 disables).",
    )
    parser.add_argument(
        "--interpeak-min-neighbor-peak-frac",
        type=float,
        default=0.20,
        help=(
            "Minimum required height for both derivative peaks surrounding an "
            "inter-peak valley-rise candidate, as a fraction of the global maximum "
            "|dY/dT| peak. This prevents one strong peak plus a small raw-noise "
            "shoulder from being split into two onsets."
        ),
    )
    parser.add_argument(
        "--max-report-peak-gap-C",
        type=float,
        default=3.0,
        help=(
            "Reject a candidate if its reported onset temperature is farther than "
            "this from the derivative peak that supposedly anchors it. This blocks "
            "raw-data artifacts where a late broad/noisy peak produces a line "
            "intersection back at an earlier unrelated transition. Use 0 to disable."
        ),
    )

    parser.add_argument("--pre-width-C", type=float, default=1.0,
                        help="Width of higher-temperature pre-transition line fit window.")
    parser.add_argument("--pre-gap-C", type=float, default=0.04,
                        help="Gap between rough onset and pre-transition fit window.")
    parser.add_argument("--trans-width-C", type=float, default=0.80,
                        help="Approximate width of transition-line fit window.")
    parser.add_argument("--trans-gap-C", type=float, default=0.02,
                        help="Gap between rough onset and transition fit window.")
    parser.add_argument("--min-fit-points", type=int, default=8,
                        help="Minimum points required for each local line fit.")
    parser.add_argument("--departure-k", type=float, default=5.0,
                        help="Baseline-departure threshold multiplier based on pre-transition residual noise.")
    parser.add_argument("--departure-min-abs", type=float, default=0.5,
                        help="Minimum absolute BW departure from the pre-transition line.")
    parser.add_argument("--hybrid-disagree-C", type=float, default=0.35,
                        help="In hybrid mode, use baseline-departure onset if it differs from intersection by more than this many °C.")

    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory. Default: same directory as each input file.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Analysis directory containing onset-config.ini. If omitted, the "
            "script infers it from --outdir or the input CSV paths."
        ),
    )
    parser.add_argument("--summary", type=Path, default=Path("bw-pchip-bend-summary.csv"),
                        help="Combined summary CSV path when multiple files are analyzed.")
    parser.add_argument("--no-reverse-x", action="store_true",
                        help="Do not reverse the temperature axis in plots.")
    parser.add_argument("--title", type=str, default="PCHIP Bend Point Analysis",
                        help="Base plot title.")
    parser.add_argument(
        "--no-title",
        dest="no_title",
        action="store_true",
        default=True,
        help="Do not show plot titles. Default for manuscript/panel figures.",
    )
    parser.add_argument(
        "--show-title",
        dest="no_title",
        action="store_false",
        help="Show plot titles for standalone audit/debug figures.",
    )
    parser.add_argument(
        "--bw",
        action="store_true",
        help=(
            "Black-and-white / grayscale plotting mode. Useful for reports and "
            "publications that should not depend on color."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution. Useful values: 150 for quick review, 300 default, 600 publication/high-res.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["smooth", "raw", "raw-smooth", "diagnostic-main", "diagnostic", "candidates"],
        default="smooth",
        help=(
            "Plot content used with --plot-set single: smooth = PCHIP curve only, "
            "raw = original source points only, raw-smooth = source points plus "
            "PCHIP curve, diagnostic-main = diagnostic main plot without derivative subplot, "
            "diagnostic = full diagnostic plot with derivative subplot."
        ),
    )
    parser.add_argument("--plot-set", choices=["default", "single", "all"],
                        default="default",
                        help=(
                            "Which plots to write. default writes clean smooth, raw-smooth, "
                            "diagnostic-main, and diagnostic plots. single writes only --plot-mode. "
                            "all writes smooth, raw, raw-smooth, diagnostic-main, and diagnostic."
                        ))
    parser.add_argument(
        "--candidate-diagnostics",
        action="store_true",
        help=(
            "Write an additional all-candidates diagnostic plot and CSV. "
            "This does not modify the normal result plots."
        ),
    )
    parser.add_argument(
        "--same-aspect-diagnostics",
        action="store_true",
        help=(
            "Use the same canvas aspect ratio for diagnostic/candidate-diagnostic "
            "figures as for the main/publication figures. Default keeps "
            "diagnostic figures taller for QC reading."
        ),
    )
    parser.add_argument(
        "--publication-plots",
        action="store_true",
        help=(
            "Write additional manuscript-style MGI publication plots: "
            "clean mono, main mono, onset mono, and clean-onset mono variants. "
            "These plots do not change onset detection or CSV outputs."
        ),
    )
    parser.add_argument("--show-diagnostics", action="store_true",
                        help=(
                            "Show diagnostic helper lines in non-diagnostic plots: rough backtracked "
                            "onset, derivative peak anchors, and fitted line segments. Final bend "
                            "lines are always shown."
                        ))
    parser.add_argument("--raw-csv", type=Path, default=None,
                        help=(
                            "Optional original source CSV for raw/raw-smooth plot modes. "
                            "If omitted, the script tries ../rgb-tr.csv or ../rgb-tr-sg.csv."
                        ))
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra diagnostic information.")

    add_view_limit_arguments(parser)

    return parser.parse_args()



# ---------------------------------------------------------------------------
# Input handling and basic numerical helpers
# ---------------------------------------------------------------------------
def autodetect_inputs(csv_files: Sequence[Path]) -> list[Path]:
    """Return explicit inputs or default inputs found in the current directory."""
    if csv_files:
        return [p for p in csv_files]

    found = [Path(name) for name in DEFAULT_INPUTS if Path(name).exists()]
    if not found:
        raise FileNotFoundError(
            "No input CSV files given and no default PCHIP CSV files found.\n"
            f"Looked for: {', '.join(DEFAULT_INPUTS)}"
        )
    return found


def infer_method(csv_path: Path, ycol: str) -> str:
    """Infer method label from filename and y column."""
    name = csv_path.name.lower()
    if "sg" in name or "smooth" in ycol.lower():
        return "sg_pchip"
    if "raw" in name:
        return "raw_pchip"
    return "pchip"


def read_xy(csv_path: Path, xcol: str | None, ycol: str | None) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Read temperature and BW columns from a PCHIP CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8")

    if df.shape[1] < 2:
        raise ValueError(f"Input CSV must have at least two columns: {csv_path}")

    columns = list(df.columns)
    x_name = xcol if xcol is not None else columns[0]
    y_name = ycol if ycol is not None else columns[1]

    if x_name not in df.columns:
        raise KeyError(f"X column '{x_name}' not found in {csv_path}; columns: {columns}")
    if y_name not in df.columns:
        raise KeyError(f"Y column '{y_name}' not found in {csv_path}; columns: {columns}")

    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        raise ValueError(f"Not enough numeric data points in {csv_path} after NaN removal.")

    # Sort by temperature ascending for derivative calculations.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse duplicate temperatures defensively.
    unique_x, start_idx = np.unique(x, return_index=True)
    if len(unique_x) != len(x):
        y_means: list[float] = []
        counts = np.diff(np.append(start_idx, len(x)))
        ptr = 0
        for count in counts:
            y_means.append(float(np.mean(y[ptr:ptr + count])))
            ptr += count
        x = unique_x
        y = np.array(y_means, dtype=float)

    return x_name, y_name, x, y


def estimate_temperature_step(x: np.ndarray) -> float:
    """Estimate a representative temperature-grid spacing.

    The PCHIP files are expected to have a mostly regular temperature grid, but
    this helper uses the median nonzero spacing so that a few duplicated or
    irregular points do not dominate distance-to-index conversions.
    """
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    dx = np.abs(dx[dx != 0])
    if dx.size == 0:
        return 1.0
    return float(np.median(dx))


def compute_derivatives(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute numerical first and second derivatives.

    ``dy_dx`` is ``dY/dT`` and ``d2y_dx2`` is ``d2Y/dT2`` on the sorted PCHIP
    grid. ``np.gradient`` is used because it accepts nonuniform x spacing and
    provides a simple central-difference estimate for interior points.
    """
    dy_dx = np.gradient(y, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    return dy_dx, d2y_dx2


def nearest_index(x: np.ndarray, value: float) -> int:
    """Return the index of the grid point closest to ``value``.

    Many onset estimates are first computed as floating-point temperatures.
    Candidate records also need a derivative value, so the temperature is mapped
    back to the nearest PCHIP-grid index.
    """
    return int(np.nanargmin(np.abs(x - value)))


def value_at_x(x: np.ndarray, y: np.ndarray, x_value: float) -> float:
    """Linearly interpolate ``y`` at ``x_value`` within the data range.

    The reported onset temperature can fall between PCHIP grid points, for
    example after a line-line intersection. The value reported in the CSV is the
    interpolated BW/BW_smooth value at that temperature.
    """
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    xv = min(max(float(x_value), x_min), x_max)
    return float(np.interp(xv, x, y))


def fit_line(x: np.ndarray, y: np.ndarray, mask: np.ndarray, min_points: int) -> FitLine | None:
    """Fit ``y = m*x + b`` using the points selected by ``mask``.

    This is an ordinary least-squares first-degree polynomial fit. It is used for
    both the local pre-transition baseline and the local transition segment. The
    function returns ``None`` instead of raising when the window is too small or
    degenerate; the caller can then fall back to a derivative-based onset.
    """
    idx = np.where(mask & np.isfinite(x) & np.isfinite(y))[0]
    if idx.size < min_points:
        return None

    xs = x[idx]
    ys = y[idx]

    if np.nanmax(xs) <= np.nanmin(xs):
        return None

    m, b = np.polyfit(xs, ys, 1)
    return FitLine(
        slope=float(m),
        intercept=float(b),
        n_points=int(idx.size),
        x_min=float(np.nanmin(xs)),
        x_max=float(np.nanmax(xs)),
    )


def line_value(line: FitLine, x_value: float) -> float:
    """Evaluate a fitted ``FitLine`` at one temperature value."""
    return float(line.slope * x_value + line.intercept)


def line_intersection_x(line1: FitLine, line2: FitLine) -> float | None:
    """Return the x-coordinate where two fitted lines intersect.

    For lines ``m1*x + b1`` and ``m2*x + b2``, the intersection is
    ``(b2 - b1) / (m1 - m2)``. A near-zero slope difference is treated as an
    invalid intersection because almost parallel lines would place the onset far
    outside the local transition region.
    """
    denom = line1.slope - line2.slope
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return None

    x_int = (line2.intercept - line1.intercept) / denom
    if not np.isfinite(x_int):
        return None
    return float(x_int)



# ---------------------------------------------------------------------------
# Onset estimation primitives
# ---------------------------------------------------------------------------
def backtrack_onset_index(abs_d: np.ndarray, peak_idx: int, frac: float) -> int:
    """
    Backtrack from a derivative peak toward the higher-temperature side.

    This is the simple fractional-peak backtracking rule. Starting at a local
    maximum of ``|dY/dT|``, the scan moves toward higher temperature, i.e. toward
    the start of the transition during cooling, until the derivative has fallen
    below ``frac * peak_height``. The returned index is the last point still
    inside the transition according to that threshold.

    This function is conservative and purely local. More robust baseline-aware
    backtracking is implemented in ``backtrack_derivative_departure_index``.
    """
    peak_height = float(abs_d[peak_idx])
    if not np.isfinite(peak_height) or peak_height <= 0:
        return peak_idx

    threshold = float(frac) * peak_height

    j = int(peak_idx)
    while j < len(abs_d) - 1 and np.isfinite(abs_d[j]) and abs_d[j] >= threshold:
        j += 1

    return max(peak_idx, j - 1)


def backtrack_derivative_departure_index(
    *,
    x: np.ndarray,
    abs_d: np.ndarray,
    peak_idx: int,
    fallback_frac: float,
    baseline_gap_C: float = 0.4,
    baseline_width_C: float = 3.0,
    k_mad: float = 6.0,
    min_peak_frac: float = 0.03,
    min_abs_threshold: float = 1.0,
) -> int:
    """
    Estimate the derivative-departure onset for one peak anchor.

    The simple fractional rule uses ``fallback_frac * peak_height``. That works
    reasonably for many smooth transitions but can put the onset too far inside a
    very steep step. This function therefore estimates a local derivative
    background on the higher-temperature side of the peak and uses a robust
    threshold:

        background_median + max(k_mad * MAD_sigma,
                                min_peak_frac * peak_height,
                                min_abs_threshold)

    where ``MAD_sigma = 1.4826 * median(abs(x - median(x)))``. The MAD term
    adapts to local derivative noise, the peak-fraction term prevents an
    unrealistically tiny threshold for strong peaks, and the absolute term keeps
    the rule stable when the local derivative background is almost flat.

    The return value is never allowed to move deeper into the transition than the
    simple fallback. It may only move the onset earlier in cooling order, i.e. to
    the higher-temperature side.
    """
    peak_height = float(abs_d[peak_idx])
    if not np.isfinite(peak_height) or peak_height <= 0:
        return int(peak_idx)

    # Conservative fallback from the old method.
    fallback_idx = backtrack_onset_index(abs_d, peak_idx, fallback_frac)

    dx = estimate_temperature_step(x)
    if not np.isfinite(dx) or dx <= 0:
        return fallback_idx

    gap_n = max(1, int(round(float(baseline_gap_C) / dx)))
    width_n = max(5, int(round(float(baseline_width_C) / dx)))

    # Local derivative background on the higher-temperature side.
    # Start after a small gap so the peak shoulder is not used as baseline.
    lo = min(len(abs_d), int(peak_idx) + gap_n)
    hi = min(len(abs_d), lo + width_n)

    bg = abs_d[lo:hi]
    bg = bg[np.isfinite(bg)]

    # If the local window is too short, use whatever exists to the right.
    if bg.size < 5:
        bg = abs_d[int(peak_idx) + 1:]
        bg = bg[np.isfinite(bg)]

    if bg.size >= 5:
        bg_med = float(np.median(bg))
        bg_mad = float(np.median(np.abs(bg - bg_med)))
        bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else float(np.std(bg))
    else:
        bg_med = 0.0
        bg_sigma = 0.0

    threshold = bg_med + max(
        float(k_mad) * bg_sigma,
        float(min_peak_frac) * peak_height,
        float(min_abs_threshold),
    )

    if not np.isfinite(threshold) or threshold <= 0:
        return fallback_idx

    # Move from peak toward the higher-temperature side until derivative
    # returns below the local-background threshold.
    j = int(peak_idx)
    while j < len(abs_d) - 1 and np.isfinite(abs_d[j]) and abs_d[j] >= threshold:
        j += 1

    departure_idx = max(int(peak_idx), j - 1)

    # Do not allow the new method to move deeper into the transition than the
    # old conservative method. It may only move the onset earlier/higher-T.
    return max(departure_idx, fallback_idx)




def interpeak_valley_departure_index(
    *,
    x: np.ndarray,
    abs_d: np.ndarray,
    peak_idx: int,
    all_peak_indices: Sequence[int],
    rise_frac: float = 0.03,
    max_neighbor_gap_C: float = 2.0,
    min_valley_drop_frac: float = 0.20,
) -> int | None:
    """
    Estimate an onset from a valley between two nearby derivative peaks.

    Multi-stage transitions can contain two adjacent ``|dY/dT|`` peaks separated
    by a local valley. For the later/lower-temperature peak, the physically
    useful onset can be the point where the derivative starts rising again after
    that valley, not the point produced by ordinary peak backtracking.

    The function only returns a valley onset when the neighboring peaks are
    close enough and the valley is deep enough. This avoids splitting one noisy
    shoulder of a single transition into a false new onset.
    """
    if not all_peak_indices:
        return None

    peak_idx = int(peak_idx)
    higher_side = sorted(int(i) for i in all_peak_indices if int(i) > peak_idx)
    if not higher_side:
        return None

    prev_peak_idx = higher_side[0]
    gap_C = float(abs(x[prev_peak_idx] - x[peak_idx]))
    if not np.isfinite(gap_C) or gap_C <= 0.0 or gap_C > max_neighbor_gap_C:
        return None

    lo = min(peak_idx, prev_peak_idx)
    hi = max(peak_idx, prev_peak_idx)
    if hi - lo < 3:
        return None

    segment = abs_d[lo:hi + 1]
    if not np.any(np.isfinite(segment)):
        return None

    valley_idx = lo + int(np.nanargmin(segment))
    if valley_idx <= peak_idx or valley_idx >= prev_peak_idx:
        return None

    peak_abs = float(abs_d[peak_idx])
    prev_abs = float(abs_d[prev_peak_idx])
    valley_abs = float(abs_d[valley_idx])
    if not (np.isfinite(peak_abs) and np.isfinite(prev_abs) and np.isfinite(valley_abs)):
        return None
    if peak_abs <= valley_abs:
        return None

    # Require that the valley is a real valley, not just a tiny shoulder.
    smaller_neighbor_peak = min(peak_abs, prev_abs)
    if smaller_neighbor_peak <= 0.0:
        return None
    valley_drop_frac = (smaller_neighbor_peak - valley_abs) / smaller_neighbor_peak
    if valley_drop_frac < min_valley_drop_frac:
        return None

    threshold = valley_abs + max(0.0, float(rise_frac)) * (peak_abs - valley_abs)

    # Walk from the valley toward the current lower-temperature peak. The first
    # sustained rise is the onset of the next stage. With very small rise_frac
    # this stays close to the valley minimum, which matches the visual rule.
    for idx in range(valley_idx, peak_idx - 1, -1):
        value = float(abs_d[idx])
        if np.isfinite(value) and value >= threshold:
            return int(idx)

    return int(valley_idx)


def build_interpeak_valley_candidates(
    *,
    x: np.ndarray,
    y: np.ndarray,
    dy_dx: np.ndarray,
    d2y_dx2: np.ndarray,
    abs_d: np.ndarray,
    peak_indices: Sequence[int],
    peak_prominence_by_index: dict[int, float],
    method: str,
    report_method: str,
    csv_path: Path,
    select_mode: str,
    rise_frac: float,
    min_neighbor_peak_frac: float = 0.20,
    max_neighbor_gap_C: float = 2.0,
    min_valley_drop_frac: float = 0.20,
    min_report_peak_distance_C: float = 0.05,
) -> list[CandidatePoint]:
    """
    Build explicit valley-rise onset candidates between adjacent derivative peaks.

    Peak-anchored candidates can report the same high-temperature onset for
    two nearby derivative peaks. In multi-stage transitions, however, the next
    physical onset may be the renewed rise from the |dY/dT| valley between two
    adjacent peaks. These candidates are therefore added as independent
    candidates, not as a replacement for the peak-anchored candidate.
    """
    if len(peak_indices) < 2:
        return []

    # Cooling order: high temperature to low temperature. x is ascending, so
    # higher temperature means larger index.
    peaks_cooling = sorted({int(i) for i in peak_indices}, key=lambda i: float(x[i]), reverse=True)
    out: list[CandidatePoint] = []

    global_peak_abs = float(np.nanmax(abs_d))
    if not np.isfinite(global_peak_abs) or global_peak_abs <= 0.0:
        return []

    for prev_peak_idx, next_peak_idx in zip(peaks_cooling[:-1], peaks_cooling[1:]):
        gap_C = float(abs(x[prev_peak_idx] - x[next_peak_idx]))
        if not np.isfinite(gap_C) or gap_C <= 0.0 or gap_C > max_neighbor_gap_C:
            continue

        lo = min(prev_peak_idx, next_peak_idx)
        hi = max(prev_peak_idx, next_peak_idx)
        if hi - lo < 3:
            continue

        segment = abs_d[lo:hi + 1]
        if not np.any(np.isfinite(segment)):
            continue

        valley_idx = lo + int(np.nanargmin(segment))
        if valley_idx <= lo or valley_idx >= hi:
            continue

        next_peak_abs = float(abs_d[next_peak_idx])
        prev_peak_abs = float(abs_d[prev_peak_idx])
        valley_abs = float(abs_d[valley_idx])
        if not (np.isfinite(next_peak_abs) and np.isfinite(prev_peak_abs) and np.isfinite(valley_abs)):
            continue
        if next_peak_abs <= valley_abs:
            continue

        smaller_neighbor_peak = min(prev_peak_abs, next_peak_abs)
        # Valley-rise onsets are only allowed between two sufficiently strong
        # adjacent derivative peaks. Otherwise a single strong transition plus
        # a small raw-data shoulder/noise feature may be incorrectly split into
        # two onsets. The ordinary peak-anchored candidates still remain
        # available; this filter only controls generated valley-rise candidates.
        min_neighbor_peak = float(min_neighbor_peak_frac) * global_peak_abs
        if smaller_neighbor_peak < min_neighbor_peak:
            continue

        if smaller_neighbor_peak <= 0.0:
            continue
        valley_drop_frac = (smaller_neighbor_peak - valley_abs) / smaller_neighbor_peak
        if valley_drop_frac < min_valley_drop_frac:
            continue

        threshold = valley_abs + max(0.0, float(rise_frac)) * (next_peak_abs - valley_abs)

        # From the valley, move in cooling direction toward the next/lower-T
        # peak. x is ascending, so this means decreasing index.
        report_idx = int(valley_idx)
        for idx in range(int(valley_idx), int(next_peak_idx) - 1, -1):
            value = float(abs_d[idx])
            if np.isfinite(value) and value >= threshold:
                report_idx = int(idx)
                break

        if abs(float(x[report_idx]) - float(x[next_peak_idx])) < min_report_peak_distance_C:
            continue

        peak_abs = float(abs_d[next_peak_idx])
        report_abs = float(abs_d[report_idx])
        derivative_fraction = float(report_abs / peak_abs) if peak_abs > 0 and np.isfinite(report_abs) else float("nan")
        prom = float(peak_prominence_by_index.get(int(next_peak_idx), peak_abs - valley_abs))

        candidate = CandidatePoint(
            candidate_id=-1,
            temperature_C=float(x[report_idx]),
            value=value_at_x(x, y, float(x[report_idx])),
            rough_onset_temperature_C=float(x[report_idx]),
            rough_onset_value=float(y[report_idx]),
            peak_temperature_C=float(x[next_peak_idx]),
            peak_value=float(y[next_peak_idx]),
            derivative_at_report=float(dy_dx[report_idx]),
            derivative_at_peak=float(dy_dx[next_peak_idx]),
            abs_derivative_at_report=float(abs_d[report_idx]),
            abs_derivative_at_peak=float(abs_d[next_peak_idx]),
            second_derivative_at_report=float(d2y_dx2[report_idx]),
            prominence=float(prom),
            report_derivative_fraction=derivative_fraction,
            method=method,
            report_mode=f"{report_method}+valley-rise",
            input_csv=csv_path.name,
            selection_mode=select_mode,
            report_point_index=int(report_idx),
            rough_onset_point_index=int(report_idx),
            peak_point_index=int(next_peak_idx),
            pre_line=None,
            transition_line=None,
            used_fallback=True,
            status="candidate",
            reason="inter-peak valley-rise candidate",
        )
        out.append(candidate)

    return out

def baseline_departure_for_peak(
    *,
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    rough_idx: int,
    pre_line: FitLine | None,
    pre_width_C: float,
    pre_gap_C: float,
    departure_k: float,
    departure_min_abs: float,
    min_fit_points: int,
) -> tuple[float | None, FitLine | None]:
    """
    Estimate onset as persistent departure from the pre-transition line.

    The pre-transition baseline is a local line on the higher-temperature side
    of the rough onset. Residuals in that fit window provide a noise estimate.
    The scan then follows cooling direction and reports the first point where a
    short run of points has departed from the baseline by at least:

        max(departure_min_abs, departure_k * robust_sigma)

    This is useful for steep transitions because it detects that the curve has
    left the baseline before the derivative reaches its maximum.
    """
    peak_T = float(x[peak_idx])
    rough_T = float(x[rough_idx])

    if pre_line is None:
        pre_lo = rough_T + pre_gap_C
        pre_hi = rough_T + pre_gap_C + pre_width_C
        pre_mask = (x >= pre_lo) & (x <= pre_hi)
        pre_line = fit_line(x, y, pre_mask, min_fit_points)

    if pre_line is None:
        return None, None

    # Estimate residual noise from the fitted pre-transition window.
    pre_mask = (x >= pre_line.x_min) & (x <= pre_line.x_max)
    residual = y[pre_mask] - (pre_line.slope * x[pre_mask] + pre_line.intercept)
    residual = residual[np.isfinite(residual)]
    if residual.size >= 3:
        # Robust sigma estimate; fallback to std if MAD degenerates.
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(residual))
    else:
        sigma = 0.0

    threshold = max(float(departure_min_abs), float(departure_k) * float(sigma))

    # Scan from high-temperature side toward peak. x is ascending, so use
    # descending indices from a point near pre_line.x_min down to peak_idx.
    start_T = min(pre_line.x_min, rough_T + pre_gap_C)
    start_idx = nearest_index(x, start_T)
    lo = min(int(peak_idx), int(start_idx))
    hi = max(int(peak_idx), int(start_idx))

    # Need a few consecutive/nearby points above threshold to avoid tiny blips.
    confirm_n = 3
    for idx in range(hi, lo - 1, -1):
        window_lo = max(lo, idx - confirm_n + 1)
        idxs = np.arange(window_lo, idx + 1)
        if idxs.size < confirm_n:
            continue
        y_line = pre_line.slope * x[idxs] + pre_line.intercept
        dev = np.abs(y[idxs] - y_line)
        if np.all(dev >= threshold):
            return float(x[idx]), pre_line

    return None, pre_line

def segmented_intersection_for_peak(
    *,
    x: np.ndarray,
    y: np.ndarray,
    abs_d: np.ndarray,
    peak_idx: int,
    rough_idx: int,
    pre_width_C: float,
    pre_gap_C: float,
    trans_width_C: float,
    trans_gap_C: float,
    min_fit_points: int,
) -> tuple[float | None, FitLine | None, FitLine | None]:
    """
    Estimate an onset by local segmented-line intersection.

    Two local lines are fitted:

    1. a pre-transition baseline/slow-ramp line on the higher-temperature side,
    2. a transition line near the steep local change.

    Their intersection approximates the temperature where the curve leaves the
    earlier trend and enters the transition. The intersection is accepted only if
    it remains in a local validity window around the peak/rough-onset region. If
    the lines are parallel, a fit is missing, or the intersection is nonlocal,
    the caller falls back to another onset estimate.
    """
    peak_T = float(x[peak_idx])
    rough_T = float(x[rough_idx])

    # Pre-transition / baseline-or-slow-ramp line on higher-temperature side.
    pre_lo = rough_T + pre_gap_C
    pre_hi = rough_T + pre_gap_C + pre_width_C
    pre_mask = (x >= pre_lo) & (x <= pre_hi)

    pre_line = fit_line(x, y, pre_mask, min_fit_points)

    # Transition line. Use region from slightly below the peak to just before
    # the rough onset. This captures the steep transition segment.
    trans_lo = max(float(np.nanmin(x)), peak_T - 0.35 * trans_width_C)
    trans_hi = max(peak_T + 0.05 * trans_width_C, rough_T - trans_gap_C)

    # If the peak and rough onset are extremely close, force a small useful window.
    if trans_hi <= trans_lo:
        trans_lo = max(float(np.nanmin(x)), peak_T - 0.5 * trans_width_C)
        trans_hi = min(float(np.nanmax(x)), peak_T + 0.5 * trans_width_C)

    trans_mask = (x >= trans_lo) & (x <= trans_hi)
    trans_line = fit_line(x, y, trans_mask, min_fit_points)

    if pre_line is None or trans_line is None:
        return None, pre_line, trans_line

    x_int = line_intersection_x(pre_line, trans_line)
    if x_int is None:
        return None, pre_line, trans_line

    # Validity window: intersection should be near the transition start,
    # not far outside both fitted segments.
    valid_lo = min(peak_T, rough_T) - 0.20 * trans_width_C
    valid_hi = rough_T + 0.50 * pre_width_C

    if not (valid_lo <= x_int <= valid_hi):
        return None, pre_line, trans_line

    return x_int, pre_line, trans_line



# ---------------------------------------------------------------------------
# Candidate generation and final T1/T2 selection
# ---------------------------------------------------------------------------
def detect_peak_anchors(
    *,
    x: np.ndarray,
    dy_dx: np.ndarray,
    max_bends: int,
    candidate_min_separation_C: float,
    min_height_frac: float,
    prominence_frac: float,
    edge_frac: float,
) -> tuple[list[int], list[float]]:
    """Find derivative-peak anchors from ``|dY/dT|``.

    The anchors are local maxima of the absolute derivative. Detection is
    intentionally permissive because final selection happens later. This allows
    the candidate diagnostics to show weak and rejected alternatives instead of
    hiding them during the first peak-search step.
    """
    abs_d = np.abs(dy_dx)

    finite = np.isfinite(abs_d)
    if not np.any(finite):
        return [], []

    max_abs = float(np.nanmax(abs_d))
    if not np.isfinite(max_abs) or max_abs <= 0:
        return [], []

    n = len(x)
    edge_n = max(0, int(round(n * edge_frac)))
    valid_mask = np.ones(n, dtype=bool)
    if edge_n > 0 and 2 * edge_n < n:
        valid_mask[:edge_n] = False
        valid_mask[-edge_n:] = False

    work = abs_d.copy()
    work[~valid_mask] = 0.0
    work[~np.isfinite(work)] = 0.0

    dx = estimate_temperature_step(x)
    distance_points = max(1, int(round(candidate_min_separation_C / dx)))

    min_height = max_abs * float(min_height_frac)
    min_prom = max_abs * float(prominence_frac)

    peaks, props = find_peaks(
        work,
        height=min_height,
        prominence=min_prom,
        distance=distance_points,
    )

    if len(peaks) == 0:
        return [], []

    prominences = props.get("prominences", np.zeros(len(peaks), dtype=float))
    heights = props.get("peak_heights", work[peaks])

    ranking = sorted(
        range(len(peaks)),
        key=lambda i: (float(prominences[i]), float(heights[i])),
        reverse=True,
    )

    # Development mode:
    # Collect more candidate anchors than finally reported.
    # Final T1/T2 selection is done later in detect_bends().
    candidate_limit = max(max_bends * 20, 50) if max_bends > 0 else 0

    if candidate_limit > 0:
        ranking = ranking[:candidate_limit]

    peak_indices = [int(peaks[i]) for i in ranking]
    peak_prominences = [float(prominences[i]) for i in ranking]

    return peak_indices, peak_prominences


def compute_final_peak_reference(
    peak_heights: Sequence[float],
    dominance_factor: float,
) -> tuple[float, str]:
    """
    Compute the robust peak-height reference for final candidate filtering.

    ``min_final_peak_height_frac`` rejects candidates whose derivative peak is
    too weak for final T1/T2 reporting. If that fraction is applied directly to
    the global maximum, one nearly vertical transition can suppress a real
    earlier T1. The capped reference prevents that failure mode.

    Algorithm:
        1. sort positive candidate peak heights from strongest to weakest,
        2. use the strongest peak as the default reference,
        3. if strongest > dominance_factor * second_strongest, cap the
           reference to dominance_factor * second_strongest.

    Set ``dominance_factor <= 0`` to disable capping and use the global maximum.
    The human-readable reason string is written into rejected-candidate messages
    so the CSV explains why a candidate passed or failed.
    """
    valid = sorted(
        (
            float(value)
            for value in peak_heights
            if np.isfinite(float(value)) and float(value) > 0.0
        ),
        reverse=True,
    )

    if not valid:
        return 0.0, "no positive candidate peaks"

    max_peak = float(valid[0])

    if len(valid) == 1:
        return max_peak, f"single candidate peak {max_peak:.6g}"

    if dominance_factor <= 0:
        return max_peak, f"global max {max_peak:.6g}; capping disabled"

    second_peak = float(valid[1])
    capped_reference = float(dominance_factor) * second_peak

    if max_peak > capped_reference:
        return capped_reference, (
            f"capped reference {capped_reference:.6g}; "
            f"global max {max_peak:.6g} > "
            f"{dominance_factor:g} * second peak {second_peak:.6g}"
        )

    return max_peak, (
        f"global max {max_peak:.6g}; "
        f"second peak {second_peak:.6g}, dominance factor {dominance_factor:g}"
    )


def qualify_forced_valley_report(
    abs_d: np.ndarray, report_idx: int, peak_idx: int, *, forced: bool, floor: float,
) -> int:
    """After the report cap, qualify a forced valley on its existing anchor."""
    if not forced or floor <= 0:
        return int(report_idx)
    peak_abs = float(abs_d[peak_idx])
    if np.isfinite(peak_abs) and peak_abs > 0 and float(abs_d[report_idx]) < floor * peak_abs:
        for idx in range(int(report_idx), int(peak_idx) - 1, -1):
            if np.isfinite(abs_d[idx]) and float(abs_d[idx]) >= floor * peak_abs:
                return int(idx)
    return int(report_idx)


def detect_bends(
    *,
    csv_path: Path,
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    dy_dx: np.ndarray,
    d2y_dx2: np.ndarray,
    max_bends: int,
    min_separation_C: float,
    candidate_min_separation_C: float,
    min_height_frac: float,
    prominence_frac: float,
    min_final_peak_height_frac: float,
    final_peak_dominance_factor: float,
    edge_frac: float,
    report_method: str,
    backtrack_frac: float,
    pre_width_C: float,
    pre_gap_C: float,
    trans_width_C: float,
    trans_gap_C: float,
    min_fit_points: int,
    departure_k: float,
    departure_min_abs: float,
    hybrid_disagree_C: float,
    select_mode: str,
    use_interpeak_valley_onsets: bool,
    interpeak_rise_frac: float,
    interpeak_min_neighbor_peak_frac: float,
    max_report_peak_gap_C: float,
    interpeak_report_rise_frac: float = 0.10,
    forced_valley_report_floor: float = 0.05,
) -> tuple[list[BendPoint], list[CandidatePoint]]:
    """Detect final bend/onset points and keep all diagnostic candidates.

    This is the central orchestration function for the mathematical part of the
    script. It deliberately separates three concepts:

    ``peak anchors``
        local maxima in ``|dY/dT|`` used as transition anchors;

    ``candidates``
        onset estimates derived from those anchors, including rejected and
        diagnostic-only alternatives;

    ``bends``
        final accepted candidates converted to ``BendPoint`` records.

    The default final selection is ``first-valid``: after quality filters, report
    the first candidates in cooling order. Peak strength is used as a validity
    filter, not as the primary ranking criterion for production T1/T2 reporting.
    """
    abs_d = np.abs(dy_dx)
    for name, value in (("interpeak_report_rise_frac", interpeak_report_rise_frac),
                        ("forced_valley_report_floor", forced_valley_report_floor)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    report_provenance: dict[int, dict] = {}

    peak_indices, peak_prominences = detect_peak_anchors(
        x=x,
        dy_dx=dy_dx,
        max_bends=max_bends,
        candidate_min_separation_C=candidate_min_separation_C,
        min_height_frac=min_height_frac,
        prominence_frac=prominence_frac,
        edge_frac=edge_frac,
    )

    peak_prominence_by_index = {
        int(peak_idx): float(prom)
        for peak_idx, prom in zip(peak_indices, peak_prominences)
    }

    bends_tmp: list[tuple[float, int, int, int, float, FitLine | None, FitLine | None, bool]] = []

    for peak_idx, prom in zip(peak_indices, peak_prominences):
        provenance = asdict(ValleyReportProvenance(
            interpeak_rise_frac=interpeak_rise_frac,
            interpeak_report_rise_frac=interpeak_report_rise_frac,
            forced_valley_report_floor=forced_valley_report_floor,
        ))
        report_provenance[int(peak_idx)] = provenance
        rough_idx = backtrack_derivative_departure_index(
            x=x,
            abs_d=abs_d,
            peak_idx=peak_idx,
            fallback_frac=backtrack_frac,
        )

        # Optional inter-peak onset rule.
        #
        # Important: this must be treated as a report-point candidate, not only
        # as a modified rough_idx. In hybrid mode the later segmented-line or
        # baseline-departure logic may otherwise override rough_idx again, which
        # was why the T2/T3 split candidate was visible in the diagnostics but
        # not selected as a reported onset.
        forced_report_idx: int | None = None
        if use_interpeak_valley_onsets:
            valley_idx = interpeak_valley_departure_index(
                x=x,
                abs_d=abs_d,
                peak_idx=peak_idx,
                all_peak_indices=peak_indices,
                rise_frac=interpeak_rise_frac,
            )
            if valley_idx is not None:
                provenance["valley_geometry_temperature_C"] = float(x[valley_idx])
            # x is ascending; higher-temperature onset corresponds to a larger
            # index. Allow the inter-peak valley rule only to move the onset
            # earlier in cooling order, not deeper into the transition.
            if valley_idx is not None and int(valley_idx) > int(rough_idx):
                rough_idx = int(valley_idx)
                # Separate final qualification from the original rough point:
                # changing a report threshold must not move fitting windows.
                forced_report_idx = interpeak_valley_departure_index(
                    x=x, abs_d=abs_d, peak_idx=peak_idx,
                    all_peak_indices=peak_indices, rise_frac=interpeak_report_rise_frac,
                )
                provenance["forced_valley_precedence"] = forced_report_idx is not None
                if forced_report_idx is not None:
                    provenance["valley_qualified_report_temperature_C"] = float(x[forced_report_idx])

        pre_line = None
        trans_line = None
        used_fallback = False

        # Compute intersection and baseline-departure candidates when needed.
        intersection_T = None
        departure_T = None

        if report_method in {"intersection", "hybrid", "departure"}:
            intersection_T, pre_line, trans_line = segmented_intersection_for_peak(
                x=x,
                y=y,
                abs_d=abs_d,
                peak_idx=peak_idx,
                rough_idx=rough_idx,
                pre_width_C=pre_width_C,
                pre_gap_C=pre_gap_C,
                trans_width_C=trans_width_C,
                trans_gap_C=trans_gap_C,
                min_fit_points=min_fit_points,
            )
            departure_T, pre_line_dep = baseline_departure_for_peak(
                x=x,
                y=y,
                peak_idx=peak_idx,
                rough_idx=rough_idx,
                pre_line=pre_line,
                pre_width_C=pre_width_C,
                pre_gap_C=pre_gap_C,
                departure_k=departure_k,
                departure_min_abs=departure_min_abs,
                min_fit_points=min_fit_points,
            )
            if pre_line is None:
                pre_line = pre_line_dep

        if forced_report_idx is not None:
            # Inter-peak valley onset: report the first renewed derivative rise
            # after the valley between adjacent derivative peaks. This is a
            # physical onset candidate in its own right and must not be moved
            # back to the derivative peak by the hybrid/intersection logic.
            bend_T = float(x[forced_report_idx])
            bend_idx = int(forced_report_idx)
            used_fallback = True
        elif report_method == "peak":
            bend_T = float(x[peak_idx])
            bend_idx = int(peak_idx)
        elif report_method == "backtrack":
            bend_T = float(x[rough_idx])
            bend_idx = int(rough_idx)
        elif report_method == "departure":
            if departure_T is None:
                bend_T = float(x[rough_idx])
                bend_idx = int(rough_idx)
                used_fallback = True
            else:
                bend_T = float(departure_T)
                bend_idx = nearest_index(x, bend_T)
        elif report_method == "intersection":
            if intersection_T is None:
                bend_T = float(x[rough_idx])
                bend_idx = int(rough_idx)
                used_fallback = True
            else:
                bend_T = float(intersection_T)
                bend_idx = nearest_index(x, bend_T)
        else:  # hybrid
            if intersection_T is None and departure_T is None:
                bend_T = float(x[rough_idx])
                bend_idx = int(rough_idx)
                used_fallback = True
            elif intersection_T is None:
                bend_T = float(departure_T)
                bend_idx = nearest_index(x, bend_T)
            elif departure_T is None:
                bend_T = float(intersection_T)
                bend_idx = nearest_index(x, bend_T)
            elif abs(float(departure_T) - float(intersection_T)) > hybrid_disagree_C:
                # Large disagreement usually means a steep/narrow transition where
                # the line intersection can be pulled too far into the transition.
                bend_T = float(departure_T)
                bend_idx = nearest_index(x, bend_T)
            else:
                bend_T = float(intersection_T)
                bend_idx = nearest_index(x, bend_T)

        bends_tmp.append((bend_T, bend_idx, rough_idx, peak_idx, prom, pre_line, trans_line, used_fallback))

    # Candidate report-point adjustment:
    #
    # The same rule is applied to every detected transition candidate
    # (T1, T2, T3, ...). If the hybrid/intersection report point is already
    # too deep inside the local transition, use the earlier derivative-
    # departure/backtracked onset instead.
    #
    # This avoids rank-specific rules and keeps the method publishable:
    # a reported onset should occur before the local derivative has reached
    # a large fraction of the corresponding derivative peak.
    # If an intersection/departure report point already sits too close to the
    # local derivative maximum, it is no longer a clean onset. In that case the
    # rough derivative-departure point is safer. The value is intentionally kept
    # internal for now because changing it affects many series.
    max_report_derivative_frac = 0.20

    adjusted_tmp = []
    for candidate in bends_tmp:
        bend_T, bend_idx, rough_idx, peak_idx, prom, pre_line, trans_line, used_fallback = candidate

        peak_abs = float(abs_d[peak_idx])
        report_abs = float(abs_d[bend_idx])

        if (
            np.isfinite(peak_abs)
            and np.isfinite(report_abs)
            and peak_abs > 0.0
            and report_abs / peak_abs > max_report_derivative_frac
        ):
            bend_T = float(x[rough_idx])
            bend_idx = int(rough_idx)

        provenance = report_provenance[int(peak_idx)]
        provenance["post_derivative_cap_temperature_C"] = float(bend_T)
        # The cap can return a forced report to a very weak valley crossing.
        # Qualify only that provenance, moving toward the same anchor; ordinary
        # intersections/departures must not acquire a global derivative floor.
        qualified_idx = qualify_forced_valley_report(
            abs_d, bend_idx, peak_idx,
            forced=provenance["forced_valley_precedence"],
            floor=forced_valley_report_floor,
        )
        if qualified_idx != bend_idx:
            bend_idx = qualified_idx
            bend_T = float(x[bend_idx])
            provenance["forced_valley_floor_applied"] = True

        adjusted_tmp.append((bend_T, bend_idx, rough_idx, peak_idx, prom, pre_line, trans_line, used_fallback))

    # Convert adjusted tuples to explicit candidate records.
    # The peak detection thresholds (--min-height-frac and --prominence-frac)
    # are the primary quality filters. The selection mode only decides which
    # quality-filtered candidates are reported as T1/T2/...
    candidates: list[CandidatePoint] = []
    for candidate_id, (bend_T, bend_idx, rough_idx, peak_idx, prom, pre_line, trans_line, used_fallback) in enumerate(adjusted_tmp, start=1):
        peak_abs = float(abs_d[peak_idx])
        report_abs = float(abs_d[bend_idx])
        if np.isfinite(peak_abs) and peak_abs > 0 and np.isfinite(report_abs):
            derivative_fraction = float(report_abs / peak_abs)
        else:
            derivative_fraction = float("nan")

        candidates.append(
            CandidatePoint(
                candidate_id=candidate_id,
                temperature_C=float(bend_T),
                value=value_at_x(x, y, float(bend_T)),
                rough_onset_temperature_C=float(x[rough_idx]),
                rough_onset_value=float(y[rough_idx]),
                peak_temperature_C=float(x[peak_idx]),
                peak_value=float(y[peak_idx]),
                derivative_at_report=float(dy_dx[bend_idx]),
                derivative_at_peak=float(dy_dx[peak_idx]),
                abs_derivative_at_report=float(abs_d[bend_idx]),
                abs_derivative_at_peak=float(abs_d[peak_idx]),
                second_derivative_at_report=float(d2y_dx2[bend_idx]),
                prominence=float(prom),
                report_derivative_fraction=derivative_fraction,
                method=method,
                report_mode=report_method,
                input_csv=csv_path.name,
                selection_mode=select_mode,
                report_point_index=int(bend_idx),
                rough_onset_point_index=int(rough_idx),
                peak_point_index=int(peak_idx),
                pre_line=pre_line,
                transition_line=trans_line,
                used_fallback=used_fallback,
                valley_report=ValleyReportProvenance(**report_provenance[int(peak_idx)]),
            )
        )

    # Add explicit inter-peak valley-rise candidates after the ordinary
    # peak-anchored candidates have been built. This is intentionally separate:
    # a split transition may need both the high-temperature onset and the
    # renewed-rise onset between two derivative peaks.
    if use_interpeak_valley_onsets:
        valley_candidates = build_interpeak_valley_candidates(
            x=x,
            y=y,
            dy_dx=dy_dx,
            d2y_dx2=d2y_dx2,
            abs_d=abs_d,
            peak_indices=peak_indices,
            peak_prominence_by_index=peak_prominence_by_index,
            method=method,
            report_method=report_method,
            csv_path=csv_path,
            select_mode=select_mode,
            rise_frac=interpeak_report_rise_frac,
            min_neighbor_peak_frac=interpeak_min_neighbor_peak_frac,
        )
        for candidate in valley_candidates:
            geometry_idx = interpeak_valley_departure_index(
                x=x, abs_d=abs_d, peak_idx=candidate.peak_point_index,
                all_peak_indices=peak_indices, rise_frac=interpeak_rise_frac,
            )
            candidate.valley_report = ValleyReportProvenance(
                interpeak_rise_frac=interpeak_rise_frac,
                interpeak_report_rise_frac=interpeak_report_rise_frac,
                forced_valley_report_floor=forced_valley_report_floor,
                valley_geometry_temperature_C=None if geometry_idx is None else float(x[geometry_idx]),
                valley_qualified_report_temperature_C=candidate.temperature_C,
            )
        candidates.extend(valley_candidates)

    # Reject candidates whose report point is implausibly far from the derivative
    # peak that anchors the candidate. This mainly protects raw/PCHIP runs: a
    # late broad/noisy derivative feature can otherwise produce a segmented-line
    # intersection back at an earlier unrelated transition, and first-valid
    # selection then accepts that spurious early report point.
    if max_report_peak_gap_C > 0:
        for candidate in candidates:
            gap_C = abs(float(candidate.temperature_C) - float(candidate.peak_temperature_C))
            if np.isfinite(gap_C) and gap_C > float(max_report_peak_gap_C):
                candidate.status = "rejected_report_peak_gap"
                candidate.reason = f"report-peak gap {gap_C:.3g} C > {max_report_peak_gap_C:g} C"

    # Final significance filter:
    # Candidate detection is intentionally permissive so diagnostics can show
    # weak derivative features. Final T1/T2 selection, however, should not let
    # a weak intermediate shoulder consume the next reported rank before a later
    # stronger transition. The derivative peak that anchors a reported onset
    # must therefore exceed a separate final-selection threshold. This keeps
    # "first-valid" from becoming "first-anything" while preserving the weak
    # candidates in the diagnostics CSV/plot.
    if min_final_peak_height_frac > 0:
        eligible_peak_heights = [
            float(candidate.abs_derivative_at_peak)
            for candidate in candidates
            if not candidate.status.startswith("rejected_")
        ]
        final_peak_reference, final_peak_reference_reason = compute_final_peak_reference(
            eligible_peak_heights,
            dominance_factor=float(final_peak_dominance_factor),
        )
        min_final_peak_abs = float(min_final_peak_height_frac) * final_peak_reference
        if np.isfinite(min_final_peak_abs) and min_final_peak_abs > 0.0:
            for candidate in candidates:
                if candidate.status.startswith("rejected_"):
                    continue
                peak_abs = float(candidate.abs_derivative_at_peak)
                if (not np.isfinite(peak_abs)) or peak_abs < min_final_peak_abs:
                    candidate.status = "rejected_weak_final_peak"
                    candidate.reason = (
                        f"peak |dY/dT| {peak_abs:.6g} < "
                        f"{min_final_peak_height_frac:g} * final reference "
                        f"{final_peak_reference:.6g} ({final_peak_reference_reason})"
                    )

    # Re-number candidates after adding optional generated candidates.
    for candidate_id, candidate in enumerate(candidates, start=1):
        candidate.candidate_id = candidate_id

    ranked_by_strength = sorted(
        candidates,
        key=lambda item: (
            float(item.prominence),
            float(item.abs_derivative_at_peak),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked_by_strength, start=1):
        candidate.candidate_rank_by_strength = rank

    ranked_by_cooling = sorted(candidates, key=lambda item: item.temperature_C, reverse=True)
    for rank, candidate in enumerate(ranked_by_cooling, start=1):
        candidate.candidate_rank_by_cooling_order = rank

    # Physical onset filter:
    #
    # Default T1/T2 mode is first-valid: report the first quality-filtered
    # onset candidates in cooling order. Derivative peak height and prominence
    # are used to reject noise before this stage, not to let later T3/T4
    # transitions replace earlier valid T1/T2 onsets.
    #
    # strongest mode is kept for diagnostics and method comparison.
    if max_bends > 0:
        distinct_min_sep_C = float(min_separation_C)
        selection_order = ranked_by_cooling if select_mode == "first-valid" else ranked_by_strength

        selected: list[CandidatePoint] = []
        for candidate in selection_order:
            if candidate.status.startswith("rejected_"):
                continue

            duplicate_of: CandidatePoint | None = None
            for accepted in selected:
                if abs(candidate.temperature_C - accepted.temperature_C) < distinct_min_sep_C:
                    duplicate_of = accepted
                    break

            if duplicate_of is not None:
                candidate.status = "rejected_duplicate_transition"
                candidate.reason = f"within {distinct_min_sep_C:g} C of accepted candidate"
                continue

            if len(selected) >= max_bends:
                candidate.status = "not_selected_after_limit"
                candidate.reason = f"maximum reported bends reached: {max_bends}"
                continue

            candidate.accepted = True
            candidate.status = "accepted"
            candidate.reason = "selected in cooling order" if select_mode == "first-valid" else "selected by transition strength"
            selected.append(candidate)

        selected = sorted(selected, key=lambda item: item.temperature_C, reverse=True)
        for rank, candidate in enumerate(selected, start=1):
            candidate.final_rank = rank
        accepted_candidates = selected
    else:
        accepted_candidates = ranked_by_cooling
        for rank, candidate in enumerate(accepted_candidates, start=1):
            candidate.accepted = True
            candidate.final_rank = rank
            candidate.status = "accepted"
            candidate.reason = "all accepted because --bends 0"

    bends: list[BendPoint] = []

    for candidate in accepted_candidates:
        rank = int(candidate.final_rank if candidate.final_rank is not None else len(bends) + 1)
        bend_T = candidate.temperature_C
        bend_idx = candidate.report_point_index
        rough_idx = candidate.rough_onset_point_index
        peak_idx = candidate.peak_point_index
        prom = candidate.prominence
        pre_line = candidate.pre_line
        trans_line = candidate.transition_line
        used_fallback = candidate.used_fallback

        bend_value = value_at_x(x, y, bend_T)
        bends.append(
            BendPoint(
                rank=rank,
                temperature_C=float(bend_T),
                value=float(bend_value),
                rough_onset_temperature_C=float(x[rough_idx]),
                rough_onset_value=float(y[rough_idx]),
                peak_temperature_C=float(x[peak_idx]),
                peak_value=float(y[peak_idx]),
                derivative_at_bend=float(dy_dx[bend_idx]),
                derivative_at_peak=float(dy_dx[peak_idx]),
                abs_derivative_at_bend=float(abs_d[bend_idx]),
                abs_derivative_at_peak=float(abs_d[peak_idx]),
                second_derivative_at_bend=float(d2y_dx2[bend_idx]),
                prominence=float(prom),
                method=method,
                report_mode=report_method,
                input_csv=csv_path.name,
                bend_point_index=int(bend_idx),
                rough_onset_point_index=int(rough_idx),
                peak_point_index=int(peak_idx),
                pre_line=pre_line,
                transition_line=trans_line,
                used_fallback=used_fallback,
                valley_report=candidate.valley_report,
            )
        )

    return bends, candidates



# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_bends_csv(path: Path, bends: Sequence[BendPoint]) -> None:
    """Write per-input bend point CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "input_csv",
        "method",
        "report_mode",
        "bend_index",
        "temperature_C",
        "value",
        "rough_onset_temperature_C",
        "rough_onset_value",
        "peak_temperature_C",
        "peak_value",
        "dY_dT_at_bend",
        "dY_dT_at_peak",
        "abs_dY_dT_at_bend",
        "abs_dY_dT_at_peak",
        "d2Y_dT2_at_bend",
        "prominence",
        "bend_point_index",
        "rough_onset_point_index",
        "peak_point_index",
        "pre_line_slope",
        "pre_line_intercept",
        "pre_line_n_points",
        "transition_line_slope",
        "transition_line_intercept",
        "transition_line_n_points",
        "used_fallback",
        *ValleyReportProvenance.__dataclass_fields__,
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for bend in bends:
            writer.writerow(
                {
                    "input_csv": bend.input_csv,
                    "method": bend.method,
                    "report_mode": bend.report_mode,
                    "bend_index": bend.rank,
                    "temperature_C": f"{bend.temperature_C:.6g}",
                    "value": f"{bend.value:.6g}",
                    "rough_onset_temperature_C": f"{bend.rough_onset_temperature_C:.6g}",
                    "rough_onset_value": f"{bend.rough_onset_value:.6g}",
                    "peak_temperature_C": f"{bend.peak_temperature_C:.6g}",
                    "peak_value": f"{bend.peak_value:.6g}",
                    "dY_dT_at_bend": f"{bend.derivative_at_bend:.6g}",
                    "dY_dT_at_peak": f"{bend.derivative_at_peak:.6g}",
                    "abs_dY_dT_at_bend": f"{bend.abs_derivative_at_bend:.6g}",
                    "abs_dY_dT_at_peak": f"{bend.abs_derivative_at_peak:.6g}",
                    "d2Y_dT2_at_bend": f"{bend.second_derivative_at_bend:.6g}",
                    "prominence": f"{bend.prominence:.6g}",
                    "bend_point_index": bend.bend_point_index,
                    "rough_onset_point_index": bend.rough_onset_point_index,
                    "peak_point_index": bend.peak_point_index,
                    "pre_line_slope": "" if bend.pre_line is None else f"{bend.pre_line.slope:.6g}",
                    "pre_line_intercept": "" if bend.pre_line is None else f"{bend.pre_line.intercept:.6g}",
                    "pre_line_n_points": "" if bend.pre_line is None else bend.pre_line.n_points,
                    "transition_line_slope": "" if bend.transition_line is None else f"{bend.transition_line.slope:.6g}",
                    "transition_line_intercept": "" if bend.transition_line is None else f"{bend.transition_line.intercept:.6g}",
                    "transition_line_n_points": "" if bend.transition_line is None else bend.transition_line.n_points,
                    "used_fallback": bend.used_fallback,
                    **({} if bend.valley_report is None else asdict(bend.valley_report)),
                }
            )





def write_candidates_csv(path: Path, candidates: Sequence[CandidatePoint]) -> None:
    """Write per-input all-candidates diagnostic CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "input_csv",
        "method",
        "report_mode",
        "selection_mode",
        "candidate_id",
        "candidate_rank_by_cooling_order",
        "candidate_rank_by_strength",
        "accepted",
        "final_rank",
        "status",
        "reason",
        "temperature_C",
        "value",
        "rough_onset_temperature_C",
        "rough_onset_value",
        "peak_temperature_C",
        "peak_value",
        "dY_dT_at_report",
        "dY_dT_at_peak",
        "abs_dY_dT_at_report",
        "abs_dY_dT_at_peak",
        "report_derivative_fraction",
        "d2Y_dT2_at_report",
        "prominence",
        "report_point_index",
        "rough_onset_point_index",
        "peak_point_index",
        "pre_line_slope",
        "pre_line_intercept",
        "pre_line_n_points",
        "transition_line_slope",
        "transition_line_intercept",
        "transition_line_n_points",
        "used_fallback",
        *ValleyReportProvenance.__dataclass_fields__,
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for candidate in sorted(candidates, key=lambda item: item.candidate_rank_by_cooling_order):
            writer.writerow(
                {
                    "input_csv": candidate.input_csv,
                    "method": candidate.method,
                    "report_mode": candidate.report_mode,
                    "selection_mode": candidate.selection_mode,
                    "candidate_id": candidate.candidate_id,
                    "candidate_rank_by_cooling_order": candidate.candidate_rank_by_cooling_order,
                    "candidate_rank_by_strength": candidate.candidate_rank_by_strength,
                    "accepted": candidate.accepted,
                    "final_rank": "" if candidate.final_rank is None else candidate.final_rank,
                    "status": candidate.status,
                    "reason": candidate.reason,
                    "temperature_C": f"{candidate.temperature_C:.6g}",
                    "value": f"{candidate.value:.6g}",
                    "rough_onset_temperature_C": f"{candidate.rough_onset_temperature_C:.6g}",
                    "rough_onset_value": f"{candidate.rough_onset_value:.6g}",
                    "peak_temperature_C": f"{candidate.peak_temperature_C:.6g}",
                    "peak_value": f"{candidate.peak_value:.6g}",
                    "dY_dT_at_report": f"{candidate.derivative_at_report:.6g}",
                    "dY_dT_at_peak": f"{candidate.derivative_at_peak:.6g}",
                    "abs_dY_dT_at_report": f"{candidate.abs_derivative_at_report:.6g}",
                    "abs_dY_dT_at_peak": f"{candidate.abs_derivative_at_peak:.6g}",
                    "report_derivative_fraction": f"{candidate.report_derivative_fraction:.6g}",
                    "d2Y_dT2_at_report": f"{candidate.second_derivative_at_report:.6g}",
                    "prominence": f"{candidate.prominence:.6g}",
                    "report_point_index": candidate.report_point_index,
                    "rough_onset_point_index": candidate.rough_onset_point_index,
                    "peak_point_index": candidate.peak_point_index,
                    "pre_line_slope": "" if candidate.pre_line is None else f"{candidate.pre_line.slope:.6g}",
                    "pre_line_intercept": "" if candidate.pre_line is None else f"{candidate.pre_line.intercept:.6g}",
                    "pre_line_n_points": "" if candidate.pre_line is None else candidate.pre_line.n_points,
                    "transition_line_slope": "" if candidate.transition_line is None else f"{candidate.transition_line.slope:.6g}",
                    "transition_line_intercept": "" if candidate.transition_line is None else f"{candidate.transition_line.intercept:.6g}",
                    "transition_line_n_points": "" if candidate.transition_line is None else candidate.transition_line.n_points,
                    "used_fallback": candidate.used_fallback,
                    **({} if candidate.valley_report is None else asdict(candidate.valley_report)),
                }
            )



# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def plot_styles(bw: bool) -> dict[str, object]:
    """Return plotting style values for color or black-and-white mode."""
    if bw:
        return {
            "main_line_color": "black",
            "raw_color": "0.45",
            "bend_line_color": "black",
            "bend_marker_color": "black",
            "rough_line_color": "0.35",
            "peak_line_color": "0.60",
            "pre_line_color": "0.20",
            "transition_line_color": "0.20",
            "derivative_color": "black",
            "abs_derivative_color": "0.55",
            "grid_color": "0.75",
        }

    return {
        # Keep the main BW/PCHIP line black also in color mode.
        "main_line_color": "black",
        "raw_color": "tab:blue",
        "bend_line_color": "tab:blue",
        "bend_marker_color": "black",
        "rough_line_color": "tab:blue",
        "peak_line_color": "tab:blue",
        "pre_line_color": "tab:green",
        "transition_line_color": "tab:red",
        "derivative_color": "tab:blue",
        "abs_derivative_color": "tab:orange",
        "grid_color": "0.75",
    }


def set_title_if_enabled(ax, title: str, no_title: bool) -> None:
    """Set plot title unless disabled."""
    if not no_title:
        ax.set_title(title)


def display_y_label(ycol: str) -> str:
    """Return figure-facing label for the selected intensity column."""
    if ycol == "BW":
        return "MGI"
    if ycol == "BW_smooth":
        return "Savitzky-Golay-filtered MGI"
    return ycol


def display_derivative_y_label(ycol: str) -> str:
    """Return figure-facing label for derivative legends."""
    if ycol == "BW":
        return "MGI"
    if ycol == "BW_smooth":
        return "Savitzky-Golay-filtered MGI"
    return display_y_label(ycol)


PLOT_FONT_SIZES = {
    "axis_label": 13,
    "tick_label": 11,
    "legend": 11,
    "annotation": 10,
    "title": 14,
}

PUBLICATION_GRID_ALPHA = 0.20


def annotate_onset_label(
    ax,
    x: float,
    y: float,
    text: str,
    rank: int,
    *,
    fontsize: int = PLOT_FONT_SIZES["annotation"],
    alpha: float = 0.95,
) -> None:
    """Annotate onset temperature with a horizontal, offset label."""
    if rank % 2 == 1:
        xytext = (18, 18)
        va = "bottom"
    else:
        xytext = (18, -24)
        va = "top"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha="left",
        va=va,
        fontsize=fontsize,
        alpha=alpha,
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.7,
            "alpha": alpha,
        },
        annotation_clip=True,
        clip_on=True,
    )


def apply_publication_axis_style(ax, *, legend: bool = True) -> None:
    """Apply publication-friendly font sizes to one Matplotlib axis."""
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick_label"])
    ax.xaxis.label.set_size(PLOT_FONT_SIZES["axis_label"])
    ax.yaxis.label.set_size(PLOT_FONT_SIZES["axis_label"])

    title = ax.title
    if title is not None:
        title.set_fontsize(PLOT_FONT_SIZES["title"])

    if legend:
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(PLOT_FONT_SIZES["legend"])



def infer_raw_source_csv(input_csv: Path, method: str, raw_csv: Path | None) -> Path | None:
    """
    Infer original source CSV for raw/raw-smooth plot modes.

    Typical layout:
        analysis/
            rgb-tr.csv
            rgb-tr-sg.csv
            pchip/
                bw-raw-vs-temp-pchip.csv
                bw-sg-vs-temp-pchip.csv
    """
    if raw_csv is not None:
        return raw_csv

    # Important: input_csv may be a relative filename such as
    # "bw-sg-vs-temp-pchip.csv" when the script is run inside analysis/pchip.
    # Resolve it so input_csv.parent.parent points to the analysis directory.
    input_csv = input_csv.expanduser().resolve()

    candidates: list[Path] = []

    if method == "sg_pchip":
        candidates.extend([
            input_csv.parent.parent / "rgb-tr-sg.csv",
            input_csv.parent / "rgb-tr-sg.csv",
        ])
    elif method == "raw_pchip":
        candidates.extend([
            input_csv.parent.parent / "rgb-tr.csv",
            input_csv.parent / "rgb-tr.csv",
        ])

    # Fallbacks for unusual names.
    candidates.extend([
        input_csv.parent.parent / "rgb-tr.csv",
        input_csv.parent.parent / "rgb-tr-sg.csv",
        input_csv.parent / "rgb-tr.csv",
        input_csv.parent / "rgb-tr-sg.csv",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_raw_points_for_plot(
    *,
    input_csv: Path,
    method: str,
    xcol: str,
    ycol: str,
    raw_csv: Path | None,
) -> tuple[np.ndarray, np.ndarray, Path] | None:
    """
    Load raw/source points for plotting.

    The function first tries to use xcol/ycol directly. If the PCHIP y column is
    BW_smooth but the raw file only has BW, it falls back to BW.
    """
    source = infer_raw_source_csv(input_csv, method, raw_csv)
    if source is None or not source.exists():
        return None

    try:
        df = pd.read_csv(source, encoding="utf-8")
    except Exception:
        return None

    if xcol not in df.columns:
        return None

    y_candidates = [ycol]
    if ycol == "BW_smooth":
        y_candidates.append("BW")
    elif ycol == "BW":
        y_candidates.append("BW_smooth")

    y_source = None
    for candidate in y_candidates:
        if candidate in df.columns:
            y_source = candidate
            break

    if y_source is None:
        return None

    x_raw = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df[y_source], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]

    if len(x_raw) == 0:
        return None

    return x_raw, y_raw, source



def plot_line_segment(
    ax,
    line: FitLine,
    *,
    label: str | None = None,
    color: object | None = None,
) -> None:
    """Plot a fitted line segment over its fit range."""
    xs = np.array([line.x_min, line.x_max], dtype=float)
    ys = line.slope * xs + line.intercept
    ax.plot(xs, ys, "--", lw=1.0, alpha=0.7, label=label, color=color)




def outward_temperature_limits(
    tr: np.ndarray,
    major_step_C: float = 5.0,
    snap_tolerance_C: float = 1.0,
) -> tuple[float, float]:
    """Return non-clipping, data-aware low/high temperature limits.

    The limits are derived from actual MGI source temperatures when available,
    otherwise from the plotted PCHIP temperature grid. The returned numeric
    order is always ``(low, high)``. Axis display direction is applied by
    ``apply_temperature_xlim``.

    The rule is deliberately conservative:

    - never snap inward;
    - snap outward to the next 5 °C grid boundary only when the data edge is
      already close to that outward boundary;
    - otherwise use the actual data edge so the figure ends at the data.

    Examples with the default 1 °C tolerance:

    - data minimum 24.8 °C -> right/lower limit 24.8 °C, not 20 °C and not 25 °C;
    - data minimum 21.0 °C -> right/lower limit 20.0 °C;
    - data maximum 59.7 °C -> left/upper limit 60.0 °C;
    - data maximum 60.6 °C -> left/upper limit 60.6 °C, not 65 °C.
    """
    x = np.asarray(tr, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Cannot determine x-limits from empty temperature data.")

    data_hi = float(np.max(x))
    data_lo = float(np.min(x))

    lower_grid = float(np.floor(data_lo / major_step_C) * major_step_C)
    upper_grid = float(np.ceil(data_hi / major_step_C) * major_step_C)

    # Snap only outward and only when the data edge is already close to the
    # outward grid boundary. Otherwise keep the true data edge. This preserves
    # all MGI data without forcing a large empty 5 °C interval.
    x_lo = lower_grid if (data_lo - lower_grid) <= snap_tolerance_C else data_lo
    x_hi = upper_grid if (upper_grid - data_hi) <= snap_tolerance_C else data_hi

    return x_lo, x_hi


def xlim_temperature_source(
    plotted_x: np.ndarray,
    raw_loaded: tuple[np.ndarray, np.ndarray, Path] | None,
) -> np.ndarray:
    """Prefer original MGI source temperatures for figure x-limits."""
    if raw_loaded is not None:
        x_raw, _y_raw, _source = raw_loaded
        x = np.asarray(x_raw, dtype=float)
        if np.isfinite(x).any():
            return x
    return np.asarray(plotted_x, dtype=float)


def apply_temperature_xlim(
    ax,
    tr: np.ndarray,
    *,
    reverse_x: bool,
    major_step_C: float = 5.0,
    snap_tolerance_C: float = 1.0,
    view_limits: ViewLimits = NONE_LIMITS,
) -> tuple[float, float]:
    """Apply data-aware display x-limits with explicit axis direction."""
    return apply_temperature_view_limits(
        ax,
        tr,
        view_limits,
        reverse_x=reverse_x,
        major_step_C=major_step_C,
        snap_tolerance_C=snap_tolerance_C,
    )


def _manual_axis_value_is_visible(value: float, axis_limits: object | None) -> bool:
    """Return True if value is inside optional manually requested axis bounds."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(v):
        return False

    if axis_limits is None:
        return True

    lower = getattr(axis_limits, "lower", None)
    upper = getattr(axis_limits, "upper", None)

    if lower is not None:
        try:
            lo = float(lower)
        except (TypeError, ValueError):
            lo = float("nan")
        if np.isfinite(lo) and v < lo:
            return False

    if upper is not None:
        try:
            hi = float(upper)
        except (TypeError, ValueError):
            hi = float("nan")
        if np.isfinite(hi) and v > hi:
            return False

    return True


def _mgi_point_visible_for_view(x_value: float, y_value: float, view_limits: ViewLimits) -> bool:
    """Return True if an MGI data-anchored annotation belongs to the current view."""
    if not _manual_axis_value_is_visible(x_value, getattr(view_limits, "x", None)):
        return False
    if not _manual_axis_value_is_visible(y_value, getattr(view_limits, "mgi_y", None)):
        return False
    return True


def _mgi_bend_visible_for_view(bend: BendPoint, view_limits: ViewLimits) -> bool:
    """Return True when a final MGI onset/bend is visible in the current view."""
    return _mgi_point_visible_for_view(bend.temperature_C, bend.value, view_limits)


def save_plot(
    *,
    path: Path,
    title: str,
    input_csv: Path,
    method: str,
    report_method: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    bends: Sequence[BendPoint],
    reverse_x: bool,
    dpi: int,
    plot_mode: str,
    show_diagnostics: bool,
    raw_csv: Path | None,
    no_title: bool,
    bw: bool,
    view_limits: ViewLimits = NONE_LIMITS,
) -> None:
    """Save bend point analysis plot."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = plot_styles(bw)
    y_label = display_y_label(ycol)
    y_derivative_label = display_derivative_y_label(ycol)

    raw_loaded = None
    if plot_mode in {"raw", "raw-smooth"}:
        raw_loaded = load_raw_points_for_plot(
            input_csv=input_csv,
            method=method,
            xcol=xcol,
            ycol=ycol,
            raw_csv=raw_csv,
        )

    if raw_loaded is not None:
        x_raw, y_raw, source = raw_loaded
        ax.plot(
            x_raw,
            y_raw,
            ".",
            ms=2,
            alpha=0.30,
            color=styles["raw_color"],
            label="Data",
        )
    elif plot_mode in {"raw", "raw-smooth"}:
        # Fall back safely so the plot is still useful.
        ax.text(
            0.02,
            0.98,
            "raw source CSV not found; showing PCHIP curve",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )

    if plot_mode in {"smooth", "raw-smooth"} or raw_loaded is None:
        ax.plot(x, y, "-", lw=1.2, color=styles["main_line_color"], label=y_label)

    for bend in bends:
        # Final bend point.
        ax.axvline(
            bend.temperature_C,
            linestyle="--",
            alpha=0.85,
            color=styles["bend_line_color"],
        )
        ax.plot(
            bend.temperature_C,
            bend.value,
            "o",
            ms=5,
            color=styles["bend_marker_color"],
        )

        txt = f"T{bend.rank}: {bend.temperature_C:.2f} °C"
        annotate_onset_label(
            ax,
            bend.temperature_C,
            bend.value,
            txt,
            bend.rank,
        )

        # Diagnostic rough onset, derivative peak anchors and fitted line segments.
        if show_diagnostics and report_method == "intersection":
            ax.axvline(
                bend.rough_onset_temperature_C,
                linestyle=":",
                alpha=0.35,
                color=styles["rough_line_color"],
            )
            ax.axvline(
                bend.peak_temperature_C,
                linestyle=":",
                alpha=0.25,
                color=styles["peak_line_color"],
            )

            if bend.pre_line is not None:
                plot_line_segment(ax, bend.pre_line, color=styles["pre_line_color"])
            if bend.transition_line is not None:
                plot_line_segment(ax, bend.transition_line, color=styles["transition_line_color"])

    ax.set_xlabel(xcol)
    ax.set_ylabel(y_label)
    set_title_if_enabled(
        ax,
        f"{title}\n{input_csv.name} ({method}, {report_method}, {plot_mode})",
        no_title,
    )
    ax.grid(True, which="major", linestyle="--", alpha=PUBLICATION_GRID_ALPHA)
    ax.legend(loc="best")
    apply_publication_axis_style(ax)

    xlim_source = xlim_temperature_source(x, raw_loaded)
    x_limits = apply_temperature_xlim(ax, xlim_source, reverse_x=reverse_x, view_limits=view_limits)
    y_series: list[tuple[np.ndarray, np.ndarray]] = []
    if plot_mode in {"smooth", "raw-smooth"} or raw_loaded is None:
        y_series.append((x, y))
    if raw_loaded is not None:
        x_raw, y_raw, _source = raw_loaded
        y_series.append((x_raw, y_raw))
    apply_signal_y_view_limits(ax, y_series, x_limits, view_limits.mgi_y)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



def save_diagnostic_main_plot(
    *,
    path: Path,
    title: str,
    input_csv: Path,
    method: str,
    report_method: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    bends: Sequence[BendPoint],
    reverse_x: bool,
    dpi: int,
    raw_csv: Path | None,
    no_title: bool,
    bw: bool,
    view_limits: ViewLimits = NONE_LIMITS,
) -> None:
    """
    Save diagnostic main plot without derivative subplot.

    This plot is useful for inspecting bend locations, fitted regression
    segments and raw/PCHIP agreement without losing vertical space to the
    derivative subplot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = plot_styles(bw)
    y_label = display_y_label(ycol)
    y_derivative_label = display_derivative_y_label(ycol)

    raw_loaded = load_raw_points_for_plot(
        input_csv=input_csv,
        method=method,
        xcol=xcol,
        ycol=ycol,
        raw_csv=raw_csv,
    )

    if raw_loaded is not None:
        x_raw, y_raw, source = raw_loaded
        ax.plot(
            x_raw,
            y_raw,
            ".",
            ms=2,
            alpha=0.25,
            color=styles["raw_color"],
            label="Data",
        )

    ax.plot(x, y, "-", lw=1.2, color=styles["main_line_color"], label=y_label)

    for bend in bends:
        # Final bend / onset.
        ax.axvline(
            bend.temperature_C,
            linestyle="--",
            alpha=0.9,
            color=styles["bend_line_color"],
        )
        ax.plot(
            bend.temperature_C,
            bend.value,
            "o",
            ms=5,
            color=styles["bend_marker_color"],
        )

        txt = f"T{bend.rank}: {bend.temperature_C:.2f} °C"
        annotate_onset_label(
            ax,
            bend.temperature_C,
            bend.value,
            txt,
            bend.rank,
        )

        # Rough onset and derivative peak anchors.
        ax.axvline(
            bend.rough_onset_temperature_C,
            linestyle=":",
            alpha=0.45,
            color=styles["rough_line_color"],
        )
        ax.axvline(
            bend.peak_temperature_C,
            linestyle=":",
            alpha=0.35,
            color=styles["peak_line_color"],
        )

        # Fitted regression segments.
        if bend.pre_line is not None:
            plot_line_segment(ax, bend.pre_line, color=styles["pre_line_color"])
        if bend.transition_line is not None:
            plot_line_segment(ax, bend.transition_line, color=styles["transition_line_color"])

    ax.set_xlabel(xcol)
    ax.set_ylabel(y_label)
    set_title_if_enabled(
        ax,
        f"{title} - diagnostic main\n{input_csv.name} ({method}, {report_method})",
        no_title,
    )
    ax.grid(True, which="major", linestyle="--", alpha=PUBLICATION_GRID_ALPHA)
    ax.legend(loc="best")
    apply_publication_axis_style(ax)

    xlim_source = xlim_temperature_source(x, raw_loaded)
    x_limits = apply_temperature_xlim(ax, xlim_source, reverse_x=reverse_x, view_limits=view_limits)
    y_series = [(x, y)]
    if raw_loaded is not None:
        x_raw, y_raw, _source = raw_loaded
        y_series.append((x_raw, y_raw))
    apply_signal_y_view_limits(ax, y_series, x_limits, view_limits.mgi_y)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_diagnostic_plot(
    *,
    path: Path,
    title: str,
    input_csv: Path,
    method: str,
    report_method: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    dy_dx: np.ndarray,
    bends: Sequence[BendPoint],
    reverse_x: bool,
    dpi: int,
    raw_csv: Path | None,
    no_title: bool,
    bw: bool,
    view_limits: ViewLimits = NONE_LIMITS,
    same_aspect_diagnostics: bool = False,
) -> None:
    """Save full diagnostic plot with derivative subplot."""
    path.parent.mkdir(parents=True, exist_ok=True)

    figure_size = (9, 5.5) if same_aspect_diagnostics else (9, 7.5)
    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    styles = plot_styles(bw)
    y_label = display_y_label(ycol)
    y_derivative_label = display_derivative_y_label(ycol)

    raw_loaded = load_raw_points_for_plot(
        input_csv=input_csv,
        method=method,
        xcol=xcol,
        ycol=ycol,
        raw_csv=raw_csv,
    )

    if raw_loaded is not None:
        x_raw, y_raw, source = raw_loaded
        ax.plot(
            x_raw,
            y_raw,
            ".",
            ms=2,
            alpha=0.25,
            color=styles["raw_color"],
            label="Data",
        )

    ax.plot(x, y, "-", lw=1.2, color=styles["main_line_color"], label=y_label)

    abs_d = np.abs(dy_dx)
    axd.plot(
        x,
        dy_dx,
        "-",
        lw=0.9,
        alpha=0.85,
        color=styles["derivative_color"],
        label=f"d({y_derivative_label})/dT",
    )
    axd.plot(
        x,
        abs_d,
        "-",
        lw=0.9,
        alpha=0.55,
        color=styles["abs_derivative_color"],
        label=f"|d({y_derivative_label})/dT|",
    )
    axd.axhline(0.0, lw=0.8, alpha=0.5)

    for bend in bends:
        # Final bend / onset.
        ax.axvline(
            bend.temperature_C,
            linestyle="--",
            alpha=0.9,
            color=styles["bend_line_color"],
        )
        ax.plot(
            bend.temperature_C,
            bend.value,
            "o",
            ms=5,
            color=styles["bend_marker_color"],
        )
        axd.axvline(
            bend.temperature_C,
            linestyle="--",
            alpha=0.9,
            color=styles["bend_line_color"],
        )

        txt = f"T{bend.rank}: {bend.temperature_C:.2f} °C"
        annotate_onset_label(
            ax,
            bend.temperature_C,
            bend.value,
            txt,
            bend.rank,
        )

        # Rough onset and derivative peak anchors.
        ax.axvline(
            bend.rough_onset_temperature_C,
            linestyle=":",
            alpha=0.45,
            color=styles["rough_line_color"],
        )
        ax.axvline(
            bend.peak_temperature_C,
            linestyle=":",
            alpha=0.35,
            color=styles["peak_line_color"],
        )
        axd.axvline(
            bend.rough_onset_temperature_C,
            linestyle=":",
            alpha=0.45,
            color=styles["rough_line_color"],
        )
        axd.axvline(
            bend.peak_temperature_C,
            linestyle=":",
            alpha=0.35,
            color=styles["peak_line_color"],
        )

        # Fitted regression segments.
        if bend.pre_line is not None:
            plot_line_segment(ax, bend.pre_line, color=styles["pre_line_color"])
        if bend.transition_line is not None:
            plot_line_segment(ax, bend.transition_line, color=styles["transition_line_color"])

        # Derivative peak marker.
        axd.plot(
            bend.peak_temperature_C,
            bend.abs_derivative_at_peak,
            "o",
            ms=4,
            alpha=0.8,
            color=styles["bend_marker_color"],
        )

    ax.set_ylabel(y_label)
    set_title_if_enabled(
        ax,
        f"{title} - diagnostic\n{input_csv.name} ({method}, {report_method})",
        no_title,
    )
    ax.grid(True, which="major", linestyle="--", alpha=PUBLICATION_GRID_ALPHA)
    ax.legend(loc="best")

    axd.set_xlabel(xcol)
    axd.set_ylabel("Derivative")
    axd.grid(True, which="major", linestyle="--", alpha=PUBLICATION_GRID_ALPHA)
    axd.legend(loc="best")
    apply_publication_axis_style(axd)

    xlim_source = xlim_temperature_source(x, raw_loaded)
    x_limits = apply_temperature_xlim(ax, xlim_source, reverse_x=reverse_x, view_limits=view_limits)
    axd.set_xlim(ax.get_xlim())
    y_series = [(x, y)]
    if raw_loaded is not None:
        x_raw, y_raw, _source = raw_loaded
        y_series.append((x_raw, y_raw))
    apply_signal_y_view_limits(ax, y_series, x_limits, view_limits.mgi_y)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



def save_candidate_diagnostic_plot(
    *,
    path: Path,
    title: str,
    input_csv: Path,
    method: str,
    report_method: str,
    selection_mode: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    dy_dx: np.ndarray,
    candidates: Sequence[CandidatePoint],
    reverse_x: bool,
    dpi: int,
    raw_csv: Path | None,
    no_title: bool,
    bw: bool,
    view_limits: ViewLimits = NONE_LIMITS,
    same_aspect_diagnostics: bool = False,
) -> None:
    """Save all-candidates diagnostic plot with derivative curves."""
    path.parent.mkdir(parents=True, exist_ok=True)

    figure_size = (9, 5.5) if same_aspect_diagnostics else (9, 8.0)
    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    styles = plot_styles(bw)
    y_label = display_y_label(ycol)
    y_derivative_label = display_derivative_y_label(ycol)

    raw_loaded = load_raw_points_for_plot(
        input_csv=input_csv,
        method=method,
        xcol=xcol,
        ycol=ycol,
        raw_csv=raw_csv,
    )

    if raw_loaded is not None:
        x_raw, y_raw, source = raw_loaded
        ax.plot(
            x_raw,
            y_raw,
            ".",
            ms=2,
            alpha=0.20,
            color=styles["raw_color"],
            label="Data",
        )

    ax.plot(x, y, "-", lw=1.2, color=styles["main_line_color"], label=y_label)

    abs_d = np.abs(dy_dx)
    axd.plot(
        x,
        dy_dx,
        "-",
        lw=0.9,
        alpha=0.85,
        color=styles["derivative_color"],
        label=f"d({y_derivative_label})/dT",
    )
    axd.plot(
        x,
        abs_d,
        "-",
        lw=0.9,
        alpha=0.55,
        color=styles["abs_derivative_color"],
        label=f"|d({y_derivative_label})/dT|",
    )
    axd.axhline(0.0, lw=0.8, alpha=0.5)

    for candidate in sorted(candidates, key=lambda item: item.candidate_rank_by_cooling_order):
        if candidate.accepted:
            line_style = "--"
            alpha = 0.90
            marker = "o"
            label_text = f"T{candidate.final_rank}: {candidate.temperature_C:.2f} °C"
            line_color = styles["bend_line_color"]
            marker_color = styles["bend_marker_color"]
        else:
            line_style = ":"
            alpha = 0.35
            marker = "x"
            label_text = f"C{candidate.candidate_rank_by_cooling_order}"
            line_color = styles["peak_line_color"]
            marker_color = styles["peak_line_color"]

        ax.axvline(
            candidate.temperature_C,
            linestyle=line_style,
            alpha=alpha,
            color=line_color,
        )
        axd.axvline(
            candidate.temperature_C,
            linestyle=line_style,
            alpha=alpha,
            color=line_color,
        )
        ax.plot(
            candidate.temperature_C,
            candidate.value,
            marker,
            ms=5 if candidate.accepted else 4,
            alpha=0.9 if candidate.accepted else 0.45,
            color=marker_color,
        )
        axd.plot(
            candidate.peak_temperature_C,
            candidate.abs_derivative_at_peak,
            marker,
            ms=5 if candidate.accepted else 4,
            alpha=0.9 if candidate.accepted else 0.45,
            color=marker_color,
        )
        annotate_onset_label(
            ax,
            candidate.temperature_C,
            candidate.value,
            label_text,
            candidate.final_rank if candidate.accepted and candidate.final_rank is not None else candidate.candidate_rank_by_cooling_order,
            fontsize=7 if candidate.accepted else 6,
            alpha=0.95 if candidate.accepted else 0.65,
        )

    ax.set_ylabel(y_label)
    set_title_if_enabled(
        ax,
        f"{title} - candidate diagnostics\n{input_csv.name} ({method}, {report_method}, {selection_mode})",
        no_title,
    )
    ax.grid(True, which="major", linestyle="--", alpha=PUBLICATION_GRID_ALPHA)
    ax.legend(loc="best")

    axd.set_xlabel(xcol)
    axd.set_ylabel("Derivative")
    axd.grid(True, which="major", linestyle="--", alpha=PUBLICATION_GRID_ALPHA)
    axd.legend(loc="best")
    apply_publication_axis_style(axd)

    xlim_source = xlim_temperature_source(x, raw_loaded)
    x_limits = apply_temperature_xlim(ax, xlim_source, reverse_x=reverse_x, view_limits=view_limits)
    axd.set_xlim(ax.get_xlim())
    y_series = [(x, y)]
    if raw_loaded is not None:
        x_raw, y_raw, _source = raw_loaded
        y_series.append((x_raw, y_raw))
    apply_signal_y_view_limits(ax, y_series, x_limits, view_limits.mgi_y)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)




def save_mgi_publication_plot(
    *,
    path: Path,
    input_csv: Path,
    method: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    bends: Sequence[BendPoint],
    reverse_x: bool,
    dpi: int,
    raw_csv: Path | None,
    variant: str,
    view_limits: ViewLimits = NONE_LIMITS,
) -> None:
    """Save a clean manuscript-style MGI/PCHIP publication plot.

    These plots are deliberately presentation variants only. They reuse the
    already detected onset points and never modify detection, candidate
    selection, CSV output, or PCHIP data. The variants mirror the combo-figure
    idea: a clean curve-only view, a main view with faint source points, and
    onset-marked views with compact T-labels on the vertical lines and a
    temperature value box instead of arrows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    mono = "mono" in variant
    clean = "clean" in variant
    show_onsets = "onsets" in variant
    show_raw = not clean

    raw_loaded = load_raw_points_for_plot(
        input_csv=input_csv,
        method=method,
        xcol=xcol,
        ycol=ycol,
        raw_csv=raw_csv,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if mono:
        line_color = "black"
        raw_color = "0.70"
        onset_color = "0.20"
        marker_color = "black"
        grid_color = "0.82"
    else:
        line_color = "#003f7f"
        raw_color = "tab:blue"
        onset_color = "tab:blue"
        marker_color = "black"
        grid_color = "0.82"

    if show_raw and raw_loaded is not None:
        x_raw, y_raw, _source = raw_loaded
        ax.plot(
            x_raw,
            y_raw,
            ".",
            ms=2.2,
            alpha=0.24 if mono else 0.30,
            color=raw_color,
            label="Data" if not clean else None,
        )

    ax.plot(x, y, "-", lw=1.45, color=line_color, label="MGI")

    if show_onsets:
        onset_lines: list[str] = []
        for bend in bends:
            if not _mgi_bend_visible_for_view(bend, view_limits):
                continue
            ax.axvline(
                bend.temperature_C,
                linestyle="--",
                lw=1.0,
                alpha=0.80,
                color=onset_color,
            )
            ax.plot(
                bend.temperature_C,
                bend.value,
                "o",
                ms=4.5,
                color=marker_color,
                zorder=5,
            )
            # Add a compact T-label directly on the vertical line so the
            # reader can map each line to the value box without arrows.
            label_y = 0.92 if bend.rank % 2 == 1 else 0.84
            ax.text(
                bend.temperature_C,
                label_y,
                f"T{bend.rank}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=PLOT_FONT_SIZES["annotation"],
                color=onset_color,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                clip_on=True,
                zorder=6,
            )

            onset_lines.append(f"T{bend.rank} = {bend.temperature_C:.2f} °C")

        if onset_lines:
            ax.text(
                0.03,
                0.96,
                "\n".join(onset_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PLOT_FONT_SIZES["annotation"],
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.80",
                    "alpha": 0.90,
                },
                zorder=6,
            )

    ax.set_xlabel(xcol)
    ax.set_ylabel(display_y_label(ycol))

    # Legend is useful for the main variants; omit it from the cleanest curve-only
    # view where the y-axis already identifies the single plotted series.
    if not clean or show_raw:
        ax.legend(loc="best")

    ax.grid(False)
    apply_publication_axis_style(ax)

    xlim_source = xlim_temperature_source(x, raw_loaded)
    x_limits = apply_temperature_xlim(ax, xlim_source, reverse_x=reverse_x, view_limits=view_limits)
    y_series = [(x, y)]
    if show_raw and raw_loaded is not None:
        x_raw, y_raw, _source = raw_loaded
        y_series.append((x_raw, y_raw))
    apply_signal_y_view_limits(ax, y_series, x_limits, view_limits.mgi_y)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_mgi_publication_plots(
    *,
    csv_path: Path,
    args: argparse.Namespace,
    method: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    bends: Sequence[BendPoint],
    outdir: Path,
) -> list[Path]:
    """Write all manuscript-style MGI publication variants."""
    variants = [
        "publication-clean-color",
        "publication-main-color",
        "publication-onsets-color",
        "publication-onsets-clean-color",
        "publication-clean-mono",
        "publication-main-mono",
        "publication-onsets-mono",
        "publication-onsets-clean-mono",
    ]
    out_paths: list[Path] = []
    for variant in variants:
        out_png = outdir / f"{csv_path.stem}-{variant}.png"
        save_mgi_publication_plot(
            path=out_png,
            input_csv=csv_path,
            method=method,
            xcol=xcol,
            ycol=ycol,
            x=x,
            y=y,
            bends=bends,
            reverse_x=not args.no_reverse_x,
            dpi=args.dpi,
            raw_csv=args.raw_csv,
            variant=variant,
            view_limits=getattr(args, "view_limits", NONE_LIMITS),
        )
        out_paths.append(out_png)
    return out_paths

def save_requested_plots(
    *,
    csv_path: Path,
    args: argparse.Namespace,
    method: str,
    xcol: str,
    ycol: str,
    x: np.ndarray,
    y: np.ndarray,
    dy_dx: np.ndarray,
    bends: Sequence[BendPoint],
    candidates: Sequence[CandidatePoint],
    outdir: Path,
) -> Path:
    """
    Save one or more plots according to --plot-set.

    Returns the primary plot path for summaries.
    """
    if args.plot_set == "single":
        modes = [args.plot_mode]
    elif args.plot_set == "all":
        modes = ["smooth", "raw", "raw-smooth", "diagnostic-main", "diagnostic"]
    else:
        modes = ["smooth", "raw-smooth", "diagnostic-main", "diagnostic"]

    if (args.candidate_diagnostics or args.plot_mode == "candidates") and "candidates" not in modes:
        modes.append("candidates")

    primary_path: Path | None = None

    for mode in modes:
        if mode == "smooth":
            out_png = outdir / f"{csv_path.stem}-bends.png"
            save_plot(
                path=out_png,
                title=args.title,
                input_csv=csv_path,
                method=method,
                report_method=args.method,
                xcol=xcol,
                ycol=ycol,
                x=x,
                y=y,
                bends=bends,
                reverse_x=not args.no_reverse_x,
                dpi=args.dpi,
                plot_mode="smooth",
                show_diagnostics=args.show_diagnostics,
                raw_csv=args.raw_csv,
                no_title=args.no_title,
                bw=args.bw,
                view_limits=getattr(args, "view_limits", NONE_LIMITS),
            )
        elif mode == "raw":
            out_png = outdir / f"{csv_path.stem}-raw-bends.png"
            save_plot(
                path=out_png,
                title=args.title,
                input_csv=csv_path,
                method=method,
                report_method=args.method,
                xcol=xcol,
                ycol=ycol,
                x=x,
                y=y,
                bends=bends,
                reverse_x=not args.no_reverse_x,
                dpi=args.dpi,
                plot_mode="raw",
                show_diagnostics=args.show_diagnostics,
                raw_csv=args.raw_csv,
                no_title=args.no_title,
                bw=args.bw,
                view_limits=getattr(args, "view_limits", NONE_LIMITS),
            )
        elif mode == "raw-smooth":
            out_png = outdir / f"{csv_path.stem}-raw-smooth-bends.png"
            save_plot(
                path=out_png,
                title=args.title,
                input_csv=csv_path,
                method=method,
                report_method=args.method,
                xcol=xcol,
                ycol=ycol,
                x=x,
                y=y,
                bends=bends,
                reverse_x=not args.no_reverse_x,
                dpi=args.dpi,
                plot_mode="raw-smooth",
                show_diagnostics=args.show_diagnostics,
                raw_csv=args.raw_csv,
                no_title=args.no_title,
                bw=args.bw,
                view_limits=getattr(args, "view_limits", NONE_LIMITS),
            )
        elif mode == "diagnostic-main":
            out_png = outdir / f"{csv_path.stem}-diagnostic-main-bends.png"
            save_diagnostic_main_plot(
                path=out_png,
                title=args.title,
                input_csv=csv_path,
                method=method,
                report_method=args.method,
                xcol=xcol,
                ycol=ycol,
                x=x,
                y=y,
                bends=bends,
                reverse_x=not args.no_reverse_x,
                dpi=args.dpi,
                raw_csv=args.raw_csv,
                no_title=args.no_title,
                bw=args.bw,
                view_limits=getattr(args, "view_limits", NONE_LIMITS),
            )
        elif mode == "diagnostic":
            out_png = outdir / f"{csv_path.stem}-diagnostic-bends.png"
            save_diagnostic_plot(
                path=out_png,
                title=args.title,
                input_csv=csv_path,
                method=method,
                report_method=args.method,
                xcol=xcol,
                ycol=ycol,
                x=x,
                y=y,
                dy_dx=dy_dx,
                bends=bends,
                reverse_x=not args.no_reverse_x,
                dpi=args.dpi,
                raw_csv=args.raw_csv,
                no_title=args.no_title,
                bw=args.bw,
                view_limits=getattr(args, "view_limits", NONE_LIMITS),
                same_aspect_diagnostics=args.same_aspect_diagnostics,
            )
        elif mode == "candidates":
            out_png = outdir / f"{csv_path.stem}-candidate-diagnostics.png"
            save_candidate_diagnostic_plot(
                path=out_png,
                title=args.title,
                input_csv=csv_path,
                method=method,
                report_method=args.method,
                selection_mode=args.select_mode,
                xcol=xcol,
                ycol=ycol,
                x=x,
                y=y,
                dy_dx=dy_dx,
                candidates=candidates,
                reverse_x=not args.no_reverse_x,
                dpi=args.dpi,
                raw_csv=args.raw_csv,
                no_title=args.no_title,
                bw=args.bw,
                view_limits=getattr(args, "view_limits", NONE_LIMITS),
                same_aspect_diagnostics=args.same_aspect_diagnostics,
            )
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

        if primary_path is None:
            primary_path = out_png

    if args.publication_plots:
        save_mgi_publication_plots(
            csv_path=csv_path,
            args=args,
            method=method,
            xcol=xcol,
            ycol=ycol,
            x=x,
            y=y,
            bends=bends,
            outdir=outdir,
        )

    assert primary_path is not None
    return primary_path



# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------
def analyze_one(csv_path: Path, args: argparse.Namespace) -> AnalysisResult:
    """Analyze one PCHIP CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    xcol, ycol, x, y = read_xy(csv_path, args.xcol, args.ycol)
    method = infer_method(csv_path, ycol)
    dy_dx, d2y_dx2 = compute_derivatives(x, y)

    bends, candidates = detect_bends(
        csv_path=csv_path,
        method=method,
        x=x,
        y=y,
        dy_dx=dy_dx,
        d2y_dx2=d2y_dx2,
        max_bends=args.bends,
        min_separation_C=args.min_separation_C,
        candidate_min_separation_C=args.candidate_min_separation_C,
        min_height_frac=args.min_height_frac,
        prominence_frac=args.prominence_frac,
        min_final_peak_height_frac=args.min_final_peak_height_frac,
        final_peak_dominance_factor=args.final_peak_dominance_factor,
        edge_frac=args.edge_frac,
        report_method=args.method,
        backtrack_frac=args.backtrack_frac,
        pre_width_C=args.pre_width_C,
        pre_gap_C=args.pre_gap_C,
        trans_width_C=args.trans_width_C,
        trans_gap_C=args.trans_gap_C,
        min_fit_points=args.min_fit_points,
        departure_k=args.departure_k,
        departure_min_abs=args.departure_min_abs,
        hybrid_disagree_C=args.hybrid_disagree_C,
        select_mode=args.select_mode,
        use_interpeak_valley_onsets=not args.no_interpeak_valley_onsets,
        interpeak_rise_frac=args.interpeak_rise_frac,
        interpeak_min_neighbor_peak_frac=args.interpeak_min_neighbor_peak_frac,
        max_report_peak_gap_C=args.max_report_peak_gap_C,
        interpeak_report_rise_frac=getattr(args, "interpeak_report_rise_frac", 0.10),
        forced_valley_report_floor=getattr(args, "forced_valley_report_floor", 0.05),
    )

    base_outdir = args.outdir if args.outdir is not None else csv_path.parent
    outdir = resolve_output_dir_for_limits(base_outdir, args, getattr(args, "view_limits", NONE_LIMITS))
    out_csv = outdir / f"{csv_path.stem}-bends.csv"
    out_candidates_csv = outdir / f"{csv_path.stem}-candidates.csv"

    write_bends_csv(out_csv, bends)
    if args.candidate_diagnostics or args.plot_mode == "candidates":
        write_candidates_csv(out_candidates_csv, candidates)
    else:
        out_candidates_csv = None

    out_png = save_requested_plots(
        csv_path=csv_path,
        args=args,
        method=method,
        xcol=xcol,
        ycol=ycol,
        x=x,
        y=y,
        dy_dx=dy_dx,
        bends=bends,
        candidates=candidates,
        outdir=outdir,
    )

    write_view_limits_json(
        outdir / "view-limits.json",
        limits=getattr(args, "view_limits", NONE_LIMITS),
        content_type="mgi",
        output_mode="limited-view" if getattr(args, "view_limits", NONE_LIMITS).any else "base",
        extra={"primary_y_axis": "MGI", "derivative_y_axis_manual_limits_applied": False,
               "onset_reporting": {
                   "interpeak_rise_frac": args.interpeak_rise_frac,
                   "interpeak_report_rise_frac": getattr(args, "interpeak_report_rise_frac", 0.10),
                   "forced_valley_report_floor": getattr(args, "forced_valley_report_floor", 0.05),
                   "rule": "separate_valley_report_with_forced_only_floor",
               }},
    )

    return AnalysisResult(
        input_csv=csv_path,
        method=method,
        xcol=xcol,
        ycol=ycol,
        x=x,
        y=y,
        dy_dx=dy_dx,
        d2y_dx2=d2y_dx2,
        bends=bends,
        candidates=candidates,
        out_csv=out_csv,
        out_png=out_png,
        out_candidates_csv=out_candidates_csv,
        out_candidates_png=(outdir / f"{csv_path.stem}-candidate-diagnostics.png") if (args.candidate_diagnostics or args.plot_mode == "candidates") else None,
    )


def write_summary(path: Path, results: Sequence[AnalysisResult]) -> None:
    """Write combined bend point summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "input_csv",
        "method",
        "report_mode",
        "bend_index",
        "temperature_C",
        "value",
        "rough_onset_temperature_C",
        "peak_temperature_C",
        "dY_dT_at_bend",
        "dY_dT_at_peak",
        "abs_dY_dT_at_bend",
        "abs_dY_dT_at_peak",
        "prominence",
        "used_fallback",
        "output_csv",
        "output_png",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            for bend in result.bends:
                writer.writerow(
                    {
                        "input_csv": bend.input_csv,
                        "method": bend.method,
                        "report_mode": bend.report_mode,
                        "bend_index": bend.rank,
                        "temperature_C": f"{bend.temperature_C:.6g}",
                        "value": f"{bend.value:.6g}",
                        "rough_onset_temperature_C": f"{bend.rough_onset_temperature_C:.6g}",
                        "peak_temperature_C": f"{bend.peak_temperature_C:.6g}",
                        "dY_dT_at_bend": f"{bend.derivative_at_bend:.6g}",
                        "dY_dT_at_peak": f"{bend.derivative_at_peak:.6g}",
                        "abs_dY_dT_at_bend": f"{bend.abs_derivative_at_bend:.6g}",
                        "abs_dY_dT_at_peak": f"{bend.abs_derivative_at_peak:.6g}",
                        "prominence": f"{bend.prominence:.6g}",
                        "used_fallback": bend.used_fallback,
                        "output_csv": str(result.out_csv),
                        "output_png": str(result.out_png),
                    }
                )


def print_result(result: AnalysisResult) -> None:
    """Print user-friendly result summary."""
    print(f"Input CSV:    {result.input_csv}")
    print(f"Method:       {result.method}")
    print(f"Columns:      X='{result.xcol}', Y='{result.ycol}'")
    print(f"Wrote CSV:    {result.out_csv}")
    if result.out_candidates_csv is not None:
        print(f"Wrote candidates CSV:  {result.out_candidates_csv}")
    print(f"Wrote plot:   {result.out_png}")
    if result.out_candidates_png is not None:
        print(f"Wrote candidates plot: {result.out_candidates_png}")

    if not result.bends:
        print("Bend points:  none detected")
    else:
        print("Bend points:")
        for bend in result.bends:
            fallback = " fallback" if bend.used_fallback else ""
            print(
                f"  b{bend.rank}: {bend.temperature_C:.3f} °C, "
                f"{result.ycol}={bend.value:.6g}, "
                f"rough={bend.rough_onset_temperature_C:.3f} °C, "
                f"peak={bend.peak_temperature_C:.3f} °C{fallback}"
            )
    print()


def resolve_mgi_bends_from_config(args: argparse.Namespace) -> None:
    """Resolve MGI bend/onset count from analysis/onset-config.ini."""
    analysis_dir = infer_analysis_dir_from_paths(
        analysis_dir=args.analysis_dir,
        input_csv=args.csv_files[0] if getattr(args, "csv_files", None) else None,
        output_dir=args.outdir,
        fallback=Path.cwd(),
    )
    cfg = ensure_onset_config(
        analysis_dir,
        dry_run=False,
        experiment_name=analysis_dir.parent.parent.parent.parent.name
        if len(analysis_dir.parts) >= 4
        else str(analysis_dir),
    )

    explicit_bends = args.bends
    if explicit_bends is None:
        args.bends = cfg.mgi_onsets
    else:
        if int(explicit_bends) < 0:
            raise ValueError(f"--bends must be an integer >= 0, got: {explicit_bends}")
        if int(explicit_bends) != int(cfg.mgi_onsets):
            print(
                "!" * 72
                + "\n"
                + "[WARNING:onset-config] explicit --bends differs from onset-config.ini\n\n"
                + f"Config file: {cfg.path}\n"
                + f"[MGI] onsets in config: {cfg.mgi_onsets}\n"
                + f"Explicit --bends for this run: {explicit_bends}\n\n"
                + "This command-line value is a one-off override. Edit "
                + "analysis/onset-config.ini to make the decision persistent.\n"
                + "!" * 72
            )
        args.bends = int(explicit_bends)

    args.onset_config_status = cfg.status
    args.onset_config_path = str(cfg.path)
    args.onset_config_mgi_onsets = cfg.mgi_onsets
    args.onset_config_fbrm_onsets = cfg.fbrm_onsets


def main() -> int:
    """Run command-line tool."""
    args = parse_args()
    try:
        resolve_mgi_bends_from_config(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    try:
        args.view_limits = parse_view_limits_from_args(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not (0 < args.backtrack_frac <= 1):
        print("[error] --backtrack-frac must be in range 0 < value <= 1", file=sys.stderr)
        return 2
    if args.min_fit_points < 2:
        print("[error] --min-fit-points must be at least 2", file=sys.stderr)
        return 2
    if args.interpeak_rise_frac < 0:
        print("[error] --interpeak-rise-frac must be non-negative", file=sys.stderr)
        return 2
    if args.interpeak_min_neighbor_peak_frac < 0:
        print("[error] --interpeak-min-neighbor-peak-frac must be non-negative", file=sys.stderr)
        return 2
    if args.max_report_peak_gap_C < 0:
        print("[error] --max-report-peak-gap-C must be non-negative", file=sys.stderr)
        return 2

    try:
        inputs = autodetect_inputs(args.csv_files)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    results: list[AnalysisResult] = []

    for csv_path in inputs:
        try:
            result = analyze_one(csv_path, args)
        except Exception as exc:
            print(f"[error] Failed to analyze {csv_path}: {exc}", file=sys.stderr)
            return 1

        print_result(result)
        results.append(result)

    if len(results) > 1:
        summary_path = args.summary
        if not summary_path.is_absolute() and args.outdir is not None:
            summary_path = args.outdir / summary_path
        write_summary(summary_path, results)
        print(f"Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
