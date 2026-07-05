#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fbrm-onsets-v14.6.py
-----------------

Sequential FBRM Total Counts onset detection using measured/interpolated
temperature while preserving the original measurement order.

Why v6 exists
=============
Earlier versions sorted FBRM data by ``Tr (°C)`` before analysis. That is not
safe for FBRM data because Total Counts vs. temperature can become multivalued,
especially after the high-count / compromised region. v5 preserved
the input row order as the measurement/cooling order and uses ``Tr (°C)`` only
as the reported physical temperature axis.

Critical temperature-axis rule
==============================
``Tr_regular`` is deliberately not used. It belongs to the old SSR
audit/prototype workflow and is an external linearized temperature model, not
the measured/interpolated physical temperature axis used for final reported
FBRM onset temperatures.

FBRM onset model
================
The detector is sequential:

1. Preserve input row order. Do not sort by temperature.
2. Detect the actual cooling segment after any high-temperature plateau.
3. Stop onset search at the first high-count threshold crossing inside that cooling segment.
3. Detect T1 as the first sustained departure from the initial clear-solution
   baseline. A baseline dip is not an onset. An isolated bump is not enough.
4. Detect T2 and later onsets locally after the previous accepted onset.
   T2 may be a level shift, slope increase, sustained rise, or a shoulder inside
   an already steep rise.
5. Report 0..N valid onsets, where N is ``--onsets``. The default N is 2, but
   two onsets are never forced.

High-count search limit
=======================
By default, onset search stops when raw Total Counts first reaches 10000:

    --max-search-count 10000

This is an analysis/search limit unless separately confirmed as an instrument
validity limit. Use ``--max-search-count 0`` to disable it or provide a different
value if the accepted FBRM count limit is known.

Typical use
===========

    cd /path/to/experiment
    $REPO/scripts/fbrm-onsets.py

Outputs
=======

    fbrm_onsets/fbrm-onsets.csv
    fbrm_onsets/fbrm-onset-candidates.csv
    fbrm_onsets/fbrm-onset-params.json
    fbrm_onsets/fbrm-onset-report.txt
    fbrm_onsets/fbrm-onset-publication.png
    fbrm_onsets/fbrm-onset-diagnostic.png
    fbrm_onsets/fbrm-onset-candidates.png

With ``--publication-plots``:

    fbrm_onsets/fbrm-onset-publication-*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from onset_config import ensure_onset_config, infer_analysis_dir_from_paths

from onset_view_limits import (
    NONE_LIMITS,
    add_view_limit_arguments,
    apply_signal_y_view_limits,
    apply_temperature_view_limits,
    parse_view_limits_from_args,
    resolve_output_dir_for_limits,
    write_view_limits_json,
)
from matplotlib.ticker import MultipleLocator, NullLocator

try:
    from scipy.signal import savgol_filter
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[error] scipy is required. Run this in the py314/lab314 environment."
    ) from exc


DEFAULT_INPUT = Path("ts-fbrm-tr.csv")
DEFAULT_OUTPUT_DIR = Path("fbrm_onsets")

TEMPERATURE_COLUMNS = ("Tr (°C)", "Tr_C", "Tr", "Temperature", "Temperature (°C)")
COUNT_COLUMNS = ("Total Counts", "FBRM Total Counts", "Total_Counts", "Counts")
TIME_COLUMNS = ("Timestamp", "timestamp", "time", "Time", "ts")
FORBIDDEN_COLUMNS = ("Tr_regular",)


@dataclass(frozen=True)
class ReferenceOnset:
    """Optional IA/PCHIP reference onset used only for comparison/plotting."""

    label: str
    Tr_C: float
    source_file: str


@dataclass(frozen=True)
class ReferencePlotGroup:
    """One plotted reference-onset marker.

    Multiple RAW/SG reference rows can map to one physical onset marker when
    their temperatures are close enough. This keeps reference overlays readable
    without removing the reference-onset information from CSV/JSON outputs.
    """

    label: str
    Tr_C: float
    members: tuple[ReferenceOnset, ...]


@dataclass(frozen=True)
class InitialBaseline:
    """Initial clear-solution baseline model."""

    start_index: int
    end_index: int
    n_points: int
    progress_start_C: float
    progress_end_C: float
    Tr_start_C: float
    Tr_end_C: float
    median: float
    slope: float
    intercept: float
    noise_sigma: float
    threshold: float


@dataclass(frozen=True)
class SeriesData:
    """FBRM data in original measurement/cooling order."""

    input_csv: str
    xcol: str
    ycol: str
    order_mode: str
    cooling_start_method: str
    cooling_start_input_row: int
    cooling_start_original_index: int
    cooling_start_Tr_C: float
    cooling_start_count: float
    cooling_start_note: str
    original_rows: np.ndarray
    Tr_C: np.ndarray
    progress_C: np.ndarray
    y_raw: np.ndarray
    y_smooth: np.ndarray
    dy_dprogress: np.ndarray
    d2y_dprogress2: np.ndarray
    analysis_mask: np.ndarray
    cutoff_index: int | None
    cutoff_Tr_C: float | None
    cutoff_count: float | None
    initial_baseline: InitialBaseline


@dataclass
class CandidatePoint:
    """One candidate event before/after final selection."""

    candidate_id: int
    target_onset: str
    status: str
    reason: str
    acceptance_mode: str
    accepted: bool
    final_rank: int | None

    input_row: int
    Tr_C: float
    cooling_progress_C: float
    total_counts_raw: float
    total_counts_smooth: float

    initial_baseline_pred: float
    initial_shift: float
    initial_future_min_shift: float
    t1_post_level_shift: float
    t1_post_level_threshold: float
    t1_post_level_ok: bool

    pre_baseline_at_onset: float
    pre_median: float
    post_median: float
    confirm_median: float
    confirm_min: float

    level_shift: float
    sustained_shift: float
    future_min_shift: float
    post_rise: float

    pre_slope: float
    post_slope: float
    confirm_slope: float
    slope_increase: float
    slope_ratio: float
    shoulder_score: float

    derivative_at_onset: float
    second_derivative_at_onset: float
    local_noise_sigma: float
    level_threshold: float
    sustained_threshold: float
    slope_threshold: float
    post_rise_threshold: float

    onset_index: int
    run_start_index: int
    run_end_index: int
    pre_window_start_Tr_C: float | None
    pre_window_end_Tr_C: float | None
    post_window_start_Tr_C: float | None
    post_window_end_Tr_C: float | None
    confirm_window_start_Tr_C: float | None
    confirm_window_end_Tr_C: float | None
    n_pre_points: int
    n_post_points: int
    n_confirm_points: int

    nearest_reference_onset: str | None = None
    nearest_reference_Tr_C: float | None = None
    delta_to_reference_C: float | None = None
    reference_file: str | None = None


@dataclass(frozen=True)
class OnsetPoint:
    """Final accepted onset record."""

    onset_label: str
    rank: int
    status: str
    reason: str
    input_row: int | None
    Tr_C: float | None
    total_counts_raw: float | None
    total_counts_smooth: float | None
    cooling_progress_C: float | None
    candidate_id: int | None
    acceptance_mode: str | None
    method: str
    source_file: str
    x_column: str
    y_column: str
    nearest_reference_onset: str | None = None
    nearest_reference_Tr_C: float | None = None
    delta_to_reference_C: float | None = None
    reference_file: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Sequential FBRM Total Counts onset detection from measured/"
            "interpolated Tr (°C). Tr_regular is deliberately not used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", "--csv", dest="input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--xcol", type=str, default=None)
    p.add_argument("--ycol", type=str, default=None)
    p.add_argument("--time-col", type=str, default=None)
    p.add_argument("--output-dir", "--outdir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Analysis directory containing onset-config.ini. If omitted, the "
            "script infers it from --output-dir or --input."
        ),
    )

    p.add_argument(
        "--order",
        choices=("input", "time"),
        default="input",
        help="Measurement order. Default preserves input row order. Use 'time' only if a time column is available.",
    )
    p.add_argument(
        "--reverse-input",
        action="store_true",
        help="Reverse rows after loading. Use only if input file is known to be in reverse measurement order.",
    )

    p.add_argument(
        "--cooling-start",
        choices=("auto", "max-temp", "row", "none"),
        default="auto",
        help=(
            "How to choose the start of the actual cooling segment. auto detects "
            "the end of a high-temperature plateau; max-temp starts at maximum Tr; "
            "row uses --cooling-start-row; none starts at the first loaded row."
        ),
    )
    p.add_argument(
        "--cooling-start-row",
        type=int,
        default=None,
        help=(
            "CSV/input row for --cooling-start row. If it matches an original CSV "
            "row index, that row is used; otherwise it is interpreted as a zero-based "
            "loaded-row position."
        ),
    )
    p.add_argument(
        "--cooling-high-band-C",
        type=float,
        default=2.0,
        help="For auto cooling start, only consider points within this many °C of the maximum Tr.",
    )
    p.add_argument(
        "--cooling-min-drop-C",
        type=float,
        default=0.75,
        help="Minimum confirmed temperature drop required to detect actual cooling start.",
    )
    p.add_argument(
        "--cooling-confirm-points",
        type=int,
        default=300,
        help="Look-ahead points used to confirm sustained cooling from a high-temperature plateau.",
    )
    p.add_argument(
        "--cooling-savgol-window",
        type=int,
        default=301,
        help="Savitzky-Golay window for temperature smoothing used only for cooling-start detection.",
    )

    p.add_argument(
        "--onsets",
        type=int,
        default=None,
        help=(
            "Maximum accepted onsets to report. If omitted, [FBRM] onsets from "
            "analysis/onset-config.ini is used. Use 0 for no accepted onsets."
        ),
    )
    p.add_argument("--min-separation-C", type=float, default=1.0)

    p.add_argument(
        "--max-search-count",
        "--max-valid-count",
        dest="max_search_count",
        type=float,
        default=10000.0,
        help=(
            "Stop onset search when raw Total Counts first reaches this value in measurement order. "
            "Use 0 to disable. This is an analysis search limit unless separately confirmed."
        ),
    )
    p.add_argument("--min-points-before-limit", type=int, default=80)

    p.add_argument("--smooth-method", choices=("savgol", "none"), default="savgol")
    p.add_argument("--savgol-window", type=int, default=101)
    p.add_argument("--savgol-polyorder", type=int, default=3)

    p.add_argument("--baseline-start-after-C", type=float, default=0.20, help="Ignore this much cooling progress after detected cooling start before estimating initial baseline.")
    p.add_argument("--initial-baseline-window-C", type=float, default=3.0)
    p.add_argument("--initial-baseline-min-points", type=int, default=20)
    p.add_argument("--initial-baseline-k-sigma", type=float, default=6.0)
    p.add_argument("--min-initial-shift-abs", type=float, default=150.0)
    p.add_argument("--min-search-progress-C", type=float, default=0.75)
    p.add_argument(
        "--t1-post-level-delay-C",
        type=float,
        default=0.25,
        help=(
            "Delay after a T1 candidate before testing whether the post-candidate "
            "level has truly left the clear-solution baseline."
        ),
    )
    p.add_argument(
        "--t1-post-level-window-C",
        type=float,
        default=1.50,
        help=(
            "End of the delayed T1 post-level persistence window in cooling-progress °C. "
            "The tested window is [delay, window] after the candidate."
        ),
    )
    p.add_argument(
        "--t1-post-level-percentile",
        type=float,
        default=25.0,
        help=(
            "Percentile of the delayed post-candidate residuals used for T1 persistence. "
            "A low percentile requires the level to stay elevated, not only spike."
        ),
    )
    p.add_argument(
        "--t1-min-post-level-shift-abs",
        type=float,
        default=60.0,
        help="Minimum delayed T1 post-level residual shift in Total Counts.",
    )
    p.add_argument(
        "--t1-min-post-level-shift-frac",
        type=float,
        default=0.01,
        help="Minimum delayed T1 post-level residual shift as a fraction of prefix dynamic range.",
    )
    p.add_argument(
        "--t1-post-level-k-sigma",
        type=float,
        default=3.0,
        help="Noise multiplier for the delayed T1 post-level persistence gate.",
    )
    p.add_argument(
        "--onset-threshold-frac",
        type=float,
        default=0.50,
        help=(
            "Fraction of the confirmation threshold used for reporting the onset position. "
            "The full threshold is still used for accepting/confirming the event. "
            "The default is conservative enough to avoid reporting before a pre-rise valley, "
            "while still usually being earlier than the final confirmation point."
        ),
    )

    p.add_argument("--pre-window-C", type=float, default=1.20)
    p.add_argument("--pre-gap-C", type=float, default=0.05)
    p.add_argument("--post-window-C", type=float, default=0.80)
    p.add_argument("--post-gap-C", type=float, default=0.05)
    p.add_argument("--confirm-window-C", type=float, default=1.20)
    p.add_argument("--confirm-gap-C", type=float, default=0.15)
    p.add_argument("--min-pre-points", type=int, default=8)
    p.add_argument("--min-post-points", type=int, default=8)
    p.add_argument("--min-confirm-points", type=int, default=8)

    p.add_argument("--level-k-sigma", type=float, default=4.0)
    p.add_argument("--min-level-shift-abs", type=float, default=60.0)
    p.add_argument("--min-sustained-shift-abs", type=float, default=60.0)
    p.add_argument("--max-return-below-baseline-abs", type=float, default=20.0)
    p.add_argument("--min-post-rise-abs", type=float, default=40.0)
    p.add_argument("--slope-k-sigma", type=float, default=3.0)
    p.add_argument("--min-slope-increase", type=float, default=0.0)
    p.add_argument("--min-shoulder-pre-slope", type=float, default=20.0)
    p.add_argument("--shoulder-slope-factor", type=float, default=1.6)
    p.add_argument("--min-valid-run-points", type=int, default=3)
    p.add_argument("--refine-window-C", type=float, default=0.90)
    p.add_argument("--persistent-points", type=int, default=3)
    p.add_argument(
        "--no-valley-aware-refine",
        action="store_true",
        help=(
            "Disable valley-aware onset refinement. By default, if a local minimum "
            "exists inside the accepted event window, the reported onset is not allowed "
            "to move before that pre-rise valley."
        ),
    )

    p.add_argument("--reference-onsets", type=Path, default=None)
    p.add_argument("--reference-xcol", type=str, default=None)
    p.add_argument("--reference-label-col", type=str, default=None)
    p.add_argument(
        "--reference-display",
        choices=("grouped", "individual", "none"),
        default="grouped",
        help=(
            "How to draw optional reference-onset markers. grouped combines "
            "nearby RAW/SG references for the same physical onset and labels "
            "them as Ref T1, Ref T2, etc.; individual draws every reference row; "
            "none reads references for CSV comparison but does not plot them."
        ),
    )
    p.add_argument(
        "--reference-cluster-tolerance-C",
        type=float,
        default=0.35,
        help=(
            "Maximum temperature spread for grouping reference-onset rows into "
            "one plotted marker. If same-rank RAW/SG references differ more than "
            "this, separate markers such as ref T1a/ref T1b are drawn."
        ),
    )
    p.add_argument(
        "--reference-rank-cycle",
        type=int,
        default=None,
        help=(
            "Fallback T-rank cycle length for reference CSVs without a label/rank "
            "column. By default, the script infers a likely RAW/SG cycle from the "
            "number of reference rows instead of blindly using --onsets."
        ),
    )
    p.add_argument(
        "--reference-label-prefix",
        type=str,
        default="Ref ",
        help="Prefix used for plotted reference-onset labels.",
    )

    p.add_argument("--no-reverse-x", action="store_true")
    p.add_argument("--no-title", action="store_true")
    p.add_argument(
        "--publication-plots",
        action="store_true",
        help=(
            "Write additional manuscript-style FBRM publication plot variants: "
            "clean/main/onsets, color/mono, with and without raw data. "
            "These plots do not change onset detection or CSV outputs."
        ),
    )
    p.add_argument(
        "--same-aspect-diagnostics",
        action="store_true",
        help=(
            "Use the same canvas aspect ratio for diagnostic/candidate plots as "
            "for the main/publication figures. Default keeps diagnostic figures "
            "taller for QC reading."
        ),
    )
    p.add_argument("--plot-full-data", action="store_true", help="Plot outside-search data as faint context.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--verbose", action="store_true")

    add_view_limit_arguments(p)

    return p.parse_args()


def autodetect_column(columns: Sequence[str], explicit: str | None, candidates: Sequence[str], role: str) -> str:
    """Return explicit column or auto-detected column."""
    if explicit is not None:
        if explicit not in columns:
            raise KeyError(f"{role} column '{explicit}' not found. Available columns: {list(columns)}")
        return explicit

    lowered = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        key = name.lower().strip()
        if key in lowered:
            return lowered[key]

    raise KeyError(
        f"Could not auto-detect {role} column. Tried {list(candidates)}. "
        f"Available columns: {list(columns)}"
    )


def validate_no_forbidden_axis(xcol: str, ycol: str) -> None:
    """Fail if Tr_regular is selected."""
    for forbidden in FORBIDDEN_COLUMNS:
        if xcol == forbidden or ycol == forbidden:
            raise ValueError(
                f"Forbidden column '{forbidden}' selected. Final FBRM onset "
                "temperatures must use measured/interpolated Tr (°C)."
            )


def adjusted_savgol_window(n: int, requested: int, polyorder: int) -> int:
    """Return a valid odd Savitzky-Golay window or 0 if impossible."""
    if n <= polyorder + 2:
        return 0
    w = int(requested)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    min_w = polyorder + 2
    if min_w % 2 == 0:
        min_w += 1
    w = max(w, min_w)
    return w if w > polyorder and w < n else 0


def smooth_array(y: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Smooth one contiguous array or return raw copy."""
    if args.smooth_method == "none":
        return y.astype(float).copy()
    w = adjusted_savgol_window(len(y), args.savgol_window, args.savgol_polyorder)
    if w <= 0:
        return y.astype(float).copy()
    return savgol_filter(y, window_length=w, polyorder=args.savgol_polyorder, mode="interp")


def robust_sigma(values: np.ndarray) -> float:
    """Robust sigma estimate using MAD; falls back to std."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(values))


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = m*x + b."""
    if len(x) < 2 or float(np.nanmax(x)) <= float(np.nanmin(x)):
        return float("nan"), float("nan")
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def line_value(m: float, b: float, x: float) -> float:
    """Evaluate fitted line."""
    if not (np.isfinite(m) and np.isfinite(b)):
        return float("nan")
    return float(m * x + b)


def build_progress(Tr: np.ndarray) -> np.ndarray:
    """Build monotonic cooling progress from measurement-order Tr.

    The physical Tr values are retained for reporting. For windowing, small
    nonmonotonic temperature jitter is handled by cumulative maximum cooling
    progress.
    """
    raw_progress = float(Tr[0]) - Tr
    return np.maximum.accumulate(raw_progress)


def find_cutoff_index(y_raw: np.ndarray, max_count: float) -> int | None:
    """Return first index where raw counts reach the high-count search limit."""
    if max_count <= 0:
        return None
    hits = np.where(y_raw >= max_count)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def make_analysis_mask(n: int, cutoff_index: int | None) -> np.ndarray:
    """Return mask for prefix data before cutoff."""
    mask = np.ones(n, dtype=bool)
    if cutoff_index is not None:
        mask[cutoff_index:] = False
    return mask


def compute_initial_baseline(
    progress: np.ndarray,
    Tr: np.ndarray,
    y_smooth: np.ndarray,
    analysis_mask: np.ndarray,
    args: argparse.Namespace,
) -> InitialBaseline:
    """Estimate initial clear-solution baseline after detected cooling start.

    A long high-temperature plateau before actual cooling must not dominate the
    clear-solution baseline. The baseline window therefore starts after
    ``--baseline-start-after-C`` cooling progress and spans
    ``--initial-baseline-window-C``.
    """
    start_p = max(0.0, float(args.baseline_start_after_C))
    end_p = start_p + float(args.initial_baseline_window_C)

    idx = np.where(
        analysis_mask
        & (progress >= start_p)
        & (progress <= end_p)
    )[0]

    if idx.size < int(args.initial_baseline_min_points):
        valid = np.where(analysis_mask & (progress >= start_p))[0]
        idx = valid[: max(int(args.initial_baseline_min_points), 1)]

    if idx.size == 0:
        raise ValueError(
            "No points available for initial baseline after cooling-start selection. "
            "Check cooling-start detection, --baseline-start-after-C, and the high-count search limit."
        )

    p = progress[idx]
    y = y_smooth[idx]
    m, b = fit_line(p, y)
    pred = m * p + b if np.isfinite(m) and np.isfinite(b) else np.full_like(y, np.median(y))
    resid = y - pred
    noise = robust_sigma(resid)
    if not np.isfinite(noise):
        noise = 0.0
    threshold = max(float(args.min_initial_shift_abs), float(args.initial_baseline_k_sigma) * noise)

    return InitialBaseline(
        start_index=int(idx[0]),
        end_index=int(idx[-1]),
        n_points=int(idx.size),
        progress_start_C=float(progress[idx[0]]),
        progress_end_C=float(progress[idx[-1]]),
        Tr_start_C=float(Tr[idx[0]]),
        Tr_end_C=float(Tr[idx[-1]]),
        median=float(np.median(y)),
        slope=float(m),
        intercept=float(b),
        noise_sigma=float(noise),
        threshold=float(threshold),
    )


def smooth_temperature_for_cooling_start(Tr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Smooth temperature only for cooling-start detection."""
    n = len(Tr)
    if n < 7:
        return Tr.astype(float).copy()
    poly = 2
    w = adjusted_savgol_window(n, int(args.cooling_savgol_window), poly)
    if w <= 0:
        return Tr.astype(float).copy()
    return savgol_filter(Tr.astype(float), window_length=w, polyorder=poly, mode="interp")


def detect_cooling_start_index(
    Tr: np.ndarray,
    y_raw: np.ndarray,
    original_rows: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, str, str]:
    """Detect the start of the actual cooling segment.

    The auto method looks for the first point near the maximum temperature from
    which a confirmed temperature drop occurs within a limited look-ahead
    window. This skips long high-temperature plateaus but does not sort by
    temperature.
    """
    n = len(Tr)
    if n == 0:
        raise ValueError("No data rows available for cooling-start detection.")

    mode = str(args.cooling_start)
    if mode == "none":
        return 0, "none", "started at first loaded row"

    if mode == "row":
        if args.cooling_start_row is None:
            raise ValueError("--cooling-start row requires --cooling-start-row N")
        row = int(args.cooling_start_row)
        matches = np.where(original_rows == row)[0]
        if matches.size:
            idx = int(matches[0])
            return idx, "row", f"used original CSV row {row}"
        if 0 <= row < n:
            return row, "row", f"used loaded-row position {row}"
        raise ValueError(f"--cooling-start-row {row} is outside loaded data range 0..{n-1}")

    if mode == "max-temp":
        idx = int(np.nanargmax(Tr))
        return idx, "max-temp", "started at maximum Tr"

    if mode != "auto":
        raise ValueError(f"Unknown cooling-start mode: {mode}")

    Tr_s = smooth_temperature_for_cooling_start(Tr, args)
    max_Tr = float(np.nanmax(Tr_s))
    high_limit = max_Tr - float(args.cooling_high_band_C)
    min_drop = float(args.cooling_min_drop_C)
    look = max(5, int(args.cooling_confirm_points))

    best_idx = None
    for i in range(0, n - 2):
        if not np.isfinite(Tr_s[i]) or Tr_s[i] < high_limit:
            continue

        j2 = min(n, i + look)
        if j2 <= i + 2:
            continue

        future = Tr_s[i:j2]
        future_min = float(np.nanmin(future))
        if Tr_s[i] - future_min < min_drop:
            continue

        # Confirm that the look-ahead segment has a negative trend, not just one
        # isolated point.
        rel = np.arange(len(future), dtype=float)
        slope, _ = fit_line(rel, future)
        if np.isfinite(slope) and slope < 0:
            best_idx = i
            break

    if best_idx is not None:
        return int(best_idx), "auto", (
            f"first high-temperature point with >= {min_drop:g} °C confirmed "
            f"drop within {look} points"
        )

    # Conservative fallback: maximum temperature. Report clearly.
    idx = int(np.nanargmax(Tr_s))
    return idx, "auto-fallback-max-temp", (
        "auto cooling-start detection failed; fell back to maximum smoothed Tr"
    )


def read_data(args: argparse.Namespace) -> SeriesData:
    """Read FBRM CSV, preserve measurement order, choose cooling segment, and build prefix data."""
    df = pd.read_csv(args.input_csv, encoding="utf-8")
    columns = list(df.columns)

    xcol = autodetect_column(columns, args.xcol, TEMPERATURE_COLUMNS, "temperature")
    ycol = autodetect_column(columns, args.ycol, COUNT_COLUMNS, "FBRM Total Counts")
    validate_no_forbidden_axis(xcol, ycol)

    if args.order == "time":
        tcol = autodetect_column(columns, args.time_col, TIME_COLUMNS, "time")
        df = df.sort_values(tcol, kind="mergesort")

    original_rows = df.index.to_numpy(dtype=int)

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    original_rows = original_rows[mask]

    if args.reverse_input:
        x = x[::-1]
        y = y[::-1]
        original_rows = original_rows[::-1]

    if len(x) < 20:
        raise ValueError(f"Not enough numeric data in {args.input_csv}")

    cooling_start_idx, cooling_method, cooling_note = detect_cooling_start_index(
        x, y, original_rows, args
    )
    cooling_start_original_index = int(cooling_start_idx)
    cooling_start_input_row = int(original_rows[cooling_start_idx])
    cooling_start_Tr = float(x[cooling_start_idx])
    cooling_start_count = float(y[cooling_start_idx])

    # Trim to actual cooling segment. Original row numbers are kept for output.
    x = x[cooling_start_idx:]
    y = y[cooling_start_idx:]
    original_rows = original_rows[cooling_start_idx:]

    if len(x) < 20:
        raise ValueError(
            "Too few rows remain after cooling-start selection. "
            "Try --cooling-start max-temp, --cooling-start none, or --cooling-start-row N."
        )

    progress = build_progress(x)
    cutoff_index = find_cutoff_index(y, float(args.max_search_count))
    analysis_mask = make_analysis_mask(len(y), cutoff_index)

    # Smooth only the analyzable prefix. This prevents high-count/loop data after
    # the cutoff from affecting the onset-region smooth curve.
    y_smooth = np.full_like(y, np.nan, dtype=float)
    prefix_idx = np.where(analysis_mask)[0]
    if prefix_idx.size:
        y_smooth[prefix_idx] = smooth_array(y[prefix_idx], args)

    # For optional context plotting, smooth the outside part separately so no
    # smoothing kernel crosses the cutoff boundary.
    outside_idx = np.where(~analysis_mask)[0]
    if outside_idx.size:
        y_smooth[outside_idx] = smooth_array(y[outside_idx], args)

    # Derivatives are meaningful only for the prefix analysis region.
    dy = np.full_like(y, np.nan, dtype=float)
    d2y = np.full_like(y, np.nan, dtype=float)
    if prefix_idx.size >= 3:
        pfx_p = progress[prefix_idx]
        pfx_y = y_smooth[prefix_idx]
        pfx_p2 = pfx_p.astype(float).copy()
        for i in range(1, len(pfx_p2)):
            if pfx_p2[i] <= pfx_p2[i - 1]:
                pfx_p2[i] = pfx_p2[i - 1] + 1e-9
        dy_p = np.gradient(pfx_y, pfx_p2)
        d2_p = np.gradient(dy_p, pfx_p2)
        dy[prefix_idx] = dy_p
        d2y[prefix_idx] = d2_p

    initial = compute_initial_baseline(progress, x, y_smooth, analysis_mask, args)

    cutoff_Tr = None
    cutoff_count = None
    if cutoff_index is not None:
        cutoff_Tr = float(x[cutoff_index])
        cutoff_count = float(y[cutoff_index])

    return SeriesData(
        input_csv=str(args.input_csv),
        xcol=xcol,
        ycol=ycol,
        order_mode=args.order,
        cooling_start_method=cooling_method,
        cooling_start_input_row=cooling_start_input_row,
        cooling_start_original_index=cooling_start_original_index,
        cooling_start_Tr_C=cooling_start_Tr,
        cooling_start_count=cooling_start_count,
        cooling_start_note=cooling_note,
        original_rows=original_rows,
        Tr_C=x,
        progress_C=progress,
        y_raw=y,
        y_smooth=y_smooth,
        dy_dprogress=dy,
        d2y_dprogress2=d2y,
        analysis_mask=analysis_mask,
        cutoff_index=cutoff_index,
        cutoff_Tr_C=cutoff_Tr,
        cutoff_count=cutoff_count,
        initial_baseline=initial,
    )


def window_indices(progress: np.ndarray, start: float, end: float) -> np.ndarray:
    """Indices with progress in [start, end]."""
    lo = min(float(start), float(end))
    hi = max(float(start), float(end))
    return np.where((progress >= lo) & (progress <= hi))[0]


def local_metrics(idx: int, target_rank: int, data: SeriesData, args: argparse.Namespace) -> dict[str, Any]:
    """Compute local and initial-baseline metrics around an index."""
    p = float(data.progress_C[idx])

    if target_rank == 1 and p < float(args.min_search_progress_C):
        return {"valid_windows": False, "skip_reason": "before_min_search_progress"}

    pre_idx = window_indices(data.progress_C, p - args.pre_gap_C - args.pre_window_C, p - args.pre_gap_C)
    post_idx = window_indices(data.progress_C, p + args.post_gap_C, p + args.post_gap_C + args.post_window_C)
    conf_idx = window_indices(
        data.progress_C,
        p + args.confirm_gap_C,
        p + args.confirm_gap_C + args.confirm_window_C,
    )

    pre_idx = pre_idx[data.analysis_mask[pre_idx]]
    post_idx = post_idx[data.analysis_mask[post_idx]]
    conf_idx = conf_idx[data.analysis_mask[conf_idx]]

    out: dict[str, Any] = {
        "valid_windows": False,
        "n_pre": int(len(pre_idx)),
        "n_post": int(len(post_idx)),
        "n_confirm": int(len(conf_idx)),
    }

    if (
        len(pre_idx) < args.min_pre_points
        or len(post_idx) < args.min_post_points
        or len(conf_idx) < args.min_confirm_points
    ):
        return out

    pre_p = data.progress_C[pre_idx]
    post_p = data.progress_C[post_idx]
    conf_p = data.progress_C[conf_idx]
    pre_y = data.y_smooth[pre_idx]
    post_y = data.y_smooth[post_idx]
    conf_y = data.y_smooth[conf_idx]

    pre_m, pre_b = fit_line(pre_p, pre_y)
    post_m, post_b = fit_line(post_p, post_y)
    conf_m, conf_b = fit_line(conf_p, conf_y)

    pre_baseline_at_onset = line_value(pre_m, pre_b, p)
    if not np.isfinite(pre_baseline_at_onset):
        pre_baseline_at_onset = float(np.median(pre_y))

    post_pred = pre_m * post_p + pre_b if np.isfinite(pre_m) and np.isfinite(pre_b) else np.full_like(post_y, np.median(pre_y))
    conf_pred = pre_m * conf_p + pre_b if np.isfinite(pre_m) and np.isfinite(pre_b) else np.full_like(conf_y, np.median(pre_y))

    pre_resid = pre_y - (pre_m * pre_p + pre_b) if np.isfinite(pre_m) and np.isfinite(pre_b) else pre_y - np.median(pre_y)
    noise = robust_sigma(pre_resid)
    if not np.isfinite(noise):
        noise = 0.0

    pre_median = float(np.median(pre_y))
    post_median = float(np.median(post_y))
    conf_median = float(np.median(conf_y))
    conf_min = float(np.min(conf_y))

    level_shift = float(np.median(post_y - post_pred))
    sustained_shift = float(np.median(conf_y - conf_pred))
    future_min_shift = float(np.min(conf_y - conf_pred))
    post_rise = float(post_y[-1] - post_y[0])

    # Defaults for the T1 delayed post-level persistence gate.
    # These must exist even for non-T1 candidates or early metric paths.
    t1_post_level_shift = float("nan")
    t1_post_level_threshold = float("nan")
    t1_post_level_ok = True

    prefix_idx = np.where(data.analysis_mask)[0]
    if prefix_idx.size:
        prefix_range = float(np.nanmax(data.y_smooth[prefix_idx]) - np.nanmin(data.y_smooth[prefix_idx]))
    else:
        prefix_range = 0.0

    slope_increase = float(post_m - pre_m) if np.isfinite(post_m) and np.isfinite(pre_m) else float("nan")
    slope_ratio = float(post_m / pre_m) if np.isfinite(post_m) and np.isfinite(pre_m) and abs(pre_m) > 1e-12 else float("nan")

    level_threshold = max(float(args.min_level_shift_abs), float(args.level_k_sigma) * noise)
    sustained_threshold = max(float(args.min_sustained_shift_abs), float(args.level_k_sigma) * noise)
    post_rise_threshold = max(float(args.min_post_rise_abs), float(args.level_k_sigma) * noise)

    auto_slope_threshold = float(args.slope_k_sigma) * noise / max(float(args.post_window_C), 1e-9)
    slope_threshold = float(args.min_slope_increase) if args.min_slope_increase > 0 else auto_slope_threshold

    no_return_to_baseline = future_min_shift >= sustained_threshold - float(args.max_return_below_baseline_abs)

    level_ok = level_shift >= level_threshold and sustained_shift >= sustained_threshold and no_return_to_baseline
    slow_rise_ok = post_rise >= post_rise_threshold and sustained_shift >= sustained_threshold and no_return_to_baseline
    slope_ok = (
        np.isfinite(slope_increase)
        and slope_increase >= slope_threshold
        and post_m > pre_m
        and sustained_shift >= sustained_threshold
        and no_return_to_baseline
    )
    shoulder_ok = (
        target_rank >= 2
        and np.isfinite(pre_m)
        and np.isfinite(post_m)
        and pre_m >= float(args.min_shoulder_pre_slope)
        and post_m >= float(args.shoulder_slope_factor) * max(pre_m, 1e-9)
        and slope_increase >= slope_threshold
        and post_rise > 0
        and sustained_shift >= 0.5 * sustained_threshold
        and no_return_to_baseline
    )

    dip = level_shift < 0 and sustained_shift < 0

    # T1 must also clear the initial clear-solution baseline, not only a local
    # drift/bump. This prevents baseline settling from becoming T1.
    init = data.initial_baseline
    init_pred_conf = init.slope * conf_p + init.intercept if np.isfinite(init.slope) and np.isfinite(init.intercept) else np.full_like(conf_y, init.median)
    init_pred_onset = line_value(init.slope, init.intercept, p)
    if not np.isfinite(init_pred_onset):
        init_pred_onset = init.median
    initial_shift = float(np.median(conf_y - init_pred_conf))
    initial_future_min_shift = float(np.min(conf_y - init_pred_conf))
    initial_gate_ok = True
    if target_rank == 1:
        initial_gate_ok = (
            initial_shift >= init.threshold
            and initial_future_min_shift >= init.threshold - float(args.max_return_below_baseline_abs)
            and p >= float(args.min_search_progress_C)
        )

        # Delayed post-level persistence check:
        # test a window after the candidate, skipping the immediate neighborhood.
        # The low percentile of residuals must remain above the local pre-candidate
        # trend. This rejects local baseline bumps that do not actually lift the
        # FBRM level.
        t1_idx = window_indices(
            data.progress_C,
            p + float(args.t1_post_level_delay_C),
            p + float(args.t1_post_level_window_C),
        )
        t1_idx = t1_idx[data.analysis_mask[t1_idx]]
        if len(t1_idx) < int(args.min_confirm_points):
            t1_post_level_ok = False
            t1_post_level_shift = float("nan")
            t1_post_level_threshold = max(
                float(args.t1_min_post_level_shift_abs),
                float(args.t1_min_post_level_shift_frac) * prefix_range,
                float(args.t1_post_level_k_sigma) * noise,
            )
        else:
            t1_p = data.progress_C[t1_idx]
            t1_y = data.y_smooth[t1_idx]
            if np.isfinite(pre_m) and np.isfinite(pre_b):
                t1_pred = pre_m * t1_p + pre_b
            else:
                t1_pred = np.full_like(t1_y, pre_median)
            t1_resid = t1_y - t1_pred
            pct = min(100.0, max(0.0, float(args.t1_post_level_percentile)))
            t1_post_level_shift = float(np.nanpercentile(t1_resid, pct))
            t1_post_level_threshold = max(
                float(args.t1_min_post_level_shift_abs),
                float(args.t1_min_post_level_shift_frac) * prefix_range,
                float(args.t1_post_level_k_sigma) * noise,
            )
            t1_post_level_ok = bool(t1_post_level_shift >= t1_post_level_threshold)

    modes: list[str] = []
    if level_ok:
        modes.append("level_shift")
    if slow_rise_ok:
        modes.append("sustained_rise")
    if slope_ok:
        modes.append("slope_increase")
    if shoulder_ok:
        modes.append("shoulder")

    if dip:
        modes.clear()
    if target_rank == 1 and (not initial_gate_ok or not t1_post_level_ok):
        modes.clear()

    score_parts = []
    if level_threshold > 0:
        score_parts.append(level_shift / level_threshold)
    if sustained_threshold > 0:
        score_parts.append(sustained_shift / sustained_threshold)
    if post_rise_threshold > 0:
        score_parts.append(post_rise / post_rise_threshold)
    if slope_threshold > 0 and np.isfinite(slope_increase):
        score_parts.append(slope_increase / slope_threshold)
    if target_rank == 1 and init.threshold > 0:
        score_parts.append(initial_shift / init.threshold)
    if target_rank == 1 and np.isfinite(t1_post_level_threshold) and t1_post_level_threshold > 0:
        score_parts.append(t1_post_level_shift / t1_post_level_threshold)
    score = float(max(score_parts)) if score_parts else 0.0

    out.update(
        {
            "valid_windows": True,
            "pre_start_Tr": float(data.Tr_C[pre_idx[0]]),
            "pre_end_Tr": float(data.Tr_C[pre_idx[-1]]),
            "post_start_Tr": float(data.Tr_C[post_idx[0]]),
            "post_end_Tr": float(data.Tr_C[post_idx[-1]]),
            "confirm_start_Tr": float(data.Tr_C[conf_idx[0]]),
            "confirm_end_Tr": float(data.Tr_C[conf_idx[-1]]),
            "initial_baseline_pred": float(init_pred_onset),
            "initial_shift": initial_shift,
            "initial_future_min_shift": initial_future_min_shift,
            "initial_gate_ok": bool(initial_gate_ok),
            "t1_post_level_shift": t1_post_level_shift,
            "t1_post_level_threshold": t1_post_level_threshold,
            "t1_post_level_ok": bool(t1_post_level_ok),
            "pre_baseline_at_onset": float(pre_baseline_at_onset),
            "pre_median": pre_median,
            "post_median": post_median,
            "confirm_median": conf_median,
            "confirm_min": conf_min,
            "level_shift": level_shift,
            "sustained_shift": sustained_shift,
            "future_min_shift": future_min_shift,
            "post_rise": post_rise,
            "pre_slope": float(pre_m),
            "post_slope": float(post_m),
            "confirm_slope": float(conf_m),
            "slope_increase": slope_increase,
            "slope_ratio": slope_ratio,
            "shoulder_score": float(slope_ratio) if np.isfinite(slope_ratio) else float("nan"),
            "noise": float(noise),
            "level_threshold": float(level_threshold),
            "sustained_threshold": float(sustained_threshold),
            "slope_threshold": float(slope_threshold),
            "post_rise_threshold": float(post_rise_threshold),
            "no_return_to_baseline": bool(no_return_to_baseline),
            "dip": bool(dip),
            "modes": modes,
            "valid_event": bool(modes),
            "score": score,
        }
    )
    return out


def refine_onset_index(run_start: int, run_end: int, target_rank: int, data: SeriesData, args: argparse.Namespace) -> int:
    """Move a valid-run event to the first persistent threshold crossing."""
    base = local_metrics(run_start, target_rank, data, args)
    if not base.get("valid_windows", False):
        return run_start

    p0 = float(data.progress_C[run_start])
    p1 = min(float(data.progress_C[run_end]), p0 + float(args.refine_window_C))
    search_idx = np.where((data.progress_C >= p0) & (data.progress_C <= p1) & data.analysis_mask)[0]
    if search_idx.size == 0:
        return run_start

    # Valley-aware reporting:
    # If the accepted event window contains a small dip / local minimum before
    # the sustained rise, do not report the onset before that valley. This
    # prevents a pre-rise baseline recovery or small pre-valley excursion from
    # becoming the reported crystallization onset.
    if not args.no_valley_aware_refine and search_idx.size >= 3:
        y_search = data.y_smooth[search_idx]
        if np.any(np.isfinite(y_search)):
            valley_pos = int(np.nanargmin(y_search))
            if 0 < valley_pos < len(search_idx) - 1:
                search_idx = search_idx[valley_pos:]

    n_persist = max(1, int(args.persistent_points))

    onset_frac = max(0.01, min(1.0, float(args.onset_threshold_frac)))

    if target_rank == 1:
        init = data.initial_baseline
        report_threshold = onset_frac * float(init.threshold)
        for pos, idx in enumerate(search_idx):
            chunk = search_idx[pos : pos + n_persist]
            if len(chunk) < n_persist:
                break
            pred = init.slope * data.progress_C[chunk] + init.intercept if np.isfinite(init.slope) and np.isfinite(init.intercept) else np.full(len(chunk), init.median)
            if np.all(data.y_smooth[chunk] >= pred + report_threshold):
                return int(idx)

    threshold = onset_frac * float(base["sustained_threshold"])
    baseline = float(base["pre_baseline_at_onset"])
    for pos, idx in enumerate(search_idx):
        chunk = search_idx[pos : pos + n_persist]
        if len(chunk) < n_persist:
            break
        if np.all(data.y_smooth[chunk] >= baseline + threshold):
            return int(idx)

    # Slope/shoulder fallback.
    pre_idx = window_indices(data.progress_C, p0 - args.pre_gap_C - args.pre_window_C, p0 - args.pre_gap_C)
    pre_idx = pre_idx[data.analysis_mask[pre_idx]]
    d_bg = data.dy_dprogress[pre_idx]
    d_bg = d_bg[np.isfinite(d_bg)]
    d_med = float(np.median(d_bg)) if d_bg.size else 0.0
    d_sig = robust_sigma(d_bg) if d_bg.size else 0.0
    d_thr = d_med + max(3.0 * d_sig, 0.0)

    for pos, idx in enumerate(search_idx):
        chunk = search_idx[pos : pos + n_persist]
        if len(chunk) < n_persist:
            break
        if np.all(data.dy_dprogress[chunk] > d_thr) and np.all(data.dy_dprogress[chunk] > 0):
            return int(idx)

    return run_start


def attach_nearest_reference(candidate: CandidatePoint, refs: Sequence[ReferenceOnset]) -> None:
    """Attach nearest reference onset info."""
    if not refs:
        return
    ref = min(refs, key=lambda r: abs(r.Tr_C - candidate.Tr_C))
    candidate.nearest_reference_onset = ref.label
    candidate.nearest_reference_Tr_C = ref.Tr_C
    candidate.delta_to_reference_C = candidate.Tr_C - ref.Tr_C
    candidate.reference_file = ref.source_file


def make_candidate(
    *,
    candidate_id: int,
    target_rank: int,
    run_start: int,
    run_end: int,
    onset_idx: int,
    data: SeriesData,
    args: argparse.Namespace,
    refs: Sequence[ReferenceOnset],
) -> CandidatePoint:
    """Create a candidate object from a valid event run."""
    m = local_metrics(onset_idx, target_rank, data, args)
    if not m.get("valid_windows", False):
        m = local_metrics(run_start, target_rank, data, args)

    modes = list(m.get("modes", []))
    mode_text = "+".join(modes) if modes else "candidate"

    cand = CandidatePoint(
        candidate_id=candidate_id,
        target_onset=f"T{target_rank}",
        status="candidate",
        reason="valid_sequential_event_candidate",
        acceptance_mode=f"T{target_rank}_{mode_text}",
        accepted=False,
        final_rank=None,
        input_row=int(data.original_rows[onset_idx]),
        Tr_C=float(data.Tr_C[onset_idx]),
        cooling_progress_C=float(data.progress_C[onset_idx]),
        total_counts_raw=float(data.y_raw[onset_idx]),
        total_counts_smooth=float(data.y_smooth[onset_idx]),
        initial_baseline_pred=float(m.get("initial_baseline_pred", float("nan"))),
        initial_shift=float(m.get("initial_shift", float("nan"))),
        initial_future_min_shift=float(m.get("initial_future_min_shift", float("nan"))),
        t1_post_level_shift=float(m.get("t1_post_level_shift", float("nan"))),
        t1_post_level_threshold=float(m.get("t1_post_level_threshold", float("nan"))),
        t1_post_level_ok=bool(m.get("t1_post_level_ok", True)),
        pre_baseline_at_onset=float(m.get("pre_baseline_at_onset", float("nan"))),
        pre_median=float(m.get("pre_median", float("nan"))),
        post_median=float(m.get("post_median", float("nan"))),
        confirm_median=float(m.get("confirm_median", float("nan"))),
        confirm_min=float(m.get("confirm_min", float("nan"))),
        level_shift=float(m.get("level_shift", float("nan"))),
        sustained_shift=float(m.get("sustained_shift", float("nan"))),
        future_min_shift=float(m.get("future_min_shift", float("nan"))),
        post_rise=float(m.get("post_rise", float("nan"))),
        pre_slope=float(m.get("pre_slope", float("nan"))),
        post_slope=float(m.get("post_slope", float("nan"))),
        confirm_slope=float(m.get("confirm_slope", float("nan"))),
        slope_increase=float(m.get("slope_increase", float("nan"))),
        slope_ratio=float(m.get("slope_ratio", float("nan"))),
        shoulder_score=float(m.get("shoulder_score", float("nan"))),
        derivative_at_onset=float(data.dy_dprogress[onset_idx]),
        second_derivative_at_onset=float(data.d2y_dprogress2[onset_idx]),
        local_noise_sigma=float(m.get("noise", float("nan"))),
        level_threshold=float(m.get("level_threshold", float("nan"))),
        sustained_threshold=float(m.get("sustained_threshold", float("nan"))),
        slope_threshold=float(m.get("slope_threshold", float("nan"))),
        post_rise_threshold=float(m.get("post_rise_threshold", float("nan"))),
        onset_index=int(onset_idx),
        run_start_index=int(run_start),
        run_end_index=int(run_end),
        pre_window_start_Tr_C=m.get("pre_start_Tr", None),
        pre_window_end_Tr_C=m.get("pre_end_Tr", None),
        post_window_start_Tr_C=m.get("post_start_Tr", None),
        post_window_end_Tr_C=m.get("post_end_Tr", None),
        confirm_window_start_Tr_C=m.get("confirm_start_Tr", None),
        confirm_window_end_Tr_C=m.get("confirm_end_Tr", None),
        n_pre_points=int(m.get("n_pre", 0)),
        n_post_points=int(m.get("n_post", 0)),
        n_confirm_points=int(m.get("n_confirm", 0)),
    )
    attach_nearest_reference(cand, refs)
    return cand


def index_after_progress(data: SeriesData, min_progress: float) -> int:
    """Return first analyzable index at or after a cooling-progress value."""
    hits = np.where((data.progress_C >= min_progress) & data.analysis_mask)[0]
    if not hits.size:
        return len(data.progress_C)
    return int(hits[0])


def find_next_event(
    *,
    target_rank: int,
    search_start_idx: int,
    candidate_id_start: int,
    data: SeriesData,
    args: argparse.Namespace,
    refs: Sequence[ReferenceOnset],
) -> tuple[CandidatePoint | None, list[CandidatePoint]]:
    """Find next accepted event candidate from a search start index."""
    valid_indices = np.where(data.analysis_mask)[0]
    if not valid_indices.size:
        return None, []
    last_valid = int(valid_indices[-1])
    if search_start_idx > last_valid:
        return None, []

    valid = np.zeros(len(data.Tr_C), dtype=bool)
    for idx in range(max(0, search_start_idx), last_valid + 1):
        if not data.analysis_mask[idx]:
            continue
        m = local_metrics(idx, target_rank, data, args)
        valid[idx] = bool(m.get("valid_event", False))

    all_candidates: list[CandidatePoint] = []
    event_count = 0
    min_run = max(1, int(args.min_valid_run_points))

    i = max(0, search_start_idx)
    while i <= last_valid:
        if not valid[i]:
            i += 1
            continue

        run_start = i
        while i <= last_valid and valid[i]:
            i += 1
        run_end = i - 1

        if run_end - run_start + 1 >= min_run:
            onset_idx = refine_onset_index(run_start, run_end, target_rank, data, args)
            cand = make_candidate(
                candidate_id=candidate_id_start + event_count,
                target_rank=target_rank,
                run_start=run_start,
                run_end=run_end,
                onset_idx=onset_idx,
                data=data,
                args=args,
                refs=refs,
            )
            event_count += 1
            all_candidates.append(cand)

        i += 1

    if not all_candidates:
        return None, []

    accepted = all_candidates[0]
    accepted.status = "accepted"
    accepted.accepted = True
    accepted.final_rank = target_rank
    accepted.reason = "selected_first_valid_event_in_measurement_order"

    for cand in all_candidates[1:]:
        cand.status = "rejected"
        cand.reason = f"not_selected_for_T{target_rank}_after_earlier_valid_event"
        cand.acceptance_mode = f"rejected_{cand.acceptance_mode}"

    return accepted, all_candidates


def make_not_found(label: str, rank: int, reason: str, data: SeriesData) -> OnsetPoint:
    """Create explicit not-found row."""
    return OnsetPoint(
        onset_label=label,
        rank=rank,
        status="not_found",
        reason=reason,
        input_row=None,
        Tr_C=None,
        total_counts_raw=None,
        total_counts_smooth=None,
        cooling_progress_C=None,
        candidate_id=None,
        acceptance_mode=None,
        method="fbrm_total_counts_sequential_v14_6",
        source_file=data.input_csv,
        x_column=data.xcol,
        y_column=data.ycol,
    )


def detect_onsets(data: SeriesData, args: argparse.Namespace, refs: Sequence[ReferenceOnset]) -> tuple[list[OnsetPoint], list[CandidatePoint], str]:
    """Sequentially detect 0..N onsets."""
    n_allowed = max(0, int(args.onsets))
    if n_allowed == 0:
        return [], [], "onsets_disabled_by_argument"

    analyzable = np.where(data.analysis_mask)[0]
    if analyzable.size == 0:
        return [make_not_found("not_found", 0, "no_data_before_search_limit", data)], [], "no_data_before_search_limit"
    if data.cutoff_index == 0:
        return [make_not_found("not_found", 0, "already_above_search_limit_at_start", data)], [], "already_above_search_limit_at_start"
    if analyzable.size < int(args.min_points_before_limit):
        return [make_not_found("not_found", 0, "too_few_points_before_search_limit", data)], [], "too_few_points_before_search_limit"

    accepted_onsets: list[OnsetPoint] = []
    candidates: list[CandidatePoint] = []
    search_start_idx = int(analyzable[0])
    candidate_id = 1

    for target_rank in range(1, n_allowed + 1):
        found, new_candidates = find_next_event(
            target_rank=target_rank,
            search_start_idx=search_start_idx,
            candidate_id_start=candidate_id,
            data=data,
            args=args,
            refs=refs,
        )
        candidates.extend(new_candidates)
        candidate_id += len(new_candidates)

        if found is None:
            if not accepted_onsets:
                accepted_onsets.append(make_not_found("not_found", 0, f"no_valid_T{target_rank}_before_search_limit", data))
            break

        onset = OnsetPoint(
            onset_label=f"T{target_rank}",
            rank=target_rank,
            status="accepted",
            reason=found.reason,
            input_row=found.input_row,
            Tr_C=found.Tr_C,
            total_counts_raw=found.total_counts_raw,
            total_counts_smooth=found.total_counts_smooth,
            cooling_progress_C=found.cooling_progress_C,
            candidate_id=found.candidate_id,
            acceptance_mode=found.acceptance_mode,
            method="fbrm_total_counts_sequential_v14_6",
            source_file=data.input_csv,
            x_column=data.xcol,
            y_column=data.ycol,
            nearest_reference_onset=found.nearest_reference_onset,
            nearest_reference_Tr_C=found.nearest_reference_Tr_C,
            delta_to_reference_C=found.delta_to_reference_C,
            reference_file=found.reference_file,
        )
        accepted_onsets.append(onset)

        next_progress = float(found.cooling_progress_C) + float(args.min_separation_C)
        search_start_idx = index_after_progress(data, next_progress)
        if search_start_idx >= len(data.Tr_C):
            break

    if accepted_onsets and accepted_onsets[0].status == "not_found":
        status = accepted_onsets[0].reason
    else:
        n = len([o for o in accepted_onsets if o.status == "accepted"])
        status = f"accepted_{n}_of_requested_{n_allowed}"
    return accepted_onsets, candidates, status


def infer_reference_rank_cycle(n_reference_rows: int, args: argparse.Namespace) -> int:
    """Infer fallback T-rank cycle length for unlabeled reference rows.

    MGI/PCHIP reference summaries commonly contain RAW T1..TN followed by
    SG T1..TN, but some historical summaries do not have an explicit onset-label
    column. In that common case, four rows should be interpreted as
    T1,T2,T1,T2, not as T1,T2,T3,T1 when FBRM is run with --onsets 3.

    The user can override this with --reference-rank-cycle N.
    """
    override = getattr(args, "reference_rank_cycle", None)
    if override is not None:
        return max(1, int(override))

    n = int(n_reference_rows)
    requested = max(1, int(getattr(args, "onsets", 2)))

    if n <= 0:
        return requested

    # Strong common case: two branches (RAW and SG), each with the same onset
    # ranks. Four rows -> T1,T2,T1,T2. Six rows -> T1,T2,T3,T1,T2,T3.
    if n >= 4 and n % 2 == 0:
        half = n // 2
        if 1 <= half <= max(requested, 3):
            return half

    # Single-branch or ambiguous small files: do not invent more ranks than rows.
    return max(1, min(requested, n))


def read_reference_onsets(args: argparse.Namespace) -> list[ReferenceOnset]:
    """Read optional reference onset CSV for visual/CSV comparison."""
    path = args.reference_onsets
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Reference onset CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    columns = list(df.columns)

    if args.reference_xcol is not None:
        xcol = args.reference_xcol
        if xcol not in columns:
            raise KeyError(f"Reference x column '{xcol}' not found: {columns}")
    else:
        xcol = None
        for name in ("Tr_C", "temperature_C", "Temperature_C", "temperature_C"):
            if name in columns:
                xcol = name
                break
        if xcol is None:
            for col in columns:
                low = col.lower()
                if "temp" in low or "tr" in low:
                    xcol = col
                    break
        if xcol is None:
            raise KeyError(f"Could not detect reference temperature column. Columns: {columns}")

    if args.reference_label_col is not None:
        lcol = args.reference_label_col
        if lcol not in columns:
            raise KeyError(f"Reference label column '{lcol}' not found: {columns}")
    else:
        lcol = None
        for name in ("onset_label", "label", "rank", "final_rank"):
            if name in columns:
                lcol = name
                break

    valid_rows: list[tuple[int, Any, float]] = []
    for i, row in df.iterrows():
        val = pd.to_numeric(pd.Series([row[xcol]]), errors="coerce").iloc[0]
        if not np.isfinite(val):
            continue
        valid_rows.append((int(i), row, float(val)))

    refs: list[ReferenceOnset] = []
    fallback_rank_count = infer_reference_rank_cycle(len(valid_rows), args)

    for _i, row, val in valid_rows:
        if lcol is not None:
            label = str(row[lcol])
        else:
            # MGI/PCHIP summaries may contain RAW T1..TN followed by SG T1..TN
            # without a dedicated onset-label column. In that common case,
            # cyclic T labels are more informative than ref1/ref2/ref3/ref4
            # and allow plotting to group RAW+SG references by physical onset.
            label = f"T{(len(refs) % fallback_rank_count) + 1}"

        refs.append(ReferenceOnset(label=label, Tr_C=float(val), source_file=str(path)))
    return refs


def json_safe(value: Any) -> Any:
    """Convert dataclass/NumPy/NaN values to JSON/CSV safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_dataclass_csv(path: Path, rows: Sequence[Any]) -> None:
    """Write dataclass rows to CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    dict_rows = [asdict(row) for row in rows]
    fieldnames = list(dict_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dict_rows:
            writer.writerow({k: json_safe(v) for k, v in row.items()})


def write_params(path: Path, args: argparse.Namespace, data: SeriesData, refs: Sequence[ReferenceOnset], run_status: str) -> None:
    """Write parameter JSON."""
    params = {
        "script": "fbrm-onsets-v14.6.py",
        "method": "fbrm_total_counts_sequential_v14_6",
        "run_status": run_status,
        "input_file": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "x_column": data.xcol,
        "y_column": data.ycol,
        "temperature_axis": "measured_or_interpolated_Tr_C",
        "forbidden_axis": "Tr_regular_not_used",
        "order": data.order_mode,
        "order_rule": "input_measurement_order_preserved_temperature_not_sorted",
        "cooling_start": {
            "method": data.cooling_start_method,
            "input_row": data.cooling_start_input_row,
            "loaded_index_before_trim": data.cooling_start_original_index,
            "Tr_C": data.cooling_start_Tr_C,
            "total_counts": data.cooling_start_count,
            "note": data.cooling_start_note,
            "cooling_high_band_C": args.cooling_high_band_C,
            "cooling_min_drop_C": args.cooling_min_drop_C,
            "cooling_confirm_points": args.cooling_confirm_points,
        },
        "requested_onsets": args.onsets,
        "requested_onsets_rule": "report_0_to_N_valid_onsets_not_forced",
        "onset_config": {
            "path": getattr(args, "onset_config_path", None),
            "status": getattr(args, "onset_config_status", None),
            "mgi_onsets": getattr(args, "onset_config_mgi_onsets", None),
            "fbrm_onsets": getattr(args, "onset_config_fbrm_onsets", None),
            "fbrm_effective_onsets": args.onsets,
        },
        "validity_cutoff": {
            "max_search_count": args.max_search_count,
            "cutoff_source": "raw_total_counts",
            "cutoff_index": data.cutoff_index,
            "cutoff_Tr_C": data.cutoff_Tr_C,
            "cutoff_count": data.cutoff_count,
            "analyzed_rows": int(np.sum(data.analysis_mask)),
            "note": "This is an analysis/search limit unless independently confirmed as an instrument validity limit.",
        },
        "initial_baseline": asdict(data.initial_baseline),
        "smoothing": {
            "method": args.smooth_method,
            "savgol_window_requested": args.savgol_window,
            "savgol_window_used_prefix": adjusted_savgol_window(int(np.sum(data.analysis_mask)), args.savgol_window, args.savgol_polyorder)
            if args.smooth_method == "savgol"
            else None,
            "savgol_polyorder": args.savgol_polyorder,
            "note": "Smoothing is applied separately to the analyzable prefix and outside-search context so the kernel does not cross the cutoff. Diagnostic derivatives are computed per sample to avoid artifacts from repeated or jittery temperature values.",
        },
        "candidate_detection": {
            "sequential_rebaselining": True,
            "T1_rule": "first_sustained_departure_from_initial_clear_solution_baseline",
            "T2_plus_rule": "next_local_sustained_level_shift_slope_increase_or_shoulder",
            "baseline_start_after_C": args.baseline_start_after_C,
            "initial_baseline_window_C": args.initial_baseline_window_C,
            "initial_baseline_k_sigma": args.initial_baseline_k_sigma,
            "min_initial_shift_abs": args.min_initial_shift_abs,
            "min_search_progress_C": args.min_search_progress_C,
            "t1_post_level_delay_C": args.t1_post_level_delay_C,
            "t1_post_level_window_C": args.t1_post_level_window_C,
            "t1_post_level_percentile": args.t1_post_level_percentile,
            "t1_min_post_level_shift_abs": args.t1_min_post_level_shift_abs,
            "t1_min_post_level_shift_frac": args.t1_min_post_level_shift_frac,
            "t1_post_level_k_sigma": args.t1_post_level_k_sigma,
            "t1_post_level_note": "T1 must be followed by a persistent delayed level increase; rejected T1-like local events do not consume the T1 slot.",
            "onset_threshold_frac": args.onset_threshold_frac,
            "onset_threshold_note": "Controls report position only; event acceptance still uses full confirmation thresholds.",
            "valley_aware_refine": not args.no_valley_aware_refine,
            "valley_aware_refine_note": "When enabled, reported onset cannot be moved before a local pre-rise minimum inside the accepted event window.",
            "pre_window_C": args.pre_window_C,
            "pre_gap_C": args.pre_gap_C,
            "post_window_C": args.post_window_C,
            "post_gap_C": args.post_gap_C,
            "confirm_window_C": args.confirm_window_C,
            "confirm_gap_C": args.confirm_gap_C,
            "level_k_sigma": args.level_k_sigma,
            "min_level_shift_abs": args.min_level_shift_abs,
            "min_sustained_shift_abs": args.min_sustained_shift_abs,
            "max_return_below_baseline_abs": args.max_return_below_baseline_abs,
            "min_post_rise_abs": args.min_post_rise_abs,
            "slope_k_sigma": args.slope_k_sigma,
            "min_slope_increase": args.min_slope_increase,
            "min_shoulder_pre_slope": args.min_shoulder_pre_slope,
            "shoulder_slope_factor": args.shoulder_slope_factor,
            "min_valid_run_points": args.min_valid_run_points,
            "refine_window_C": args.refine_window_C,
            "persistent_points": args.persistent_points,
            "min_separation_C": args.min_separation_C,
        },
        "reference_onsets": [asdict(ref) for ref in refs],
        "reference_plotting": {
            "display": args.reference_display,
            "cluster_tolerance_C": args.reference_cluster_tolerance_C,
            "rank_cycle_override": args.reference_rank_cycle,
            "rank_cycle_inferred": infer_reference_rank_cycle(len(refs), args) if refs else None,
            "label_prefix": args.reference_label_prefix,
            "plotted_groups": [
                {
                    "label": group.label,
                    "Tr_C": group.Tr_C,
                    "n_members": len(group.members),
                    "member_labels": [member.label for member in group.members],
                    "member_Tr_C": [member.Tr_C for member in group.members],
                }
                for group in build_reference_plot_groups(refs, args)
            ],
        },
    }
    path.write_text(json.dumps(json_safe(params), indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    data: SeriesData,
    onsets: Sequence[OnsetPoint],
    candidates: Sequence[CandidatePoint],
    run_status: str,
) -> None:
    """Write human-readable text report."""
    lines: list[str] = []
    lines.append("FBRM Total Counts onset analysis v6")
    lines.append("=" * 42)
    lines.append(f"Input CSV: {args.input_csv}")
    lines.append(f"Columns: X='{data.xcol}', Y='{data.ycol}'")
    lines.append(f"Order: {data.order_mode}; temperature sorting not used")
    lines.append(
        f"Cooling start: method={data.cooling_start_method}, row={data.cooling_start_input_row}, "
        f"Tr={data.cooling_start_Tr_C:.3f} °C, Total Counts={data.cooling_start_count:.3f}"
    )
    lines.append(f"Cooling-start note: {data.cooling_start_note}")
    lines.append("Temperature axis: measured/interpolated Tr (°C)")
    lines.append("Tr_regular: not used")
    lines.append(f"Requested onsets: 0..{args.onsets} valid onsets; not forced")
    lines.append(f"Run status: {run_status}")
    lines.append("")
    lines.append("Initial baseline:")
    lines.append(
        f"  rows {data.initial_baseline.start_index}..{data.initial_baseline.end_index}, "
        f"Tr {data.initial_baseline.Tr_start_C:.3f}..{data.initial_baseline.Tr_end_C:.3f} °C, "
        f"median={data.initial_baseline.median:.3f}, threshold={data.initial_baseline.threshold:.3f}"
    )

    if float(args.max_search_count) > 0:
        lines.append("")
        lines.append(
            f"Search limit: raw Total Counts < {args.max_search_count:g} "
            "(analysis/search limit; not necessarily instrument specification)"
        )
        if data.cutoff_Tr_C is not None:
            lines.append(f"First cutoff hit: Tr={data.cutoff_Tr_C:.3f} °C, Total Counts={data.cutoff_count:.3f}")
        lines.append(f"Rows before search limit: {int(np.sum(data.analysis_mask))}")
    else:
        lines.append("Search limit: disabled")

    lines.append("")
    lines.append("Accepted onsets:")
    accepted = [o for o in onsets if o.status == "accepted"]
    if accepted:
        for o in accepted:
            lines.append(
                f"  {o.onset_label}: row={o.input_row}, Tr={o.Tr_C:.3f} °C, "
                f"Total Counts={o.total_counts_smooth:.3f}, mode={o.acceptance_mode}"
            )
        if len(accepted) < int(args.onsets):
            lines.append(f"  Only {len(accepted)} accepted onset(s) found before search limit.")
    else:
        lines.append("  none")

    lines.append("")
    lines.append(f"Candidates: {len(candidates)}")
    for c in candidates:
        lines.append(
            f"  candidate {c.candidate_id} ({c.target_onset}): {c.status}, row={c.input_row}, "
            f"Tr={c.Tr_C:.3f} °C, mode={c.acceptance_mode}, "
            f"initial_shift={c.initial_shift:.3f}, t1_post_shift={c.t1_post_level_shift:.3f}, "
            f"t1_post_thr={c.t1_post_level_threshold:.3f}, sustained_shift={c.sustained_shift:.3f}, "
            f"future_min_shift={c.future_min_shift:.3f}, slope_increase={c.slope_increase:.3f}, "
            f"reason={c.reason}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def outward_temperature_limits(
    tr: np.ndarray,
    major_step_C: float = 5.0,
    snap_tolerance_C: float = 1.0,
) -> tuple[float, float]:
    """Return non-clipping, data-aware low/high temperature limits.

    The limits are derived from the actual FBRM analysis window, not from the
    optional outside-search context. The returned numeric order is always
    ``(low, high)``; display direction is applied by ``apply_temperature_xlim``.

    The rule is deliberately conservative:

    - never snap inward;
    - snap outward to the next 5 °C grid boundary only when the data edge is
      already close to that outward boundary;
    - otherwise use the actual data edge so the figure ends at the data.
    """
    x = np.asarray(tr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot determine x-limits from empty temperature data.")

    data_lo = float(np.min(x))
    data_hi = float(np.max(x))

    lower_grid = float(np.floor(data_lo / major_step_C) * major_step_C)
    upper_grid = float(np.ceil(data_hi / major_step_C) * major_step_C)

    x_lo = lower_grid if (data_lo - lower_grid) <= snap_tolerance_C else data_lo
    x_hi = upper_grid if (upper_grid - data_hi) <= snap_tolerance_C else data_hi
    return x_lo, x_hi


def analysis_temperature_source(data: SeriesData) -> np.ndarray:
    """Return the temperature range that should define FBRM plot x-limits."""
    x = np.asarray(data.Tr_C[data.analysis_mask], dtype=float)
    if np.isfinite(x).any():
        return x
    return np.asarray(data.Tr_C, dtype=float)


def apply_temperature_xlim(
    ax: plt.Axes,
    tr: np.ndarray,
    *,
    reverse_x: bool,
    major_step_C: float = 5.0,
    snap_tolerance_C: float = 1.0,
) -> None:
    """Apply data-aware non-clipping x-limits with explicit axis direction."""
    x_lo, x_hi = outward_temperature_limits(
        tr,
        major_step_C=major_step_C,
        snap_tolerance_C=snap_tolerance_C,
    )
    ax.xaxis.set_major_locator(MultipleLocator(major_step_C))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.minorticks_off()
    ax.grid(False, which="minor")

    if reverse_x:
        ax.set_xlim(x_hi, x_lo)
    else:
        ax.set_xlim(x_lo, x_hi)


def setup_x_axis(ax: plt.Axes, args: argparse.Namespace, xlim_source: np.ndarray | None = None) -> tuple[float, float] | None:
    """Set temperature x-axis without allowing outside-search context to expand it."""
    ax.set_xlabel("Tr (°C) - cooling")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.minorticks_off()
    view_limits = getattr(args, "view_limits", NONE_LIMITS)
    if xlim_source is not None:
        return apply_temperature_view_limits(
            ax,
            xlim_source,
            view_limits,
            reverse_x=not args.no_reverse_x,
        )
    if not args.no_reverse_x:
        ax.invert_xaxis()
    return None


def apply_publication_axis_style(ax: plt.Axes, *, legend: bool = True) -> None:
    """Apply publication-friendly axis styling."""
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    if legend:
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(11)


def _limit_bounds(limit_obj: object | None) -> tuple[float | None, float | None]:
    """Return optional manual lower/upper bounds from a view-limit object."""
    if limit_obj is None:
        return None, None

    if isinstance(limit_obj, dict):
        lower = limit_obj.get("lower")
        upper = limit_obj.get("upper")
    else:
        lower = getattr(limit_obj, "lower", None)
        upper = getattr(limit_obj, "upper", None)

    lower_f = float(lower) if lower is not None else None
    upper_f = float(upper) if upper is not None else None
    return lower_f, upper_f


def _value_in_manual_bounds(value: float | None, lower: float | None, upper: float | None) -> bool:
    """Return True when value is inside optional lower/upper manual bounds."""
    if value is None:
        return False
    value_f = float(value)
    if not np.isfinite(value_f):
        return False

    if lower is None and upper is None:
        return True

    lo = -np.inf if lower is None else min(float(lower), float(upper)) if upper is not None else float(lower)
    hi = np.inf if upper is None else max(float(lower), float(upper)) if lower is not None else float(upper)
    eps = 1e-9
    return (lo - eps) <= value_f <= (hi + eps)


def _x_visible_in_view(x_value: float | None, args: argparse.Namespace) -> bool:
    """Return True when an x-position should be annotated in the current view."""
    view_limits = getattr(args, "view_limits", NONE_LIMITS)
    lower, upper = _limit_bounds(getattr(view_limits, "x", None))
    return _value_in_manual_bounds(x_value, lower, upper)


def _fbrm_y_visible_in_view(y_value: float | None, args: argparse.Namespace) -> bool:
    """Return True when a Total Counts y-position should be annotated."""
    view_limits = getattr(args, "view_limits", NONE_LIMITS)
    lower, upper = _limit_bounds(getattr(view_limits, "fbrm_y", None))
    return _value_in_manual_bounds(y_value, lower, upper)


def _fbrm_point_visible_in_view(x_value: float | None, y_value: float | None, args: argparse.Namespace) -> bool:
    """Return True when an FBRM point annotation belongs to the current view."""
    return _x_visible_in_view(x_value, args) and _fbrm_y_visible_in_view(y_value, args)


def _reference_rank(label: str) -> int | None:
    """Return T-rank parsed from a reference label such as T1 or ref T2."""
    match = re.search(r"\bT\s*(\d+)\b", str(label), flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _format_reference_plot_label(base_label: str, args: argparse.Namespace) -> str:
    """Return a plotted reference label with the configured prefix."""
    prefix = str(getattr(args, "reference_label_prefix", "ref "))
    base = str(base_label).strip()
    if not base:
        base = "reference"

    if base.lower().startswith(prefix.strip().lower()):
        return base
    return f"{prefix}{base}"


def _cluster_refs_by_temperature(
    refs: Sequence[ReferenceOnset],
    *,
    tolerance_C: float,
) -> list[tuple[float, list[ReferenceOnset]]]:
    """Cluster references by close temperatures in cooling order."""
    finite_refs = [
        ref
        for ref in refs
        if ref.Tr_C is not None and np.isfinite(float(ref.Tr_C))
    ]
    if not finite_refs:
        return []

    ordered = sorted(finite_refs, key=lambda ref: float(ref.Tr_C), reverse=True)
    clusters: list[list[ReferenceOnset]] = []

    for ref in ordered:
        if not clusters:
            clusters.append([ref])
            continue

        center = float(np.median([member.Tr_C for member in clusters[-1]]))
        if abs(float(ref.Tr_C) - center) <= tolerance_C:
            clusters[-1].append(ref)
        else:
            clusters.append([ref])

    out: list[tuple[float, list[ReferenceOnset]]] = []
    for cluster in clusters:
        x = float(np.median([member.Tr_C for member in cluster]))
        out.append((x, cluster))
    return out


def build_reference_plot_groups(
    refs: Sequence[ReferenceOnset],
    args: argparse.Namespace,
) -> list[ReferencePlotGroup]:
    """Build reference markers for plotting.

    The default grouped mode is meant for MGI RAW+SG reference rows:
    close same-rank references are drawn as a single marker, e.g. ``Ref T1``.
    If same-rank references differ by more than the tolerance, they remain
    separate and receive suffixes such as ``Ref T1a`` and ``Ref T1b``.
    """
    display = str(getattr(args, "reference_display", "grouped"))
    if display == "none":
        return []

    finite_refs = [
        ref
        for ref in refs
        if ref.Tr_C is not None and np.isfinite(float(ref.Tr_C))
    ]
    if not finite_refs:
        return []

    tolerance = max(0.0, float(getattr(args, "reference_cluster_tolerance_C", 0.35)))

    if display == "individual":
        ordered = sorted(finite_refs, key=lambda ref: float(ref.Tr_C), reverse=True)
        return [
            ReferencePlotGroup(
                label=_format_reference_plot_label(str(ref.label), args),
                Tr_C=float(ref.Tr_C),
                members=(ref,),
            )
            for ref in ordered
        ]

    ranked: dict[int, list[ReferenceOnset]] = {}
    unranked: list[ReferenceOnset] = []
    for ref in finite_refs:
        rank = _reference_rank(str(ref.label))
        if rank is None:
            unranked.append(ref)
        else:
            ranked.setdefault(rank, []).append(ref)

    groups: list[ReferencePlotGroup] = []

    if ranked:
        for rank in sorted(ranked):
            clusters = _cluster_refs_by_temperature(ranked[rank], tolerance_C=tolerance)
            if len(clusters) <= 1:
                for x, cluster in clusters:
                    groups.append(
                        ReferencePlotGroup(
                            label=_format_reference_plot_label(f"T{rank}", args),
                            Tr_C=x,
                            members=tuple(cluster),
                        )
                    )
            else:
                # Same-rank references are far enough apart to be shown separately.
                # Use stable, compact suffixes rather than raw ref1/ref3 labels.
                for suffix_index, (x, cluster) in enumerate(clusters):
                    suffix = chr(ord("a") + suffix_index)
                    groups.append(
                        ReferencePlotGroup(
                            label=_format_reference_plot_label(f"T{rank}{suffix}", args),
                            Tr_C=x,
                            members=tuple(cluster),
                        )
                    )

    if unranked:
        clusters = _cluster_refs_by_temperature(unranked, tolerance_C=tolerance)
        start_rank = max(ranked) + 1 if ranked else 1
        for offset, (x, cluster) in enumerate(clusters):
            groups.append(
                ReferencePlotGroup(
                    label=_format_reference_plot_label(f"T{start_rank + offset}", args),
                    Tr_C=x,
                    members=tuple(cluster),
                )
            )

    # Display in cooling order: higher Tr first.
    return sorted(groups, key=lambda group: float(group.Tr_C), reverse=True)


def plot_reference_lines(
    ax,
    refs: Sequence[ReferenceOnset],
    args: argparse.Namespace,
    *,
    annotate: bool,
) -> None:
    """Draw grouped reference-onset lines on one axis."""
    groups = build_reference_plot_groups(refs, args)
    if not groups:
        return

    visible_groups = [
        group
        for group in groups
        if _x_visible_in_view(group.Tr_C, args)
    ]
    if not visible_groups:
        return

    for idx, group in enumerate(visible_groups):
        ax.axvline(group.Tr_C, linestyle=":", linewidth=1.0, alpha=0.65)
        if not annotate:
            continue

        # Use axis-fraction coordinates so labels stay inside the axes and do
        # not depend on the data y-scale. Stagger only if several reference
        # markers are shown.
        label_y = 0.985 - 0.055 * (idx % 3)
        ax.text(
            group.Tr_C,
            label_y,
            group.label,
            rotation=90,
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=8,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )


def plot_publication(
    path: Path,
    data: SeriesData,
    onsets: Sequence[OnsetPoint],
    candidates: Sequence[CandidatePoint],
    refs: Sequence[ReferenceOnset],
    args: argparse.Namespace,
) -> None:
    """Write publication-style plot using analyzable prefix as primary curve."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=args.dpi)

    prefix = data.analysis_mask
    outside = ~data.analysis_mask

    if args.plot_full_data and np.any(outside):
        ax.scatter(data.Tr_C[outside], data.y_raw[outside], s=4, alpha=0.08, label="Outside search raw")
        ax.plot(data.Tr_C[outside], data.y_smooth[outside], linewidth=0.8, alpha=0.25, label="Outside search smooth")
        ax.axvspan(
            float(np.nanmin(data.Tr_C[outside])),
            float(np.nanmax(data.Tr_C[outside])),
            alpha=0.08,
            label="Outside onset search",
        )

    ax.scatter(data.Tr_C[prefix], data.y_raw[prefix], s=5, alpha=0.30, label="Raw Total Counts")
    ax.plot(data.Tr_C[prefix], data.y_smooth[prefix], linewidth=1.5, label="Smoothed Total Counts")

    # Initial baseline line over its window.
    init = data.initial_baseline
    init_idx = np.arange(init.start_index, init.end_index + 1)
    init_pred = init.slope * data.progress_C[init_idx] + init.intercept if np.isfinite(init.slope) and np.isfinite(init.intercept) else np.full(len(init_idx), init.median)
    ax.plot(data.Tr_C[init_idx], init_pred, linestyle=":", linewidth=1.1, label="Initial baseline")

    plot_reference_lines(ax, refs, args, annotate=True)

    for onset in onsets:
        if onset.status != "accepted" or onset.Tr_C is None:
            continue
        if not _x_visible_in_view(onset.Tr_C, args):
            continue
        ax.axvline(onset.Tr_C, linestyle="--", linewidth=1.2)
        if _fbrm_y_visible_in_view(onset.total_counts_smooth, args):
            ax.scatter([onset.Tr_C], [onset.total_counts_smooth], s=38, zorder=5)
            ax.text(
                onset.Tr_C,
                onset.total_counts_smooth,
                f"{onset.onset_label}\n{onset.Tr_C:.2f} °C",
                fontsize=9,
                ha="left",
                va="bottom",
                clip_on=True,
            )

    ax.set_ylabel("Total Counts")
    if not args.no_title:
        ax.set_title("FBRM Total Counts Onset Analysis")
    x_limits = setup_x_axis(ax, args, analysis_temperature_source(data))
    if x_limits is not None:
        apply_signal_y_view_limits(
            ax,
            [(data.Tr_C[prefix], data.y_raw[prefix]), (data.Tr_C[prefix], data.y_smooth[prefix])],
            x_limits,
            getattr(args, "view_limits", NONE_LIMITS).fbrm_y,
        )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic(
    path: Path,
    data: SeriesData,
    onsets: Sequence[OnsetPoint],
    candidates: Sequence[CandidatePoint],
    refs: Sequence[ReferenceOnset],
    args: argparse.Namespace,
) -> None:
    """Write diagnostic plot."""
    figure_size = (9, 5.5) if args.same_aspect_diagnostics else (9, 8.0)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figure_size,
        dpi=args.dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    ax, dax, ddax = axes
    prefix = data.analysis_mask
    outside = ~data.analysis_mask

    if args.plot_full_data and np.any(outside):
        ax.scatter(data.Tr_C[outside], data.y_raw[outside], s=4, alpha=0.08, label="Outside search raw")
        ax.plot(data.Tr_C[outside], data.y_smooth[outside], linewidth=0.8, alpha=0.25, label="Outside search smooth")

    ax.scatter(data.Tr_C[prefix], data.y_raw[prefix], s=4, alpha=0.25, label="Raw")
    ax.plot(data.Tr_C[prefix], data.y_smooth[prefix], linewidth=1.2, label="Smooth")

    for c in candidates:
        if not _fbrm_point_visible_in_view(c.Tr_C, c.total_counts_smooth, args):
            continue
        marker = "o" if c.status == "accepted" else "x"
        ax.scatter([c.Tr_C], [c.total_counts_smooth], marker=marker, s=36, zorder=4)
        ax.text(c.Tr_C, c.total_counts_smooth, str(c.candidate_id), fontsize=7, ha="left", va="bottom", clip_on=True)

    for onset in onsets:
        if onset.status == "accepted" and onset.Tr_C is not None and _x_visible_in_view(onset.Tr_C, args):
            ax.axvline(onset.Tr_C, linestyle="--", linewidth=1.2)

    plot_reference_lines(ax, refs, args, annotate=False)

    dax.plot(data.Tr_C[prefix], data.dy_dprogress[prefix], linewidth=0.9)
    dax.axhline(0, linewidth=0.8)
    dax.set_ylabel("dCounts/sample")

    ddax.plot(data.Tr_C[prefix], data.d2y_dprogress2[prefix], linewidth=0.9)
    ddax.axhline(0, linewidth=0.8)
    ddax.set_ylabel("d²Counts/sample²")

    ax.set_ylabel("Total Counts")
    ax.legend(loc="best")
    if not args.no_title:
        ax.set_title("FBRM Total Counts Diagnostic")
    for a in axes:
        a.grid(True, alpha=0.3)
    x_limits = setup_x_axis(ddax, args, analysis_temperature_source(data))
    if x_limits is not None:
        apply_signal_y_view_limits(
            ax,
            [(data.Tr_C[prefix], data.y_raw[prefix]), (data.Tr_C[prefix], data.y_smooth[prefix])],
            x_limits,
            getattr(args, "view_limits", NONE_LIMITS).fbrm_y,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_candidates(path: Path, data: SeriesData, candidates: Sequence[CandidatePoint], args: argparse.Namespace) -> None:
    """Write candidate overview plot."""
    figure_size = (9, 5.5) if args.same_aspect_diagnostics else (9, 8.0)
    fig, ax = plt.subplots(figsize=figure_size, dpi=args.dpi)
    prefix = data.analysis_mask
    ax.plot(data.Tr_C[prefix], data.y_smooth[prefix], linewidth=1.2, label="Smoothed Total Counts")

    accepted_plotted = False
    rejected_plotted = False

    for c in candidates:
        if not _fbrm_point_visible_in_view(c.Tr_C, c.total_counts_smooth, args):
            continue
        marker = "o" if c.status == "accepted" else "x"
        legend_label = None
        if c.status == "accepted" and not accepted_plotted:
            legend_label = "Accepted candidate"
            accepted_plotted = True
        elif c.status != "accepted" and not rejected_plotted:
            legend_label = "Rejected candidate"
            rejected_plotted = True

        ax.scatter(
            [c.Tr_C],
            [c.total_counts_smooth],
            marker=marker,
            s=42 if c.status == "accepted" else 36,
            label=legend_label,
            zorder=5,
        )

        # Keep the original diagnostic label style. The prefix is the candidate
        # id and the suffix is the target onset for that sequential pass, e.g.
        # 2:T2. This is clearer than C2/T2 when several T2 candidates overlap.
        label = f"{c.candidate_id}:{c.target_onset}"
        y_offset = 8 if c.status == "accepted" else -10
        ax.annotate(
            label,
            xy=(c.Tr_C, c.total_counts_smooth),
            xytext=(4, y_offset),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom" if y_offset >= 0 else "top",
            clip_on=True,
        )

    ax.set_ylabel("Total Counts")
    if not args.no_title:
        ax.set_title("FBRM Onset Candidates")
    x_limits = setup_x_axis(ax, args, analysis_temperature_source(data))
    if x_limits is not None:
        apply_signal_y_view_limits(
            ax,
            [(data.Tr_C[prefix], data.y_raw[prefix]), (data.Tr_C[prefix], data.y_smooth[prefix])],
            x_limits,
            getattr(args, "view_limits", NONE_LIMITS).fbrm_y,
        )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def save_fbrm_publication_plot(
    *,
    path: Path,
    data: SeriesData,
    onsets: Sequence[OnsetPoint],
    args: argparse.Namespace,
    variant: str,
) -> None:
    """Save a manuscript-style FBRM publication plot variant.

    These are presentation variants only. They reuse already detected onsets and
    never modify detection, candidate selection, CSV output, smoothing, or the
    analysis/search cutoff. X-limits are always derived from the analyzable
    prefix so optional outside-search context cannot stretch the figure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    mono = "mono" in variant
    clean = "clean" in variant
    show_onsets = "onsets" in variant
    show_raw = not clean

    prefix = data.analysis_mask
    xlim_source = analysis_temperature_source(data)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=args.dpi)

    if mono:
        line_color = "black"
        raw_color = "0.70"
        onset_color = "0.20"
        marker_color = "black"
        grid_color = "0.82"
    else:
        line_color = "#1f4e79"
        raw_color = "#1f4e79"
        onset_color = "#1f4e79"
        marker_color = "black"
        grid_color = "0.82"

    if show_raw:
        ax.plot(
            data.Tr_C[prefix],
            data.y_raw[prefix],
            ".",
            ms=2.2,
            alpha=0.24 if mono else 0.30,
            color=raw_color,
            label="Data",
        )

    ax.plot(
        data.Tr_C[prefix],
        data.y_smooth[prefix],
        "-",
        lw=1.45,
        color=line_color,
        label="FBRM",
    )

    if show_onsets:
        onset_lines: list[str] = []
        accepted = [
            o
            for o in onsets
            if o.status == "accepted" and o.Tr_C is not None and _x_visible_in_view(o.Tr_C, args)
        ]
        for idx, onset in enumerate(accepted, start=1):
            ax.axvline(
                onset.Tr_C,
                linestyle="--",
                lw=1.0,
                alpha=0.80,
                color=onset_color,
            )
            if onset.total_counts_smooth is not None and _fbrm_y_visible_in_view(onset.total_counts_smooth, args):
                ax.plot(
                    onset.Tr_C,
                    onset.total_counts_smooth,
                    "o",
                    ms=4.5,
                    color=marker_color,
                    zorder=5,
                )
            label_y = 0.92 if idx % 2 == 1 else 0.84
            ax.text(
                onset.Tr_C,
                label_y,
                onset.onset_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=10,
                color=onset_color,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                zorder=6,
                clip_on=True,
            )
            onset_lines.append(f"{onset.onset_label} = {onset.Tr_C:.2f} °C")

        if onset_lines:
            ax.text(
                0.03,
                0.96,
                "\n".join(onset_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.80",
                    "alpha": 0.90,
                },
                zorder=6,
            )

    ax.set_xlabel("Tr (°C) - cooling")
    ax.set_ylabel("Total Counts")

    if not clean or show_raw:
        ax.legend(loc="best")

    ax.grid(True, which="major", linestyle="--", alpha=0.18, color=grid_color)
    ax.grid(False, which="minor")
    x_limits = apply_temperature_view_limits(
        ax,
        xlim_source,
        getattr(args, "view_limits", NONE_LIMITS),
        reverse_x=not args.no_reverse_x,
    )
    apply_signal_y_view_limits(
        ax,
        [(data.Tr_C[prefix], data.y_raw[prefix]), (data.Tr_C[prefix], data.y_smooth[prefix])],
        x_limits,
        getattr(args, "view_limits", NONE_LIMITS).fbrm_y,
    )
    apply_publication_axis_style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def save_fbrm_publication_plots(
    *,
    outdir: Path,
    data: SeriesData,
    onsets: Sequence[OnsetPoint],
    args: argparse.Namespace,
) -> list[Path]:
    """Write all manuscript-style FBRM publication variants."""
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
    paths: list[Path] = []
    for variant in variants:
        out_png = outdir / f"fbrm-onset-{variant}.png"
        save_fbrm_publication_plot(
            path=out_png,
            data=data,
            onsets=onsets,
            args=args,
            variant=variant,
        )
        paths.append(out_png)
    return paths

def print_summary(
    args: argparse.Namespace,
    data: SeriesData,
    onsets: Sequence[OnsetPoint],
    candidates: Sequence[CandidatePoint],
    run_status: str,
    outdir: Path,
) -> None:
    """Print console summary."""
    print(f"Input CSV:    {args.input_csv}")
    print(f"Columns:      X='{data.xcol}', Y='{data.ycol}'")
    print(f"Order:        {data.order_mode}; temperature sorting not used")
    print(
        f"Cooling start: method={data.cooling_start_method}, row={data.cooling_start_input_row}, "
        f"Tr={data.cooling_start_Tr_C:.3f} °C, Total Counts={data.cooling_start_count:.3f}"
    )
    print("Axis:         measured/interpolated Tr (°C); Tr_regular not used")
    print(
        f"Initial baseline: Tr {data.initial_baseline.Tr_start_C:.3f}..{data.initial_baseline.Tr_end_C:.3f} °C, "
        f"median={data.initial_baseline.median:.3f}, threshold={data.initial_baseline.threshold:.3f}"
    )
    if float(args.max_search_count) > 0:
        print(f"Search limit: raw Total Counts < {args.max_search_count:g}; analyzed rows={int(np.sum(data.analysis_mask))}")
    print(f"Output dir:   {outdir}")
    print(f"Run status:   {run_status}")
    print(f"Candidates:   {len(candidates)}")

    accepted = [o for o in onsets if o.status == "accepted"]
    if accepted:
        print("Accepted onsets:")
        for o in accepted:
            print(
                f"  {o.onset_label}: row={o.input_row}, Tr={o.Tr_C:.3f} °C, "
                f"Total Counts={o.total_counts_smooth:.3f}, mode={o.acceptance_mode}"
            )
        if len(accepted) < int(args.onsets):
            print(f"  Only {len(accepted)} accepted onset(s) found; no later valid onset before search limit.")
    else:
        print("Accepted onsets: none")

    print(f"Wrote CSV:    {outdir / 'fbrm-onsets.csv'}")
    print(f"Wrote candidates CSV: {outdir / 'fbrm-onset-candidates.csv'}")
    print(f"Wrote params: {outdir / 'fbrm-onset-params.json'}")
    print(f"Wrote report: {outdir / 'fbrm-onset-report.txt'}")
    print(
        "Wrote plots:  "
        f"{outdir / 'fbrm-onset-publication.png'}, "
        f"{outdir / 'fbrm-onset-diagnostic.png'}, "
        f"{outdir / 'fbrm-onset-candidates.png'}"
    )
    if args.publication_plots:
        print(f"Wrote publication variants: {outdir / 'fbrm-onset-publication-*.png'}")


def resolve_fbrm_onsets_from_config(args: argparse.Namespace) -> None:
    """Resolve FBRM onset count from analysis/onset-config.ini.

    The config file is the persistent truth source for workflow reruns. A direct
    command-line --onsets value is still accepted for explicit one-off testing,
    but a warning is printed if it differs from the INI value.
    """
    analysis_dir = infer_analysis_dir_from_paths(
        analysis_dir=args.analysis_dir,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        fallback=Path.cwd(),
    )
    cfg = ensure_onset_config(
        analysis_dir,
        dry_run=False,
        experiment_name=analysis_dir.parent.parent.parent.parent.name
        if len(analysis_dir.parts) >= 4
        else str(analysis_dir),
    )

    explicit_onsets = args.onsets
    if explicit_onsets is None:
        args.onsets = cfg.fbrm_onsets
        args.onset_config_status = cfg.status
        args.onset_config_path = str(cfg.path)
        args.onset_config_fbrm_onsets = cfg.fbrm_onsets
        args.onset_config_mgi_onsets = cfg.mgi_onsets
        return

    if int(explicit_onsets) < 0:
        raise ValueError(f"--onsets must be an integer >= 0, got: {explicit_onsets}")

    if int(explicit_onsets) != int(cfg.fbrm_onsets):
        print(
            "!" * 72
            + "\n"
            + "[WARNING:onset-config] explicit --onsets differs from onset-config.ini\n\n"
            + f"Config file: {cfg.path}\n"
            + f"[FBRM] onsets in config: {cfg.fbrm_onsets}\n"
            + f"Explicit --onsets for this run: {explicit_onsets}\n\n"
            + "This command-line value is a one-off override. Edit "
            + "analysis/onset-config.ini to make the decision persistent.\n"
            + "!" * 72
        )

    args.onsets = int(explicit_onsets)
    args.onset_config_status = cfg.status
    args.onset_config_path = str(cfg.path)
    args.onset_config_fbrm_onsets = cfg.fbrm_onsets
    args.onset_config_mgi_onsets = cfg.mgi_onsets


def main() -> int:
    """Run CLI."""
    args = parse_args()
    try:
        resolve_fbrm_onsets_from_config(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    try:
        args.view_limits = parse_view_limits_from_args(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    data = read_data(args)
    refs = read_reference_onsets(args)
    onsets, candidates, run_status = detect_onsets(data, args, refs)

    base_outdir = args.output_dir
    outdir = resolve_output_dir_for_limits(base_outdir, args, args.view_limits)
    outdir.mkdir(parents=True, exist_ok=True)

    write_dataclass_csv(outdir / "fbrm-onsets.csv", onsets)
    write_dataclass_csv(outdir / "fbrm-onset-candidates.csv", candidates)
    write_params(outdir / "fbrm-onset-params.json", args, data, refs, run_status)
    write_report(outdir / "fbrm-onset-report.txt", args, data, onsets, candidates, run_status)
    write_view_limits_json(
        outdir / "view-limits.json",
        limits=args.view_limits,
        content_type="fbrm",
        output_mode="limited-view" if args.view_limits.any else "base",
        extra={"primary_y_axis": "FBRM Total Counts", "derivative_y_axis_manual_limits_applied": False},
    )
    plot_publication(outdir / "fbrm-onset-publication.png", data, onsets, candidates, refs, args)
    plot_diagnostic(outdir / "fbrm-onset-diagnostic.png", data, onsets, candidates, refs, args)
    plot_candidates(outdir / "fbrm-onset-candidates.png", data, candidates, args)
    if args.publication_plots:
        save_fbrm_publication_plots(outdir=outdir, data=data, onsets=onsets, args=args)
    print_summary(args, data, onsets, candidates, run_status, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
