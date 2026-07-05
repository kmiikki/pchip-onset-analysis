#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-mgi-fbrm-combo.py
----------------------

Create manuscript-style combo figures from already analyzed MGI/PCHIP and FBRM
onset-analysis results.

This script is a post-analysis figure generator. It does not detect onsets.

Default expected working directory:

    experiment/tl/roi/rgb/analysis/

Default inputs:

    rgb-tr.csv
    ts-fbrm-tr.csv

    pchip/bw-raw-vs-temp-pchip.csv
    pchip/bw-raw-vs-temp-pchip-bends.csv

    fbrm_onsets/fbrm-onsets.csv
    fbrm_onsets/fbrm-onset-params.json

Default outputs:

    combo/mgi-fbrm-combo.png
    combo/mgi-fbrm-combo-data.csv
    combo/mgi-fbrm-combo-params.json
    combo/mgi-fbrm-combo-report.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

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

try:
    from scipy.signal import savgol_filter
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[error] scipy is required for FBRM Savitzky-Golay smoothing. "
        "Run this in the py314/lab314 environment."
    ) from exc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MGI_SOURCE_CSV = Path("rgb-tr.csv")
DEFAULT_FBRM_CSV = Path("ts-fbrm-tr.csv")

DEFAULT_MGI_PCHIP_CSV = Path("pchip/bw-raw-vs-temp-pchip.csv")
DEFAULT_MGI_ONSETS_CSV = Path("pchip/bw-raw-vs-temp-pchip-bends.csv")

DEFAULT_FBRM_ONSETS_CSV = Path("fbrm_onsets/fbrm-onsets.csv")
DEFAULT_FBRM_PARAMS_JSON = Path("fbrm_onsets/fbrm-onset-params.json")

DEFAULT_OUTPUT_DIR = Path("combo")

TEMPERATURE_COLUMNS = ("Tr (°C)", "Tr_C", "Tr", "Temperature", "Temperature (°C)")
MGI_COLUMNS = ("BW", "MGI", "BW_smooth", "Mean gray intensity", "mean_gray_intensity")
FBRM_COLUMNS = ("Total Counts", "FBRM Total Counts", "Total_Counts", "Counts")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PLOT_FONT_SIZES = {
    "axis_label": 13,
    "tick_label": 11,
    "legend": 11,
    "annotation": 10,
}

RAW_POINT_ALPHA = 0.22
MGI_RAW_POINT_ALPHA = 0.08
FBRM_RAW_POINT_ALPHA = 0.08
RAW_POINT_SIZE = 6
MAIN_LINE_WIDTH = 1.6
DERIVATIVE_LINE_WIDTH = 1.3
ONSET_LINE_ALPHA = 0.65
ONSET_LINE_WIDTH = 1.0

# Manuscript combo colors.
MGI_RAW_COLOR = "#1f77b4"      # blue raw/source points
MGI_COLOR = "#003f7f"          # dark blue PCHIP curve
FBRM_RAW_COLOR = "#ff7f0e"     # orange raw points
FBRM_COLOR = "#b22222"         # dark red SG curve
ONSET_ANNOTATION_ALPHA = 0.85


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Onset:
    source: str
    label: str
    temperature_C: float


@dataclass(frozen=True)
class ComboInputs:
    mgi_source_csv: str | None
    mgi_pchip_csv: str
    mgi_onsets_csv: str
    fbrm_csv: str
    fbrm_onsets_csv: str
    fbrm_params_json: str | None


@dataclass(frozen=True)
class FbrmSmoothing:
    method: str
    savgol_window_requested: int | None
    savgol_window_used_prefix: int | None
    savgol_polyorder: int | None
    source: str
    note: str


@dataclass(frozen=True)
class FbrmAnalysisWindow:
    """FBRM plotting window reused from the FBRM onset analysis metadata."""

    source: str
    applied: bool
    cooling_start_input_row: int | None
    start_position: int | None
    analyzed_rows: int | None
    cutoff_Tr_C: float | None
    cutoff_count: float | None
    note: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create manuscript-style MGI/FBRM combo figure from already analyzed "
            "PCHIP/MGI and FBRM onset results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "analysis_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Analysis directory. Default: current directory.",
    )

    p.add_argument("--mgi-source-csv", type=Path, default=None)
    p.add_argument("--mgi-pchip-csv", type=Path, default=None)
    p.add_argument("--mgi-onsets-csv", type=Path, default=None)

    p.add_argument("--fbrm-csv", type=Path, default=None)
    p.add_argument("--fbrm-onsets-csv", type=Path, default=None)
    p.add_argument("--fbrm-params-json", type=Path, default=None)

    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--output-name", type=str, default="mgi-fbrm-combo")

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-reverse-x", action="store_true")
    p.add_argument("--show-title", action="store_true")
    p.add_argument("--no-raw-points", action="store_true")
    p.add_argument("--no-derivative-panel", action="store_true")
    p.add_argument(
        "--same-aspect-diagnostics",
        action="store_true",
        help=(
            "Use the same canvas aspect ratio for derivative-panel combo figures "
            "as for main-only figures. Default keeps derivative-panel figures "
            "taller for diagnostic/QC reading."
        ),
    )

    p.add_argument(
        "--fbrm-count-unit",
        type=str,
        default="counts/s",
        help="Unit shown on the FBRM y-axis. Use an empty string for no unit.",
    )

    p.add_argument(
        "--fallback-savgol-window",
        type=int,
        default=101,
        help="Fallback FBRM Savitzky-Golay window if params JSON is missing.",
    )
    p.add_argument(
        "--fallback-savgol-polyorder",
        type=int,
        default=3,
        help="Fallback FBRM Savitzky-Golay polyorder if params JSON is missing.",
    )

    add_view_limit_arguments(p)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def resolve_path(base: Path, explicit: Path | None, default: Path) -> Path:
    """Resolve explicit or default path against the analysis directory."""
    path = explicit if explicit is not None else default
    if path.is_absolute():
        return path
    return base / path


def require_file(path: Path, role: str) -> Path:
    """Return path if it exists, otherwise fail clearly."""
    if not path.exists():
        raise FileNotFoundError(f"{role} not found: {path}")
    return path


def optional_file(path: Path) -> Path | None:
    """Return existing optional path or None."""
    return path if path.exists() else None


def autodetect_column(columns: Sequence[str], candidates: Sequence[str], role: str) -> str:
    """Return first matching column name."""
    lowered = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        key = name.lower().strip()
        if key in lowered:
            return lowered[key]
    raise KeyError(f"Could not auto-detect {role} column. Columns: {list(columns)}")


def read_xy_csv(path: Path, x_candidates: Sequence[str], y_candidates: Sequence[str]) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Read x/y numeric columns from CSV."""
    df = pd.read_csv(path, encoding="utf-8")
    columns = list(df.columns)
    xcol = autodetect_column(columns, x_candidates, "x")
    ycol = autodetect_column(columns, y_candidates, "y")

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError(f"Too few numeric rows in {path}")

    return xcol, ycol, x, y


def read_mgi_pchip(path: Path) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Read MGI PCHIP curve. The PCHIP file is normally already sorted."""
    return read_xy_csv(path, TEMPERATURE_COLUMNS, MGI_COLUMNS)


def read_mgi_source(path: Path | None) -> tuple[str, str, np.ndarray, np.ndarray] | None:
    """Read optional original MGI source points."""
    if path is None:
        return None
    return read_xy_csv(path, TEMPERATURE_COLUMNS, MGI_COLUMNS)


def read_fbrm_curve(path: Path) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Read FBRM Tr/Total Counts curve in file order."""
    return read_xy_csv(path, TEMPERATURE_COLUMNS, FBRM_COLUMNS)


def read_fbrm_analysis_window(params_path: Path | None) -> FbrmAnalysisWindow:
    """Read FBRM cooling-start and analyzed-prefix metadata from params JSON."""
    if params_path is None or not params_path.exists():
        return FbrmAnalysisWindow(
            source="none",
            applied=False,
            cooling_start_input_row=None,
            start_position=None,
            analyzed_rows=None,
            cutoff_Tr_C=None,
            cutoff_count=None,
            note="FBRM params JSON not found; full FBRM curve was used.",
        )

    data = json.loads(params_path.read_text(encoding="utf-8"))
    cooling = data.get("cooling_start", {})
    cutoff = data.get("validity_cutoff", {})

    input_row = cooling.get("input_row", None)
    analyzed_rows = cutoff.get("analyzed_rows", None)

    return FbrmAnalysisWindow(
        source=str(params_path),
        applied=True,
        cooling_start_input_row=None if input_row is None else int(input_row),
        start_position=None,
        analyzed_rows=None if analyzed_rows is None else int(analyzed_rows),
        cutoff_Tr_C=cutoff.get("cutoff_Tr_C", None),
        cutoff_count=cutoff.get("cutoff_count", None),
        note=(
            "FBRM curve is cropped to the cooling segment and analyzed prefix "
            "defined by fbrm-onset-params.json."
        ),
    )


def read_fbrm_curves_for_combo(
    path: Path,
    params_path: Path | None,
) -> tuple[
    tuple[str, str, np.ndarray, np.ndarray],
    tuple[str, str, np.ndarray, np.ndarray],
    FbrmAnalysisWindow,
]:
    """Read FBRM curves for combo plotting.

    Returns two FBRM curves:

    1. Main-panel curve:
       cooling start -> end of data. This follows the manuscript-style visual
       comparison and does not clip the displayed curve at the high-count
       search limit.

    2. Derivative/analyzed curve:
       cooling start -> analyzed prefix before the high-count search limit.
       This reuses the FBRM onset-analysis window for derivative diagnostics.
    """
    df = pd.read_csv(path, encoding="utf-8")
    columns = list(df.columns)

    xcol = autodetect_column(columns, TEMPERATURE_COLUMNS, "FBRM temperature")
    ycol = autodetect_column(columns, FBRM_COLUMNS, "FBRM total counts")

    original_rows = df.index.to_numpy(dtype=int)
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    original_rows = original_rows[mask]

    if len(x) < 3:
        raise ValueError(f"Too few numeric FBRM rows in {path}")

    window = read_fbrm_analysis_window(params_path)

    start = 0
    if window.applied and window.cooling_start_input_row is not None:
        matches = np.where(original_rows == int(window.cooling_start_input_row))[0]
        start = int(matches[0]) if matches.size else 0

    # Main manuscript comparison curve: cooling start -> end.
    x_main = x[start:]
    y_main = y[start:]

    # Derivative/analyzed curve: cooling start -> analyzed prefix.
    if window.applied and window.analyzed_rows is not None and window.analyzed_rows > 0:
        end = min(len(x), start + int(window.analyzed_rows))
    else:
        end = len(x)

    x_deriv = x[start:end]
    y_deriv = y[start:end]

    window = FbrmAnalysisWindow(
        source=window.source,
        applied=window.applied,
        cooling_start_input_row=window.cooling_start_input_row,
        start_position=start,
        analyzed_rows=len(x_deriv),
        cutoff_Tr_C=window.cutoff_Tr_C,
        cutoff_count=window.cutoff_count,
        note=(
            "Main-panel FBRM curve uses cooling start -> end. "
            "Derivative-panel FBRM curve uses the analyzed prefix from "
            "fbrm-onset-params.json."
        ),
    )

    return (xcol, ycol, x_main, y_main), (xcol, ycol, x_deriv, y_deriv), window



def read_mgi_onsets(path: Path) -> list[Onset]:
    """Read accepted MGI/PCHIP onset temperatures from bends CSV."""
    df = pd.read_csv(path, encoding="utf-8")
    if "temperature_C" not in df.columns:
        raise KeyError(f"temperature_C missing in {path}; columns: {list(df.columns)}")

    label_col = "bend_index" if "bend_index" in df.columns else None

    out: list[Onset] = []
    for i, row in df.iterrows():
        temp = pd.to_numeric(pd.Series([row["temperature_C"]]), errors="coerce").iloc[0]
        if not np.isfinite(temp):
            continue
        if label_col is not None:
            label = f"T{int(row[label_col])}"
        else:
            label = f"T{i + 1}"
        out.append(Onset(source="MGI", label=label, temperature_C=float(temp)))
    return out


def read_fbrm_onsets(path: Path) -> list[Onset]:
    """Read accepted FBRM onset temperatures from fbrm-onsets.csv."""
    df = pd.read_csv(path, encoding="utf-8")

    if "Tr_C" not in df.columns:
        raise KeyError(f"Tr_C missing in {path}; columns: {list(df.columns)}")

    out: list[Onset] = []
    for i, row in df.iterrows():
        status = str(row.get("status", "accepted"))
        if status != "accepted":
            continue
        temp = pd.to_numeric(pd.Series([row["Tr_C"]]), errors="coerce").iloc[0]
        if not np.isfinite(temp):
            continue
        label = str(row.get("onset_label", f"T{i + 1}"))
        out.append(Onset(source="FBRM", label=label, temperature_C=float(temp)))
    return out


def adjusted_savgol_window(n: int, requested: int, polyorder: int) -> int:
    """Return valid odd Savitzky-Golay window or 0 if impossible."""
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


def read_fbrm_smoothing(params_path: Path | None, fallback_window: int, fallback_polyorder: int) -> FbrmSmoothing:
    """Read FBRM smoothing parameters from params JSON, or fallback."""
    if params_path is not None and params_path.exists():
        data = json.loads(params_path.read_text(encoding="utf-8"))
        smoothing = data.get("smoothing", {})
        return FbrmSmoothing(
            method=str(smoothing.get("method", "savgol")),
            savgol_window_requested=smoothing.get("savgol_window_requested", None),
            savgol_window_used_prefix=smoothing.get("savgol_window_used_prefix", None),
            savgol_polyorder=smoothing.get("savgol_polyorder", None),
            source=str(params_path),
            note=str(smoothing.get("note", "")),
        )

    return FbrmSmoothing(
        method="savgol",
        savgol_window_requested=int(fallback_window),
        savgol_window_used_prefix=int(fallback_window),
        savgol_polyorder=int(fallback_polyorder),
        source="fallback_cli_defaults",
        note="FBRM params JSON was not found; fallback Savitzky-Golay parameters were used for figure preparation.",
    )


def smooth_fbrm_for_combo(y: np.ndarray, smoothing: FbrmSmoothing) -> np.ndarray:
    """Apply FBRM smoothing for visualization/derivative preparation."""
    if smoothing.method == "none":
        return y.astype(float).copy()

    if smoothing.method != "savgol":
        raise ValueError(f"Unsupported FBRM smoothing method for combo: {smoothing.method}")

    requested = smoothing.savgol_window_used_prefix or smoothing.savgol_window_requested or 101
    poly = smoothing.savgol_polyorder or 3

    w = adjusted_savgol_window(len(y), int(requested), int(poly))
    if w <= 0:
        return y.astype(float).copy()

    return savgol_filter(y.astype(float), window_length=w, polyorder=int(poly), mode="interp")


def sort_by_temperature(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort x/y by increasing temperature and collapse duplicate x values."""
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    unique_x, start_idx = np.unique(xs, return_index=True)
    if len(unique_x) == len(xs):
        return xs, ys

    y_means: list[float] = []
    counts = np.diff(np.append(start_idx, len(xs)))
    ptr = 0
    for count in counts:
        y_means.append(float(np.mean(ys[ptr : ptr + count])))
        ptr += count

    return unique_x, np.array(y_means, dtype=float)


def derivative_normalized_abs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x and normalized absolute dY/dT for derivative panel."""
    xs, ys = sort_by_temperature(x, y)
    if len(xs) < 3:
        return xs, np.full_like(xs, np.nan, dtype=float)

    dy = np.gradient(ys, xs)
    abs_d = np.abs(dy)
    finite = np.isfinite(abs_d)
    if not np.any(finite):
        return xs, abs_d

    max_v = float(np.nanmax(abs_d[finite]))
    if not np.isfinite(max_v) or max_v <= 0:
        return xs, np.zeros_like(abs_d)

    return xs, abs_d / max_v


def json_safe(value: Any) -> Any:
    """Convert numpy/path values to JSON-safe form."""
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


def apply_axis_style(ax: plt.Axes) -> None:
    """Apply common manuscript figure font sizes."""
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick_label"])
    ax.xaxis.label.set_size(PLOT_FONT_SIZES["axis_label"])
    ax.yaxis.label.set_size(PLOT_FONT_SIZES["axis_label"])

    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(PLOT_FONT_SIZES["legend"])


def combined_legend(ax1: plt.Axes, ax2: plt.Axes, loc: str = "best") -> None:
    """Create combined legend from twin axes."""
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc=loc, fontsize=PLOT_FONT_SIZES["legend"])



def view_x_contains(x_value: float, view_limits: ViewLimits = NONE_LIMITS) -> bool:
    """Return True if x_value is visible in the requested manual x view.

    Missing/automatic bounds are treated as open intervals. This helper is used
    only for display annotations; it must not affect data processing, onset
    values, CSV output, or smoothing/interpolation.
    """
    try:
        value = float(x_value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value):
        return False

    x_limits = getattr(view_limits, "x", None)
    if x_limits is None:
        return True

    lower = getattr(x_limits, "lower", None)
    upper = getattr(x_limits, "upper", None)

    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def add_derivative_onset_lines(
    ax: plt.Axes,
    onsets: Sequence[Onset],
    linestyle: str,
    alpha: float,
    *,
    view_limits: ViewLimits = NONE_LIMITS,
) -> None:
    """Add accepted onset lines to derivative panel only."""
    for onset in onsets:
        if not view_x_contains(onset.temperature_C, view_limits):
            continue
        color = MGI_COLOR if onset.source == "MGI" else FBRM_COLOR
        zorder = 5 if onset.source == "MGI" else 4
        ax.axvline(
            onset.temperature_C,
            linestyle=linestyle,
            linewidth=ONSET_LINE_WIDTH,
            alpha=alpha,
            color=color,
            zorder=zorder,
            clip_on=True,
        )


def annotate_derivative_onsets(
    ax: plt.Axes,
    onsets: Sequence[Onset],
    y: float,
    x_offset: float = 0.0,
    *,
    view_limits: ViewLimits = NONE_LIMITS,
) -> None:
    """Add small onset labels to derivative panel."""
    for onset in onsets:
        x_pos = onset.temperature_C + x_offset
        if not view_x_contains(x_pos, view_limits):
            continue
        color = MGI_COLOR if onset.source == "MGI" else FBRM_COLOR
        zorder = 7 if onset.source == "MGI" else 6
        ax.text(
            x_pos,
            y,
            f"{onset.source} {onset.label}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            alpha=ONSET_ANNOTATION_ALPHA,
            color=color,
            zorder=zorder,
            clip_on=True,
        )

# ---------------------------------------------------------------------------
# Plotting and outputs
# ---------------------------------------------------------------------------

def outward_temperature_limits(
    tr: np.ndarray,
    major_step_C: float = 5.0,
    snap_threshold_C: float = 1.0,
) -> tuple[float, float]:
    """Return outward-rounded low/high temperature limits.

    The limits are derived from actual MGI temperature data. They may snap
    outward to readable grid boundaries, but never inward:

    - high-temperature side snaps upward to the next grid boundary
    - low-temperature side snaps downward to the previous grid boundary

    This function returns limits in numeric order: (low, high). It does not
    decide the displayed x-axis direction.
    """
    x = np.asarray(tr, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Cannot determine x-limits from empty MGI temperature data.")

    data_hi = float(np.max(x))
    data_lo = float(np.min(x))

    upper_grid = float(np.ceil(data_hi / major_step_C) * major_step_C)
    lower_grid = float(np.floor(data_lo / major_step_C) * major_step_C)

    x_hi = upper_grid if (upper_grid - data_hi) <= snap_threshold_C else data_hi
    x_lo = lower_grid if (data_lo - lower_grid) <= snap_threshold_C else data_lo

    return x_lo, x_hi


def apply_temperature_xlim_preserve_direction(
    ax,
    tr: np.ndarray,
    major_step_C: float = 5.0,
    snap_threshold_C: float = 1.0,
) -> None:
    """Apply outward-rounded x-limits while preserving current axis direction."""
    x_lo, x_hi = outward_temperature_limits(
        tr,
        major_step_C=major_step_C,
        snap_threshold_C=snap_threshold_C,
    )

    old_left, old_right = ax.get_xlim()
    if old_left > old_right:
        ax.set_xlim(x_hi, x_lo)
    else:
        ax.set_xlim(x_lo, x_hi)


def xlim_source_from_mgi(
    src_x: np.ndarray | None,
    mgi_x: np.ndarray,
) -> np.ndarray:
    """Prefer original MGI source temperatures for x-axis limits when available."""
    if src_x is not None:
        x = np.asarray(src_x, dtype=float)
        if np.isfinite(x).any():
            return x
    return np.asarray(mgi_x, dtype=float)


def make_plot(
    *,
    output_png: Path,
    title: str | None,
    reverse_x: bool,
    show_raw_points: bool,
    derivative_panel: bool,
    same_aspect_diagnostics: bool = False,
    fbrm_count_unit: str,
    mono: bool,
    clean_mono: bool,
    mgi_source: tuple[str, str, np.ndarray, np.ndarray] | None,
    mgi_pchip: tuple[str, str, np.ndarray, np.ndarray],
    fbrm_curve_main: tuple[str, str, np.ndarray, np.ndarray],
    fbrm_curve_deriv: tuple[str, str, np.ndarray, np.ndarray],
    fbrm_smooth_y_main: np.ndarray,
    fbrm_smooth_y_deriv: np.ndarray,
    mgi_onsets: Sequence[Onset],
    fbrm_onsets: Sequence[Onset],
    view_limits: ViewLimits = NONE_LIMITS,
) -> None:
    """Create combo figure."""
    if derivative_panel:
        figure_size = (9.0, 5.2) if same_aspect_diagnostics else (9.0, 7.0)
        height_ratios = [3.0, 0.85] if same_aspect_diagnostics else [3.0, 1.25]
        fig, (ax, axd) = plt.subplots(
            2,
            1,
            figsize=figure_size,
            dpi=plt.rcParams.get("figure.dpi", 300),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.06},
        )
    else:
        fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=plt.rcParams.get("figure.dpi", 300))
        axd = None

    ax2 = ax.twinx()

    # Draw the MGI axis above the FBRM twin axis. Line zorder alone is not
    # enough across twin axes; the axes themselves also have a draw order.
    # The MGI axis patch must be transparent so the FBRM data remains visible.
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    _, _, mgi_x, mgi_y = mgi_pchip
    fbrm_xcol, fbrm_ycol, fbrm_x, fbrm_y = fbrm_curve_main

    if clean_mono:
        mgi_raw_color = "0.75"
        mgi_line_color = "0.00"
        fbrm_raw_color = "0.78"
        fbrm_line_color = "0.28"
        fbrm_line_style = (0, (3, 1.5))
    elif mono:
        mgi_raw_color = "0.70"
        mgi_line_color = "0.00"
        fbrm_raw_color = "0.78"
        fbrm_line_color = "0.25"
        fbrm_line_style = (0, (3, 1.5))
    else:
        mgi_raw_color = MGI_RAW_COLOR
        mgi_line_color = MGI_COLOR
        fbrm_raw_color = FBRM_RAW_COLOR
        fbrm_line_color = FBRM_COLOR
        fbrm_line_style = "-"

    # Main panel: no onset lines.
    if show_raw_points and not clean_mono and mgi_source is not None:
        _, _, src_x, src_y = mgi_source
        ax.scatter(src_x, src_y, s=RAW_POINT_SIZE, alpha=MGI_RAW_POINT_ALPHA, color=mgi_raw_color, zorder=1)

    ax.plot(mgi_x, mgi_y, linewidth=MAIN_LINE_WIDTH, label="MGI", color=mgi_line_color, zorder=3)

    if show_raw_points and not clean_mono:
        ax2.scatter(fbrm_x, fbrm_y, s=RAW_POINT_SIZE, alpha=FBRM_RAW_POINT_ALPHA, color=fbrm_raw_color, zorder=1)

    ax2.plot(
        fbrm_x,
        fbrm_smooth_y_main,
        linewidth=MAIN_LINE_WIDTH,
        label="FBRM",
        color=fbrm_line_color,
        linestyle=fbrm_line_style,
        zorder=3,
    )

    ax.set_ylabel("MGI")

    fbrm_ylabel = "FBRM total counts"
    if fbrm_count_unit:
        fbrm_ylabel += f" ({fbrm_count_unit})"
    ax2.set_ylabel(fbrm_ylabel)

    if title:
        ax.set_title(title)

    ax.grid(False)
    ax2.grid(False)
    combined_legend(ax, ax2, loc="best")
    apply_axis_style(ax)
    apply_axis_style(ax2)

    # Derivative panel.
    if axd is not None:
        mgi_dx, mgi_abs_norm = derivative_normalized_abs(mgi_x, mgi_y)
        # Diagnostic derivatives:
        #   MGI  -> PCHIP curve
        #   FBRM -> Savitzky-Golay-filtered full cooling curve
        _, _, fbrm_xd, _ = fbrm_curve_main
        fbrm_dx, fbrm_abs_norm = derivative_normalized_abs(fbrm_xd, fbrm_smooth_y_main)

        # Draw SG/FBRM first and PCHIP/MGI last so MGI remains visible on top.
        sg_line = axd.plot(
            fbrm_dx,
            fbrm_abs_norm,
            linewidth=DERIVATIVE_LINE_WIDTH,
            label="SG",
            color=FBRM_COLOR,
            zorder=2,
        )
        pchip_line = axd.plot(
            mgi_dx,
            mgi_abs_norm,
            linewidth=DERIVATIVE_LINE_WIDTH,
            label="PCHIP",
            color=MGI_COLOR,
            zorder=3,
        )

        add_derivative_onset_lines(
            axd,
            fbrm_onsets,
            linestyle=":",
            alpha=ONSET_ANNOTATION_ALPHA,
            view_limits=view_limits,
        )
        add_derivative_onset_lines(
            axd,
            mgi_onsets,
            linestyle="--",
            alpha=ONSET_ANNOTATION_ALPHA,
            view_limits=view_limits,
        )

        annotate_derivative_onsets(axd, fbrm_onsets, y=0.78, view_limits=view_limits)
        annotate_derivative_onsets(axd, mgi_onsets, y=0.98, view_limits=view_limits)

        axd.set_ylabel("Normalized |dY/dT|")
        axd.set_ylim(-0.03, 1.05)
        axd.grid(False)
        axd.legend(
            handles=[pchip_line[0], sg_line[0]],
            labels=["PCHIP", "SG"],
            loc="best",
            fontsize=PLOT_FONT_SIZES["legend"],
        )
        apply_axis_style(axd)

    bottom_ax = axd if axd is not None else ax
    bottom_ax.set_xlabel("Tr (°C)")

    # Use the MGI acquisition temperature range as the x-axis truth source.
    # FBRM is already clipped to the MGI timestamp window; it must not stretch
    # the displayed x-axis beyond the MGI-defined domain. Manual x limits are
    # display-only and are resolved against the same MGI source.
    mgi_xlim_source = xlim_source_from_mgi(locals().get("src_x"), mgi_x)
    x_limits = apply_temperature_view_limits(
        ax,
        mgi_xlim_source,
        view_limits,
        reverse_x=reverse_x,
    )
    for axis in fig.axes:
        if axis is not ax:
            axis.set_xlim(ax.get_xlim())

    # Primary signal y-limits are signal-specific.  They autoscale from the
    # currently visible x-range unless manual bounds were provided.
    mgi_series: list[tuple[np.ndarray, np.ndarray]] = [(mgi_x, mgi_y)]
    if show_raw_points and not clean_mono and mgi_source is not None:
        mgi_series.append((src_x, src_y))
    apply_signal_y_view_limits(ax, mgi_series, x_limits, view_limits.mgi_y)

    fbrm_series: list[tuple[np.ndarray, np.ndarray]] = [(fbrm_x, fbrm_smooth_y_main)]
    if show_raw_points and not clean_mono:
        fbrm_series.append((fbrm_x, fbrm_y))
    apply_signal_y_view_limits(ax2, fbrm_series, x_limits, view_limits.fbrm_y)


    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def write_combo_data(
    path: Path,
    mgi_pchip: tuple[str, str, np.ndarray, np.ndarray],
    fbrm_curve_main: tuple[str, str, np.ndarray, np.ndarray],
    fbrm_curve_deriv: tuple[str, str, np.ndarray, np.ndarray],
    fbrm_smooth_y_main: np.ndarray,
    fbrm_smooth_y_deriv: np.ndarray,
) -> None:
    """Write long-form plotted data for reproducibility."""
    rows: list[dict[str, Any]] = []

    _, _, mgi_x, mgi_y = mgi_pchip
    mgi_dx, mgi_abs_norm = derivative_normalized_abs(mgi_x, mgi_y)
    for x, y in zip(mgi_x, mgi_y):
        rows.append({"series": "MGI", "kind": "curve", "Tr_C": x, "value": y})
    for x, y in zip(mgi_dx, mgi_abs_norm):
        rows.append({"series": "MGI", "kind": "normalized_abs_derivative", "Tr_C": x, "value": y})

    _, _, fbrm_x, fbrm_y = fbrm_curve_main
    for x, y_raw, y_smooth in zip(fbrm_x, fbrm_y, fbrm_smooth_y_main):
        rows.append({"series": "FBRM", "kind": "raw_curve_main_panel", "Tr_C": x, "value": y_raw})
        rows.append({"series": "FBRM", "kind": "smoothed_curve_main_panel", "Tr_C": x, "value": y_smooth})

    _, _, fbrm_xd, _ = fbrm_curve_main
    fbrm_dx, fbrm_abs_norm = derivative_normalized_abs(fbrm_xd, fbrm_smooth_y_main)
    for x, y in zip(fbrm_dx, fbrm_abs_norm):
        rows.append({"series": "FBRM", "kind": "normalized_abs_derivative_full_sg_cooling_curve", "Tr_C": x, "value": y})

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_params(
    path: Path,
    *,
    args: argparse.Namespace,
    inputs: ComboInputs,
    fbrm_smoothing: FbrmSmoothing,
    fbrm_analysis_window: FbrmAnalysisWindow,
    mgi_onsets: Sequence[Onset],
    fbrm_onsets: Sequence[Onset],
    outputs: dict[str, str],
) -> None:
    """Write combo generation metadata JSON."""
    payload = {
        "script": "make-mgi-fbrm-combo.py",
        "purpose": "post_analysis_manuscript_combo_figure_generator",
        "analysis_dir": str(args.analysis_dir),
        "does_not_detect_onsets": True,
        "uses_candidate_onsets": False,
        "main_panel": {
            "x_axis": "Tr (°C)",
            "left_y_axis": "MGI",
            "right_y_axis": "FBRM total counts"
            + (f" ({args.fbrm_count_unit})" if args.fbrm_count_unit else ""),
            "legend": ["MGI", "FBRM"],
            "curve_meaning": {
                "MGI": "PCHIP curve over MGI raw/source points",
                "FBRM": "Savitzky-Golay-filtered FBRM curve over raw FBRM points"
            },
            "title": "shown" if args.show_title else "not shown",
            "grid": "off",
            "onset_vertical_lines": "not shown",
            "fbrm_curve_window": "cooling start to end of data; not clipped at high-count search limit",
        },
        "derivative_panel": {
            "enabled": not args.no_derivative_panel,
            "y_axis": "Normalized |dY/dT|",
            "normalization": "each derivative series normalized to its own maximum absolute dY/dT",
            "onset_vertical_lines": "accepted MGI and FBRM onsets only",
            "candidate_onsets": "not shown",
            "derivative_sources": {
                "PCHIP": "MGI derivative from pchip/bw-raw-vs-temp-pchip.csv",
                "SG": "FBRM derivative from Savitzky-Golay-filtered cooling-start-to-end FBRM curve"
            },
            "fbrm_derivative_window": "cooling start to end of data for diagnostic display; accepted onset lines still come from FBRM onset analysis",
            "grid": "off",
        },
        "inputs": asdict(inputs),
        "fbrm_smoothing_for_combo": asdict(fbrm_smoothing),
        "fbrm_analysis_window": asdict(fbrm_analysis_window),
        "mgi_onsets": [asdict(o) for o in mgi_onsets],
        "fbrm_onsets": [asdict(o) for o in fbrm_onsets],
        "outputs": outputs,
    }

    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    *,
    inputs: ComboInputs,
    fbrm_smoothing: FbrmSmoothing,
    fbrm_analysis_window: FbrmAnalysisWindow,
    mgi_onsets: Sequence[Onset],
    fbrm_onsets: Sequence[Onset],
    outputs: dict[str, str],
) -> None:
    """Write short human-readable report."""
    lines = [
        "MGI/FBRM combo figure",
        "=" * 22,
        "",
        "This is a post-analysis figure generator.",
        "It does not detect onsets and it does not use onset candidates.",
        "",
        "Inputs:",
        f"  MGI source CSV:  {inputs.mgi_source_csv}",
        f"  MGI PCHIP CSV:   {inputs.mgi_pchip_csv}",
        f"  MGI onsets CSV:  {inputs.mgi_onsets_csv}",
        f"  FBRM CSV:        {inputs.fbrm_csv}",
        f"  FBRM onsets CSV: {inputs.fbrm_onsets_csv}",
        f"  FBRM params:     {inputs.fbrm_params_json}",
        "",
        "FBRM smoothing used for combo derivative preparation:",
        f"  method: {fbrm_smoothing.method}",
        f"  window requested: {fbrm_smoothing.savgol_window_requested}",
        f"  window used prefix: {fbrm_smoothing.savgol_window_used_prefix}",
        f"  polyorder: {fbrm_smoothing.savgol_polyorder}",
        f"  source: {fbrm_smoothing.source}",
        "",
        "FBRM analysis window reused for combo plotting:",
        f"  source: {fbrm_analysis_window.source}",
        f"  cooling_start_input_row: {fbrm_analysis_window.cooling_start_input_row}",
        f"  start_position_after_numeric_filter: {fbrm_analysis_window.start_position}",
        f"  plotted/analyzed rows: {fbrm_analysis_window.analyzed_rows}",
        f"  cutoff_Tr_C: {fbrm_analysis_window.cutoff_Tr_C}",
        f"  cutoff_count: {fbrm_analysis_window.cutoff_count}",
        f"  note: {fbrm_analysis_window.note}",
        "",
        "Accepted MGI onsets:",
    ]
    for o in mgi_onsets:
        lines.append(f"  {o.label}: {o.temperature_C:.3f} °C")
    if not mgi_onsets:
        lines.append("  none")

    lines.append("")
    lines.append("Accepted FBRM onsets:")
    for o in fbrm_onsets:
        lines.append(f"  {o.label}: {o.temperature_C:.3f} °C")
    if not fbrm_onsets:
        lines.append("  none")

    lines.append("")
    lines.append("Outputs:")
    for key, value in outputs.items():
        lines.append(f"  {key}: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    try:
        view_limits = parse_view_limits_from_args(args)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2

    analysis_dir = args.analysis_dir.resolve()

    mgi_source_path = resolve_path(analysis_dir, args.mgi_source_csv, DEFAULT_MGI_SOURCE_CSV)
    mgi_source_path_opt = None if args.no_raw_points else optional_file(mgi_source_path)

    mgi_pchip_path = require_file(
        resolve_path(analysis_dir, args.mgi_pchip_csv, DEFAULT_MGI_PCHIP_CSV),
        "MGI PCHIP CSV",
    )
    mgi_onsets_path = require_file(
        resolve_path(analysis_dir, args.mgi_onsets_csv, DEFAULT_MGI_ONSETS_CSV),
        "MGI onsets CSV",
    )
    fbrm_path = require_file(
        resolve_path(analysis_dir, args.fbrm_csv, DEFAULT_FBRM_CSV),
        "FBRM CSV",
    )
    fbrm_onsets_path = require_file(
        resolve_path(analysis_dir, args.fbrm_onsets_csv, DEFAULT_FBRM_ONSETS_CSV),
        "FBRM onsets CSV",
    )
    fbrm_params_path = optional_file(
        resolve_path(analysis_dir, args.fbrm_params_json, DEFAULT_FBRM_PARAMS_JSON)
    )

    base_outdir = resolve_path(analysis_dir, args.outdir, DEFAULT_OUTPUT_DIR)
    outdir = resolve_output_dir_for_limits(base_outdir, args, view_limits)
    outdir.mkdir(parents=True, exist_ok=True)

    output_png = outdir / f"{args.output_name}.png"
    output_png_main = outdir / f"{args.output_name}-main.png"
    output_png_main_mono = outdir / f"{args.output_name}-main-mono.png"
    output_png_main_clean_mono = outdir / f"{args.output_name}-main-clean-mono.png"
    output_csv = outdir / f"{args.output_name}-data.csv"
    output_json = outdir / f"{args.output_name}-params.json"
    output_report = outdir / f"{args.output_name}-report.txt"

    mgi_source = read_mgi_source(mgi_source_path_opt)
    mgi_pchip = read_mgi_pchip(mgi_pchip_path)
    fbrm_curve_main, fbrm_curve_deriv, fbrm_analysis_window = read_fbrm_curves_for_combo(
        fbrm_path,
        fbrm_params_path,
    )

    # Clip FBRM data to the MGI imaging timestamp window before smoothing.
    # The combo figure compares simultaneous MGI and FBRM measurements; FBRM
    # data outside the MGI acquisition time range is not part of that comparison.
    if mgi_source_path_opt is not None:
        mgi_ts_df = pd.read_csv(mgi_source_path_opt, usecols=["Timestamp"])
        fbrm_ts_df = pd.read_csv(fbrm_path, usecols=["Timestamp"])

        mgi_ts = pd.to_numeric(mgi_ts_df["Timestamp"], errors="coerce").to_numpy(dtype=float)
        fbrm_ts_all = pd.to_numeric(fbrm_ts_df["Timestamp"], errors="coerce").to_numpy(dtype=float)

        mgi_ts = mgi_ts[np.isfinite(mgi_ts)]
        if mgi_ts.size:
            t0 = float(np.nanmin(mgi_ts))
            t1 = float(np.nanmax(mgi_ts))
            t_lo = min(t0, t1)
            t_hi = max(t0, t1)

            def clip_curve_to_mgi_time_window(
                curve: tuple[str, str, np.ndarray, np.ndarray],
                label: str,
            ) -> tuple[str, str, np.ndarray, np.ndarray]:
                xcol, ycol, x, y = curve

                # FBRM curves returned by read_fbrm_curves_for_combo() start at
                # the cooling-start position. The derivative curve may be a
                # shorter prefix of the same segment. Therefore timestamps must
                # be sliced from the same start position, not from the tail.
                start = fbrm_analysis_window.start_position
                if start is None:
                    # Conservative fallback for unexpected old parameter files.
                    start = max(0, len(fbrm_ts_all) - len(x))
                start = int(start)

                ts = fbrm_ts_all[start : start + len(x)]

                if len(ts) != len(x):
                    raise ValueError(
                        f"Cannot align FBRM timestamps for {label}: "
                        f"timestamps={len(ts)}, curve={len(x)}, start={start}"
                    )

                mask = np.isfinite(ts) & (ts >= t_lo) & (ts <= t_hi)
                return xcol, ycol, x[mask], y[mask]

            fbrm_curve_main = clip_curve_to_mgi_time_window(fbrm_curve_main, "main")
            fbrm_curve_deriv = clip_curve_to_mgi_time_window(fbrm_curve_deriv, "derivative")

    fbrm_smoothing = read_fbrm_smoothing(
        fbrm_params_path,
        fallback_window=args.fallback_savgol_window,
        fallback_polyorder=args.fallback_savgol_polyorder,
    )
    _, _, _, fbrm_y_main = fbrm_curve_main
    _, _, _, fbrm_y_deriv = fbrm_curve_deriv
    fbrm_smooth_y_main = smooth_fbrm_for_combo(fbrm_y_main, fbrm_smoothing)
    fbrm_smooth_y_deriv = smooth_fbrm_for_combo(fbrm_y_deriv, fbrm_smoothing)

    mgi_onsets = read_mgi_onsets(mgi_onsets_path)
    fbrm_onsets = read_fbrm_onsets(fbrm_onsets_path)

    inputs = ComboInputs(
        mgi_source_csv=None if mgi_source_path_opt is None else str(mgi_source_path_opt),
        mgi_pchip_csv=str(mgi_pchip_path),
        mgi_onsets_csv=str(mgi_onsets_path),
        fbrm_csv=str(fbrm_path),
        fbrm_onsets_csv=str(fbrm_onsets_path),
        fbrm_params_json=None if fbrm_params_path is None else str(fbrm_params_path),
    )

    # Matplotlib dpi is set here so make_plot can use it for both panel variants.
    plt.rcParams["figure.dpi"] = args.dpi

    make_plot(
        output_png=output_png,
        title="MGI and FBRM comparison" if args.show_title else None,
        reverse_x=not args.no_reverse_x,
        show_raw_points=not args.no_raw_points,
        derivative_panel=not args.no_derivative_panel,
        same_aspect_diagnostics=args.same_aspect_diagnostics,
        fbrm_count_unit=args.fbrm_count_unit,
        mono=False,
        clean_mono=False,
        mgi_source=mgi_source,
        mgi_pchip=mgi_pchip,
        fbrm_curve_main=fbrm_curve_main,
        fbrm_curve_deriv=fbrm_curve_deriv,
        fbrm_smooth_y_main=fbrm_smooth_y_main,
        fbrm_smooth_y_deriv=fbrm_smooth_y_deriv,
        mgi_onsets=mgi_onsets,
        fbrm_onsets=fbrm_onsets,
        view_limits=view_limits,
    )

    # Also write a main-panel-only manuscript figure.
    if not args.no_derivative_panel:
        make_plot(
            output_png=output_png_main,
            title="MGI and FBRM comparison" if args.show_title else None,
            reverse_x=not args.no_reverse_x,
            show_raw_points=not args.no_raw_points,
            derivative_panel=False,
            fbrm_count_unit=args.fbrm_count_unit,
            mono=False,
            clean_mono=False,
            mgi_source=mgi_source,
            mgi_pchip=mgi_pchip,
            fbrm_curve_main=fbrm_curve_main,
            fbrm_curve_deriv=fbrm_curve_deriv,
            fbrm_smooth_y_main=fbrm_smooth_y_main,
            fbrm_smooth_y_deriv=fbrm_smooth_y_deriv,
            mgi_onsets=mgi_onsets,
            fbrm_onsets=fbrm_onsets,
            view_limits=view_limits,
        )

    # Also write a monochrome main-panel-only figure for grayscale publication use.
    make_plot(
        output_png=output_png_main_mono,
        title="MGI and FBRM comparison" if args.show_title else None,
        reverse_x=not args.no_reverse_x,
        show_raw_points=not args.no_raw_points,
        derivative_panel=False,
        fbrm_count_unit=args.fbrm_count_unit,
        mono=True,
        clean_mono=False,
        mgi_source=mgi_source,
        mgi_pchip=mgi_pchip,
        fbrm_curve_main=fbrm_curve_main,
        fbrm_curve_deriv=fbrm_curve_deriv,
        fbrm_smooth_y_main=fbrm_smooth_y_main,
        fbrm_smooth_y_deriv=fbrm_smooth_y_deriv,
        mgi_onsets=mgi_onsets,
        fbrm_onsets=fbrm_onsets,
        view_limits=view_limits,
    )

    # Also write a clean monochrome main-panel-only figure without raw points.
    make_plot(
        output_png=output_png_main_clean_mono,
        title="MGI and FBRM comparison" if args.show_title else None,
        reverse_x=not args.no_reverse_x,
        show_raw_points=False,
        derivative_panel=False,
        fbrm_count_unit=args.fbrm_count_unit,
        mono=True,
        clean_mono=True,
        mgi_source=mgi_source,
        mgi_pchip=mgi_pchip,
        fbrm_curve_main=fbrm_curve_main,
        fbrm_curve_deriv=fbrm_curve_deriv,
        fbrm_smooth_y_main=fbrm_smooth_y_main,
        fbrm_smooth_y_deriv=fbrm_smooth_y_deriv,
        mgi_onsets=mgi_onsets,
        fbrm_onsets=fbrm_onsets,
        view_limits=view_limits,
    )

    write_combo_data(
        output_csv,
        mgi_pchip,
        fbrm_curve_main,
        fbrm_curve_deriv,
        fbrm_smooth_y_main,
        fbrm_smooth_y_deriv,
    )

    outputs = {
        "png": str(output_png),
        "png_main_only": str(output_png_main),
        "png_main_only_mono": str(output_png_main_mono),
        "png_main_only_clean_mono": str(output_png_main_clean_mono),
        "data_csv": str(output_csv),
        "params_json": str(output_json),
        "report_txt": str(output_report),
    }


    write_view_limits_json(
        outdir / "view-limits.json",
        limits=view_limits,
        content_type="combo",
        output_mode="limited-view" if view_limits.any else "base",
        extra={
            "mgi_y_axis": "left primary signal axis",
            "fbrm_y_axis": "right primary signal axis",
            "derivative_y_axis_manual_limits_applied": False,
        },
    )

    write_params(
        output_json,
        args=args,
        inputs=inputs,
        fbrm_smoothing=fbrm_smoothing,
        fbrm_analysis_window=fbrm_analysis_window,
        mgi_onsets=mgi_onsets,
        fbrm_onsets=fbrm_onsets,
        outputs=outputs,
    )
    write_report(
        output_report,
        inputs=inputs,
        fbrm_smoothing=fbrm_smoothing,
        fbrm_analysis_window=fbrm_analysis_window,
        mgi_onsets=mgi_onsets,
        fbrm_onsets=fbrm_onsets,
        outputs=outputs,
    )

    print(f"[ok] wrote {output_png}")
    if output_png_main.exists():
        print(f"[ok] wrote {output_png_main}")
    if output_png_main_mono.exists():
        print(f"[ok] wrote {output_png_main_mono}")
    if output_png_main_clean_mono.exists():
        print(f"[ok] wrote {output_png_main_clean_mono}")
    print(f"[ok] wrote {output_csv}")
    print(f"[ok] wrote {output_json}")
    print(f"[ok] wrote {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
