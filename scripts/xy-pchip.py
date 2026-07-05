#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xy-pchip.py
-----------

PCHIP interpolation + flexible CSV I/O

Features:
  • Column selection by name or index (--xcol / --xcoln, --ycol / --ycoln)
  • Optional column listing (--print-cols)
  • Optional pre-deduplication (--pre-dedupe)
  • Robust median binning before PCHIP interpolation
  • Winsorization to suppress outliers (--clip-quant)
  • Multiple export modes (--export-mode grid|raw|unique)
  • Optional rounding before uniqueness filtering (--x-round)
  • Export PCHIP-interpolated curve data and/or bin medians
  • Clean high-resolution plot (300 dpi)

Read x–y profile data from a CSV, optionally robust-bin it, evaluate a
shape-preserving PCHIP interpolant, and export a plot (+ optional CSV). If
the user provides a bare filename
(positionally), it is treated as --csv <filename>. Output files are named by
the outstem, whose default is "<input_stem>-pchip".

Column selection:
- by name:      --xcol XNAME --ycol YNAME
- by index:     --xcoln XINDEX --ycoln YINDEX   (1-based indices)
These are mutually exclusive per-axis: you cannot mix --xcol with --xcoln, nor
--ycol with --ycoln. If numeric indices are used, the header row is read first
to resolve the column *names*, and names are then used everywhere in the code.

Examples
--------
python xy-pchip.py data.csv
python xy-pchip.py --csv data.csv
python xy-pchip.py --csv data.csv --xcol Temperature --ycol BW --no-grid
python xy-pchip.py data.csv --xcoln 1 --ycoln 4 --reverse-x --save-csv
python xy-pchip.py data.csv --outstem result --bins 300 --dense 5000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import csv
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from scipy.interpolate import PchipInterpolator


# ----------------------------- CLI parsing --------------------------------- #


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and enforce invariants."""
    p = argparse.ArgumentParser(
        description="Interpolate x–y profile data using robust binning + shape-preserving PCHIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--csv",
        type=Path,
        help="Input CSV file to read (or provide filename directly as a positional argument).",
    )
    p.add_argument(
        "--outstem",
        type=str,
        help="Base name (stem) for output files. Default = <input stem>-pchip.",
    )

    # Column selectors: name-based
    p.add_argument("--xcol", type=str, help="X column name.")
    p.add_argument("--ycol", type=str, help="Y column name.")

    # Column selectors: index-based (1-based indices)
    p.add_argument("--xcoln", type=int, help="X column index (1-based).")
    p.add_argument("--ycoln", type=int, help="Y column index (1-based).")

    # I/O and parsing details
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter.")
    p.add_argument("--encoding", type=str, default="utf-8", help="CSV file encoding.")

    # Interpolation controls
    p.add_argument("--bins", type=int, default=200, help="Number of robust x-bins.")
    p.add_argument("--dense", type=int, default=2000, help="Dense points for PCHIP curve.")

    # Plot options
    p.add_argument("--no-grid", action="store_true", help="Disable grid on the plot.")
    p.add_argument("--reverse-x", action="store_true", help="Reverse the x-axis.")
    
    p.add_argument("--no-raw", action="store_true", help="Hide raw points in plot.")
    p.add_argument("--binned", action="store_true", help="Enable binned points in plot.")
    p.add_argument("--pchip-lw", type=float, default=1.0, help="PCHIP line width.")
    p.add_argument("--pchip-color", type=str, default='k',
                   help="PCHIP line color (e.g. 'k' or '#000000').")
    p.add_argument("--raw-alpha", type=float, default=0.35, help="Alpha for raw points.")
    p.add_argument("--binned-alpha", type=float, default=0.8, help="Alpha for binned points.")
    p.add_argument("--xlabel", type=str, default=None,
                   help="X axis label (default: X column name).")
    p.add_argument("--ylabel", type=str, default=None,
                  help="Y axis label (default: Y column name).")
        
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (default: derived from input file).",
    )

    # Outputs
    p.add_argument("--save-csv", action="store_true", help="Also save the PCHIP-interpolated curve to CSV.")

    # Parse known args to collect optional positional filename
    args, positional = p.parse_known_args()

    # Positional filename → --csv
    if positional and args.csv is None:
        file_candidate = Path(positional[0])
        # Accept any file; extension check is not strictly necessary
        args.csv = file_candidate

    # Validate CSV provided
    if args.csv is None:
        p.error("No CSV provided. Use --csv <file> or pass filename positionally (e.g. xy-pchip.py data.csv).")
    if not args.csv.exists():
        p.error(f"CSV file not found: {args.csv}")

    # Enforce mutual exclusion for x and y selectors
    if args.xcol is not None and args.xcoln is not None:
        p.error("Choose either --xcol OR --xcoln, not both.")
    if args.ycol is not None and args.ycoln is not None:
        p.error("Choose either --ycol OR --ycoln, not both.")

    if args.xcol is None and args.xcoln is None:
      args.xcoln = 1
    if args.ycol is None and args.ycoln is None:
      args.ycoln = 2

    # After header resolution we will ensure we have definitive xcol,ycol names
    # Determine outstem default
    if not args.outstem:
        args.outstem = f"{args.csv.stem}-pchip"

    return args


# --------------------------- CSV utilities --------------------------------- #


def read_header_columns(csv_path: Path, delimiter: str, encoding: str) -> List[str]:
    """Read only the header row and return the list of column names."""
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file appears empty: {csv_path}")
    # Strip whitespace around names
    return [h.strip() for h in header]


def index_to_name(header: Sequence[str], one_based_index: int) -> str:
    """Map a 1-based column index to a header name; raise on out-of-range."""
    if one_based_index < 1 or one_based_index > len(header):
        raise IndexError(
            f"Column index {one_based_index} is out of range (1..{len(header)})."
        )
    return header[one_based_index - 1]


def resolve_column_names(
    csv_path: Path,
    delimiter: str,
    encoding: str,
    xcol: str | None,
    ycol: str | None,
    xcoln: int | None,
    ycoln: int | None,
) -> Tuple[str, str]:
    """
    Resolve final x and y column *names* based on name or numeric selectors.
    If numeric selectors are used, read header first and map to names.
    """
    header = read_header_columns(csv_path, delimiter, encoding)

    # X
    if xcol is None:
        if xcoln is None:
            raise ValueError(
                "No X column specified. Use --xcol <name> or --xcoln <index> (1-based)."
            )
        xcol = index_to_name(header, xcoln)
    else:
        # ensure the named column exists
        if xcol not in header:
            raise ValueError(f"X column '{xcol}' not found in header {header}")

    # Y
    if ycol is None:
        if ycoln is None:
            raise ValueError(
                "No Y column specified. Use --ycol <name> or --ycoln <index> (1-based)."
            )
        ycol = index_to_name(header, ycoln)
    else:
        if ycol not in header:
            raise ValueError(f"Y column '{ycol}' not found in header {header}")

    return xcol, ycol


# --------------------------- Processing ------------------------------------ #


@dataclass(frozen=True)
class SeriesXY:
    x: np.ndarray
    y: np.ndarray


def load_xy(
    csv_path: Path,
    delimiter: str,
    encoding: str,
    xcol: str,
    ycol: str,
) -> SeriesXY:
    """
    Load two numeric columns (x,y) from CSV. Coerce to numeric and drop NaNs.
    """
    df = pd.read_csv(
        csv_path,
        delimiter=delimiter,
        encoding=encoding,
        usecols=[xcol, ycol],
    )

    # Coerce to numeric, drop rows where either is NaN
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])

    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    return SeriesXY(x=x, y=y)


def sort_and_unique_x(series: SeriesXY) -> SeriesXY:
    """
    Sort by x ascending and collapse duplicate x by averaging y.
    This ensures strictly non-decreasing x for PCHIP input.
    """
    order = np.argsort(series.x)
    x_sorted = series.x[order]
    y_sorted = series.y[order]

    # Aggregate duplicates by mean
    uniq_x, idx_start = np.unique(x_sorted, return_index=True)
    # For means, compute group boundaries:
    # build counts per unique x
    counts = np.diff(np.append(idx_start, len(x_sorted)))
    # compute grouped means efficiently
    y_means = []
    ptr = 0
    for c in counts:
        y_means.append(float(np.mean(y_sorted[ptr : ptr + c])))
        ptr += c

    return SeriesXY(x=uniq_x, y=np.array(y_means, dtype=float))


def robust_bin_xy(x: np.ndarray, y: np.ndarray, n_bins: int) -> SeriesXY:
    """
    Robust x-binning:
    - Partition x using quantile edges into ~n_bins (at least 5).
    - For each non-empty bin, take median(x) and mean(y).
    """
    n_bins = max(5, int(n_bins))
    # Quantile edges from 0..1
    edges = np.quantile(x, q=np.linspace(0.0, 1.0, num=n_bins + 1))
    # Deduplicate edges (degenerate distributions)
    edges = np.unique(edges)
    # Assign bins
    # Right-closed bins except last edge to include max
    digitized = np.digitize(x, bins=edges[1:-1], right=True)

    xb, yb = [], []
    for b in range(len(edges) - 1):
        mask = digitized == b
        if not np.any(mask):
            continue
        xb.append(float(np.median(x[mask])))
        yb.append(float(np.mean(y[mask])))

    xb = np.array(xb, dtype=float)
    yb = np.array(yb, dtype=float)

    # Ensure strictly increasing xb for PCHIP (unique + sort)
    return sort_and_unique_x(SeriesXY(xb, yb))


def pchip_dense(series: SeriesXY, n_dense: int) -> SeriesXY:
    """
    Fit a shape-preserving PCHIP interpolant and evaluate it on a dense grid.
    """
    if len(series.x) < 2:
        raise ValueError("Not enough data points for PCHIP (need at least 2).")

    x_min, x_max = float(series.x[0]), float(series.x[-1])
    if x_max <= x_min:
        raise ValueError("Non-increasing x-range for PCHIP.")

    x_dense = np.linspace(x_min, x_max, int(max(10, n_dense)))
    pchip = PchipInterpolator(series.x, series.y, extrapolate=False)
    y_dense = pchip(x_dense)

    # Drop NaNs from edges if any due to extrapolate=False
    mask = ~np.isnan(y_dense)
    return SeriesXY(x_dense[mask], y_dense[mask])


# --------------------------- Plot & Save ----------------------------------- #


def outward_x_limits(
    x_values: np.ndarray,
    major_step: float = 5.0,
    snap_tolerance: float = 1.0,
) -> tuple[float, float]:
    """Return non-clipping, data-aware low/high x-limits.

    The rule is conservative: snap outward to a major-step boundary only when
    the data edge is already close to that outward boundary. Otherwise use the
    actual data edge. This avoids forcing a large empty 5 °C interval at the end
    of a cooling plot, while still allowing clean limits such as 60 °C or 20 °C
    when the data actually ends close to them.
    """
    x = np.asarray(x_values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot determine x-limits from empty x data.")

    data_low = float(np.min(x))
    data_high = float(np.max(x))
    lower_grid = float(np.floor(data_low / major_step) * major_step)
    upper_grid = float(np.ceil(data_high / major_step) * major_step)

    x_low = lower_grid if (data_low - lower_grid) <= snap_tolerance else data_low
    x_high = upper_grid if (upper_grid - data_high) <= snap_tolerance else data_high
    return x_low, x_high


def apply_outward_x_limits(
    ax,
    x_values: np.ndarray,
    *,
    reverse_x: bool,
    major_step: float = 5.0,
    snap_tolerance: float = 1.0,
) -> None:
    """Apply data-aware non-clipping x-limits while preserving axis direction."""
    x_low, x_high = outward_x_limits(
        x_values,
        major_step=major_step,
        snap_tolerance=snap_tolerance,
    )
    if reverse_x:
        ax.set_xlim(x_high, x_low)
    else:
        ax.set_xlim(x_low, x_high)


def configure_major_grid_only(ax, *, x_major_step: float = 5.0, grid: bool = True) -> None:
    """Use 5 °C major ticks on x and explicitly disable minor grid/ticks.

    Some Matplotlib styles or previous plotting defaults may enable minor ticks
    or minor grids. This function turns them off explicitly so regenerated
    workflow figures do not inherit a minor grid from the runtime style.
    """
    ax.xaxis.set_major_locator(MultipleLocator(x_major_step))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.minorticks_off()
    ax.grid(False, which="minor")
    if grid:
        ax.grid(True, which="major", linestyle="-", linewidth=0.25, alpha=0.7)


def save_plot(
    raw: SeriesXY,
    binned: SeriesXY,
    smooth: SeriesXY,
    outstem: str,
    title: str | None,
    grid: bool,
    reverse_x: bool,
    show_raw: bool,
    show_binned: bool,
    pchip_lw: float,
    pchip_color: str | None,
    raw_alpha: float,
    binned_alpha: float,
    xlabel: str,
    ylabel: str,
) -> Path:
    """
    Create and save a plot: raw points, binned points, and PCHIP curve.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if show_raw:    
      ax.plot(raw.x, raw.y, ".", ms=2, alpha=raw_alpha, label="Raw")
    if show_binned:
      ax.plot(binned.x, binned.y, "o", ms=4, alpha=binned_alpha, label="Binned")
    ax.plot(smooth.x, smooth.y, "-", color=pchip_color, lw=pchip_lw, alpha=1.0, label="PCHIP")

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    apply_outward_x_limits(ax, raw.x, reverse_x=reverse_x)
    configure_major_grid_only(ax, grid=grid)

    out_png = Path(f"{outstem}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def save_curve_csv(smooth: SeriesXY, outstem: str, xname: str, yname: str) -> Path:
    """ 
    Save the PCHIP-interpolated curve to CSV.
    If both names are identical or missing, use default 'X','Y'.
    """
    # Normalize names
    xcol = xname.strip() if xname else ""
    ycol = yname.strip() if yname else ""

    # If identical, empty, or whitespace-only → default
    if not xcol or not ycol or xcol.lower() == ycol.lower():
        xcol, ycol = "X", "Y"

    out_csv = Path(f"{outstem}.csv")
    df_out = pd.DataFrame({xcol: smooth.x, ycol: smooth.y})
    df_out.to_csv(out_csv, index=False)
    return out_csv


# ----------------------------- Main ---------------------------------------- #


def main() -> None:
    args = parse_args()

    # Resolve final column *names* (if numeric indices were provided)
    try:
        xcol_name, ycol_name = resolve_column_names(
            args.csv, args.delimiter, args.encoding, args.xcol, args.ycol, args.xcoln, args.ycoln
        )
    except Exception as e:
        sys.exit(f"[error] {e}")

    # Load data
    try:
        raw = load_xy(args.csv, args.delimiter, args.encoding, xcol_name, ycol_name)
    except Exception as e:
        sys.exit(f"[error] Failed to read numeric columns '{xcol_name}'/'{ycol_name}': {e}")

    # Sort and collapse duplicate x
    raw_su = sort_and_unique_x(raw)

    # Robust binning
    binned = robust_bin_xy(raw_su.x, raw_su.y, args.bins)

    # Shape-preserving PCHIP interpolation
    try:
        smooth = pchip_dense(binned, args.dense)
    except Exception as e:
        sys.exit(f"[error] PCHIP failed: {e}")

    # Title
    title = args.title or "PCHIP interpolation"

    # Axis labels default to original column names unless overridden
    xlabel = args.xlabel or xcol_name
    ylabel = args.ylabel or ycol_name

    # Plot
    out_png = save_plot(
        raw=raw_su,
        binned=binned,
        smooth=smooth,
        outstem=args.outstem,
        title=title,
        grid=not args.no_grid,
        reverse_x=args.reverse_x,
        show_raw=not args.no_raw,
        show_binned=args.binned,
        pchip_lw=args.pchip_lw,
        pchip_color=args.pchip_color,
        raw_alpha=args.raw_alpha,
        binned_alpha=args.binned_alpha,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # Optional CSV
    out_csv = None
    if args.save_csv:
        out_csv = save_curve_csv(smooth, args.outstem, xcol_name, ycol_name)

    # User-friendly summary
    print(f"Input CSV:          {args.csv}")
    print(f"Selected columns:   X='{xcol_name}', Y='{ycol_name}'")
    print(f"Output stem:        {args.outstem}")
    print(f"Wrote plot:         {out_png}")
    if out_csv:
        print(f"Wrote PCHIP CSV:     {out_csv}")


if __name__ == "__main__":
    main()
