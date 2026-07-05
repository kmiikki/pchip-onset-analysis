#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onset_view_limits.py
--------------------

Shared display-only axis-limit helpers for MGI/PCHIP, FBRM and MGI-FBRM
combo onset figures.

These helpers deliberately separate figure view limits from analysis data
selection.  They must not be used for onset detection, candidate detection,
smoothing, interpolation or CSV result generation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from matplotlib.ticker import MultipleLocator, NullLocator


@dataclass(frozen=True)
class AxisRange:
    """Optional numeric lower/upper display bounds for one axis."""

    lower: float | None = None
    upper: float | None = None

    @property
    def any(self) -> bool:
        return self.lower is not None or self.upper is not None

    @property
    def complete(self) -> bool:
        return self.lower is not None and self.upper is not None

    def label(self) -> str:
        """Return directory-name component using lower-upper positions."""
        if not self.any:
            return "auto"
        return f"{format_limit_value(self.lower)}-{format_limit_value(self.upper)}"

    def as_dict(self) -> dict[str, float | None]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class ViewLimits:
    """All display-only view limits for an onset figure set."""

    x: AxisRange
    mgi_y: AxisRange
    fbrm_y: AxisRange

    @property
    def any(self) -> bool:
        return self.x.any or self.mgi_y.any or self.fbrm_y.any

    @property
    def view_id(self) -> str:
        return view_id_from_limits(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.as_dict(),
            "mgi_y": self.mgi_y.as_dict(),
            "fbrm_y": self.fbrm_y.as_dict(),
        }


NONE_LIMITS = ViewLimits(x=AxisRange(), mgi_y=AxisRange(), fbrm_y=AxisRange())


def add_view_limit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared display-only view-limit arguments to an ArgumentParser."""
    group = parser.add_argument_group(
        "display-only axis/view limits",
        description=(
            "Axis limits affect only generated figures/view outputs. They do not "
            "change onset detection, candidate detection, smoothing, interpolation "
            "or analysis CSV values."
        ),
    )

    group.add_argument("--x-range", nargs=2, type=float, metavar=("LOWER", "UPPER"), help="Numeric x/temperature display range. Values are lower upper, not plot left right.")
    group.add_argument("--x-lower", type=float, default=None, help="Numeric lower x/temperature display bound. The upper bound remains automatic if not given.")
    group.add_argument("--x-upper", type=float, default=None, help="Numeric upper x/temperature display bound. The lower bound remains automatic if not given.")

    group.add_argument("--mgi-y-range", nargs=2, type=float, metavar=("LOWER", "UPPER"), help="MGI signal y-axis display range for MGI and combo primary MGI axes.")
    group.add_argument("--mgi-y-lower", type=float, default=None, help="MGI signal y-axis lower display bound.")
    group.add_argument("--mgi-y-upper", type=float, default=None, help="MGI signal y-axis upper display bound.")

    group.add_argument("--fbrm-y-range", nargs=2, type=float, metavar=("LOWER", "UPPER"), help="FBRM Total Counts y-axis display range for FBRM and combo primary FBRM axes.")
    group.add_argument("--fbrm-y-lower", type=float, default=None, help="FBRM Total Counts y-axis lower display bound.")
    group.add_argument("--fbrm-y-upper", type=float, default=None, help="FBRM Total Counts y-axis upper display bound.")

    group.add_argument("--view-output-dir", type=Path, default=None, help="Explicit output directory for an axis-limited view run. If omitted, limited plots are written under <base-output>/views/<view-id>.")


def _range_from_args(range_pair: Sequence[float] | None, lower: float | None, upper: float | None, name: str) -> AxisRange:
    if range_pair is not None:
        if lower is not None or upper is not None:
            raise ValueError(f"Use either --{name}-range or --{name}-lower/--{name}-upper, not both.")
        lo = float(range_pair[0])
        hi = float(range_pair[1])
    else:
        lo = None if lower is None else float(lower)
        hi = None if upper is None else float(upper)

    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"--{name} lower bound must be <= upper bound: {lo:g} > {hi:g}")

    return AxisRange(lower=lo, upper=hi)


def parse_view_limits_from_args(args: argparse.Namespace) -> ViewLimits:
    """Parse shared view-limit CLI arguments from a Namespace."""
    return ViewLimits(
        x=_range_from_args(getattr(args, "x_range", None), getattr(args, "x_lower", None), getattr(args, "x_upper", None), "x"),
        mgi_y=_range_from_args(getattr(args, "mgi_y_range", None), getattr(args, "mgi_y_lower", None), getattr(args, "mgi_y_upper", None), "mgi-y"),
        fbrm_y=_range_from_args(getattr(args, "fbrm_y_range", None), getattr(args, "fbrm_y_lower", None), getattr(args, "fbrm_y_upper", None), "fbrm-y"),
    )


def format_limit_value(value: float | None) -> str:
    """Format a limit value for a readable directory-name component."""
    if value is None:
        return "auto"
    if not math.isfinite(float(value)):
        raise ValueError(f"Axis limit must be finite, got {value!r}")
    text = f"{float(value):.10g}"
    # Keep ordinary decimal points. They are valid and clearer than 0p20.
    return text.replace("-", "minus")


def view_id_from_limits(limits: ViewLimits) -> str:
    """Return canonical project view identifier for the given limits."""
    return f"x_{limits.x.label()}__mgi_{limits.mgi_y.label()}__fbrm_{limits.fbrm_y.label()}"


def default_view_output_dir(base_output_dir: Path, limits: ViewLimits) -> Path:
    """Return default output directory for a limited view run."""
    return base_output_dir / "views" / limits.view_id


def resolve_output_dir_for_limits(base_output_dir: Path, args: argparse.Namespace, limits: ViewLimits) -> Path:
    """Return output directory for a base or limited-view run."""
    if not limits.any:
        return base_output_dir
    explicit = getattr(args, "view_output_dir", None)
    if explicit is not None:
        return Path(explicit)
    return default_view_output_dir(base_output_dir, limits)


def finite_values(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def outward_limits(values: np.ndarray | Sequence[float], *, major_step: float = 5.0, snap_tolerance: float = 1.0) -> tuple[float, float]:
    """Return readable outward numeric lower/upper limits from finite values."""
    x = finite_values(values)
    if x.size == 0:
        raise ValueError("Cannot determine automatic axis limits from empty data.")
    data_lo = float(np.min(x))
    data_hi = float(np.max(x))
    lower_grid = float(np.floor(data_lo / major_step) * major_step)
    upper_grid = float(np.ceil(data_hi / major_step) * major_step)
    lo = lower_grid if (data_lo - lower_grid) <= snap_tolerance else data_lo
    hi = upper_grid if (upper_grid - data_hi) <= snap_tolerance else data_hi
    return lo, hi


def resolved_numeric_range(
    source_values: np.ndarray | Sequence[float],
    manual: AxisRange,
    *,
    major_step: float = 5.0,
    snap_tolerance: float = 1.0,
    outward: bool = True,
) -> tuple[float, float]:
    """Resolve optional manual lower/upper bounds against automatic data limits."""
    if outward:
        auto_lo, auto_hi = outward_limits(source_values, major_step=major_step, snap_tolerance=snap_tolerance)
    else:
        finite = finite_values(source_values)
        if finite.size == 0:
            raise ValueError("Cannot determine automatic axis limits from empty data.")
        auto_lo = float(np.min(finite))
        auto_hi = float(np.max(finite))
    lo = auto_lo if manual.lower is None else float(manual.lower)
    hi = auto_hi if manual.upper is None else float(manual.upper)
    if lo > hi:
        raise ValueError(f"Resolved axis lower bound is greater than upper bound: {lo:g} > {hi:g}")
    return lo, hi


def apply_temperature_view_limits(
    ax,
    source_x: np.ndarray | Sequence[float],
    limits: ViewLimits,
    *,
    reverse_x: bool,
    major_step_C: float = 5.0,
    snap_tolerance_C: float = 1.0,
) -> tuple[float, float]:
    """Apply x/temperature view limits to one matplotlib axis."""
    x_lo, x_hi = resolved_numeric_range(
        source_x,
        limits.x,
        major_step=major_step_C,
        snap_tolerance=snap_tolerance_C,
        outward=True,
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
    return x_lo, x_hi


def visible_x_mask(x: np.ndarray | Sequence[float], x_limits: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    lo, hi = x_limits
    return np.isfinite(arr) & (arr >= min(lo, hi)) & (arr <= max(lo, hi))


def _visible_y_values(
    series: Iterable[tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]]],
    x_limits: tuple[float, float],
) -> np.ndarray:
    collected: list[np.ndarray] = []
    for x, y in series:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if xa.shape != ya.shape:
            continue
        mask = visible_x_mask(xa, x_limits) & np.isfinite(ya)
        if np.any(mask):
            collected.append(ya[mask])
    if not collected:
        return np.array([], dtype=float)
    return np.concatenate(collected)


def apply_signal_y_view_limits(
    ax,
    series: Iterable[tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]]],
    x_limits: tuple[float, float],
    y_limits: AxisRange,
    *,
    pad_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """Apply y-limits to a primary signal axis.

    If y_limits are automatic, autoscale from y values visible inside x_limits.
    This is intentionally *not* used for derivative axes.
    """
    visible_y = _visible_y_values(series, x_limits)
    if visible_y.size == 0:
        print("[warning] selected x-limits contain no data for y autoscaling; keeping matplotlib y-limits.")
        return None

    data_lo = float(np.min(visible_y))
    data_hi = float(np.max(visible_y))
    if data_lo == data_hi:
        pad = max(abs(data_lo) * pad_fraction, 1.0)
    else:
        pad = (data_hi - data_lo) * pad_fraction

    auto_lo = data_lo - pad
    auto_hi = data_hi + pad
    lo = auto_lo if y_limits.lower is None else float(y_limits.lower)
    hi = auto_hi if y_limits.upper is None else float(y_limits.upper)
    if lo > hi:
        raise ValueError(f"Resolved y-axis lower bound is greater than upper bound: {lo:g} > {hi:g}")
    ax.set_ylim(lo, hi)
    return lo, hi


def write_view_limits_json(
    path: Path,
    *,
    limits: ViewLimits,
    content_type: str,
    output_mode: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write metadata describing a base or axis-limited view output."""
    payload: dict[str, Any] = {
        "view_type": "axis-limited" if limits.any else "base",
        "view_id": limits.view_id if limits.any else None,
        "content_type": content_type,
        "output_mode": output_mode,
        "limits": limits.as_dict(),
        "directory_syntax": "x_<lower>-<upper>__mgi_<lower>-<upper>__fbrm_<lower>-<upper>; auto means that bound was not manually set",
        "semantics": {
            "axis_limits_are_display_only": True,
            "analysis_changed": False,
            "onset_detection_changed": False,
            "candidate_detection_changed": False,
            "smoothing_or_interpolation_changed": False,
            "csv_result_values_changed": False,
            "manual_y_limits_apply_to_derivative_panels": False,
        },
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
