#!/usr/bin/env python3
"""Render saved results only; never import or run onset detectors.

Examples (paths are explicit; manuscript mappings are not inferred)::

    python scripts/render-publication-figures.py mgi --branch raw --curve CURVE.csv \
        --bends BENDS.csv --points hide --view full --name mgi-review
    python scripts/render-publication-figures.py combo --data COMBO-DATA.csv \
        --view limited --x-range 45 49 --name combo-review

Outputs are exclusive-created beneath generated/ or data analysis view directories
unless --force is explicit.
FBRM reconstructs the production analysis prefix with verified saved-parameter replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
import re
import struct
import subprocess
import zlib
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from onset_view_limits import apply_publication_temperature_ticks
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "generated"
ANALYSIS_ROOT = ROOT / "data"
LW_SCALE = 0.50
OUTPUT_DPI = {"color": 300, "gray": 600, "bw": 1200}
TEMPERATURE_LABEL = "Tr [°C]"
PUBLICATION_RC = {
    "font.size": 8, "font.family": "DejaVu Sans", "font.weight": "normal",
    "axes.labelsize": 8, "axes.labelweight": "normal",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.marker": "None", "figure.autolayout": False,
    "figure.constrained_layout.use": False, "savefig.bbox": None,
    # Preserve the saved polyline, including nonmonotonic temperature loops.
    "path.simplify": False,
}


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SavedResult:
    mgi: Curve | None
    fbrm: Curve | None
    measurements: Curve | None
    onsets: tuple[dict, ...]
    sources: tuple[dict, ...]
    replay: dict | None = None


def read_rows(path: Path, required: tuple[str, ...]) -> tuple[list[dict], dict]:
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not set(required).issubset(reader.fieldnames or []):
        raise ValueError(f"{path}: required columns {required}")
    return list(reader), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def curve_from_rows(rows: list[dict], xcol: str, ycol: str) -> Curve:
    # Reject invalid samples instead of silently changing scientific selection.
    values = np.array([(float(r[xcol]), float(r[ycol])) for r in rows], dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Curve requires at least two finite saved samples")
    x, y = values[:, 0].copy(), values[:, 1].copy()
    x.flags.writeable = y.flags.writeable = False
    return Curve(x, y)


def read_onsets(path: Path, family: str, curve_name: str | None = None,
                branch: str | None = None) -> tuple[tuple[dict, ...], dict]:
    columns = (("bend_index", "temperature_C", "value") if family == "MGI"
               else ("status", "onset_label", "Tr_C"))
    rows, source = read_rows(path, columns)
    accepted = []
    for row in rows:
        if family == "FBRM" and row["status"] != "accepted":
            continue
        if family == "MGI":
            if curve_name and row.get("input_csv") and Path(row["input_csv"]).name != curve_name:
                raise ValueError("Bends file does not belong to the selected curve")
            if branch and row.get("method") and row["method"] != f"{branch}_pchip":
                raise ValueError("Bends method conflicts with explicit RAW/SG branch")
            rank = int(row["bend_index"])
            if rank < 1:
                raise ValueError("Invalid saved onset rank")
            label, temp, value = f"T{rank}", float(row["temperature_C"]), float(row["value"])
        else:
            label, temp, value = row["onset_label"], float(row["Tr_C"]), None
        if not math.isfinite(temp) or (value is not None and not math.isfinite(value)):
            raise ValueError("Nonfinite accepted onset")
        if any(o["label"] == label for o in accepted):
            raise ValueError("Ambiguous repeated onset label; select a single result file")
        accepted.append({"family": family, "label": label, "temperature_C": temp, "value": value})
    return tuple(accepted), source


def load_mgi(args: argparse.Namespace) -> SavedResult:
    rows, source = read_rows(args.curve, (args.xcol, args.ycol))
    curve = curve_from_rows(rows, args.xcol, args.ycol)
    onsets, bends_source = read_onsets(args.bends, "MGI", args.curve.name, args.branch)
    sources = [source, bends_source]
    measurements = None
    if args.points == "show" or args.acquisition_extent:
        if args.measurements is None or args.measurement_ycol is None:
            raise ValueError("Measurement display/extent requires --measurements and --measurement-ycol")
        rows, measurement_source = read_rows(args.measurements, (args.xcol, args.measurement_ycol))
        measurements = curve_from_rows(rows, args.xcol, args.measurement_ycol)
        sources.append(measurement_source)
    elif args.measurements is not None or args.measurement_ycol is not None:
        raise ValueError("Measurements may only be supplied for points or acquisition extent")
    return SavedResult(curve, None, measurements, onsets, tuple(sources))


def load_combo(args: argparse.Namespace) -> SavedResult:
    rows, source = read_rows(args.data, ("series", "kind", "Tr_C", "value"))
    mgi = curve_from_rows([r for r in rows if (r["series"], r["kind"]) == ("MGI", "curve")], "Tr_C", "value")
    fbrm = curve_from_rows([r for r in rows if (r["series"], r["kind"]) ==
                            ("FBRM", "smoothed_curve_main_panel")], "Tr_C", "value")
    sources, onsets = [source], []
    for path, family in ((args.mgi_bends, "MGI"), (args.fbrm_onsets, "FBRM")):
        if path is not None:
            accepted, onset_source = read_onsets(path, family)
            onsets.extend(accepted)
            sources.append(onset_source)
    return SavedResult(mgi, fbrm, None, tuple(onsets), tuple(sources))


def load_fbrm(args: argparse.Namespace) -> SavedResult:
    from saved_fbrm_replay import Sources, production_smoothing, replay_fbrm
    sources = Sources()
    paths = dict(fbrm_input=args.data, fbrm_result=args.onsets_file,
                 fbrm_candidates=args.candidates_file, fbrm_params=args.params_file)
    data = replay_fbrm(sources, paths, *production_smoothing(sources))
    n = data["prefix_n"]
    curve = Curve(data["x"][:n], data["y"][:n])
    measurements = Curve(data["x"][:n], data["raw"][:n]) if args.points == "show" else None
    onsets = tuple(dict(family="FBRM", label=o["label"], temperature_C=o["x"], value=o["y"])
                   for o in data["onsets"])
    sources.add(Path(__file__).with_name("saved_fbrm_replay.py"))
    sources.verify()
    return SavedResult(None, curve, measurements, onsets,
                       tuple(dict(path=p, sha256=h) for p, h in sources.hashes().items()),
                       data["replay"] | {"rendered_region": "saved analysis prefix"})


def place_onset_text(fig, ax, onsets: tuple[dict, ...], markers: bool,
                     family_prefix: bool = False) -> str | None:
    if not onsets:
        return None
    text = "\n".join((f"{o['family']} " if family_prefix else "") +
                     f"{o['label']} = {o['temperature_C']:.2f} °C" for o in onsets)
    artist = ax.text(.97, .03, text, transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    # Prefer lower-right; reject corners overlapping marker lines or curve samples.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [("lower-right", .97, .03, "right", "bottom"),
                  ("lower-left", .03, .03, "left", "bottom"),
                  ("upper-right", .97, .97, "right", "top"),
                  ("upper-left", .03, .97, "left", "top")]
    scores = []
    for name, x, y, ha, va in candidates:
        artist.set_position((x, y)); artist.set_ha(ha); artist.set_va(va)
        box = artist.get_window_extent(renderer).padded(5 * fig.dpi / 72)
        score = 0
        if markers:
            score += 100000 * sum(box.x0 <= ax.transData.transform((o["temperature_C"], 0))[0] <= box.x1
                                  for o in onsets)
        for axis in fig.axes:
            for line in axis.lines:
                if line.get_linestyle() != "-":
                    continue
                pts = line.get_transform().transform(line.get_xydata())
                score += int(np.sum((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) &
                                    (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)))
        scores.append(score)
    chosen = int(np.argmin(scores))
    name, x, y, ha, va = candidates[chosen]
    artist.set_position((x, y)); artist.set_ha(ha); artist.set_va(va)
    return name


def place_mgi_onset_labels(fig, ax, onsets):
    """Keep compact labels where clear; stagger crowded labels in display space."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    lo, hi = sorted(ax.get_xlim())
    visible = [o for o in onsets if lo <= o["temperature_C"] <= hi]
    curve = ax.lines[0].get_path().transformed(ax.lines[0].get_transform())
    occupied = [t.get_window_extent(renderer).padded(2 * pt) for t in ax.texts]
    labels, normal_boxes, anchors = [], [], []
    for onset in visible:
        label = ax.annotate(onset["label"], xy=(onset["temperature_C"], .97),
                            xycoords=ax.get_xaxis_transform(), xytext=(3, 0),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=8, color="black", annotation_clip=True)
        label.set_in_layout(False)
        labels.append(label)
        normal_boxes.append(label.get_window_extent(renderer))
        anchors.append(ax.get_xaxis_transform().transform((onset["temperature_C"], .97))[0])
    placed = []
    for i, label in enumerate(labels):
        normal = normal_boxes[i]
        close = [j for j in range(len(labels)) if j != i and
                 abs(anchors[i] - anchors[j]) < max(normal.width, normal_boxes[j].width) + 6 * pt]

        def clear(box):
            padded = box.padded(2 * pt + ax.lines[0].get_linewidth() * pt / 2)
            return (ax.bbox.contains(padded.x0, padded.y0) and
                    ax.bbox.contains(padded.x1, padded.y1) and
                    not curve.intersects_bbox(padded, filled=False) and
                    not any(padded.overlaps(other) for other in occupied) and
                    not any(padded.x0 <= x <= padded.x1 for x in anchors) and
                    not any(j in close and abs(box.y1 - other.y1) < box.height + 6 * pt
                            for j, other in placed))

        moved = bool(close) or not clear(normal)
        if moved:
            found = False
            offsets = [sign * distance for distance in range(12, int(ax.bbox.width / pt) + 24, 12)
                       for sign in (1, -1)]
            # Search near the top first, moving to separate rows when crowded.
            for row in range(max(1, int(ax.bbox.height / (normal.height + 6 * pt)))):
                dy = -row * (normal.height / pt + 6)
                for dx in offsets:
                    label.set_position((dx, dy))
                    label.set_ha("left" if dx > 0 else "right")
                    box = label.get_window_extent(renderer)
                    if clear(box):
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError("Cannot place onset labels without overlap in this view")
            # Attach to the upper portion of this onset line, not the data point.
            text, xy, position, ha = label.get_text(), label.xy, label.get_position(), label.get_ha()
            label.remove()
            label = ax.annotate(text, xy=xy, xycoords=ax.get_xaxis_transform(),
                                xytext=position, textcoords="offset points", ha=ha, va="top",
                                fontsize=8, color="black", annotation_clip=True,
                                arrowprops={"arrowstyle": "-", "color": "black", "lw": .35,
                                            "shrinkA": 2, "shrinkB": 0})
            label.set_in_layout(False)
        else:
            box = normal
        # Store text bounds only; a leader must not inflate the collision box.
        occupied.append(box.padded(2 * pt))
        placed.append((i, box))


def draw_combo(ax, result, mode="gray", fbrm_bw_linestyle="dashed"):
    """Draw the exact saved polylines, in their saved order, on a twin axis."""
    ax.plot(result.mgi.x, result.mgi.y, color="#0072B2" if mode == "color" else "black", linestyle="-",
            linewidth=1.6 * LW_SCALE, label="MGI")
    right = ax.twinx()
    right.plot(result.fbrm.x, result.fbrm.y, color="#D55E00" if mode == "color" else "black" if mode == "bw" else "0.40",
               linestyle=fbrm_bw_linestyle if mode == "bw" else "-",
               linewidth=1.6 * LW_SCALE, label="FBRM")
    right.set_ylabel("FBRM total counts")
    ax.legend([ax.lines[0], right.lines[0]], ["MGI", "FBRM"],
              loc="upper left", frameon=False)
    return right


def apply_signal_y_range(axis, args, family, panel=None):
    """Change only requested display bounds; retain the other autoscaled side."""
    bounds = getattr(args, f"{family}_y_range", None)
    lower = getattr(args, f"{family}_y_lower", None)
    upper = getattr(args, f"{family}_y_upper", None)
    panel_bounds = getattr(args, f"panel_{family}_y_ranges", None)
    if panel is not None and panel_bounds is not None:
        bounds = panel_bounds[2*panel:2*panel+2]
    if bounds is not None:
        lower, upper = sorted(bounds)
    if lower is None and upper is None:
        return
    auto_lower, auto_upper = axis.get_ylim()
    lower = auto_lower if lower is None else lower
    upper = auto_upper if upper is None else upper
    if not lower < upper:
        raise ValueError(f"Resolved {family.upper()} y lower bound must be less than upper bound")
    axis.set_ylim(lower, upper)


def apply_combo_y_ranges(ax, right, args, panel=None):
    for axis, family in ((ax, "mgi"), (right, "fbrm")):
        apply_signal_y_range(axis, args, family, panel)


def build_combo_panels(result, args):
    """One manuscript canvas; ranges only clip display, never select samples."""
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(1, 2, figsize=(args.width_mm / 25.4, args.width_mm / 2 / 25.4 * 5.5 / 9), dpi=OUTPUT_DPI[args.mode])
        for i, ax in enumerate(axes):
            right = draw_combo(ax, result, args.mode, getattr(args, "fbrm_bw_linestyle", "dashed"))
            apply_combo_y_ranges(ax, right, args, panel=i)
            ax.set_xlabel(TEMPERATURE_LABEL)
            ax.set_ylabel("MGI")
            ax.set_title(("a)", "b)")[i], loc="left", fontsize=8)
            ax.set_xlim(*sorted(args.panel_x_ranges[2*i:2*i+2], reverse=True))
            apply_publication_temperature_ticks(ax)
            for axis in (ax, right):
                axis.tick_params(labelsize=8)
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.minorticks_off()
                axis.grid(False)
        fig.tight_layout(pad=.6, w_pad=1.2)
        placement = None
        if args.onsets != "none":
            placement = [place_onset_text(fig, ax, result.onsets, False, family_prefix=True)
                         for ax in axes]
        return fig, placement


def build_figure(result: SavedResult, args: argparse.Namespace):
    combo = args.kind == "combo"
    fbrm = args.kind == "fbrm"
    primary = result.fbrm if fbrm else result.mgi
    if combo and args.view == "panels":
        return build_combo_panels(result, args)
    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(args.width_mm / 25.4, (60 if fbrm else args.width_mm * 5.5 / 9) / 25.4),
                               dpi=OUTPUT_DPI[args.mode])
        if combo:
            right = draw_combo(ax, result, args.mode, getattr(args, "fbrm_bw_linestyle", "dashed"))
            apply_combo_y_ranges(ax, right, args)
        else:
            ax.plot(primary.x, primary.y, color="#D55E00" if fbrm and args.mode == "color" else "black", linestyle="-",
                    linewidth=(args.mgi_linewidth if args.kind == "mgi" and getattr(args, "mgi_linewidth", None) is not None
                               else 1.45 * LW_SCALE), label="FBRM" if fbrm else "MGI")
            if args.points == "show":
                ax.plot(result.measurements.x, result.measurements.y, ".",
                        color=("#D55E00" if fbrm else "#0072B2") if args.mode == "color" else "0.70",
                        alpha=.3 if args.mode == "color" else None, ms=1,
                        zorder=1)
        ax.set_ylabel("FBRM Total Counts [counts/s]" if fbrm else "MGI")
        ax.set_xlabel(TEMPERATURE_LABEL)
        extent = (result.measurements if not combo and getattr(args, "acquisition_extent", False) else primary)
        lo, hi = sorted(args.x_range) if args.x_range else (float(extent.x.min()), float(extent.x.max()))
        if lo == hi:
            raise ValueError("Temperature extent must have positive width")
        ax.set_xlim(hi, lo)
        apply_publication_temperature_ticks(ax)
        for axis in fig.axes:
            axis.tick_params(labelsize=8)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.minorticks_off()
            axis.grid(False)
        markers = not combo and (args.onsets == "markers" or (fbrm and args.onsets == "text"))
        if markers:
            for onset in result.onsets:
                if lo <= onset["temperature_C"] <= hi:
                    ax.axvline(onset["temperature_C"], color="black", ls="--", lw=1.0 * LW_SCALE)
                    ax.plot(onset["temperature_C"], onset["value"], "o", color="black", ms=2)
        if not combo:
            apply_signal_y_range(ax, args, "fbrm" if fbrm else "mgi")
        fig.tight_layout(pad=.6)
        placement = None
        if args.onsets != "none":
            placement = place_onset_text(fig, ax, result.onsets, markers, family_prefix=combo)
        if args.kind == "mgi" and markers and getattr(args, "show_onset_labels", False):
            place_mgi_onset_labels(fig, ax, result.onsets)
        return fig, placement


def encode_png(fig, mode: str) -> bytes:
    """Adapted from font-test raster helpers; no preparation/layout imports.

    Render opaque RGB for color, grayscale otherwise. BW uses threshold 128, no dithering, and
    OpenCV bilevel encoding. Insert physical resolution without changing pixels.
    """
    dpi = OUTPUT_DPI[mode]
    buffer = io.BytesIO()
    # Saving happens after build_figure's context exits. Do not inherit a host
    # script's bbox/layout/simplification settings during deferred raster drawing.
    with plt.rc_context(PUBLICATION_RC):
        fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", transparent=False)
    gray = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError("Cannot decode rendered PNG")
    params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    if mode == "bw":
        gray = np.where(gray < 128, 0, 255).astype(np.uint8)
        params += [cv2.IMWRITE_PNG_BILEVEL, 1]
    ok, encoded = cv2.imencode(".png", gray, params)
    if not ok:
        raise RuntimeError("Cannot encode publication PNG")
    raw = encoded.tobytes()
    if raw[24:26] != bytes([1 if mode == "bw" else 8, 2 if mode == "color" else 0]):
        raise RuntimeError("Unexpected PNG bit depth/color type")
    ppm = round(dpi / .0254)
    chunk = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    return raw[:33] + struct.pack(">I", 9) + chunk + struct.pack(">I", zlib.crc32(chunk)) + raw[33:]


def validated_output_dir(path: Path) -> Path:
    """Allow generated/ or exactly a resolved data analysis/.../views/<id> directory."""
    resolved = path.resolve()
    generated_safe = OUTPUT_ROOT.resolve() == OUTPUT_ROOT and resolved.is_relative_to(OUTPUT_ROOT)
    # A generated/ symlink must not escape its original permitted tree.
    if path.absolute().is_relative_to(OUTPUT_ROOT) and not generated_safe:
        raise ValueError("Output must stay beneath repository generated/ (no symlink escape)")
    if generated_safe:
        return resolved
    if ANALYSIS_ROOT.resolve() == ANALYSIS_ROOT and resolved.is_relative_to(ANALYSIS_ROOT):
        parts = resolved.relative_to(ANALYSIS_ROOT).parts
        if (len(parts) >= 4 and parts[-2] == "views" and
                "analysis" in parts[1:-2] and not {"bin", "scripts"}.intersection(parts)):
            return resolved
    raise ValueError("Output must stay beneath repository generated/ or data analysis/.../views/<view-id>/ (no symlink escape)")


def validate(args: argparse.Namespace) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise ValueError("--name must be a simple filename stem")
    if args.kind == "mgi" and getattr(args, "mgi_linewidth", None) is not None:
        if not math.isfinite(args.mgi_linewidth) or args.mgi_linewidth <= 0:
            raise ValueError("--mgi-linewidth must be positive and finite")
    if not math.isfinite(args.width_mm) or args.width_mm < 75:
        raise ValueError("--width-mm must be finite and at least 75 mm for 8 pt layout")
    if (args.view == "limited") != (args.x_range is not None):
        raise ValueError("Only --view limited requires --x-range")
    if args.x_range and (not all(map(math.isfinite, args.x_range)) or args.x_range[0] == args.x_range[1]):
        raise ValueError("--x-range requires distinct finite bounds")
    panel_ranges = getattr(args, "panel_x_ranges", None)
    if args.view == "panels":
        if (panel_ranges is None or not all(map(math.isfinite, panel_ranges)) or
                panel_ranges[0] == panel_ranges[1] or panel_ranges[2] == panel_ranges[3]):
            raise ValueError("Two-panel combo requires four finite --panel-x-ranges bounds")
    elif panel_ranges is not None:
        raise ValueError("--panel-x-ranges requires --view panels")
    for family in ("mgi", "fbrm"):
        lower = getattr(args, f"{family}_y_lower", None)
        upper = getattr(args, f"{family}_y_upper", None)
        if getattr(args, f"{family}_y_range", None) is not None and (lower is not None or upper is not None):
            raise ValueError(f"Do not combine --{family}-y-range with --{family}-y-lower/--{family}-y-upper")
        if any(v is not None and not math.isfinite(v) for v in (lower, upper)):
            raise ValueError(f"{family.upper()} y bounds must be finite")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"{family.upper()} y lower bound must be less than upper bound")
        for prefix, count in (("", 2), ("panel_", 4)):
            name = f"{prefix}{family}_y_range" + ("s" if prefix else "")
            bounds = getattr(args, name, None)
            if bounds is None:
                continue
            flag = "--" + name.replace("_", "-")
            if prefix and args.view != "panels":
                raise ValueError(f"{flag} requires --view panels")
            if (len(bounds) != count or not all(map(math.isfinite, bounds)) or
                    any(bounds[i] == bounds[i+1] for i in range(0, count, 2))):
                raise ValueError(f"{flag} requires finite, distinct bounds for each axis")
    if args.mode == "bw" and getattr(args, "points", None) == "show":
        raise ValueError("Gray measurement points require grayscale output")
    outdir = validated_output_dir(args.output_dir)
    if args.kind == "fbrm" and not outdir.is_relative_to(OUTPUT_ROOT):
        raise ValueError("FBRM replay outputs must stay beneath generated/; analysis inputs are read-only")
    for suffix in (".png", ".manifest.json"):
        path = outdir / (args.name + suffix)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Refusing non-regular output path {path}")
        if path.exists() and not getattr(args, "force", False):
            raise ValueError(f"Refusing to overwrite {path}")
    return outdir


def render(args: argparse.Namespace) -> tuple[Path, Path]:
    outdir = validate(args)
    result = {"mgi": load_mgi, "combo": load_combo, "fbrm": load_fbrm}[args.kind](args)
    fig, placement = build_figure(result, args)
    try:
        png = encode_png(fig, args.mode)
    finally:
        plt.close(fig)
    for source in result.sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("Source changed during rendering; refusing output")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout.strip()
    manifest = {"sources": result.sources, "accepted_onsets": result.onsets,
                "git_commit": commit, "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "options": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "branch": getattr(args, "branch", "saved-fbrm-replay" if args.kind == "fbrm" else "saved-combo-export"), "view": args.view,
                "output_mode": args.mode, "dpi": OUTPUT_DPI[args.mode],
                "linewidth_scale": LW_SCALE, "font_pt": 8, "onset_text_placement": placement,
                "x_axis_label": TEMPERATURE_LABEL,
                "figure_size_mm": list(fig.get_size_inches() * 25.4),
                "axes": [{"bounds": list(ax.get_position().bounds),
                          "xlim": list(ax.get_xlim()),
                          "ylim": list(ax.get_ylim())} for ax in fig.axes],
                "publication_rc": PUBLICATION_RC,
                "applied_line_styles": [{"series": line.get_label(), "color": line.get_color(),
                                         "linestyle": line.get_linestyle()}
                                        for ax in fig.axes for line in ax.lines
                                        if line.get_label() in ("MGI", "FBRM")],
                "curve_linewidth_pt": fig.axes[0].lines[0].get_linewidth(),
                "output_sha256": hashlib.sha256(png).hexdigest(),
                "scientific_preparation": "none; saved samples and accepted results only"}
    if result.replay is not None:
        manifest["replay"] = result.replay
        manifest["scientific_preparation"] = "verified saved-parameter FBRM replay; no detection or selection"
    outdir.mkdir(parents=True, exist_ok=True)
    image_path, manifest_path = outdir / (args.name + ".png"), outdir / (args.name + ".manifest.json")
    payloads = ((image_path, png),
                (manifest_path, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")))
    if any(str(path.resolve()) == source["path"] for path, _ in payloads for source in result.sources):
        raise ValueError("Output must not replace a source file")
    if getattr(args, "force", False):
        # Stage both complete files before replacing either. Each rename is atomic;
        # the manifest is installed last and its PNG hash detects interrupted pairs.
        staged = []
        try:
            for path, data in payloads:
                with tempfile.NamedTemporaryFile(dir=outdir, prefix=f".{args.name}-", delete=False) as handle:
                    staged.append((Path(handle.name), path))
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            validate(args)  # Recheck target safety immediately before replacement.
            for temporary, path in staged:
                os.replace(temporary, path)
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
    else:
        for path, data in payloads:
            with path.open("xb") as handle:
                handle.write(data)
    return image_path, manifest_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="kind", required=True)
    for kind in ("mgi", "combo", "fbrm"):
        sub = commands.add_parser(kind)
        sub.add_argument("--name", required=True)
        sub.add_argument("-f", "--force", action="store_true", help="Replace only the named PNG and manifest")
        sub.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
        sub.add_argument("--view", choices=("full", "limited", "panels") if kind == "combo" else ("full", "limited"), required=True)
        sub.add_argument("--x-range", nargs=2, type=float)
        sub.add_argument("--mode", choices=("color", "gray", "bw"), default="gray")
        sub.add_argument("--width-mm", type=float, default=83.5)
        sub.add_argument("--onsets", choices=("none", "text", "markers") if kind == "mgi" else ("none", "text"), default="none")
        for family in (("mgi", "fbrm") if kind == "combo" else (kind,)):
            sub.add_argument(f"--{family}-y-range", nargs=2, type=float,
                             metavar=("YMIN", "YMAX"), help="Display-only bounds; default: autoscale")
            for side in ("lower", "upper"):
                sub.add_argument(f"--{family}-y-{side}", type=float,
                                 help=f"Set only the {side} bound; omitted side stays automatic. Cannot combine with --{family}-y-range")
        if kind == "mgi":
            sub.add_argument("--show-onset-labels", action="store_true",
                             help="Label accepted onset lines T1/T2/... when --onsets markers is used")
            sub.add_argument("--mgi-linewidth", type=float, default=None,
                             help="Standalone MGI curve linewidth in points; default: 1.45 × 0.50")
            sub.add_argument("--branch", choices=("raw", "sg"), required=True)
            sub.add_argument("--curve", type=Path, required=True)
            sub.add_argument("--bends", type=Path, required=True)
            sub.add_argument("--xcol", default="Tr (°C)")
            sub.add_argument("--ycol", default="BW")
            sub.add_argument("--points", choices=("show", "hide"), required=True)
            sub.add_argument("--measurements", type=Path)
            sub.add_argument("--measurement-ycol")
            sub.add_argument("--acquisition-extent", action="store_true")
        elif kind == "fbrm":
            sub.add_argument("--data", type=Path, required=True, help="Saved ts-fbrm-tr.csv")
            sub.add_argument("--onsets-file", type=Path, required=True)
            sub.add_argument("--candidates-file", type=Path, required=True)
            sub.add_argument("--params-file", type=Path, required=True)
            sub.add_argument("--points", choices=("show", "hide"), default="hide")
        elif kind == "combo":
            sub.add_argument("--fbrm-bw-linestyle", choices=("dashed", "dotted"), default="dashed",
                             help="FBRM black line style in BW mode; GRAY remains gray solid")
            for family in ("mgi", "fbrm"):
                sub.add_argument(f"--panel-{family}-y-ranges", nargs=4, type=float,
                                 metavar=("Y1MIN", "Y1MAX", "Y2MIN", "Y2MAX"),
                                 help=f"Display-only bounds for panels a/b; overrides --{family}-y-range; requires --view panels")
            sub.add_argument("--panel-x-ranges", nargs=4, type=float,
                             help="Two display-only ranges, a then b; requires --view panels")
            sub.add_argument("--data", type=Path, required=True, help="Saved combo/mgi-fbrm-combo-data.csv")
            sub.add_argument("--mgi-bends", type=Path)
            sub.add_argument("--fbrm-onsets", type=Path)
    return p


def main() -> int:
    p = parser()
    try:
        for path in render(p.parse_args()):
            print(path)
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
