#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-onset-gallery.py
---------------------

Create a static, offline HTML gallery for existing MGI/PCHIP, FBRM onset, and
MGI-FBRM combo outputs under a PCHIP project tree.

This script is a gallery/index builder only. It does not calculate onsets, does
not redraw figures, and does not modify analysis data.

Expected active project-tree layouts
------------------------------------

MGI/PCHIP base output:

    */tl/roi*/rgb/analysis/pchip/*.png

MGI/PCHIP limit-view output:

    */tl/roi*/rgb/analysis/pchip/views/<view-id>/*.png

FBRM base output:

    */tl/roi*/rgb/analysis/fbrm_onsets/*.png

FBRM limit-view output:

    */tl/roi*/rgb/analysis/fbrm_onsets/views/<view-id>/*.png

MGI-FBRM combo base output:

    */tl/roi*/rgb/analysis/combo/*.png

MGI-FBRM combo limit-view output:

    */tl/roi*/rgb/analysis/combo/views/<view-id>/*.png

The script intentionally prefers the active workflow directories under
tl/roi*/rgb/analysis/. Root-level fbrm_onsets/ directories and test/QC variants
such as fbrm_onsets_frac_* are ignored by normal discovery.

Examples
--------

All base images, no data links:

    python scripts/make-onset-gallery.py . \
        --gallery-content all \
        --gallery-view base

MGI base gallery with data links:

    python scripts/make-onset-gallery.py . \
        --gallery-content mgi \
        --gallery-view base \
        --with-data

All existing limit-view images:

    python scripts/make-onset-gallery.py . \
        --gallery-content all \
        --gallery-view limited

Only one limit-view:

    python scripts/make-onset-gallery.py . \
        --gallery-content all \
        --gallery-view limited \
        --view-id x_45-57__mgi_auto__fbrm_auto

Author: Kim Miikki / ChatGPT-assisted workflow, 2026
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
DATA_EXTENSIONS = {".csv", ".json", ".txt"}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "_archive-20260612",
    "_cleanup_archive",
    "_github",
}

PCHIP_IMAGE_PRIORITY = [
    "bw-raw-vs-temp-pchip-publication-onsets-clean-color.png",
    "bw-raw-vs-temp-pchip-publication-onsets-clean-mono.png",
    "bw-raw-vs-temp-pchip-publication-onsets-color.png",
    "bw-raw-vs-temp-pchip-publication-onsets-mono.png",
    "bw-raw-vs-temp-pchip-publication-clean-color.png",
    "bw-raw-vs-temp-pchip-publication-clean-mono.png",
    "bw-raw-vs-temp-pchip-publication-main-color.png",
    "bw-raw-vs-temp-pchip-publication-main-mono.png",
    "bw-sg-vs-temp-pchip-publication-onsets-clean-color.png",
    "bw-sg-vs-temp-pchip-publication-onsets-clean-mono.png",
    "bw-sg-vs-temp-pchip-publication-onsets-color.png",
    "bw-sg-vs-temp-pchip-publication-onsets-mono.png",
    "bw-sg-vs-temp-pchip-publication-clean-color.png",
    "bw-sg-vs-temp-pchip-publication-clean-mono.png",
    "bw-sg-vs-temp-pchip-publication-main-color.png",
    "bw-sg-vs-temp-pchip-publication-main-mono.png",
    "bw-raw-vs-temp-pchip-bends.png",
    "bw-sg-vs-temp-pchip-bends.png",
    "bw-raw-vs-temp-pchip-candidate-diagnostics.png",
    "bw-sg-vs-temp-pchip-candidate-diagnostics.png",
    "bw-raw-vs-temp-pchip-diagnostic-bends.png",
    "bw-sg-vs-temp-pchip-diagnostic-bends.png",
    "bw-raw-vs-temp-pchip-diagnostic-main-bends.png",
    "bw-sg-vs-temp-pchip-diagnostic-main-bends.png",
    "bw-raw-vs-temp-pchip-raw-smooth-bends.png",
    "bw-sg-vs-temp-pchip-raw-smooth-bends.png",
    "bw-raw-vs-temp-pchip-raw-bends.png",
    "bw-sg-vs-temp-pchip-raw-bends.png",
    "bw-raw-vs-temp-pchip.png",
    "bw-sg-vs-temp-pchip.png",
]

FBRM_IMAGE_PRIORITY = [
    "fbrm-onset-publication-onsets-clean-color.png",
    "fbrm-onset-publication-onsets-clean-mono.png",
    "fbrm-onset-publication-onsets-data-color.png",
    "fbrm-onset-publication-onsets-data-mono.png",
    "fbrm-onset-publication-clean-color.png",
    "fbrm-onset-publication-clean-mono.png",
    "fbrm-onset-publication-main-color.png",
    "fbrm-onset-publication-main-mono.png",
    "fbrm-onset-publication.png",
    "fbrm-onset-candidates.png",
    "fbrm-onset-diagnostic.png",
]

COMBO_IMAGE_PRIORITY = [
    "mgi-fbrm-combo.png",
    "mgi-fbrm-combo-main.png",
    "mgi-fbrm-combo-main-mono.png",
    "mgi-fbrm-combo-main-clean-mono.png",
]

PLOT_LABELS = {
    # MGI/PCHIP
    "bw-raw-vs-temp-pchip-publication-onsets-clean-color.png": "MGI raw + PCHIP · publication onsets · clean color",
    "bw-raw-vs-temp-pchip-publication-onsets-clean-mono.png": "MGI raw + PCHIP · publication onsets · clean mono",
    "bw-raw-vs-temp-pchip-publication-onsets-color.png": "MGI raw + PCHIP · publication onsets · data color",
    "bw-raw-vs-temp-pchip-publication-onsets-mono.png": "MGI raw + PCHIP · publication onsets · data mono",
    "bw-raw-vs-temp-pchip-publication-clean-color.png": "MGI raw + PCHIP · publication clean · color",
    "bw-raw-vs-temp-pchip-publication-clean-mono.png": "MGI raw + PCHIP · publication clean · mono",
    "bw-raw-vs-temp-pchip-publication-main-color.png": "MGI raw + PCHIP · publication main · color",
    "bw-raw-vs-temp-pchip-publication-main-mono.png": "MGI raw + PCHIP · publication main · mono",
    "bw-sg-vs-temp-pchip-publication-onsets-clean-color.png": "MGI SG + PCHIP · publication onsets · clean color",
    "bw-sg-vs-temp-pchip-publication-onsets-clean-mono.png": "MGI SG + PCHIP · publication onsets · clean mono",
    "bw-sg-vs-temp-pchip-publication-onsets-color.png": "MGI SG + PCHIP · publication onsets · data color",
    "bw-sg-vs-temp-pchip-publication-onsets-mono.png": "MGI SG + PCHIP · publication onsets · data mono",
    "bw-sg-vs-temp-pchip-publication-clean-color.png": "MGI SG + PCHIP · publication clean · color",
    "bw-sg-vs-temp-pchip-publication-clean-mono.png": "MGI SG + PCHIP · publication clean · mono",
    "bw-sg-vs-temp-pchip-publication-main-color.png": "MGI SG + PCHIP · publication main · color",
    "bw-sg-vs-temp-pchip-publication-main-mono.png": "MGI SG + PCHIP · publication main · mono",
    "bw-raw-vs-temp-pchip-bends.png": "MGI raw + PCHIP · onsets",
    "bw-sg-vs-temp-pchip-bends.png": "MGI SG + PCHIP · onsets",
    "bw-raw-vs-temp-pchip-candidate-diagnostics.png": "MGI raw + PCHIP · candidate diagnostics",
    "bw-sg-vs-temp-pchip-candidate-diagnostics.png": "MGI SG + PCHIP · candidate diagnostics",
    "bw-raw-vs-temp-pchip-diagnostic-bends.png": "MGI raw + PCHIP · diagnostic",
    "bw-sg-vs-temp-pchip-diagnostic-bends.png": "MGI SG + PCHIP · diagnostic",
    "bw-raw-vs-temp-pchip-diagnostic-main-bends.png": "MGI raw + PCHIP · diagnostic main",
    "bw-sg-vs-temp-pchip-diagnostic-main-bends.png": "MGI SG + PCHIP · diagnostic main",
    "bw-raw-vs-temp-pchip-raw-smooth-bends.png": "MGI raw + PCHIP · raw/smooth",
    "bw-sg-vs-temp-pchip-raw-smooth-bends.png": "MGI SG + PCHIP · raw/smooth",
    "bw-raw-vs-temp-pchip-raw-bends.png": "MGI raw + PCHIP · raw",
    "bw-sg-vs-temp-pchip-raw-bends.png": "MGI SG + PCHIP · raw",
    "bw-raw-vs-temp-pchip.png": "MGI raw + PCHIP · data",
    "bw-sg-vs-temp-pchip.png": "MGI SG + PCHIP · data",
    # FBRM
    "fbrm-onset-publication-onsets-clean-color.png": "FBRM · publication onsets · clean color",
    "fbrm-onset-publication-onsets-clean-mono.png": "FBRM · publication onsets · clean mono",
    "fbrm-onset-publication-onsets-data-color.png": "FBRM · publication onsets · data color",
    "fbrm-onset-publication-onsets-data-mono.png": "FBRM · publication onsets · data mono",
    "fbrm-onset-publication-clean-color.png": "FBRM · publication clean · color",
    "fbrm-onset-publication-clean-mono.png": "FBRM · publication clean · mono",
    "fbrm-onset-publication-main-color.png": "FBRM · publication main · color",
    "fbrm-onset-publication-main-mono.png": "FBRM · publication main · mono",
    "fbrm-onset-publication.png": "FBRM · publication",
    "fbrm-onset-candidates.png": "FBRM · candidate diagnostics",
    "fbrm-onset-diagnostic.png": "FBRM · full diagnostic",
    # Combo
    "mgi-fbrm-combo.png": "MGI-FBRM combo · with derivatives",
    "mgi-fbrm-combo-main.png": "MGI-FBRM combo · main color",
    "mgi-fbrm-combo-main-mono.png": "MGI-FBRM combo · main mono",
    "mgi-fbrm-combo-main-clean-mono.png": "MGI-FBRM combo · main clean mono",
}

PRIORITY_BY_KIND = {
    "mgi": {name: i for i, name in enumerate(PCHIP_IMAGE_PRIORITY)},
    "fbrm": {name: i for i, name in enumerate(FBRM_IMAGE_PRIORITY)},
    "combo": {name: i for i, name in enumerate(COMBO_IMAGE_PRIORITY)},
}

KIND_TITLE = {
    "mgi": "MGI/PCHIP",
    "fbrm": "FBRM",
    "combo": "MGI-FBRM combo",
}


@dataclass(frozen=True)
class GalleryItem:
    """One image result directory rendered as one HTML section."""

    kind: str
    view_scope: str
    view_id: str | None
    result_dir: Path
    rel_result_dir: Path
    experiment: str
    roi: str
    images: list[Path]
    data_files: list[Path]


@dataclass
class TreeNode:
    """Small in-memory representation of referenced files for HTML rendering."""

    name: str
    path: Path
    is_dir: bool
    children: dict[str, "TreeNode"] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline HTML gallery for existing MGI/PCHIP, FBRM, and combo onset outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Source PCHIP project root directory.",
    )
    parser.add_argument("--root-label", help="Display name for the source root; does not change file paths.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("galleries/onset-gallery.html"),
        help=(
            "Output HTML file or output directory. If the path has no .html/.htm suffix, "
            "onset-gallery.html is written inside that directory. Relative paths are interpreted "
            "relative to the project root."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Onset analysis gallery",
        help="HTML page title.",
    )
    parser.add_argument(
        "--gallery-content",
        choices=("all", "mgi", "fbrm", "combo"),
        default="all",
        help="Which result families to include.",
    )
    parser.add_argument(
        "--gallery-view",
        choices=("base", "limited", "all"),
        default="base",
        help="Include base outputs, limit-view outputs, or both.",
    )
    parser.add_argument(
        "--view-id",
        type=str,
        default=None,
        help="Limit gallery to one views/<view-id>/ directory. Valid only with --gallery-view limited/all.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Include only these experiment directories in the gallery. "
            "Names may be given as directory names or paths."
        ),
    )
    parser.add_argument(
        "--exclude-experiments",
        nargs="+",
        default=[],
        help=(
            "Exclude these experiment directories from the gallery. "
            "Names may be given as directory names or paths."
        ),
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Include CSV/JSON/TXT support links for MGI and/or FBRM sections.",
    )
    parser.add_argument(
        "--export-image-tree",
        type=Path,
        default=None,
        help=(
            "Create a shareable image-only gallery under this export root. "
            "Only gallery images are copied, preserving project-relative paths. "
            "The HTML output is written inside the export root."
        ),
    )
    parser.add_argument(
        "--force-export-overwrite",
        action="store_true",
        help=(
            "Allow writing into an existing non-empty --export-image-tree directory. "
            "Existing copied images may be overwritten."
        ),
    )
    parser.add_argument(
        "--export-data",
        action="store_true",
        help=(
            "When used with --export-image-tree, also copy CSV/JSON/TXT "
            "data/support files referenced by --with-data. This option implies --with-data."
        ),
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help=(
            "Embed image files directly into the HTML as base64 data URIs. "
            "This creates a self-contained image gallery, but the HTML file can become large."
        ),
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude from discovery and referenced-file tree. Can be repeated.",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Do not include the referenced-file tree.",
    )
    parser.add_argument(
        "--open-all",
        action="store_true",
        help="Render experiment details opened by default.",
    )
    parser.add_argument(
        "--thumbnail-height",
        type=int,
        default=260,
        help="Thumbnail image height in pixels.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=0,
        help="Maximum depth for referenced-file tree. 0 = unlimited.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    """Print a clear error and exit."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def normalize_experiment_names(values: list[str] | None) -> set[str]:
    """Normalize experiment names supplied as names or paths."""
    if not values:
        return set()
    return {Path(value).name for value in values if str(value).strip()}


def filter_items_by_experiment(
    items: list[GalleryItem],
    *,
    args: argparse.Namespace,
) -> list[GalleryItem]:
    """Apply explicit experiment include/exclude filters."""
    include_names = normalize_experiment_names(args.experiments)
    exclude_names = normalize_experiment_names(args.exclude_experiments)

    filtered = items

    if include_names:
        filtered = [item for item in filtered if item.experiment in include_names]

    if exclude_names:
        filtered = [item for item in filtered if item.experiment not in exclude_names]

    return filtered


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI combinations."""
    include_names = normalize_experiment_names(args.experiments)
    exclude_names = normalize_experiment_names(args.exclude_experiments)
    overlap = include_names & exclude_names
    if overlap:
        fail(
            "The same experiment cannot be listed in both --experiments and "
            f"--exclude-experiments: {', '.join(sorted(overlap))}"
        )
    if getattr(args, "export_data", False):
        if getattr(args, "export_image_tree", None) is None:
            fail("--export-data requires --export-image-tree.")
        args.with_data = True
    if args.gallery_content == "combo" and args.with_data:
        fail(
            "--with-data is not supported for --gallery-content combo. "
            "Use --gallery-content all --with-data if data files are needed."
        )
    if args.view_id is not None and args.gallery_view == "base":
        fail("--view-id is only valid with --gallery-view limited or --gallery-view all.")
    if args.thumbnail_height < 80:
        fail("--thumbnail-height must be at least 80 pixels.")
    if args.max_tree_depth < 0:
        fail("--max-tree-depth must be >= 0.")


def resolve_output_path(root: Path, output_arg: Path, *, output_root: Path | None = None) -> Path:
    """Resolve --output as either an HTML file or a directory.

    In normal mode, relative outputs are interpreted under the source project
    root. In image-export mode, relative outputs are interpreted under the
    export root so the HTML and copied images form one self-contained tree.
    """
    base = output_root if output_root is not None else root
    output = output_arg
    if not output.is_absolute():
        output = base / output
    if output.suffix.lower() not in {".html", ".htm"}:
        output = output / "onset-gallery.html"
    return output.resolve()


def is_safe_export_destination(source_root: Path, export_root: Path) -> bool:
    """Avoid exporting into the source tree or directly on top of it."""
    source_root = source_root.resolve()
    export_root = export_root.resolve()

    if source_root == export_root:
        return False

    try:
        common = Path(os.path.commonpath([source_root, export_root]))
    except ValueError:
        return True

    return common != source_root


def iter_gallery_image_paths(items: list[GalleryItem]) -> list[Path]:
    """Return unique project-relative image paths referenced by gallery items."""
    seen: set[Path] = set()
    out: list[Path] = []

    for item in items:
        for image in item.images:
            if image not in seen:
                seen.add(image)
                out.append(image)

    return out


def iter_gallery_data_paths(items: list[GalleryItem]) -> list[Path]:
    """Return unique project-relative data/support paths referenced by gallery items."""
    seen: set[Path] = set()
    out: list[Path] = []

    for item in items:
        for data_file in item.data_files:
            if data_file not in seen:
                seen.add(data_file)
                out.append(data_file)

    return out


def export_gallery_tree(
    items: list[GalleryItem],
    *,
    source_root: Path,
    export_root: Path,
    force: bool,
    include_data: bool,
) -> tuple[int, int]:
    """Copy selected gallery files to export_root, preserving relative paths."""
    source_root = source_root.resolve()
    export_root = export_root.resolve()

    if not is_safe_export_destination(source_root, export_root):
        raise RuntimeError(
            "export destination must not be the source root or inside the source tree"
        )

    if export_root.exists() and not export_root.is_dir():
        raise RuntimeError(f"export destination exists but is not a directory: {export_root}")

    if export_root.exists() and any(export_root.iterdir()) and not force:
        raise RuntimeError(
            f"export destination is not empty: {export_root}\n"
            "Use --force-export-overwrite to allow writing into it."
        )

    export_root.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_data = 0
    copied_data = 0

    for rel in iter_gallery_image_paths(items):
        src = source_root / rel
        dst = export_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_images += 1

    if include_data:
        for rel in iter_gallery_data_paths(items):
            src = source_root / rel
            dst = export_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_data += 1

    return copied_images, copied_data


def strip_data_files_for_image_export(items: list[GalleryItem]) -> list[GalleryItem]:
    """Remove CSV/JSON/TXT support links from an image-only export gallery."""
    out: list[GalleryItem] = []

    for item in items:
        out.append(
            GalleryItem(
                kind=item.kind,
                view_scope=item.view_scope,
                view_id=item.view_id,
                result_dir=item.result_dir,
                rel_result_dir=item.rel_result_dir,
                experiment=item.experiment,
                roi=item.roi,
                images=item.images,
                data_files=[],
            )
        )

    return out


def path_to_href(path: Path) -> str:
    """Convert a filesystem path to a browser-friendly href."""
    parts = [quote(part) for part in path.as_posix().split("/")]
    return "/".join(parts)


def rel_href(target_abs: Path, *, output_dir_abs: Path) -> str:
    """Return a URL-safe relative link from the HTML file directory to target."""
    rel = Path(os.path.relpath(target_abs, start=output_dir_abs))
    return path_to_href(rel)


def image_src_for_html(
    image_rel: Path,
    *,
    root_abs: Path,
    output_dir_abs: Path,
    embed_images: bool,
) -> str:
    """Return either a relative image href or an embedded data URI."""
    target_abs = root_abs / image_rel

    if not embed_images:
        return rel_href(target_abs, output_dir_abs=output_dir_abs)

    mime_type, _ = mimetypes.guess_type(target_abs.name)
    if mime_type is None:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(target_abs.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def should_skip_path(path: Path, exclude_names: set[str]) -> bool:
    """Return True if a path contains an excluded component."""
    return any(part in exclude_names for part in path.parts)


def detect_experiment_and_roi(rel_result_dir: Path) -> tuple[str, str]:
    """Infer experiment and ROI labels from a project-relative result directory."""
    parts = rel_result_dir.parts
    experiment = parts[0] if parts else "(unknown experiment)"
    roi = "roi"

    if "tl" in parts:
        tl_index = parts.index("tl")
        if tl_index + 1 < len(parts):
            roi = parts[tl_index + 1]
    else:
        for part in parts:
            if part.lower().startswith("roi"):
                roi = part
                break

    return experiment, roi


def is_active_result_dir(rel_dir: Path, *, result_name: str) -> bool:
    """Check active workflow result path shape under tl/roi*/rgb/analysis/."""
    parts = rel_dir.parts
    if len(parts) < 6:
        return False
    if parts[-1] != result_name:
        return False
    if parts[-2] != "analysis":
        return False
    if parts[-3] != "rgb":
        return False
    if "tl" not in parts:
        return False
    return True


def is_active_view_dir(rel_view_dir: Path, *, result_name: str) -> bool:
    """Check active workflow limit-view path shape under result/views/<view-id>/."""
    parts = rel_view_dir.parts
    if len(parts) < 8:
        return False
    if parts[-3:] and parts[-2] != "views":
        return False
    if parts[-3] != result_name:
        return False
    if parts[-4] != "analysis":
        return False
    if parts[-5] != "rgb":
        return False
    if "tl" not in parts:
        return False
    return True


def iter_selected_kinds(content: str) -> list[str]:
    """Return result families requested by --gallery-content."""
    if content == "all":
        return ["mgi", "fbrm", "combo"]
    return [content]


def result_name_for_kind(kind: str) -> str:
    """Return result-directory name for a gallery family."""
    if kind == "mgi":
        return "pchip"
    if kind == "fbrm":
        return "fbrm_onsets"
    if kind == "combo":
        return "combo"
    raise ValueError(f"Unknown kind: {kind}")


def sort_images(kind: str, images: list[Path]) -> list[Path]:
    """Sort known publication/diagnostic plots first."""
    ranks = PRIORITY_BY_KIND[kind]
    return sorted(
        images,
        key=lambda p: (
            ranks.get(p.name, len(ranks)),
            p.name.lower(),
        ),
    )


def collect_images(result_dir: Path, *, root: Path, kind: str) -> list[Path]:
    """Collect image files directly under one result directory."""
    images = [
        p.relative_to(root)
        for p in result_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sort_images(kind, images)


def collect_data_files(result_dir: Path, *, root: Path, kind: str, with_data: bool) -> list[Path]:
    """Collect data/support files directly under one result directory."""
    if not with_data:
        return []
    if kind == "combo":
        # Combo sections stay image-only. In all --with-data mode the useful
        # data belongs to the MGI and FBRM result directories.
        return []
    files = [
        p.relative_to(root)
        for p in result_dir.iterdir()
        if p.is_file() and p.suffix.lower() in DATA_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def discover_base_items(
    root: Path,
    *,
    kind: str,
    exclude_names: set[str],
    with_data: bool,
) -> list[GalleryItem]:
    """Discover base-output result directories for one result family."""
    result_name = result_name_for_kind(kind)
    out: list[GalleryItem] = []

    for result_dir in sorted(root.rglob(result_name)):
        if not result_dir.is_dir():
            continue

        rel = result_dir.relative_to(root)
        if should_skip_path(rel, exclude_names):
            continue
        if not is_active_result_dir(rel, result_name=result_name):
            continue

        images = collect_images(result_dir, root=root, kind=kind)
        data_files = collect_data_files(result_dir, root=root, kind=kind, with_data=with_data)
        if not images and not data_files:
            continue

        experiment, roi = detect_experiment_and_roi(rel)
        out.append(
            GalleryItem(
                kind=kind,
                view_scope="base",
                view_id=None,
                result_dir=result_dir,
                rel_result_dir=rel,
                experiment=experiment,
                roi=roi,
                images=images,
                data_files=data_files,
            )
        )

    return out


def discover_limited_items(
    root: Path,
    *,
    kind: str,
    exclude_names: set[str],
    with_data: bool,
    view_id: str | None,
) -> list[GalleryItem]:
    """Discover limit-view result directories for one result family."""
    result_name = result_name_for_kind(kind)
    out: list[GalleryItem] = []

    for views_dir in sorted(root.rglob("views")):
        if not views_dir.is_dir():
            continue

        parent = views_dir.parent
        if parent.name != result_name:
            continue

        rel_views = views_dir.relative_to(root)
        if should_skip_path(rel_views, exclude_names):
            continue

        for view_dir in sorted(p for p in views_dir.iterdir() if p.is_dir()):
            if view_id is not None and view_dir.name != view_id:
                continue

            rel = view_dir.relative_to(root)
            if should_skip_path(rel, exclude_names):
                continue
            if not is_active_view_dir(rel, result_name=result_name):
                continue

            images = collect_images(view_dir, root=root, kind=kind)
            data_files = collect_data_files(view_dir, root=root, kind=kind, with_data=with_data)
            if not images and not data_files:
                continue

            experiment, roi = detect_experiment_and_roi(rel)
            out.append(
                GalleryItem(
                    kind=kind,
                    view_scope="limited",
                    view_id=view_dir.name,
                    result_dir=view_dir,
                    rel_result_dir=rel,
                    experiment=experiment,
                    roi=roi,
                    images=images,
                    data_files=data_files,
                )
            )

    return out


def discover_items(args: argparse.Namespace, root: Path) -> list[GalleryItem]:
    """Discover all requested gallery sections."""
    exclude_names = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude_dir)
    items: list[GalleryItem] = []

    for kind in iter_selected_kinds(args.gallery_content):
        if args.gallery_view in {"base", "all"}:
            items.extend(
                discover_base_items(
                    root,
                    kind=kind,
                    exclude_names=exclude_names,
                    with_data=args.with_data,
                )
            )
        if args.gallery_view in {"limited", "all"}:
            items.extend(
                discover_limited_items(
                    root,
                    kind=kind,
                    exclude_names=exclude_names,
                    with_data=args.with_data,
                    view_id=args.view_id,
                )
            )

    items = filter_items_by_experiment(items, args=args)

    return sorted(
        items,
        key=lambda item: (
            item.experiment.lower(),
            item.roi.lower(),
            {"mgi": 0, "fbrm": 1, "combo": 2}[item.kind],
            {"base": 0, "limited": 1}[item.view_scope],
            item.view_id or "",
            item.rel_result_dir.as_posix(),
        ),
    )


def label_for_image(kind: str, image: Path) -> str:
    """Return human-readable plot-card label."""
    if image.name in PLOT_LABELS:
        return PLOT_LABELS[image.name]

    stem = image.stem.replace("-", " ").replace("_", " ").strip()
    prefix = KIND_TITLE.get(kind, kind.upper())
    return f"{prefix} · {stem}" if stem else image.name


def group_by_experiment(items: list[GalleryItem]) -> dict[str, list[GalleryItem]]:
    """Group items by experiment label, preserving sorted item order."""
    grouped: dict[str, list[GalleryItem]] = {}
    for item in items:
        grouped.setdefault(item.experiment, []).append(item)
    return grouped


def insert_tree_path(root_node: TreeNode, rel_path: Path) -> None:
    """Insert one referenced relative path into a compact tree."""
    node = root_node
    parts = rel_path.parts
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        child = node.children.get(part)
        if child is None:
            child_path = Path(*parts[: index + 1])
            child = TreeNode(
                name=part,
                path=child_path,
                is_dir=not is_last,
            )
            node.children[part] = child
        node = child


def build_referenced_tree(items: list[GalleryItem]) -> TreeNode:
    """Build a compact tree from files referenced by the gallery."""
    root_node = TreeNode(name=".", path=Path("."), is_dir=True)
    for item in items:
        for rel_path in [*item.images, *item.data_files]:
            insert_tree_path(root_node, rel_path)
    return root_node


def render_tree_node(
    node: TreeNode,
    *,
    root_abs: Path,
    output_dir_abs: Path,
    max_depth: int,
    depth: int = 0,
) -> str:
    """Render a compact referenced-file tree."""
    if max_depth and depth > max_depth:
        return ""

    if not node.children:
        return ""

    lines: list[str] = ["<ul>"]
    for child in sorted(node.children.values(), key=lambda c: (not c.is_dir, c.name.lower())):
        escaped_name = html.escape(child.name)
        if child.is_dir:
            lines.append(f"<li><strong>{escaped_name}/</strong>")
            lines.append(
                render_tree_node(
                    child,
                    root_abs=root_abs,
                    output_dir_abs=output_dir_abs,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            )
            lines.append("</li>")
        else:
            target_abs = root_abs / child.path
            href = rel_href(target_abs, output_dir_abs=output_dir_abs)
            lines.append(f'<li><a href="{href}">{escaped_name}</a></li>')
    lines.append("</ul>")
    return "\n".join(lines)


def render_css(thumbnail_height: int) -> str:
    """Return embedded CSS."""
    return f"""
:root {{
  color-scheme: light;
}}
body {{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 2rem;
  line-height: 1.45;
  color: #202124;
  background: #ffffff;
}}
h1 {{
  margin-bottom: 0.2rem;
}}
h2 {{
  border-bottom: 1px solid #ddd;
  margin-top: 2.0rem;
  padding-bottom: 0.25rem;
}}
h3 {{
  margin-top: 1.2rem;
  margin-bottom: 0.35rem;
}}
.summary {{
  color: #444;
  margin-bottom: 1.5rem;
}}
.meta {{
  color: #555;
  font-size: 0.92rem;
}}
.badge {{
  display: inline-block;
  border: 1px solid #bbb;
  border-radius: 999px;
  padding: 0.08rem 0.55rem;
  margin-right: 0.35rem;
  margin-bottom: 0.25rem;
  font-size: 0.82rem;
  background: #f7f7f7;
}}
details {{
  margin: 1.0rem 0;
}}
summary {{
  cursor: pointer;
  font-weight: 650;
}}
.section-card {{
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 0.9rem;
  margin: 0.9rem 0;
  background: #fcfcfc;
}}
.gallery-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 1rem;
  margin-top: 0.8rem;
}}
.figure-card {{
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 0.65rem;
  background: #ffffff;
}}
.figure-card img {{
  max-width: 100%;
  height: {thumbnail_height}px;
  object-fit: contain;
  display: block;
  margin: 0 auto 0.5rem auto;
}}
.figure-title {{
  font-weight: 650;
  margin: 0.2rem 0;
}}
.figure-card.publication {{
  grid-column: 1 / -1;
}}
.figure-card.publication img {{
  height: auto;
  margin-left: 0;
}}
.filename {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85rem;
  color: #555;
  word-break: break-all;
}}
.file-list {{
  margin-top: 0.7rem;
  font-size: 0.92rem;
}}
.file-list code {{
  font-size: 0.86rem;
}}
a {{
  color: #0645ad;
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
.tree {{
  font-size: 0.9rem;
}}
.tree ul {{
  list-style-type: none;
  padding-left: 1.2rem;
}}
.warning {{
  border: 1px solid #e6b800;
  background: #fff9db;
  border-radius: 8px;
  padding: 0.8rem;
  margin: 1rem 0;
}}
.controls {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: #ffffff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 0.6rem;
  margin: 1rem 0 1.5rem 0;
}}
.controls button {{
  font: inherit;
  border: 1px solid #aaa;
  border-radius: 6px;
  background: #f7f7f7;
  padding: 0.35rem 0.65rem;
  margin: 0.15rem 0.25rem 0.15rem 0;
  cursor: pointer;
}}
.controls button:hover {{
  background: #eeeeee;
}}
footer {{
  margin-top: 2.5rem;
  color: #777;
  font-size: 0.85rem;
}}
"""


def publication_width_mm(image: Path) -> float | None:
    """Use saved-renderer provenance, not a filename guess, for preview scale."""
    try:
        manifest = json.loads(image.with_suffix('.manifest.json').read_text())
        if manifest.get('scientific_preparation') != 'none; saved samples and accepted results only':
            return None
        width = float(manifest['options']['width_mm'])
        return width if 75 <= width <= 300 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render_item(
    item: GalleryItem,
    *,
    root_abs: Path,
    output_dir_abs: Path,
    embed_images: bool,
) -> str:
    """Render one gallery result section."""
    badges = [
        KIND_TITLE[item.kind],
        item.view_scope,
        item.roi,
    ]
    if item.view_id:
        badges.append(item.view_id)

    badge_html = " ".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)
    rel_dir = html.escape(item.rel_result_dir.as_posix())

    lines: list[str] = [
        '<div class="section-card">',
        f"<h3>{badge_html}</h3>",
        f'<div class="meta"><strong>Result directory:</strong> <code>{rel_dir}</code></div>',
    ]

    if item.images:
        lines.append('<div class="gallery-grid">')
        for image in item.images:
            src = image_src_for_html(
                image,
                root_abs=root_abs,
                output_dir_abs=output_dir_abs,
                embed_images=embed_images,
            )
            title = html.escape(label_for_image(item.kind, image))
            filename = html.escape(image.name)
            rel_path = html.escape(image.as_posix())
            width = publication_width_mm(root_abs / image)
            card = '<div class="figure-card publication">' if width is not None else '<div class="figure-card">'
            image_style = f' style="width:{width:g}mm"' if width is not None else ''
            if width is not None:
                title += f' · publication preview {width:g} mm (responsive on narrow screens)'

            if embed_images:
                # Do not duplicate the large data URI in both href and src.
                # The image itself is embedded; filename/path remain as text.
                lines.extend(
                    [
                        card,
                        f'<img src="{src}" alt="{title}"{image_style}>',
                        f'<div class="figure-title">{title}</div>',
                        f'<div class="filename">{filename}</div>',
                        f'<div class="filename">{rel_path}</div>',
                        '</div>',
                    ]
                )
            else:
                lines.extend(
                    [
                        card,
                        f'<a href="{src}"><img src="{src}" alt="{title}"{image_style}></a>',
                        f'<div class="figure-title">{title}</div>',
                        f'<div class="filename"><a href="{src}">{filename}</a></div>',
                        f'<div class="filename">{rel_path}</div>',
                        '</div>',
                    ]
                )
        lines.append("</div>")
    else:
        lines.append('<div class="warning">No images found in this result directory.</div>')

    if item.data_files:
        lines.append('<div class="file-list"><strong>Data/support files:</strong><ul>')
        for data_file in item.data_files:
            target_abs = root_abs / data_file
            href = rel_href(target_abs, output_dir_abs=output_dir_abs)
            lines.append(
                f'<li><a href="{href}"><code>{html.escape(data_file.name)}</code></a> '
                f'<span class="meta">{html.escape(data_file.as_posix())}</span></li>'
            )
        lines.append("</ul></div>")

    lines.append("</div>")
    return "\n".join(lines)


def render_html(
    *,
    args: argparse.Namespace,
    root_abs: Path,
    output_path: Path,
    items: list[GalleryItem],
) -> str:
    """Render the complete HTML gallery."""
    output_dir_abs = output_path.parent
    grouped = group_by_experiment(items)

    image_count = sum(len(item.images) for item in items)
    data_count = sum(len(item.data_files) for item in items)
    view_ids = sorted({item.view_id for item in items if item.view_id})
    content = args.gallery_content
    gallery_view = args.gallery_view
    with_data = "yes" if args.with_data else "no"

    css = render_css(args.thumbnail_height)
    title = html.escape(args.title)
    root_display = html.escape(getattr(args, "root_label", None) or root_abs.as_posix())

    lines: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title>",
        "<style>",
        css,
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        '<div class="summary">',
        f"<p><strong>Root:</strong> <code>{root_display}</code></p>",
        (
            f"<p><strong>Sections:</strong> {len(items)} · "
            f"<strong>Images:</strong> {image_count} · "
            f"<strong>Data/support links:</strong> {data_count}</p>"
        ),
        (
            f'<p><span class="badge">content: {html.escape(content)}</span> '
            f'<span class="badge">view: {html.escape(gallery_view)}</span> '
            f'<span class="badge">with data: {with_data}</span></p>'
        ),
        "</div>",
        '<div class="controls">',
        '<button type="button" onclick="setAllExperimentDetails(true)">Open all experiments</button>',
        '<button type="button" onclick="setAllExperimentDetails(false)">Close all experiments</button>',
        '<button type="button" onclick="scrollToReferencedFiles()">Scroll to referenced files</button>',
        '<button type="button" onclick="window.scrollTo({top: 0, behavior: \'smooth\'})">Back to top</button>',
        '</div>',
    ]

    if args.view_id:
        lines.append(
            f'<p><span class="badge">requested view-id: {html.escape(args.view_id)}</span></p>'
        )
    elif view_ids:
        lines.append(
            "<p><strong>Included view IDs:</strong> "
            + ", ".join(f"<code>{html.escape(v)}</code>" for v in view_ids)
            + "</p>"
        )

    if not items:
        lines.append(
            '<div class="warning">'
            "No matching gallery items were found. Check --gallery-content, "
            "--gallery-view, --view-id, and whether outputs exist under "
            "<code>*/tl/roi*/rgb/analysis/</code>."
            "</div>"
        )

    open_attr = " open" if args.open_all else ""

    for experiment, exp_items in grouped.items():
        exp_image_count = sum(len(item.images) for item in exp_items)
        detail_key = html.escape(experiment, quote=True)
        lines.append(f'<details{open_attr} data-detail-key="{detail_key}">')
        lines.append(
            f"<summary>{html.escape(experiment)} "
            f'<span class="meta">({len(exp_items)} sections, {exp_image_count} images)</span></summary>'
        )

        for item in exp_items:
            lines.append(
                render_item(
                    item,
                    root_abs=root_abs,
                    output_dir_abs=output_dir_abs,
                    embed_images=args.embed_images,
                )
            )

        lines.append("</details>")

    if not args.no_tree and items:
        tree = build_referenced_tree(items)
        tree_html = render_tree_node(
            tree,
            root_abs=root_abs,
            output_dir_abs=output_dir_abs,
            max_depth=args.max_tree_depth,
        )
        if tree_html:
            lines.append('<h2 id="referenced-files">Referenced files</h2>')
            lines.append('<div class="tree">')
            lines.append(tree_html)
            lines.append("</div>")

    lines.extend(
        [
            "<footer>",
            "Generated by <code>make-onset-gallery.py</code>. "
            "This is an offline static HTML gallery; all links are relative.",
            "</footer>",
            "<script>",
            "(function() {",
            "  var storageKey = 'make-onset-gallery:details:' + window.location.pathname + ':' + document.title;",
            "  function getDetails() {",
            "    return Array.prototype.slice.call(document.querySelectorAll('body > details[data-detail-key]'));",
            "  }",
            "  function loadState() {",
            "    try { return JSON.parse(sessionStorage.getItem(storageKey) || '{}'); }",
            "    catch (err) { return {}; }",
            "  }",
            "  function saveState() {",
            "    var state = {};",
            "    getDetails().forEach(function(el) { state[el.dataset.detailKey] = !!el.open; });",
            "    try { sessionStorage.setItem(storageKey, JSON.stringify(state)); } catch (err) {}",
            "  }",
            "  function restoreState() {",
            "    var state = loadState();",
            "    getDetails().forEach(function(el) {",
            "      var key = el.dataset.detailKey;",
            "      if (Object.prototype.hasOwnProperty.call(state, key)) { el.open = !!state[key]; }",
            "    });",
            "  }",
            "  window.setAllExperimentDetails = function(openState) {",
            "    getDetails().forEach(function(el) { el.open = openState; });",
            "    saveState();",
            "  };",
            "  window.scrollToReferencedFiles = function() {",
            "    var el = document.getElementById('referenced-files');",
            "    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }",
            "  };",
            "  document.addEventListener('DOMContentLoaded', function() {",
            "    restoreState();",
            "    getDetails().forEach(function(el) { el.addEventListener('toggle', saveState); });",
            "  });",
            "  window.addEventListener('pageshow', restoreState);",
            "})();",
            "</script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    validate_args(args)

    root = args.root.resolve()
    if not root.is_dir():
        fail(f"Project root is not a directory: {root}")

    items = discover_items(args, root)

    export_mode = args.export_image_tree is not None
    export_root: Path | None = None
    copied_images = 0

    if export_mode:
        export_root = args.export_image_tree.expanduser().resolve()

        try:
            copied_images, copied_data = export_gallery_tree(
                items,
                source_root=root,
                export_root=export_root,
                force=args.force_export_overwrite,
                include_data=args.export_data,
            )
        except RuntimeError as exc:
            fail(str(exc))

        if not args.export_data:
            # Image-only export means exactly that: no CSV/JSON/TXT support links
            # are copied or referenced from the exported HTML.
            items = strip_data_files_for_image_export(items)

    output_path = resolve_output_path(root, args.output, output_root=export_root)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail(f"Could not create output directory {output_path.parent}: {exc}")

    html_root = export_root if export_root is not None else root
    html_text = render_html(
        args=args,
        root_abs=html_root,
        output_path=output_path,
        items=items,
    )

    try:
        output_path.write_text(html_text, encoding="utf-8")
    except OSError as exc:
        fail(f"Could not write output HTML {output_path}: {exc}")

    image_count = sum(len(item.images) for item in items)
    data_count = sum(len(item.data_files) for item in items)
    print(f"Wrote: {output_path}")
    if export_mode:
        print(f"Export:  {export_root}")
        print(f"Copied:  {copied_images} images")
        if args.export_data:
            print(f"Copied:  {copied_data} data/support files")
    print(f"Sections: {len(items)}")
    print(f"Images:   {image_count}")
    print(f"Data:     {data_count}")
    if args.gallery_view in {"limited", "all"}:
        view_ids = sorted({item.view_id for item in items if item.view_id})
        if view_ids:
            print("Views:    " + ", ".join(view_ids))
        elif args.view_id:
            print(f"Views:    no matches for {args.view_id!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
