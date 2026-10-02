#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-pchip-gallery.py
---------------------

Create a static, offline HTML gallery for PCHIP analysis results and optional
method-comparison result images.

Two use cases are supported:

1. Normal in-place gallery

       python3 make-pchip-gallery.py --root . --output pchip-gallery.html

   The gallery is written inside --root and links to the existing full tree.
   PCHIP result images are always included. Method-comparison images are included
   by default when directories named "method-comparison" are found.

2. Shareable image-only gallery

       python3 make-pchip-gallery.py \
           --root . \
           --output pchip-gallery.html \
           --export-image-tree /path/to/gallery-share

   The script copies only gallery images into the export directory, preserving
   the original relative directory structure, and writes a matching HTML gallery
   into that export directory. This makes a compact shareable package without
   copying raw data, CSV files, logs, or other analysis files.

Design notes
------------
- PCHIP images are always included.
- Method-comparison images are included by default, unless --no-method-comparison
  is used.
- The HTML uses relative links and works directly from file:// without a web
  server.
- In normal mode, the full file tree can include all files under --root.
- In export-image-tree mode, the full tree shows the exported image-only tree.

Author: Kim Miikki / ChatGPT-assisted workflow, 2026
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "_archive-20260612",
    "_github",
}

PCHIP_IMAGE_PRIORITY = [
    # Manuscript/publication-oriented MGI/PCHIP figures first.
    # These images are intentionally titleless, so the gallery card labels
    # below carry the plot-role information for visual review.
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

    # Standard onset and diagnostic figures.
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

METHOD_IMAGE_PRIORITY = [
    "old-vs-pchip-raw.png",
    "old-vs-pchip-sg.png",
    "pchip-raw-vs-sg.png",
    # old/temporary names are accepted and sorted later
    "method-comparison.png",
]

PCHIP_SUPPORT_FILES = [
    "bw-pchip-bend-summary.csv",
    "bw-pchip-bend-report.txt",
    "bw-pchip-workflow.log",
    "bw-raw-vs-temp-pchip-bends.csv",
    "bw-sg-vs-temp-pchip-bends.csv",
    "bw-raw-vs-temp-pchip-candidates.csv",
    "bw-sg-vs-temp-pchip-candidates.csv",
]

METHOD_SUPPORT_FILES = [
    "method-comparison.csv",
]


@dataclass(frozen=True)
class GalleryDir:
    """One discovered result directory rendered as an image gallery section."""

    kind: str
    path: Path
    rel: Path
    experiment: str
    roi: str
    images: list[Path]
    support_files: list[Path]


@dataclass
class TreeNode:
    """Small in-memory representation of a file tree for HTML rendering."""

    name: str
    path: Path
    is_dir: bool
    children: list["TreeNode"] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline HTML gallery/index for PCHIP and method-comparison results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Source PCHIP project root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pchip-gallery.html"),
        help=(
            "Output HTML file. In normal mode, relative paths are written under --root. "
            "In --export-image-tree mode, relative paths are written under the export root."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="PCHIP analysis gallery",
        help="HTML page title.",
    )
    parser.add_argument(
        "--no-method-comparison",
        action="store_true",
        help="Do not include method-comparison directories. PCHIP is always included.",
    )
    parser.add_argument(
        "--export-image-tree",
        type=Path,
        default=None,
        help=(
            "Create a shareable image-only tree in this destination directory. "
            "Only gallery images are copied, preserving relative paths."
        ),
    )
    parser.add_argument(
        "--force-export-overwrite",
        action="store_true",
        help="Allow writing into an existing export directory. Existing files may be overwritten.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=0,
        help="Maximum depth for full file tree. 0 = unlimited.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude from discovery and full tree rendering. Can be repeated.",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Do not include the clickable file tree.",
    )
    parser.add_argument(
        "--open-all",
        action="store_true",
        help="Render experiment details opened by default.",
    )
    parser.add_argument(
        "--thumbnail-height",
        type=int,
        default=240,
        help="Thumbnail image height in pixels.",
    )
    return parser.parse_args()


def path_to_href(path: Path) -> str:
    """Convert a relative filesystem path to a browser-friendly href."""
    parts = [quote(part) for part in path.as_posix().split("/")]
    return "/".join(parts)


def rel_link(target: Path, *, output_dir_rel_to_root: Path) -> str:
    """Return a relative hyperlink from the HTML output directory to target."""
    if output_dir_rel_to_root == Path("."):
        return path_to_href(target)
    href = Path(os.path.relpath(target, start=output_dir_rel_to_root))
    return path_to_href(href)


def detect_experiment_and_roi(rel_dir: Path) -> tuple[str, str]:
    """Infer experiment and ROI labels from a relative result directory."""
    parts = rel_dir.parts
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


def should_skip_rel_path(path: Path, exclude_names: set[str]) -> bool:
    """Return True if a relative path contains an excluded directory name."""
    return any(part in exclude_names for part in path.parts)


def priority_key(path: Path, priority: list[str]) -> tuple[int, str]:
    """Sort images in stable, analysis-friendly order."""
    name = path.name
    try:
        index = priority.index(name)
    except ValueError:
        index = len(priority)
    return index, name.lower()


def discover_result_dirs(
    root: Path,
    *,
    include_method_comparison: bool,
    exclude_names: set[str] | None = None,
) -> tuple[list[GalleryDir], list[GalleryDir]]:
    """Discover PCHIP and method-comparison result directories."""
    pchip_dirs: list[GalleryDir] = []
    method_dirs: list[GalleryDir] = []
    exclude_names = exclude_names or set()

    # PCHIP is always discovered and included.
    for pchip_dir in sorted(root.rglob("pchip")):
        if not pchip_dir.is_dir():
            continue

        rel = pchip_dir.relative_to(root)
        if should_skip_rel_path(rel, exclude_names):
            continue

        images = sorted(
            [p for p in pchip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: priority_key(p, PCHIP_IMAGE_PRIORITY),
        )
        support_files = [pchip_dir / name for name in PCHIP_SUPPORT_FILES if (pchip_dir / name).is_file()]

        if not images and not support_files:
            continue

        experiment, roi = detect_experiment_and_roi(rel)
        pchip_dirs.append(
            GalleryDir(
                kind="pchip",
                path=pchip_dir,
                rel=rel,
                experiment=experiment,
                roi=roi,
                images=[p.relative_to(root) for p in images],
                support_files=[p.relative_to(root) for p in support_files],
            )
        )

    if include_method_comparison:
        for method_dir in sorted(root.rglob("method-comparison")):
            if not method_dir.is_dir():
                continue

            rel = method_dir.relative_to(root)
            if should_skip_rel_path(rel, exclude_names):
                continue

            images = sorted(
                [p for p in method_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                key=lambda p: priority_key(p, METHOD_IMAGE_PRIORITY),
            )
            support_files = [method_dir / name for name in METHOD_SUPPORT_FILES if (method_dir / name).is_file()]

            if not images and not support_files:
                continue

            experiment, roi = detect_experiment_and_roi(rel)
            method_dirs.append(
                GalleryDir(
                    kind="method-comparison",
                    path=method_dir,
                    rel=rel,
                    experiment=experiment,
                    roi=roi,
                    images=[p.relative_to(root) for p in images],
                    support_files=[p.relative_to(root) for p in support_files],
                )
            )

    return pchip_dirs, method_dirs


def all_gallery_images(*groups: Iterable[GalleryDir]) -> list[Path]:
    """Return unique relative gallery image paths from one or more GalleryDir groups."""
    out: list[Path] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            for rel in item.images:
                key = rel.as_posix()
                if key not in seen:
                    seen.add(key)
                    out.append(rel)
    return out


def is_safe_export_destination(source_root: Path, export_root: Path) -> bool:
    """Avoid exporting inside the source tree or into the source root."""
    source_root = source_root.resolve()
    export_root = export_root.resolve()
    if source_root == export_root:
        return False
    try:
        common = Path(os.path.commonpath([source_root, export_root]))
        if common == source_root:
            return False
    except ValueError:
        pass
    return True


def export_image_tree(
    *,
    source_root: Path,
    export_root: Path,
    image_paths: Iterable[Path],
    force: bool,
) -> int:
    """Copy only selected image files to export_root, preserving relative paths."""
    if not is_safe_export_destination(source_root, export_root):
        raise RuntimeError("export destination must not be source root or inside source tree")

    if export_root.exists() and not export_root.is_dir():
        raise RuntimeError(f"export destination exists but is not a directory: {export_root}")

    if export_root.exists() and any(export_root.iterdir()) and not force:
        raise RuntimeError(
            f"export destination is not empty: {export_root}\n"
            "Use --force-export-overwrite to allow writing into it."
        )

    copied = 0
    for rel in image_paths:
        src = source_root / rel
        dst = export_root / rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def strip_support_for_export(items: Iterable[GalleryDir]) -> list[GalleryDir]:
    """Remove support file links for image-only exports."""
    return [
        GalleryDir(
            kind=item.kind,
            path=item.path,
            rel=item.rel,
            experiment=item.experiment,
            roi=item.roi,
            images=item.images,
            support_files=[],
        )
        for item in items
    ]


def should_exclude_dir(path: Path, exclude_names: set[str]) -> bool:
    return path.name in exclude_names


def build_tree(
    root: Path,
    *,
    max_depth: int = 0,
    exclude_names: set[str] | None = None,
    current_depth: int = 0,
) -> TreeNode:
    """Build a filesystem tree rooted at root."""
    exclude_names = exclude_names or set()
    node = TreeNode(name=root.name or str(root), path=root, is_dir=True)

    if max_depth > 0 and current_depth >= max_depth:
        return node

    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return node

    for entry in entries:
        if entry.is_dir():
            if should_exclude_dir(entry, exclude_names):
                continue
            node.children.append(
                build_tree(
                    entry,
                    max_depth=max_depth,
                    exclude_names=exclude_names,
                    current_depth=current_depth + 1,
                )
            )
        else:
            node.children.append(TreeNode(name=entry.name, path=entry, is_dir=False))

    return node


def render_tree_node(
    node: TreeNode,
    *,
    root: Path,
    output_dir_rel_to_root: Path,
    open_dirs: bool,
) -> str:
    """Render a TreeNode recursively as nested details/ul HTML."""
    if node.is_dir:
        if node.path == root:
            label = html.escape(node.name)
        else:
            rel = node.path.relative_to(root)
            href = rel_link(rel, output_dir_rel_to_root=output_dir_rel_to_root)
            label = f'<a href="{href}">{html.escape(node.name)}/</a>'

        open_attr = " open" if open_dirs else ""
        children_html = "\n".join(
            render_tree_node(
                child,
                root=root,
                output_dir_rel_to_root=output_dir_rel_to_root,
                open_dirs=False,
            )
            for child in node.children
        )
        return (
            f'<details class="tree-dir"{open_attr}>'
            f'<summary>{label}</summary>'
            f'<ul>{children_html}</ul>'
            f'</details>'
        )

    rel = node.path.relative_to(root)
    href = rel_link(rel, output_dir_rel_to_root=output_dir_rel_to_root)
    cls = "image-file" if node.path.suffix.lower() in IMAGE_EXTENSIONS else "file"
    return f'<li class="{cls}"><a href="{href}">{html.escape(node.name)}</a></li>'


def render_support_links(
    support_files: Iterable[Path],
    *,
    output_dir_rel_to_root: Path,
) -> str:
    """Render compact links to CSV/log/report files for one result directory."""
    items = []
    for rel in support_files:
        href = rel_link(rel, output_dir_rel_to_root=output_dir_rel_to_root)
        items.append(f'<a href="{href}">{html.escape(rel.name)}</a>')
    if not items:
        return ""
    return '<div class="support-links">' + " · ".join(items) + "</div>"


def pchip_image_label(filename: str) -> str:
    """Return a human-readable gallery card label for a PCHIP image."""
    stem = filename
    prefix = ""
    if stem.startswith("bw-raw-vs-temp-pchip"):
        prefix = "Raw MGI/PCHIP"
        stem = stem.replace("bw-raw-vs-temp-pchip", "", 1)
    elif stem.startswith("bw-sg-vs-temp-pchip"):
        prefix = "SG MGI/PCHIP"
        stem = stem.replace("bw-sg-vs-temp-pchip", "", 1)
    else:
        return filename

    stem = stem.removesuffix(".png").strip("-")

    if not stem:
        return f"{prefix} base plot"

    labels = {
        "publication-onsets-clean-color": "Publication onsets · clean color",
        "publication-onsets-clean-mono": "Publication onsets · clean mono",
        "publication-onsets-color": "Publication onsets · data color",
        "publication-onsets-mono": "Publication onsets · data mono",
        "publication-clean-color": "Publication clean · color",
        "publication-clean-mono": "Publication clean · mono",
        "publication-main-color": "Publication main · data color",
        "publication-main-mono": "Publication main · data mono",
        "bends": "Standard onset plot",
        "candidate-diagnostics": "Candidate diagnostics",
        "diagnostic-bends": "Full diagnostic plot",
        "diagnostic-main-bends": "Diagnostic main plot",
        "raw-smooth-bends": "Raw + PCHIP onset plot",
        "raw-bends": "Raw-only onset plot",
    }
    return f"{prefix}: {labels.get(stem, stem.replace('-', ' '))}"


def render_image_card(rel: Path, *, output_dir_rel_to_root: Path) -> str:
    """Render one linked image thumbnail card."""
    href = rel_link(rel, output_dir_rel_to_root=output_dir_rel_to_root)
    name = html.escape(rel.name)
    label = html.escape(pchip_image_label(rel.name))
    return f"""
    <figure class="card">
      <h4><a href="{href}">{label}</a></h4>
      <a href="{href}">
        <img loading="lazy" src="{href}" alt="{label}">
      </a>
      <figcaption>{name}</figcaption>
    </figure>
    """


def group_by_experiment(items: Iterable[GalleryDir]) -> dict[str, list[GalleryDir]]:
    """Group result directories by experiment."""
    out: dict[str, list[GalleryDir]] = {}
    for item in items:
        out.setdefault(item.experiment, []).append(item)
    return out


def render_dir_section(
    item: GalleryDir,
    *,
    output_dir_rel_to_root: Path,
) -> str:
    """Render one ROI/result-directory gallery section."""
    href = rel_link(item.rel, output_dir_rel_to_root=output_dir_rel_to_root)
    label = "PCHIP" if item.kind == "pchip" else "method comparison"

    parts: list[str] = []
    parts.append('<section class="roi-section">')
    parts.append(
        f'<h3>{html.escape(item.roi)} <span class="badge">{html.escape(label)}</span> '
        f'<small><a href="{href}">{html.escape(item.rel.as_posix())}/</a></small></h3>'
    )
    parts.append(render_support_links(item.support_files, output_dir_rel_to_root=output_dir_rel_to_root))

    if item.images:
        parts.append('<div class="grid">')
        for image in item.images:
            parts.append(render_image_card(image, output_dir_rel_to_root=output_dir_rel_to_root))
        parts.append('</div>')
    else:
        parts.append('<p class="muted">No image files found in this result directory.</p>')

    parts.append('</section>')
    return "\n".join(parts)


def render_combined_gallery(
    pchip_dirs: list[GalleryDir],
    method_dirs: list[GalleryDir],
    *,
    output_dir_rel_to_root: Path,
    open_details: bool,
) -> str:
    """Render experiment/ROI grouped gallery with PCHIP first, then method comparisons."""
    if not pchip_dirs and not method_dirs:
        return "<p>No PCHIP or method-comparison result directories were found.</p>"

    by_exp_pchip = group_by_experiment(pchip_dirs)
    by_exp_method = group_by_experiment(method_dirs)
    experiments = sorted(set(by_exp_pchip) | set(by_exp_method))

    html_parts: list[str] = []

    for experiment in experiments:
        open_attr = " open" if open_details else ""
        html_parts.append(f'<details class="experiment"{open_attr}>')
        html_parts.append(f'<summary><h2>{html.escape(experiment)}</h2></summary>')

        p_items = sorted(by_exp_pchip.get(experiment, []), key=lambda item: (item.roi, item.rel.as_posix()))
        m_items = sorted(by_exp_method.get(experiment, []), key=lambda item: (item.roi, item.rel.as_posix()))

        # PCHIP first, always.
        for item in p_items:
            html_parts.append(render_dir_section(item, output_dir_rel_to_root=output_dir_rel_to_root))

        for item in m_items:
            html_parts.append(render_dir_section(item, output_dir_rel_to_root=output_dir_rel_to_root))

        html_parts.append('</details>')

    return "\n".join(html_parts)


def make_html(
    *,
    title: str,
    root: Path,
    output_path: Path,
    pchip_dirs: list[GalleryDir],
    method_dirs: list[GalleryDir],
    tree_root: TreeNode | None,
    open_details: bool,
    thumbnail_height: int,
    export_mode: bool,
) -> str:
    """Assemble complete static HTML page."""
    output_dir_rel_to_root = output_path.parent.relative_to(root)
    if output_dir_rel_to_root == Path(""):
        output_dir_rel_to_root = Path(".")

    gallery_html = render_combined_gallery(
        pchip_dirs,
        method_dirs,
        output_dir_rel_to_root=output_dir_rel_to_root,
        open_details=open_details,
    )

    if tree_root is not None:
        tree_html = render_tree_node(
            tree_root,
            root=root,
            output_dir_rel_to_root=output_dir_rel_to_root,
            open_dirs=True,
        )
    else:
        tree_html = "<p>File tree disabled.</p>"

    pchip_count = len(pchip_dirs)
    pchip_image_count = sum(len(item.images) for item in pchip_dirs)
    method_count = len(method_dirs)
    method_image_count = sum(len(item.images) for item in method_dirs)
    mode_text = "image-only export" if export_mode else "full source tree"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --bg: #f7f7f7;
  --panel: #ffffff;
  --text: #222;
  --muted: #666;
  --border: #ddd;
  --badge: #eef2ff;
}}
body {{
  margin: 0;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
}}
header {{
  padding: 1rem 1.25rem;
  background: #222;
  color: white;
}}
header h1 {{
  margin: 0 0 0.25rem 0;
  font-size: 1.45rem;
}}
header p {{
  margin: 0.15rem 0;
  color: #ddd;
}}
main {{
  padding: 1rem;
  max-width: 1700px;
  margin: 0 auto;
}}
nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 1rem;
}}
nav a, .button {{
  display: inline-block;
  padding: 0.35rem 0.6rem;
  border: 1px solid var(--border);
  border-radius: 0.4rem;
  background: var(--panel);
  text-decoration: none;
  color: var(--text);
}}
details {{
  margin: 0.75rem 0;
}}
details.experiment, section.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 0.6rem;
  padding: 0.75rem 1rem;
}}
details.experiment > summary {{
  cursor: pointer;
}}
summary h2 {{
  display: inline;
  font-size: 1.25rem;
}}
.roi-section {{
  border-top: 1px solid var(--border);
  margin-top: 0.75rem;
  padding-top: 0.75rem;
}}
.roi-section h3 {{
  margin: 0 0 0.5rem 0;
}}
.roi-section small {{
  font-weight: normal;
  color: var(--muted);
}}
.badge {{
  display: inline-block;
  padding: 0.1rem 0.35rem;
  border: 1px solid var(--border);
  border-radius: 0.35rem;
  background: var(--badge);
  font-size: 0.75rem;
  font-weight: 600;
  color: #333;
}}
.support-links {{
  margin: 0.4rem 0 0.65rem 0;
  font-size: 0.9rem;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 0.9rem;
}}
.card {{
  margin: 0;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.5rem;
  background: #fff;
}}
.card h4 {{
  margin: 0 0 0.45rem 0;
  font-size: 0.95rem;
  line-height: 1.25;
}}
.card h4 a {{
  color: #111;
}}
.card img {{
  width: 100%;
  height: {thumbnail_height}px;
  object-fit: contain;
  display: block;
  background: #fafafa;
  border: 1px solid #eee;
}}
.card figcaption {{
  margin-top: 0.35rem;
  font-size: 0.86rem;
  color: var(--muted);
  overflow-wrap: anywhere;
}}
.tree {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.88rem;
  line-height: 1.35;
}}
.tree ul {{
  list-style: none;
  margin: 0 0 0 1.2rem;
  padding: 0;
}}
.tree li {{
  margin: 0.12rem 0;
}}
.tree summary {{
  cursor: pointer;
}}
a {{
  color: #0645ad;
}}
.muted {{
  color: var(--muted);
}}
.footer {{
  margin-top: 2rem;
  color: var(--muted);
  font-size: 0.9rem;
}}
</style>
<script>
function openAllDetails() {{
  document.querySelectorAll('details').forEach(d => d.open = true);
}}
function closeExperiments() {{
  document.querySelectorAll('details.experiment').forEach(d => d.open = false);
}}
</script>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>Root: {html.escape(root.resolve().as_posix())}</p>
  <p>Mode: {html.escape(mode_text)}</p>
  <p>PCHIP directories: {pchip_count} · PCHIP images: {pchip_image_count}</p>
  <p>Method-comparison directories: {method_count} · Method-comparison images: {method_image_count}</p>
</header>
<main>
<nav>
  <a href="#gallery">Gallery</a>
  <a href="#tree">File tree</a>
  <button class="button" onclick="openAllDetails()">Open all</button>
  <button class="button" onclick="closeExperiments()">Close experiments</button>
</nav>

<section id="gallery">
  <h1>PCHIP + method-comparison gallery</h1>
  {gallery_html}
</section>

<section id="tree" class="panel">
  <h1>File tree</h1>
  <p class="muted">Every link is relative to this HTML file. The page works without a web server.</p>
  <div class="tree">
  {tree_html}
  </div>
</section>

<p class="footer">
Generated by make-pchip-gallery.py. This is a static offline index.
</p>
</main>
</body>
</html>
"""


def resolve_output_path(*, root: Path, output: Path, export_root: Path | None) -> tuple[Path, Path]:
    """Resolve output path and the gallery root it must live inside."""
    gallery_root = export_root if export_root is not None else root
    assert gallery_root is not None

    if output.is_absolute():
        output_path = output.expanduser().resolve()
    else:
        output_path = (gallery_root / output).expanduser().resolve()

    try:
        output_path.parent.relative_to(gallery_root)
    except ValueError as exc:
        raise RuntimeError(
            "output must be inside the active gallery root so relative links stay portable\n"
            f"gallery root: {gallery_root}\n"
            f"output:       {output_path}"
        ) from exc

    return output_path, gallery_root


def main() -> int:
    args = parse_args()

    source_root = args.root.expanduser().resolve()
    if not source_root.is_dir():
        print(f"[error] root is not a directory: {source_root}", file=sys.stderr)
        return 2

    exclude_names = set(args.exclude_dir)
    exclude_names.update(DEFAULT_EXCLUDE_DIRS)

    include_method = not args.no_method_comparison
    pchip_dirs, method_dirs = discover_result_dirs(
        source_root,
        include_method_comparison=include_method,
        exclude_names=exclude_names,
    )

    export_root: Path | None = None
    export_mode = args.export_image_tree is not None

    if export_mode:
        export_root = args.export_image_tree.expanduser().resolve()
        try:
            copied = export_image_tree(
                source_root=source_root,
                export_root=export_root,
                image_paths=all_gallery_images(pchip_dirs, method_dirs),
                force=args.force_export_overwrite,
            )
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2

        # In image-only export mode, support files are intentionally not copied.
        pchip_dirs_for_html = strip_support_for_export(pchip_dirs)
        method_dirs_for_html = strip_support_for_export(method_dirs)
        gallery_root = export_root
        print(f"Copied gallery images: {copied}")
    else:
        pchip_dirs_for_html = pchip_dirs
        method_dirs_for_html = method_dirs
        gallery_root = source_root

    try:
        output_path, gallery_root = resolve_output_path(
            root=source_root,
            output=args.output,
            export_root=export_root,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    tree_root = None
    if not args.no_tree:
        tree_root = build_tree(
            gallery_root,
            max_depth=args.max_tree_depth,
            exclude_names=exclude_names,
        )

    html_text = make_html(
        title=args.title,
        root=gallery_root,
        output_path=output_path,
        pchip_dirs=pchip_dirs_for_html,
        method_dirs=method_dirs_for_html,
        tree_root=tree_root,
        open_details=args.open_all,
        thumbnail_height=args.thumbnail_height,
        export_mode=export_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    print(f"Source root:              {source_root}")
    if export_mode:
        print(f"Export image-tree root:   {gallery_root}")
    print(f"PCHIP dirs:               {len(pchip_dirs)}")
    print(f"PCHIP images:             {sum(len(item.images) for item in pchip_dirs)}")
    print(f"Method-comparison dirs:   {len(method_dirs)}")
    print(f"Method-comparison images: {sum(len(item.images) for item in method_dirs)}")
    print(f"Wrote HTML index:         {output_path}")
    print()
    print("Open with:")
    print(f"  firefox {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
