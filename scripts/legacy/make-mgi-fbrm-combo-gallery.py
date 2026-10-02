#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-mgi-fbrm-combo-gallery.py
------------------------------

Create a static, offline HTML image gallery for MGI-FBRM combo figures.

The script discovers combo result directories under a PCHIP project tree:

    */tl/roi*/rgb/analysis/combo/

The gallery is intended for fast visual inspection of batch-generated combo
figures. It does not perform analysis. CSV/JSON/report files are included only
as compact support links.

Examples
--------

In-place gallery:

    make-mgi-fbrm-combo-gallery.py \
        --root . \
        --output galleries/mgi-fbrm-combo-gallery.html \
        --open-all

Image-only export gallery:

    make-mgi-fbrm-combo-gallery.py \
        --root . \
        --output mgi-fbrm-combo-gallery.html \
        --export-image-tree /path/to/gallery-share/mgi-fbrm-combo-gallery \
        --force-export-overwrite \
        --open-all
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

PLOT_GROUPS = [
    ("Combo + derivatives", "mgi-fbrm-combo.png"),
    ("Main color", "mgi-fbrm-combo-main.png"),
    ("Main mono", "mgi-fbrm-combo-main-mono.png"),
    ("Main clean mono", "mgi-fbrm-combo-main-clean-mono.png"),
]

SUPPORT_FILES = [
    "mgi-fbrm-combo-data.csv",
    "mgi-fbrm-combo-params.json",
    "mgi-fbrm-combo-report.txt",
]

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "_archive-20260612",
    "_github",
}


@dataclass(frozen=True)
class ComboGalleryDir:
    """One discovered MGI-FBRM combo result directory."""

    path: Path
    rel: Path
    experiment: str
    roi: str
    images: list[Path]
    support_files: list[Path]
    missing_images: list[str]


@dataclass
class TreeNode:
    """Small in-memory representation of a file tree for HTML rendering."""

    name: str
    path: Path
    is_dir: bool
    children: list["TreeNode"] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline HTML gallery for MGI-FBRM combo figures.",
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
        default=Path("galleries/mgi-fbrm-combo-gallery.html"),
        help=(
            "Output HTML file. In normal mode, relative paths are written under --root. "
            "In --export-image-tree mode, relative paths are written under the export root."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MGI-FBRM combo gallery",
        help="HTML page title.",
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
        help="Directory name to exclude from discovery and tree rendering. Can be repeated.",
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
        default=320,
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


def should_skip_path(path: Path, exclude_names: set[str]) -> bool:
    """Return True if any path component should be excluded."""
    return any(part in exclude_names for part in path.parts)


def detect_experiment_and_roi(rel_combo_dir: Path) -> tuple[str, str]:
    """Infer experiment and ROI labels from a relative combo directory."""
    parts = rel_combo_dir.parts
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


def discover_combo_dirs(root: Path, *, exclude_names: set[str]) -> list[ComboGalleryDir]:
    """Discover combo result directories under the project root."""
    items: list[ComboGalleryDir] = []

    for combo_dir in sorted(root.rglob("combo")):
        if not combo_dir.is_dir():
            continue

        rel = combo_dir.relative_to(root)
        if should_skip_path(rel, exclude_names):
            continue

        parts = rel.parts
        if len(parts) < 5:
            continue

        # Expected active workflow:
        # experiment/tl/roi/rgb/analysis/combo
        if parts[-1] != "combo":
            continue
        if len(parts) < 5 or parts[-2] != "analysis" or parts[-3] != "rgb":
            continue
        if "tl" not in parts:
            continue

        images: list[Path] = []
        missing_images: list[str] = []
        for _label, filename in PLOT_GROUPS:
            path = combo_dir / filename
            if path.is_file():
                images.append(path.relative_to(root))
            else:
                missing_images.append(filename)

        support_files = [
            (combo_dir / filename).relative_to(root)
            for filename in SUPPORT_FILES
            if (combo_dir / filename).is_file()
        ]

        if not images and not support_files:
            continue

        experiment, roi = detect_experiment_and_roi(rel)
        items.append(
            ComboGalleryDir(
                path=combo_dir,
                rel=rel,
                experiment=experiment,
                roi=roi,
                images=images,
                support_files=support_files,
                missing_images=missing_images,
            )
        )

    return items


def all_gallery_images(items: Iterable[ComboGalleryDir]) -> list[Path]:
    """Return unique relative gallery image paths."""
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
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


def strip_support_for_export(items: Iterable[ComboGalleryDir]) -> list[ComboGalleryDir]:
    """Remove support file links for image-only exports."""
    return [
        ComboGalleryDir(
            path=item.path,
            rel=item.rel,
            experiment=item.experiment,
            roi=item.roi,
            images=item.images,
            support_files=[],
            missing_images=item.missing_images,
        )
        for item in items
    ]


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
            if entry.name in exclude_names:
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
    """Render compact links to CSV/JSON/report files for one combo directory."""
    items = []
    for rel in support_files:
        href = rel_link(rel, output_dir_rel_to_root=output_dir_rel_to_root)
        items.append(f'<a href="{href}">{html.escape(rel.name)}</a>')
    if not items:
        return ""
    return '<div class="support-links">' + " · ".join(items) + "</div>"


def image_lookup(item: ComboGalleryDir) -> dict[str, Path]:
    """Return image filename -> relative path lookup."""
    return {rel.name: rel for rel in item.images}


def render_image_card(
    *,
    label: str,
    filename: str,
    rel: Path | None,
    output_dir_rel_to_root: Path,
) -> str:
    """Render one linked image thumbnail card, or a missing placeholder."""
    label_escaped = html.escape(label)

    if rel is None:
        return f"""
        <figure class="card missing-card">
          <h4>{label_escaped}</h4>
          <div class="missing-box">missing</div>
          <figcaption>{html.escape(filename)}</figcaption>
        </figure>
        """

    href = rel_link(rel, output_dir_rel_to_root=output_dir_rel_to_root)
    return f"""
    <figure class="card">
      <h4><a href="{href}">{label_escaped}</a></h4>
      <a href="{href}">
        <img loading="lazy" src="{href}" alt="{html.escape(filename)}">
      </a>
      <figcaption>{html.escape(filename)}</figcaption>
    </figure>
    """


def group_by_experiment(items: Iterable[ComboGalleryDir]) -> dict[str, list[ComboGalleryDir]]:
    """Group result directories by experiment."""
    out: dict[str, list[ComboGalleryDir]] = {}
    for item in items:
        out.setdefault(item.experiment, []).append(item)
    return out


def render_combo_section(
    item: ComboGalleryDir,
    *,
    output_dir_rel_to_root: Path,
) -> str:
    """Render one ROI/combo gallery section."""
    href = rel_link(item.rel, output_dir_rel_to_root=output_dir_rel_to_root)
    by_name = image_lookup(item)

    parts: list[str] = []
    parts.append('<section class="roi-section">')
    parts.append(
        f'<h3>{html.escape(item.roi)} <span class="badge">combo</span> '
        f'<small><a href="{href}">{html.escape(item.rel.as_posix())}/</a></small></h3>'
    )
    parts.append(render_support_links(item.support_files, output_dir_rel_to_root=output_dir_rel_to_root))

    parts.append('<div class="grid">')
    for label, filename in PLOT_GROUPS:
        parts.append(
            render_image_card(
                label=label,
                filename=filename,
                rel=by_name.get(filename),
                output_dir_rel_to_root=output_dir_rel_to_root,
            )
        )
    parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def render_gallery(
    items: list[ComboGalleryDir],
    *,
    output_dir_rel_to_root: Path,
    open_details: bool,
) -> str:
    """Render experiment/ROI grouped combo gallery."""
    if not items:
        return "<p>No MGI-FBRM combo directories were found.</p>"

    by_experiment = group_by_experiment(items)
    html_parts: list[str] = []

    for experiment in sorted(by_experiment):
        open_attr = " open" if open_details else ""
        html_parts.append(f'<details class="experiment"{open_attr}>')
        html_parts.append(f"<summary><h2>{html.escape(experiment)}</h2></summary>")

        exp_items = sorted(
            by_experiment[experiment],
            key=lambda item: (item.roi, item.rel.as_posix()),
        )
        for item in exp_items:
            html_parts.append(render_combo_section(item, output_dir_rel_to_root=output_dir_rel_to_root))

        html_parts.append("</details>")

    return "\n".join(html_parts)


def make_html(
    *,
    title: str,
    root: Path,
    output_path: Path,
    combo_dirs: list[ComboGalleryDir],
    tree_root: TreeNode | None,
    open_details: bool,
    thumbnail_height: int,
    export_mode: bool,
) -> str:
    """Assemble complete static HTML page."""
    output_dir_rel_to_root = output_path.parent.relative_to(root)
    if output_dir_rel_to_root == Path(""):
        output_dir_rel_to_root = Path(".")

    gallery_html = render_gallery(
        combo_dirs,
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

    combo_count = len(combo_dirs)
    image_count = sum(len(item.images) for item in combo_dirs)
    missing_count = sum(len(item.missing_images) for item in combo_dirs)
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
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
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
.missing-card {{
  opacity: 0.7;
}}
.missing-box {{
  height: {thumbnail_height}px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--border);
  background: #fafafa;
  color: var(--muted);
  font-style: italic;
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
  <p>Combo directories: {combo_count} · Images: {image_count} · Missing expected images: {missing_count}</p>
</header>
<main>
<nav>
  <a href="#gallery">Gallery</a>
  <a href="#tree">File tree</a>
  <button class="button" onclick="openAllDetails()">Open all</button>
  <button class="button" onclick="closeExperiments()">Close experiments</button>
</nav>

<section id="gallery">
  <h1>MGI-FBRM combo gallery</h1>
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
Generated by make-mgi-fbrm-combo-gallery.py. This is a static offline image gallery.
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

    exclude_names = set(DEFAULT_EXCLUDE_DIRS)
    exclude_names.update(args.exclude_dir)

    combo_dirs = discover_combo_dirs(source_root, exclude_names=exclude_names)

    export_root: Path | None = None
    export_mode = args.export_image_tree is not None

    if export_mode:
        export_root = args.export_image_tree.expanduser().resolve()
        try:
            copied = export_image_tree(
                source_root=source_root,
                export_root=export_root,
                image_paths=all_gallery_images(combo_dirs),
                force=args.force_export_overwrite,
            )
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2

        combo_dirs_for_html = strip_support_for_export(combo_dirs)
        gallery_root = export_root
        print(f"Copied gallery images: {copied}")
    else:
        combo_dirs_for_html = combo_dirs
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
        combo_dirs=combo_dirs_for_html,
        tree_root=tree_root,
        open_details=args.open_all,
        thumbnail_height=args.thumbnail_height,
        export_mode=export_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    image_count = sum(len(item.images) for item in combo_dirs)
    missing_count = sum(len(item.missing_images) for item in combo_dirs)

    print(f"Source root:        {source_root}")
    if export_mode:
        print(f"Export image root:  {gallery_root}")
    print(f"Combo dirs:         {len(combo_dirs)}")
    print(f"Combo images:       {image_count}")
    print(f"Missing expected:   {missing_count}")
    print(f"Wrote HTML index:   {output_path}")
    print()
    print("Open with:")
    print(f"  firefox {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
