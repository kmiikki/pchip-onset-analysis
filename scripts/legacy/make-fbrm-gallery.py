#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-fbrm-gallery.py
--------------------

Create a static, offline HTML gallery for FBRM onset-analysis plots.

The gallery is intentionally independent of plot-internal titles: publication
PNGs can remain clean/titleless, while the HTML cards provide experiment,
plot-type, variant, and filename context.

Supported layouts
-----------------

1. Result-package layout

    root/per_experiment/<experiment>/fbrm-onset-*.png

2. Direct project-tree layout

    root/**/fbrm_onsets/fbrm-onset-*.png

Export mode can copy only the gallery images into a shareable image-only tree,
preserving relative paths from the selected root.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


IMAGE_PATTERN = "fbrm-onset*.png"
DEFAULT_EXCLUDE_DIRS = ("_archive-20260612", "__pycache__")

# Publication variants first. These names are forward-compatible with the FBRM
# publication-plot family that will be generated next. Only existing files are
# shown, so the gallery remains clean before those variants exist.
PLOT_PRIORITY: list[tuple[str, str]] = [
    ("FBRM: Publication onsets · clean color", "fbrm-onset-publication-onsets-clean-color.png"),
    ("FBRM: Publication onsets · clean mono", "fbrm-onset-publication-onsets-clean-mono.png"),
    ("FBRM: Publication onsets · data color", "fbrm-onset-publication-onsets-data-color.png"),
    ("FBRM: Publication onsets · data mono", "fbrm-onset-publication-onsets-data-mono.png"),
    ("FBRM: Publication clean · color", "fbrm-onset-publication-clean-color.png"),
    ("FBRM: Publication clean · mono", "fbrm-onset-publication-clean-mono.png"),
    ("FBRM: Publication main · data color", "fbrm-onset-publication-main-color.png"),
    ("FBRM: Publication main · data mono", "fbrm-onset-publication-main-mono.png"),
    # Legacy/current outputs.
    ("FBRM: Standard publication plot", "fbrm-onset-publication.png"),
    ("FBRM: Candidate diagnostics", "fbrm-onset-candidates.png"),
    ("FBRM: Full diagnostic plot", "fbrm-onset-diagnostic.png"),
]

PLOT_LABEL_BY_NAME = {filename: label for label, filename in PLOT_PRIORITY}
PLOT_RANK_BY_NAME = {filename: rank for rank, (_label, filename) in enumerate(PLOT_PRIORITY)}


@dataclass(frozen=True)
class GalleryImage:
    """One image rendered as one gallery card."""

    experiment: str
    result_label: str
    result_rel: Path
    plot_label: str
    filename: str
    source_path: Path
    rel_from_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline HTML gallery for FBRM onset plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help=(
            "FBRM result root. Supports either root/per_experiment/<experiment>/ "
            "or a direct project tree containing fbrm_onsets/ directories."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gallery/fbrm-gallery.html"),
        help=(
            "Output HTML path. Relative paths are interpreted relative to --root "
            "in in-place mode, or relative to --export-image-tree in export mode."
        ),
    )
    parser.add_argument(
        "--title",
        default="FBRM onset gallery",
        help="HTML page title.",
    )
    parser.add_argument(
        "--export-image-tree",
        type=Path,
        default=None,
        help="Optional export root for a shareable image-only gallery.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing copied images in export-image-tree mode.",
    )
    parser.add_argument(
        "--no-file-tree",
        action="store_true",
        help="Do not include the linked file tree section.",
    )
    parser.add_argument(
        "--open-all",
        action="store_true",
        help="Render all experiment details opened by default.",
    )
    parser.add_argument(
        "--thumbnail-height",
        type=int,
        default=240,
        help="Maximum thumbnail height in pixels.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=0,
        help="Maximum depth for file tree. 0 = unlimited.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=list(DEFAULT_EXCLUDE_DIRS),
        help="Directory name to exclude from discovery and file-tree rendering. Can be repeated.",
    )
    return parser.parse_args()


def url_href(path: Path, base_dir: Path) -> str:
    """Return URL-safe relative href from base_dir to path."""
    rel = Path(os.path.relpath(path, start=base_dir))
    return quote(rel.as_posix())


def path_has_excluded_part(path: Path, exclude_names: set[str]) -> bool:
    """Return True if path contains an excluded directory name."""
    return any(part in exclude_names for part in path.parts)


def infer_experiment_from_path(path: Path, root: Path) -> str:
    """Infer experiment label from a path relative to root."""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = path.parts

    for part in parts:
        if re.search(r"\d{8}_ex_\d+", part, flags=re.IGNORECASE):
            return part
    for part in parts:
        if re.search(r"\bex_\d+\b", part, flags=re.IGNORECASE):
            return part
    if len(parts) >= 2:
        return parts[0]
    return root.name or "(root)"



def infer_result_label_from_path(result_dir: Path, root: Path, experiment: str) -> tuple[str, Path]:
    """Return a compact result-section label and relative result directory.

    Direct project trees may contain several fbrm_onsets directories for the
    same experiment, for example different ROI choices. The gallery therefore
    must not flatten everything under only the experiment name. This helper
    gives each result directory its own visible section label.
    """
    try:
        rel = result_dir.relative_to(root)
    except ValueError:
        rel = Path(result_dir.name)

    parts = rel.parts
    roi = None
    if "tl" in parts:
        i = parts.index("tl")
        if i + 1 < len(parts):
            roi = parts[i + 1]
    if roi is None:
        for part in parts:
            if part.lower().startswith("roi"):
                roi = part
                break

    # For result-package layout rel may be just per_experiment/<experiment>.
    if roi is None:
        roi = "fbrm_onsets"

    label = f"{roi} · {rel.as_posix()}"
    return label, rel

def label_for_filename(filename: str) -> str:
    """Return a human-readable card title for an FBRM plot filename."""
    if filename in PLOT_LABEL_BY_NAME:
        return PLOT_LABEL_BY_NAME[filename]

    stem = filename
    if stem.startswith("fbrm-onset-"):
        stem = stem[len("fbrm-onset-") :]
    if stem.endswith(".png"):
        stem = stem[:-4]
    text = stem.replace("-", " ").strip()
    return "FBRM: " + (text[:1].upper() + text[1:] if text else filename)


def image_sort_key(image: GalleryImage) -> tuple[int, str]:
    """Sort publication variants first, then legacy/unknown images by name."""
    return (PLOT_RANK_BY_NAME.get(image.filename, len(PLOT_PRIORITY)), image.filename.lower())


def images_from_experiment_dir(root: Path, exp_dir: Path, experiment: str) -> list[GalleryImage]:
    """Collect gallery images from one FBRM result directory."""
    out: list[GalleryImage] = []
    result_label, result_rel = infer_result_label_from_path(exp_dir, root, experiment)
    for source_path in sorted(exp_dir.glob(IMAGE_PATTERN)):
        if not source_path.is_file():
            continue
        rel_from_root = source_path.relative_to(root)
        out.append(
            GalleryImage(
                experiment=experiment,
                result_label=result_label,
                result_rel=result_rel,
                plot_label=label_for_filename(source_path.name),
                filename=source_path.name,
                source_path=source_path,
                rel_from_root=rel_from_root,
            )
        )
    return sorted(out, key=image_sort_key)


def discover_images(root: Path, exclude_names: set[str]) -> tuple[list[GalleryImage], str]:
    """Discover FBRM onset plot images from supported layouts."""
    per_experiment = root / "per_experiment"
    images: list[GalleryImage] = []

    if per_experiment.is_dir():
        experiments = sorted(p for p in per_experiment.iterdir() if p.is_dir())
        for exp_dir in experiments:
            if path_has_excluded_part(exp_dir.relative_to(root), exclude_names):
                continue
            images.extend(images_from_experiment_dir(root, exp_dir, exp_dir.name))
        return images, "per_experiment layout"

    # Direct project tree fallback. This is useful when running the gallery
    # directly against /path/to/data instead of an exported result package.
    fbrm_dirs = sorted(
        p
        for p in root.rglob("fbrm_onsets")
        if p.is_dir() and not path_has_excluded_part(p.relative_to(root), exclude_names)
    )

    for fbrm_dir in fbrm_dirs:
        experiment = infer_experiment_from_path(fbrm_dir, root)
        images.extend(images_from_experiment_dir(root, fbrm_dir, experiment))

    # Also support a root that itself is one fbrm_onsets-like directory.
    if not fbrm_dirs:
        direct_images = images_from_experiment_dir(root, root, infer_experiment_from_path(root, root))
        images.extend(direct_images)

    return images, "direct project-tree layout"


def copy_images_for_export(
    images: list[GalleryImage],
    export_root: Path,
    *,
    overwrite: bool,
) -> list[GalleryImage]:
    """Copy gallery images into export_root, preserving relative paths."""
    copied_images: list[GalleryImage] = []

    for image in images:
        dest = export_root / image.rel_from_root
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not overwrite:
            raise SystemExit(
                "ERROR: export destination already exists. Use --overwrite to replace it:\n"
                f"{dest}"
            )

        shutil.copy2(image.source_path, dest)

        copied_images.append(
            GalleryImage(
                experiment=image.experiment,
                result_label=image.result_label,
                result_rel=image.result_rel,
                plot_label=image.plot_label,
                filename=image.filename,
                source_path=dest,
                rel_from_root=image.rel_from_root,
            )
        )

    return copied_images


def render_file_tree(
    root: Path,
    base_dir: Path,
    *,
    exclude_names: set[str],
    max_depth: int,
) -> str:
    """Render a simple linked file tree rooted at root."""

    def render_node(path: Path, depth: int) -> str:
        rel = path.relative_to(root)
        label = html.escape(root.name if rel.as_posix() == "." else path.name)

        if path.is_dir():
            if max_depth > 0 and depth >= max_depth:
                return f"<li><strong>{label}/</strong></li>"

            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            rendered_children: list[str] = []
            for child in children:
                if child.is_dir() and child.name in exclude_names:
                    continue
                rendered_children.append(render_node(child, depth + 1))
            children_html = "\n".join(rendered_children)

            if rel.as_posix() == ".":
                return f"<li><strong>{label}/</strong><ul>{children_html}</ul></li>"

            href = url_href(path, base_dir)
            return f'<li><a href="{href}">{label}/</a><ul>{children_html}</ul></li>'

        href = url_href(path, base_dir)
        return f'<li><a href="{href}">{label}</a></li>'

    return f"<ul>{render_node(root, 0)}</ul>"


def render_html(
    *,
    title: str,
    root: Path,
    output: Path,
    images: list[GalleryImage],
    mode_text: str,
    layout_text: str,
    include_file_tree: bool,
    open_all: bool,
    thumbnail_height: int,
    exclude_names: set[str],
    max_tree_depth: int,
) -> str:
    """Render complete HTML gallery."""
    by_experiment: dict[str, list[GalleryImage]] = {}
    for image in images:
        by_experiment.setdefault(image.experiment, []).append(image)
    experiments = sorted(by_experiment)

    thumb_h = max(120, int(thumbnail_height))

    body: list[str] = []
    body.append("<!doctype html>")
    body.append('<html lang="en">')
    body.append("<head>")
    body.append('<meta charset="utf-8">')
    body.append(f"<title>{html.escape(title)}</title>")
    body.append(
        f"""<style>
:root {{
  --border: #d9d9d9;
  --muted: #666;
  --bg: #f7f7f7;
  --card: #fff;
  --link: #0645ad;
}}
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  line-height: 1.4;
  background: #fff;
  color: #111;
}}
header {{
  background: #202020;
  color: #fff;
  padding: 14px 18px;
}}
header h1 {{ margin: 0 0 0.35em 0; }}
header p {{ margin: 0.15em 0; }}
main {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 18px;
}}
nav {{
  display: flex;
  gap: 8px;
  margin: 0 0 20px 0;
  flex-wrap: wrap;
}}
button, .button {{
  display: inline-block;
  font-size: 0.9rem;
  padding: 5px 9px;
  border: 1px solid var(--border);
  background: var(--bg);
  border-radius: 4px;
  cursor: pointer;
  color: #111;
}}
details.experiment {{
  border: 1px solid var(--border);
  border-radius: 7px;
  margin: 12px 0;
  background: #fff;
}}
details.experiment > summary {{
  cursor: pointer;
  padding: 10px 12px;
  background: var(--bg);
  border-radius: 7px;
}}
details.experiment[open] > summary {{
  border-bottom: 1px solid var(--border);
  border-radius: 7px 7px 0 0;
}}
summary h2 {{
  display: inline;
  font-size: 1.05rem;
  margin: 0;
}}
.gallery-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 14px;
  padding: 14px;
}}
.card {{
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  padding: 8px;
  margin: 0;
}}
.card h3 {{
  font-size: 0.92rem;
  margin: 0 0 8px 0;
}}
.card img {{
  width: 100%;
  max-height: {thumb_h}px;
  object-fit: contain;
  border: 1px solid #eee;
  background: #fff;
}}
figcaption {{
  color: var(--muted);
  font-size: 0.82rem;
  margin-top: 4px;
  word-break: break-word;
}}
a {{ color: var(--link); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.badge {{
  font-size: 0.75rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 5px;
  background: #fafafa;
  color: #333;
}}
.muted {{ color: var(--muted); }}
#file-tree {{
  margin-top: 24px;
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 14px;
  background: #fff;
}}
#file-tree ul {{ margin-top: 0.25em; }}
#file-tree li {{ margin: 0.2em 0; }}
footer {{
  margin-top: 24px;
  color: var(--muted);
  font-size: 0.9rem;
}}
</style>
<script>
function openAllExperiments() {{
  document.querySelectorAll("details.experiment").forEach(d => d.open = true);
}}
function closeAllExperiments() {{
  document.querySelectorAll("details.experiment").forEach(d => d.open = false);
}}
</script>"""
    )
    body.append("</head>")
    body.append("<body>")
    body.append("<header>")
    body.append(f"<h1>{html.escape(title)}</h1>")
    body.append(f"<p>Root: {html.escape(root.as_posix())}</p>")
    body.append(f"<p>Mode: {html.escape(mode_text)} · Layout: {html.escape(layout_text)}</p>")
    body.append(f"<p>Experiments: {len(experiments)} · Images: {len(images)}</p>")
    body.append("</header>")
    body.append("<main>")
    body.append(
        '<nav>'
        '<a class="button" href="#gallery">Gallery</a>'
        '<a class="button" href="#file-tree">File tree</a>'
        '<button onclick="openAllExperiments()">Open all</button>'
        '<button onclick="closeAllExperiments()">Close experiments</button>'
        '</nav>'
    )

    body.append('<section id="gallery">')
    body.append(f"<h1>{html.escape(title)}</h1>")
    body.append(
        "<p class=\"muted\">Publication PNGs can remain titleless; "
        "the gallery card titles identify the plot type and variant. Multiple ROI/result directories are shown as separate sections under each experiment.</p>"
    )

    if not experiments:
        body.append("<p>No FBRM onset images found.</p>")

    for idx, experiment in enumerate(experiments):
        open_attr = " open" if open_all or idx == 0 else ""
        body.append(f'<details class="experiment"{open_attr}>')
        body.append(f"<summary><h2>{html.escape(experiment)}</h2></summary>")

        result_groups: dict[str, list[GalleryImage]] = {}
        for image in by_experiment[experiment]:
            result_groups.setdefault(image.result_rel.as_posix(), []).append(image)

        for result_key in sorted(result_groups):
            group_images = sorted(result_groups[result_key], key=image_sort_key)
            first = group_images[0]
            result_href = url_href(root / first.result_rel, output.parent)
            body.append('<section class="result-section">')
            body.append(
                f'<h3>{html.escape(first.result_label.split(" · ", 1)[0])} '
                f'<span class="badge">FBRM</span></h3>'
            )
            body.append(
                f'<p class="path"><a href="{result_href}">{html.escape(first.result_rel.as_posix())}/</a></p>'
            )
            body.append('<div class="gallery-grid">')

            for image in group_images:
                href = url_href(image.source_path, output.parent)
                full_label = f"{image.experiment} / {image.result_rel.as_posix()} / {image.filename}"
                body.append('<figure class="card">')
                body.append(f'<h3><a href="{href}">{html.escape(image.plot_label)}</a></h3>')
                body.append(
                    f'<a href="{href}"><img loading="lazy" src="{href}" '
                    f'alt="{html.escape(full_label)}"></a>'
                )
                body.append(f"<figcaption>{html.escape(image.filename)}</figcaption>")
                body.append("</figure>")

            body.append("</div>")
            body.append("</section>")

        body.append("</details>")

    body.append("</section>")

    if include_file_tree:
        body.append('<section id="file-tree">')
        body.append("<h1>File tree</h1>")
        body.append("<p>Every link is relative to this HTML file. The page works without a web server.</p>")
        body.append(
            render_file_tree(
                root,
                output.parent,
                exclude_names=exclude_names,
                max_depth=max_tree_depth,
            )
        )
        body.append("</section>")

    body.append("<footer>")
    body.append("<p>Generated by make-fbrm-gallery.py.</p>")
    body.append("</footer>")
    body.append("</main>")
    body.append("</body>")
    body.append("</html>")

    return "\n".join(body)


def main() -> int:
    args = parse_args()

    source_root = args.root.expanduser().resolve()
    exclude_names = set(args.exclude_dir or [])
    images, layout_text = discover_images(source_root, exclude_names)

    if args.export_image_tree is not None:
        gallery_root = args.export_image_tree.expanduser().resolve()
        gallery_root.mkdir(parents=True, exist_ok=True)
        gallery_images = copy_images_for_export(
            images,
            gallery_root,
            overwrite=args.overwrite,
        )
        mode_text = "image-only export"
    else:
        gallery_root = source_root
        gallery_images = images
        mode_text = "in-place gallery"

    output = args.output
    if not output.is_absolute():
        output = gallery_root / output
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    html_text = render_html(
        title=args.title,
        root=gallery_root,
        output=output,
        images=gallery_images,
        mode_text=mode_text,
        layout_text=layout_text,
        include_file_tree=not args.no_file_tree,
        open_all=bool(args.open_all),
        thumbnail_height=int(args.thumbnail_height),
        exclude_names=exclude_names,
        max_tree_depth=int(args.max_tree_depth),
    )

    output.write_text(html_text, encoding="utf-8")

    print(f"Wrote:        {output}")
    print(f"Mode:         {mode_text}")
    print(f"Layout:       {layout_text}")
    print(f"Source root:  {source_root}")
    print(f"Gallery root: {gallery_root}")
    print(f"Images:       {len(gallery_images)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
