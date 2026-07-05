#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run-bw-pchip-workflow-v2.py
---------------------------

Portable workflow runner for BW vs. temperature PCHIP T1/T2 onset analysis.

The workflow has two independent layers:

1. Analysis workflow for one directory

   analysis-dir/
       rgb-tr.csv              required by default
       rgb-tr-sg.csv           optional by default

   The runner creates/uses:

   analysis-dir/
       pchip/
           bw-raw-vs-temp-pchip.csv
           bw-raw-vs-temp-pchip.png
           bw-sg-vs-temp-pchip.csv
           bw-sg-vs-temp-pchip.png
           bw-*-vs-temp-pchip-bends.csv
           bw-*-vs-temp-pchip-bends.png
           bw-*-vs-temp-pchip-candidates.csv      when requested
           bw-*-candidate-diagnostics.png         when requested
           bw-pchip-bend-summary.csv
           bw-pchip-workflow.log

2. Optional batch discovery

   The user may provide --root and --discover-pattern to find many analysis
   directories. The discovery pattern points to raw input CSV files. Their
   parent directories are treated as analysis directories.

Minimum input for the default single-directory workflow
------------------------------------------------------
Required file:
    rgb-tr.csv

Required columns in rgb-tr.csv:
    Tr (°C), BW

Optional SG-control file:
    rgb-tr-sg.csv

Required columns in rgb-tr-sg.csv:
    Tr (°C), BW_smooth

Typical single-directory use
----------------------------
Assumption: xy-pchip.py and bw-pchip-onsets-t1t2.py are available from PATH.
Run from a directory containing rgb-tr.csv:

    python3 run-bw-pchip-workflow-v2.py \
        --analysis-dir . \
        --plot-set all \
        --candidate-diagnostics \
        --execute

Typical batch use for the original T7 layout
--------------------------------------------
Assumption: xy-pchip.py and bw-pchip-onsets-t1t2.py are available from PATH.

    python3 run-bw-pchip-workflow-v2.py \
        --root $PROJECT_ROOT \
        --discover-pattern "*_ex_*/tl/roi*/rgb/analysis/rgb-tr.csv" \
        --plot-set all \
        --candidate-diagnostics \
        --execute

The script path options still exist for development and special cases:
    --xy-pchip /path/to/xy-pchip.py
    --onset-script /path/to/bw-pchip-onsets-t1t2.py

Design principles
-----------------
- Dry-run by default. Add --execute to write files and run commands.
- Single-directory mode is the default mental model for ordinary users.
- Batch discovery is a separate layer and can be adapted with --discover-pattern.
- Input filenames, column names, and output directory are configurable.
- Critical scripts are expected to be available from PATH by default; explicit script paths remain optional.
- Commands are logged in each pchip output directory when --execute is used.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AnalysisDir:
    """One analysis directory to process."""

    label: str
    path: Path


@dataclass
class WorkflowResult:
    """Result of processing one analysis directory."""

    label: str
    analysis_dir: Path
    pchip_dir: Path
    raw_ok: bool = False
    sg_ok: bool = False
    onset_ok: bool = False
    skipped: bool = False
    error: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BW vs. temperature PCHIP + T1/T2 onset workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_argument_group("workflow mode")
    mode.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Process one analysis directory containing the raw input CSV. "
            "If omitted and --root is not given, the current directory is used."
        ),
    )
    mode.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Batch root directory. When given, --discover-pattern is evaluated "
            "relative to this directory and each matching file's parent is "
            "processed as one analysis directory."
        ),
    )
    mode.add_argument(
        "--discover-pattern",
        type=str,
        default="*_ex_*/tl/roi*/rgb/analysis/rgb-tr.csv",
        help="Glob pattern for batch discovery, relative to --root.",
    )
    mode.add_argument(
        "--only",
        type=str,
        default=None,
        help="Batch/single filter: process only paths containing this text.",
    )
    mode.add_argument(
        "--max-dirs",
        type=int,
        default=0,
        help="Batch mode: process at most N directories. 0 = no limit.",
    )

    inputs = parser.add_argument_group("input and output files")
    inputs.add_argument(
        "--input-raw",
        type=str,
        default="rgb-tr.csv",
        help="Raw BW input CSV filename inside each analysis directory.",
    )
    inputs.add_argument(
        "--input-sg",
        type=str,
        default="rgb-tr-sg.csv",
        help="Optional SG-smoothed input CSV filename inside each analysis directory.",
    )
    inputs.add_argument(
        "--pchip-dir",
        type=str,
        default="pchip",
        help="Output subdirectory created inside each analysis directory.",
    )
    inputs.add_argument(
        "--raw-outstem",
        type=str,
        default="bw-raw-vs-temp-pchip",
        help="Output stem for raw PCHIP files inside --pchip-dir.",
    )
    inputs.add_argument(
        "--sg-outstem",
        type=str,
        default="bw-sg-vs-temp-pchip",
        help="Output stem for SG PCHIP files inside --pchip-dir.",
    )

    columns = parser.add_argument_group("CSV columns")
    columns.add_argument("--xcol", type=str, default="Tr (°C)", help="Temperature/X column name.")
    columns.add_argument("--ycol-raw", type=str, default="BW", help="Raw BW/Y column name.")
    columns.add_argument("--ycol-sg", type=str, default="BW_smooth", help="SG BW/Y column name.")
    columns.add_argument(
        "--xlabel",
        type=str,
        default="Temperature (°C) - cooling",
        help="X-axis label forwarded to xy-pchip.py.",
    )
    columns.add_argument("--ylabel-raw", type=str, default="BW", help="Raw plot Y-axis label.")
    columns.add_argument("--ylabel-sg", type=str, default="BW_smooth", help="SG plot Y-axis label.")

    scripts = parser.add_argument_group("script paths")
    scripts.add_argument(
        "--xy-pchip",
        type=Path,
        default=Path("xy-pchip.py"),
        help="xy-pchip.py command or path. Default assumes xy-pchip.py is available from PATH.",
    )
    scripts.add_argument(
        "--onset-script",
        "--bends-script",
        dest="onset_script",
        type=Path,
        default=Path("bw-pchip-onsets-t1t2.py"),
        help="Onset/bend detection command or path. Default assumes bw-pchip-onsets-t1t2.py is available from PATH. --bends-script is kept as an alias.",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument("--execute", action="store_true", help="Actually run commands. Default is dry-run.")
    execution.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing PCHIP CSV/PNG files. Default: skip existing PCHIP generation.",
    )
    execution.add_argument("--verbose", action="store_true", help="Print commands and command output.")
    execution.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python interpreter used for child scripts. Default uses the current "
            "interpreter. Use 'python3' when generating portable example logs."
        ),
    )

    onset = parser.add_argument_group("onset-script options")
    onset.add_argument("--bends", type=int, default=2, help="Forwarded to onset script.")
    onset.add_argument(
        "--select-mode",
        choices=["first-valid", "strongest"],
        default=None,
        help="Forwarded to onset script when set.",
    )
    onset.add_argument(
        "--plot-set",
        choices=["default", "single", "all"],
        default="default",
        help="Forwarded to onset script.",
    )
    onset.add_argument(
        "--plot-mode",
        choices=["smooth", "raw", "raw-smooth", "diagnostic-main", "diagnostic", "candidates"],
        default="smooth",
        help="Forwarded to onset script when --plot-set single is used.",
    )
    onset.add_argument("--candidate-diagnostics", action="store_true", help="Forward to onset script.")
    onset.add_argument("--publication-plots", action="store_true", help="Forward --publication-plots to onset script.")
    onset.add_argument("--bw", action="store_true", help="Forward --bw to onset script.")
    onset.add_argument("--no-title", action="store_true", help="Forward --no-title to onset script.")
    onset.add_argument("--dpi", type=int, default=300, help="Forwarded to onset script where applicable.")

    reporting = parser.add_argument_group("batch reports")
    reporting.add_argument(
        "--summary",
        type=Path,
        default=Path("bw-pchip-workflow-summary.csv"),
        help="Batch/single workflow summary CSV. Relative paths are written under the run root.",
    )
    reporting.add_argument(
        "--report",
        type=Path,
        default=Path("bw-pchip-workflow-report.txt"),
        help="Human-readable workflow report. Relative paths are written under the run root.",
    )

    return parser.parse_args()


def resolve_script_path(path: Path) -> Path:
    """
    Resolve a script path.

    Resolution order:
    1. absolute path as given
    2. relative path from current working directory
    3. executable/script found from PATH

    Returning an unresolved path is allowed; the later subprocess call will then
    produce a clear execution error.
    """
    if path.is_absolute():
        return path

    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate.resolve()

    found = shutil.which(str(path))
    if found:
        return Path(found).resolve()

    return path


def display_path(path: Path) -> str:
    """Return a shell-friendly path string for dry-run display."""
    return shlex.quote(str(path))


def discover_analysis_dirs(
    *,
    root: Path | None,
    analysis_dir: Path | None,
    discover_pattern: str,
    only: str | None,
    max_dirs: int,
    input_raw: str,
) -> tuple[Path, list[AnalysisDir]]:
    """
    Resolve either a single analysis directory or a batch list.

    Returns
    -------
    tuple[Path, list[AnalysisDir]]
        The report root and the analysis directories to process.

    Notes
    -----
    - Single mode does not require any special directory structure.
    - Batch mode discovers raw input files and uses their parent directories.
    """
    if root is None:
        base = (analysis_dir or Path.cwd()).expanduser().resolve()
        if only and only not in str(base):
            return base, []
        return base, [AnalysisDir(label=base.name or str(base), path=base)]

    run_root = root.expanduser().resolve()
    found: list[AnalysisDir] = []

    # Use glob relative to root. The default pattern matches the original T7
    # layout; users can replace it with e.g. "**/rgb-tr.csv".
    for raw_csv in sorted(run_root.glob(discover_pattern)):
        if raw_csv.name != Path(input_raw).name and Path(discover_pattern).name == Path(input_raw).name:
            # Defensive guard for unusual glob behavior; normally unnecessary.
            continue

        analysis_path = raw_csv.parent
        if only and only not in str(analysis_path):
            continue

        try:
            label = str(analysis_path.relative_to(run_root))
        except ValueError:
            label = str(analysis_path)

        found.append(AnalysisDir(label=label, path=analysis_path))
        if max_dirs > 0 and len(found) >= max_dirs:
            break

    return run_root, found


def command_to_string(cmd: Sequence[str], cwd: Path) -> str:
    """Format a command for logs and dry-runs."""
    quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
    return f"(cd {display_path(cwd)} && {quoted_cmd})"


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    execute: bool,
    verbose: bool,
    log_file: Path | None = None,
) -> tuple[bool, str]:
    """Run or print a command. Return ``(ok, combined_stdout_stderr)``."""
    printable = command_to_string(cmd, cwd)

    if verbose or not execute:
        print(f"$ {printable}")

    if not execute:
        return True, ""

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output = proc.stdout or ""

    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"$ {printable}\n")
            f.write(output)
            if output and not output.endswith("\n"):
                f.write("\n")
            f.write(f"[exit code: {proc.returncode}]\n\n")

    if verbose and output:
        print(output, end="" if output.endswith("\n") else "\n")

    return proc.returncode == 0, output


def ensure_pchip_dir(analysis_dir: Path, pchip_subdir: str, execute: bool) -> Path:
    """Create and return the PCHIP output directory."""
    pchip_dir = analysis_dir / pchip_subdir
    if execute:
        pchip_dir.mkdir(parents=True, exist_ok=True)
    return pchip_dir


def run_xy_pchip(
    *,
    analysis_dir: Path,
    pchip_dir: Path,
    xy_pchip: Path,
    input_csv_name: str,
    output_stem_name: str,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    python: str,
    execute: bool,
    overwrite: bool,
    verbose: bool,
    log_file: Path,
) -> bool:
    """Run xy-pchip.py for one input CSV if the input exists."""
    input_csv = analysis_dir / input_csv_name
    if not input_csv.exists():
        if verbose:
            print(f"SKIP missing input: {input_csv}")
        return False

    out_csv = pchip_dir / f"{output_stem_name}.csv"
    out_png = pchip_dir / f"{output_stem_name}.png"

    if out_csv.exists() and out_png.exists() and not overwrite:
        if verbose:
            print(f"SKIP existing PCHIP: {out_csv}")
        return True

    # xy-pchip.py writes outstem relative to cwd. Keep the command portable by
    # using a relative outstem when pchip_dir is inside analysis_dir.
    try:
        outstem = str((pchip_dir / output_stem_name).relative_to(analysis_dir))
    except ValueError:
        outstem = str(pchip_dir / output_stem_name)

    cmd = [
        python,
        str(xy_pchip),
        input_csv_name,
        "--xcol",
        xcol,
        "--ycol",
        ycol,
        "--reverse-x",
        "--save-csv",
        "--outstem",
        outstem,
        "--xlabel",
        xlabel,
        "--ylabel",
        ylabel,
        "--title",
        title,
    ]

    ok, _ = run_command(cmd, cwd=analysis_dir, execute=execute, verbose=verbose, log_file=log_file)
    return ok


def build_onset_command(onset_script: Path, args: argparse.Namespace, *, python: str) -> list[str]:
    """Build the onset-script command line."""
    cmd = [
        python,
        str(onset_script),
        "--plot-set",
        args.plot_set,
        "--plot-mode",
        args.plot_mode,
        "--dpi",
        str(args.dpi),
        "--bends",
        str(args.bends),
    ]

    if args.select_mode is not None:
        cmd.extend(["--select-mode", args.select_mode])
    if args.candidate_diagnostics:
        cmd.append("--candidate-diagnostics")
    if args.publication_plots:
        cmd.append("--publication-plots")
    if args.bw:
        cmd.append("--bw")
    if args.no_title:
        cmd.append("--no-title")

    return cmd


def run_onset_script(
    *,
    pchip_dir: Path,
    onset_script: Path,
    execute: bool,
    verbose: bool,
    log_file: Path,
    args: argparse.Namespace,
    python: str,
) -> bool:
    """Run the onset/T1-T2 script inside the pchip directory."""
    if execute:
        pchip_csvs = sorted(pchip_dir.glob("*-vs-temp-pchip.csv"))
        if not pchip_csvs:
            if verbose:
                print(f"SKIP onset: no PCHIP CSV files in {pchip_dir}")
            return False

    cmd = build_onset_command(onset_script, args, python=python)
    ok, _ = run_command(cmd, cwd=pchip_dir, execute=execute, verbose=verbose, log_file=log_file)
    return ok


def process_analysis_dir(
    item: AnalysisDir,
    *,
    xy_pchip: Path,
    onset_script: Path,
    args: argparse.Namespace,
) -> WorkflowResult:
    """Process one analysis directory."""
    result = WorkflowResult(
        label=item.label,
        analysis_dir=item.path,
        pchip_dir=item.path / args.pchip_dir,
    )

    if not item.path.exists():
        result.error = f"analysis directory does not exist: {item.path}"
        return result

    pchip_dir = ensure_pchip_dir(item.path, args.pchip_dir, args.execute)
    result.pchip_dir = pchip_dir

    log_file = pchip_dir / "bw-pchip-workflow.log"
    if args.execute:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            "BW PCHIP workflow log\n"
            "====================\n\n"
            f"analysis_dir: {item.path}\n"
            f"pchip_dir:    {pchip_dir}\n\n",
            encoding="utf-8",
        )

    try:
        result.raw_ok = run_xy_pchip(
            analysis_dir=item.path,
            pchip_dir=pchip_dir,
            xy_pchip=xy_pchip,
            input_csv_name=args.input_raw,
            output_stem_name=args.raw_outstem,
            xcol=args.xcol,
            ycol=args.ycol_raw,
            xlabel=args.xlabel,
            ylabel=args.ylabel_raw,
            title="BW vs. Temperature - raw + PCHIP",
            python=args.python,
            execute=args.execute,
            overwrite=args.overwrite,
            verbose=args.verbose,
            log_file=log_file,
        )

        result.sg_ok = run_xy_pchip(
            analysis_dir=item.path,
            pchip_dir=pchip_dir,
            xy_pchip=xy_pchip,
            input_csv_name=args.input_sg,
            output_stem_name=args.sg_outstem,
            xcol=args.xcol,
            ycol=args.ycol_sg,
            xlabel=args.xlabel,
            ylabel=args.ylabel_sg,
            title="BW vs. Temperature - SG + PCHIP",
            python=args.python,
            execute=args.execute,
            overwrite=args.overwrite,
            verbose=args.verbose,
            log_file=log_file,
        )

        result.onset_ok = run_onset_script(
            pchip_dir=pchip_dir,
            onset_script=onset_script,
            execute=args.execute,
            verbose=args.verbose,
            log_file=log_file,
            args=args,
            python=args.python,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        result.error = str(exc)

    return result


def write_batch_summary(path: Path, results: Iterable[WorkflowResult]) -> None:
    """Write workflow summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)

    fieldnames = [
        "label",
        "analysis_dir",
        "pchip_dir",
        "raw_ok",
        "sg_ok",
        "onset_ok",
        "skipped",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "label": r.label,
                    "analysis_dir": str(r.analysis_dir),
                    "pchip_dir": str(r.pchip_dir),
                    "raw_ok": r.raw_ok,
                    "sg_ok": r.sg_ok,
                    "onset_ok": r.onset_ok,
                    "skipped": r.skipped,
                    "error": r.error,
                }
            )


def write_batch_report(path: Path, results: Iterable[WorkflowResult], *, execute: bool) -> None:
    """Write human-readable workflow report."""
    rows = list(results)
    ok_count = sum(1 for r in rows if r.raw_ok and r.onset_ok and not r.error)
    sg_count = sum(1 for r in rows if r.sg_ok)
    err_count = sum(1 for r in rows if r.error or not r.onset_ok)

    lines: list[str] = []
    lines.append("BW PCHIP Workflow Report")
    lines.append("========================")
    lines.append("")
    lines.append(f"mode:              {'EXECUTE' if execute else 'DRY-RUN'}")
    lines.append(f"analysis dirs:     {len(rows)}")
    lines.append(f"raw+pchip ok:      {ok_count}")
    lines.append(f"sg+pchip ok:       {sg_count}")
    lines.append(f"errors/warnings:   {err_count}")
    lines.append("")
    lines.append("Per-directory results")
    lines.append("---------------------")

    for r in rows:
        status = "OK" if (r.raw_ok and r.onset_ok and not r.error) else "CHECK"
        lines.append(f"{status}: {r.label} raw={r.raw_ok} sg={r.sg_ok} onset={r.onset_ok}")
        lines.append(f"  analysis: {r.analysis_dir}")
        lines.append(f"  pchip:    {r.pchip_dir}")
        if r.error:
            lines.append(f"  error:    {r.error}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_report_path(path: Path, run_root: Path) -> Path:
    """Resolve report/summary path relative to the run root."""
    if path.is_absolute():
        return path
    return run_root / path


def main() -> int:
    """Run the workflow."""
    args = parse_args()

    run_root, items = discover_analysis_dirs(
        root=args.root,
        analysis_dir=args.analysis_dir,
        discover_pattern=args.discover_pattern,
        only=args.only,
        max_dirs=args.max_dirs,
        input_raw=args.input_raw,
    )

    xy_pchip = resolve_script_path(args.xy_pchip)
    onset_script = resolve_script_path(args.onset_script)

    print(f"Run root: {run_root}")
    print(f"Analysis directories found: {len(items)}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"xy-pchip.py: {xy_pchip}")
    print(f"onset script: {onset_script}")
    print(f"input raw: {args.input_raw}")
    print(f"input SG:  {args.input_sg}")
    print(f"pchip dir: {args.pchip_dir}")
    print(f"bends: {args.bends}")
    print()

    if not items:
        print("No analysis directories found.")
        return 0

    results: list[WorkflowResult] = []
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.label}: {item.path}")
        result = process_analysis_dir(
            item,
            xy_pchip=xy_pchip,
            onset_script=onset_script,
            args=args,
        )
        results.append(result)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  raw={result.raw_ok}, sg={result.sg_ok}, onset={result.onset_ok}")

    summary_path = resolve_report_path(args.summary, run_root)
    report_path = resolve_report_path(args.report, run_root)

    if args.execute:
        write_batch_summary(summary_path, results)
        write_batch_report(report_path, results, execute=args.execute)
        print()
        print(f"Wrote workflow summary: {summary_path}")
        print(f"Wrote workflow report:  {report_path}")
    else:
        print()
        print("Dry-run only. Add --execute to actually run the workflow.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
