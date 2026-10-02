#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run-onset-workflow.py
---------------------

Version: v11, 2026-06-18

Conservative orchestrator for the PCHIP/IA/FBRM onset workflow.

This runner contains no scientific onset-detection logic. It only finds
experiment directories, checks that required input files exist, builds command
lines for the existing workflow scripts, runs them in the requested order, and
writes a workflow log plus a summary CSV.

Default mode is --dry-run: no analysis scripts are executed and no workflow output directories are created.

Expected project structure
--------------------------

Project root:
  scripts/
    xy-pchip.py
    bw-pchip-onsets-t1t2.py
    fbrm-onsets.py
    make-mgi-fbrm-combo.py
    make-onset-gallery.py
    clean-onset-outputs.py
    onset_config.py

Experiments:
  20990101_ex_99/tl/roi/rgb/analysis/
    rgb-tr.csv
    ts-fbrm-tr.csv
    merged_rgb_counts.csv
    pchip/          generated output
    fbrm_onsets/    generated output
    combo/          generated output

The cleanup step, when enabled, delegates to clean-onset-outputs.py. It never
implements its own cleanup rules.

Typical use
-----------

# Show what would be run.
python scripts/run-onset-workflow.py --dry-run

# Clean old generated outputs by archiving them, then run the workflow.
python scripts/run-onset-workflow.py --run --clean-mode archive

# Run only one experiment.
python scripts/run-onset-workflow.py --run --experiments 20990101_ex_99

# Run only selected steps.
python scripts/run-onset-workflow.py --run --steps pchip,mgi,fbrm,combo

# If mgi is selected and required PCHIP CSVs are missing, the runner
# automatically generates the missing PCHIP prerequisites by default.
python scripts/run-onset-workflow.py --run --steps mgi

# FBRM runs from ts-fbrm-tr.csv and writes analysis/fbrm_onsets/.
# MGI/PCHIP reference-onsets are passed by default when available. Plot-level
# grouping in fbrm-onsets.py keeps overlapping RAW/SG reference lines readable.
python scripts/run-onset-workflow.py --run --steps fbrm

# Combo is a post-analysis figure step. It reads existing MGI/PCHIP and FBRM
# onset outputs and writes analysis/combo/.
python scripts/run-onset-workflow.py --run --steps combo

# The pchip step generates the RAW+PCHIP branch from rgb-tr.csv and, when
# available, the SG+PCHIP branch from rgb-tr-sg.csv. Missing SG input is logged
# as a skipped optional branch and does not fail the workflow.

# Generate document/gallery-oriented diagnostic aspect ratio if supported by
# the called scripts.
python scripts/run-onset-workflow.py --run --same-aspect-diagnostics

Safety principles
-----------------

- default is dry-run
- clean-mode default is none
- clean-mode archive calls clean-onset-outputs.py --archive
- delete cleanup is intentionally not exposed by this runner
- mgi automatically generates missing PCHIP prerequisites by default
- analysis/onset-config.ini is the truth source for per-experiment MGI/FBRM onset counts
- existing outputs cause the relevant selected step to be skipped unless --force is used
- all commands are logged before execution
- stdout/stderr from child scripts are captured in the workflow log
- dry-run may write a workflow log/summary, but it must not create analysis output directories
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from onset_config import OnsetConfig, ensure_onset_config


SCRIPT_NAME = "run-onset-workflow.py"
DEFAULT_STEPS = ("pchip", "mgi", "fbrm", "combo", "gallery")
PER_EXPERIMENT_STEPS = ("pchip", "mgi", "fbrm", "combo")
ALL_STEPS = DEFAULT_STEPS


@dataclass(frozen=True)
class Experiment:
    name: str
    root: Path
    analysis_dir: Path
    rgb_tr_csv: Path
    rgb_tr_sg_csv: Path
    ts_fbrm_tr_csv: Path
    merged_rgb_counts_csv: Path
    pchip_dir: Path
    fbrm_onsets_dir: Path
    combo_dir: Path
    onset_config_ini: Path


@dataclass
class StepResult:
    experiment: str
    step: str
    command: str
    cwd: str
    status: str
    returncode: int | None
    started_at: str
    finished_at: str
    elapsed_seconds: float
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing PCHIP/IA/FBRM onset workflow scripts in order. "
            "Default is --dry-run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute them.")
    mode.add_argument("--run", action="store_true", help="Execute workflow commands.")

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Active code repository root containing scripts/. Default is current working directory.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Experiment/data root. Defaults to --project-root for single-tree usage.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for child scripts.",
    )
    parser.add_argument(
        "--steps",
        default=",".join(DEFAULT_STEPS),
        help=(
            "Comma-separated steps to run. Available: "
            "pchip,mgi,fbrm,combo,gallery. Order is normalized to workflow order."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional experiment directory names to run, for example 20990101_ex_99.",
    )
    parser.add_argument(
        "--exclude-experiments",
        nargs="+",
        default=None,
        help="Optional experiment directory names to exclude.",
    )
    parser.add_argument(
        "--clean-mode",
        choices=("none", "archive"),
        default="none",
        help="Optional cleanup before running. archive delegates to clean-onset-outputs.py.",
    )
    parser.add_argument(
        "--auto-prerequisites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically run missing prerequisite steps. Currently, mgi "
            "generates missing RAW/SG PCHIP CSVs before onset detection."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run steps even if their expected output files already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later experiments/steps after a command fails.",
    )
    parser.add_argument(
        "--same-aspect-diagnostics",
        action="store_true",
        help=(
            "Pass --same-aspect-diagnostics to MGI, FBRM and combo scripts. "
            "Use only if those scripts support the option."
        ),
    )

    # Display/view limit forwarding. These are intentionally simple and are only
    # appended to scripts that commonly generate onset figures.
    parser.add_argument("--x-limits", nargs=2, metavar=("LOW", "HIGH"), help="Forward x display limits as --x-range LOW HIGH.")
    parser.add_argument("--mgi-y-limits", nargs=2, metavar=("LOW", "HIGH"), help="Forward MGI y display limits as --mgi-y-range LOW HIGH.")
    parser.add_argument("--fbrm-y-limits", nargs=2, metavar=("LOW", "HIGH"), help="Forward FBRM y display limits as --fbrm-y-range LOW HIGH.")

    # Extra per-step arguments for local CLI differences without editing the runner.
    parser.add_argument("--pchip-extra", default="", help="Extra args appended to xy-pchip.py.")
    parser.add_argument("--mgi-extra", default="", help="Extra args appended to bw-pchip-onsets-t1t2.py.")
    parser.add_argument("--fbrm-extra", default="", help="Extra args appended to fbrm-onsets.py.")
    parser.add_argument("--combo-extra", default="", help="Extra args appended to make-mgi-fbrm-combo.py.")
    parser.add_argument("--gallery-extra", default="", help="Extra args appended to make-onset-gallery.py.")

    # PCHIP defaults are known from xy-pchip.py. The other scripts may evolve,
    # so their commands are also template-overridable.
    parser.add_argument("--pchip-xcol", default="Tr (°C)", help="X column for xy-pchip.py.")
    parser.add_argument("--pchip-ycol", default="BW", help="Y/MGI column for xy-pchip.py.")
    parser.add_argument(
        "--pchip-outstem",
        default="{analysis}/pchip/bw-raw-vs-temp-pchip",
        help="RAW output stem template for xy-pchip.py.",
    )
    parser.add_argument(
        "--pchip-sg-outstem",
        default="{analysis}/pchip/bw-sg-vs-temp-pchip",
        help="SG output stem template for xy-pchip.py.",
    )

    parser.add_argument(
        "--pchip-template",
        default=(
            "{python} {bin}/xy-pchip.py "
            "--csv {rgb_tr} "
            "--xcol {pchip_xcol} "
            "--ycol {pchip_ycol} "
            "--outstem {pchip_outstem} "
            "--reverse-x "
            "--save-csv"
        ),
        help="Command template for RAW pchip branch.",
    )
    parser.add_argument(
        "--pchip-sg-template",
        default=(
            "{python} {bin}/xy-pchip.py "
            "--csv {rgb_tr_sg} "
            "--xcol {pchip_xcol} "
            "--ycol BW_smooth "
            "--outstem {pchip_sg_outstem} "
            "--reverse-x "
            "--save-csv"
        ),
        help="Command template for optional SG pchip branch.",
    )
    parser.add_argument(
        "--mgi-template",
        default=(
            "{python} {bin}/bw-pchip-onsets-t1t2.py "
            "{mgi_pchip_csvs} "
            "--bends {mgi_onsets} "
            "--outdir {analysis}/pchip "
            "--raw-csv {rgb_tr} "
            "--plot-set all "
            "--candidate-diagnostics "
            "--publication-plots"
        ),
        help=(
            "Command template for MGI onset step. {mgi_pchip_csvs} expands to "
            "all existing MGI PCHIP CSV inputs, normally RAW and optional SG."
        ),
    )
    parser.add_argument(
        "--fbrm-template",
        default=(
            "{python} {bin}/fbrm-onsets.py "
            "--input {ts_fbrm_tr} "
            "--output-dir {analysis}/fbrm_onsets "
            "--onsets {fbrm_onsets} "
            "{fbrm_reference_args} "
            "--publication-plots"
        ),
        help=(
            "Command template for FBRM onset step. {fbrm_reference_args} expands "
            "according to --fbrm-reference-mode."
        ),
    )
    parser.add_argument(
        "--fbrm-reference-mode",
        choices=("none", "auto"),
        default="auto",
        help=(
            "Whether to pass MGI/PCHIP bend summary as --reference-onsets to "
            "fbrm-onsets.py. Default auto passes the summary when it exists; "
            "plot-level grouping in fbrm-onsets.py controls how references are drawn. "
            "Use none only for FBRM-only figures without reference overlays."
        ),
    )
    parser.add_argument(
        "--combo-template",
        default=(
            "{python} {bin}/make-mgi-fbrm-combo.py "
            "{analysis}"
        ),
        help=(
            "Command template for MGI/FBRM combo step. make-mgi-fbrm-combo.py "
            "uses the analysis directory as a positional argument and writes "
            "analysis/combo/ by default."
        ),
    )
    parser.add_argument(
        "--gallery-template",
        default="{python} {bin}/make-onset-gallery.py {data_root}",
        help="Command template for gallery step. Gallery is run once, not per experiment.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("workflow_logs"),
        help="Directory for workflow log and summary CSV. Relative paths are under project root.",
    )

    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = args.project_root

    if not args.dry_run and not args.run:
        args.dry_run = True

    args.steps = normalize_steps(args.steps)

    forbidden_extra = []
    for label, extra, forbidden in [
        ("--mgi-extra", args.mgi_extra, ("--bends",)),
        ("--fbrm-extra", args.fbrm_extra, ("--onsets",)),
    ]:
        tokens = split_extra(extra)
        for token in tokens:
            if token in forbidden or any(token.startswith(f"{name}=") for name in forbidden):
                forbidden_extra.append(f"{label} contains {token}")
    if forbidden_extra:
        raise SystemExit(
            "[error] onset counts are controlled by analysis/onset-config.ini; "
            "edit that file instead of passing onset-count options through extras: "
            + "; ".join(forbidden_extra)
        )

    return args


def normalize_steps(raw_steps: str) -> tuple[str, ...]:
    requested = [s.strip().lower() for s in raw_steps.split(",") if s.strip()]
    invalid = [s for s in requested if s not in ALL_STEPS]
    if invalid:
        raise SystemExit(f"[error] invalid step(s): {', '.join(invalid)}")

    requested_set = set(requested)
    ordered = tuple(step for step in DEFAULT_STEPS if step in requested_set)
    if not ordered:
        raise SystemExit("[error] no workflow steps selected")
    return ordered


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def split_extra(extra: str) -> list[str]:
    return shlex.split(extra) if extra.strip() else []


def format_template(template: str, context: Mapping[str, str]) -> list[str]:
    rendered = template.format_map(context)
    return shlex.split(rendered)


def looks_like_project_root(root: Path) -> bool:
    return (root / "scripts").is_dir()


def iter_experiments(data_root: Path) -> Iterable[Path]:
    """Yield experiment directories discovered by naming convention."""
    for path in sorted(data_root.iterdir()):
        if not path.is_dir() or path.is_symlink():
            continue
        if "_ex_" not in path.name:
            continue
        yield path

def discover_experiments(args: argparse.Namespace) -> list[Experiment]:
    data_root = args.data_root.resolve()

    requested = set(args.experiments or [])
    excluded = set(args.exclude_experiments or [])

    found: list[Experiment] = []

    for exp_root in iter_experiments(data_root):
        if requested and exp_root.name not in requested:
            continue
        if exp_root.name in excluded:
            continue

        analysis_dir = exp_root / "tl" / "roi" / "rgb" / "analysis"
        rgb_tr = analysis_dir / "rgb-tr.csv"
        rgb_tr_sg = analysis_dir / "rgb-tr-sg.csv"
        ts_fbrm_tr = analysis_dir / "ts-fbrm-tr.csv"
        merged = analysis_dir / "merged_rgb_counts.csv"

        # The experiment can still be listed if some files are missing; step-level
        # validation decides whether to skip or fail.
        found.append(
            Experiment(
                name=exp_root.name,
                root=exp_root,
                analysis_dir=analysis_dir,
                rgb_tr_csv=rgb_tr,
                rgb_tr_sg_csv=rgb_tr_sg,
                ts_fbrm_tr_csv=ts_fbrm_tr,
                merged_rgb_counts_csv=merged,
                pchip_dir=analysis_dir / "pchip",
                fbrm_onsets_dir=analysis_dir / "fbrm_onsets",
                combo_dir=analysis_dir / "combo",
                onset_config_ini=analysis_dir / "onset-config.ini",
            )
        )

    if requested:
        found_names = {exp.name for exp in found}
        missing = requested.difference(found_names)
        if missing:
            raise SystemExit(f"[error] requested experiment(s) not found: {', '.join(sorted(missing))}")

    return found


def ensure_script_exists(project_root: Path, script_name: str) -> Path:
    script_path = project_root / "scripts" / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Required script missing: {script_path}")
    return script_path


def step_inputs_missing(exp: Experiment, step: str) -> list[Path]:
    if step == "pchip":
        # RAW input is mandatory. SG input is optional and handled as a skipped
        # branch if rgb-tr-sg.csv is missing.
        return [p for p in [exp.rgb_tr_csv] if not p.is_file()]
    if step == "mgi":
        # MGI onset detection needs at least the RAW PCHIP CSV. The SG PCHIP CSV
        # is optional and included automatically when it exists.
        return [
            p for p in [
                exp.rgb_tr_csv,
                exp.pchip_dir / "bw-raw-vs-temp-pchip.csv",
            ]
            if not p.is_file()
        ]
    if step == "fbrm":
        return [p for p in [exp.ts_fbrm_tr_csv] if not p.is_file()]
    if step == "combo":
        return [
            p
            for p in [
                exp.rgb_tr_csv,
                exp.ts_fbrm_tr_csv,
                exp.pchip_dir / "bw-raw-vs-temp-pchip.csv",
                exp.pchip_dir / "bw-raw-vs-temp-pchip-bends.csv",
                exp.fbrm_onsets_dir / "fbrm-onsets.csv",
                exp.fbrm_onsets_dir / "fbrm-onset-params.json",
            ]
            if not p.is_file()
        ]
    return []


def expected_outputs(exp: Experiment, step: str) -> list[Path]:
    if step == "pchip":
        outputs = [
            exp.pchip_dir / "bw-raw-vs-temp-pchip.csv",
            exp.pchip_dir / "bw-raw-vs-temp-pchip.png",
        ]
        if exp.rgb_tr_sg_csv.is_file():
            outputs.extend(
                [
                    exp.pchip_dir / "bw-sg-vs-temp-pchip.csv",
                    exp.pchip_dir / "bw-sg-vs-temp-pchip.png",
                ]
            )
        return outputs
    if step == "mgi":
        outputs = [
            exp.pchip_dir / "bw-pchip-bend-summary.csv",
            exp.pchip_dir / "bw-raw-vs-temp-pchip-bends.csv",
        ]
        if (exp.pchip_dir / "bw-sg-vs-temp-pchip.csv").is_file():
            outputs.append(exp.pchip_dir / "bw-sg-vs-temp-pchip-bends.csv")
        return outputs
    if step == "fbrm":
        return [
            exp.fbrm_onsets_dir / "fbrm-onsets.csv",
        ]
    if step == "combo":
        return [
            exp.combo_dir / "mgi-fbrm-combo.png",
            exp.combo_dir / "mgi-fbrm-combo-main.png",
            exp.combo_dir / "mgi-fbrm-combo-main-mono.png",
            exp.combo_dir / "mgi-fbrm-combo-main-clean-mono.png",
            exp.combo_dir / "mgi-fbrm-combo-data.csv",
            exp.combo_dir / "mgi-fbrm-combo-params.json",
            exp.combo_dir / "mgi-fbrm-combo-report.txt",
        ]
    return []


def output_exists(exp: Experiment, step: str) -> bool:
    outputs = expected_outputs(exp, step)
    return bool(outputs) and all(p.exists() for p in outputs)


def available_mgi_pchip_csvs(exp: Experiment) -> list[Path]:
    """Return existing MGI PCHIP input CSVs in preferred reporting order."""
    candidates = [
        exp.pchip_dir / "bw-raw-vs-temp-pchip.csv",
        exp.pchip_dir / "bw-sg-vs-temp-pchip.csv",
    ]
    return [p for p in candidates if p.is_file()]


def branch_state_line(exp: Experiment) -> str:
    """Return a compact human-readable RAW/SG branch state."""
    raw_input = exp.rgb_tr_csv.is_file()
    sg_input = exp.rgb_tr_sg_csv.is_file()
    raw_pchip = (exp.pchip_dir / "bw-raw-vs-temp-pchip.csv").is_file()
    sg_pchip = (exp.pchip_dir / "bw-sg-vs-temp-pchip.csv").is_file()

    def yn(value: bool) -> str:
        return "yes" if value else "no"

    return (
        f"RAW input={yn(raw_input)}, RAW PCHIP={yn(raw_pchip)}; "
        f"SG input={yn(sg_input)}, SG PCHIP={yn(sg_pchip)}"
    )


def mgi_input_branch_names(exp: Experiment) -> list[str]:
    """Return branch names that will be passed to MGI onset detection."""
    names: list[str] = []
    if (exp.pchip_dir / "bw-raw-vs-temp-pchip.csv").is_file():
        names.append("raw")
    if (exp.pchip_dir / "bw-sg-vs-temp-pchip.csv").is_file():
        names.append("sg")
    return names


def fbrm_reference_onsets_path(exp: Experiment) -> Path:
    """Return the optional MGI/PCHIP summary used as FBRM reference onsets."""
    return exp.pchip_dir / "bw-pchip-bend-summary.csv"


def fbrm_reference_args(args: argparse.Namespace, exp: Experiment) -> str:
    """Return quoted optional reference-onsets arguments for fbrm-onsets.py."""
    if args.fbrm_reference_mode == "none":
        return ""
    ref = fbrm_reference_onsets_path(exp)
    if args.fbrm_reference_mode == "auto" and ref.is_file():
        return f"--reference-onsets {q(ref)}"
    return ""


def fbrm_state_line(exp: Experiment) -> str:
    """Return a compact human-readable FBRM input/reference state."""
    input_ok = exp.ts_fbrm_tr_csv.is_file()
    ref = fbrm_reference_onsets_path(exp)
    ref_ok = ref.is_file()
    output_ok = (exp.fbrm_onsets_dir / "fbrm-onsets.csv").is_file()

    def yn(value: bool) -> str:
        return "yes" if value else "no"

    return (
        f"FBRM input={yn(input_ok)}, "
        f"MGI reference={yn(ref_ok)}, "
        f"FBRM output={yn(output_ok)}"
    )


def combo_state_line(exp: Experiment) -> str:
    """Return a compact human-readable combo input/output state."""
    raw_pchip = (exp.pchip_dir / "bw-raw-vs-temp-pchip.csv").is_file()
    raw_mgi_onsets = (exp.pchip_dir / "bw-raw-vs-temp-pchip-bends.csv").is_file()
    fbrm_onsets = (exp.fbrm_onsets_dir / "fbrm-onsets.csv").is_file()
    fbrm_params = (exp.fbrm_onsets_dir / "fbrm-onset-params.json").is_file()
    combo_output = (exp.combo_dir / "mgi-fbrm-combo.png").is_file()

    def yn(value: bool) -> str:
        return "yes" if value else "no"

    return (
        f"MGI PCHIP={yn(raw_pchip)}, "
        f"MGI onsets={yn(raw_mgi_onsets)}, "
        f"FBRM onsets={yn(fbrm_onsets)}, "
        f"FBRM params={yn(fbrm_params)}, "
        f"combo output={yn(combo_output)}"
    )


def pchip_branch_specs(args: argparse.Namespace, exp: Experiment) -> list[tuple[str, Path, Path, str]]:
    """Return pchip branches as (branch_name, input_csv, expected_output_csv, template)."""
    specs = [
        (
            "raw",
            exp.rgb_tr_csv,
            exp.pchip_dir / "bw-raw-vs-temp-pchip.csv",
            args.pchip_template,
        )
    ]

    if exp.rgb_tr_sg_csv.is_file():
        specs.append(
            (
                "sg",
                exp.rgb_tr_sg_csv,
                exp.pchip_dir / "bw-sg-vs-temp-pchip.csv",
                args.pchip_sg_template,
            )
        )

    return specs


def missing_pchip_prerequisite_specs(
    args: argparse.Namespace,
    exp: Experiment,
) -> list[tuple[str, Path, Path, str]]:
    """Return only missing PCHIP branches required before MGI onset detection."""
    return [
        spec
        for spec in pchip_branch_specs(args, exp)
        if spec[1].is_file() and not spec[2].is_file()
    ]


def mgi_needs_pchip_prerequisites(args: argparse.Namespace, exp: Experiment) -> bool:
    """Return True if MGI needs runner-generated PCHIP CSVs first."""
    return bool(missing_pchip_prerequisite_specs(args, exp))


def combo_missing_mgi_outputs(exp: Experiment) -> bool:
    """Return True if combo needs MGI/PCHIP onset outputs generated first."""
    required = [
        exp.pchip_dir / "bw-raw-vs-temp-pchip.csv",
        exp.pchip_dir / "bw-raw-vs-temp-pchip-bends.csv",
    ]
    return any(not p.is_file() for p in required)


def combo_missing_fbrm_outputs(exp: Experiment) -> bool:
    """Return True if combo needs FBRM onset outputs generated first."""
    required = [
        exp.fbrm_onsets_dir / "fbrm-onsets.csv",
        exp.fbrm_onsets_dir / "fbrm-onset-params.json",
    ]
    return any(not p.is_file() for p in required)


def get_onset_config_for_experiment(args: argparse.Namespace, exp: Experiment) -> OnsetConfig:
    """Ensure/read analysis/onset-config.ini once per experiment."""
    cache = getattr(args, "_onset_config_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_onset_config_cache", cache)

    if exp.name not in cache:
        cache[exp.name] = ensure_onset_config(
            exp.analysis_dir,
            dry_run=args.dry_run,
            experiment_name=exp.name,
        )

    return cache[exp.name]


def onset_config_state_line(args: argparse.Namespace, exp: Experiment) -> str:
    """Return compact resolved onset config status for logs and summaries."""
    cfg = get_onset_config_for_experiment(args, exp)
    return (
        f"onset_config={cfg.status}; "
        f"MGI onsets={cfg.mgi_onsets}; "
        f"FBRM onsets={cfg.fbrm_onsets}"
    )


def make_context(args: argparse.Namespace, exp: Experiment | None = None) -> dict[str, str]:
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    bin_dir = project_root / "scripts"

    context: dict[str, str] = {
        "python": q(args.python),
        "project_root": q(project_root),
        "data_root": q(data_root),
        "bin": q(bin_dir),
        "pchip_xcol": q(args.pchip_xcol),
        "pchip_ycol": q(args.pchip_ycol),
    }

    if exp is not None:
        onset_cfg = get_onset_config_for_experiment(args, exp)

        pchip_outstem = args.pchip_outstem.format(
            project_root=project_root,
            data_root=data_root,
            experiment=exp.root,
            analysis=exp.analysis_dir,
            name=exp.name,
        )
        pchip_sg_outstem = args.pchip_sg_outstem.format(
            project_root=project_root,
            data_root=data_root,
            experiment=exp.root,
            analysis=exp.analysis_dir,
            name=exp.name,
        )
        mgi_pchip_csvs = " ".join(q(path) for path in available_mgi_pchip_csvs(exp))

        context.update(
            {
                "experiment_name": q(exp.name),
                "experiment": q(exp.root),
                "analysis": q(exp.analysis_dir),
                "rgb_tr": q(exp.rgb_tr_csv),
                "rgb_tr_sg": q(exp.rgb_tr_sg_csv),
                "ts_fbrm_tr": q(exp.ts_fbrm_tr_csv),
                "merged_rgb_counts": q(exp.merged_rgb_counts_csv),
                "pchip_dir": q(exp.pchip_dir),
                "fbrm_onsets_dir": q(exp.fbrm_onsets_dir),
                "combo_dir": q(exp.combo_dir),
                "onset_config": q(exp.onset_config_ini),
                "mgi_onsets": str(onset_cfg.mgi_onsets),
                "fbrm_onsets": str(onset_cfg.fbrm_onsets),
                "onset_config_status": q(onset_cfg.status),
                "pchip_outstem": q(pchip_outstem),
                "pchip_sg_outstem": q(pchip_sg_outstem),
                "mgi_pchip_csvs": mgi_pchip_csvs,
                "fbrm_reference_args": fbrm_reference_args(args, exp),
            }
        )

    return context


def build_pchip_branch_command(
    args: argparse.Namespace,
    exp: Experiment,
    *,
    branch: str,
    template: str,
) -> list[str]:
    context = make_context(args, exp)
    cmd = format_template(template, context)
    cmd.extend(split_extra(args.pchip_extra))
    return cmd


def build_step_command(args: argparse.Namespace, exp: Experiment, step: str) -> list[str]:
    context = make_context(args, exp)

    if step == "pchip":
        cmd = format_template(args.pchip_template, context)
        cmd.extend(split_extra(args.pchip_extra))
    elif step == "mgi":
        cmd = format_template(args.mgi_template, context)
        cmd.extend(build_view_args(args, include_mgi=True, include_fbrm=False))
        if args.same_aspect_diagnostics:
            cmd.append("--same-aspect-diagnostics")
        cmd.extend(split_extra(args.mgi_extra))
    elif step == "fbrm":
        cmd = format_template(args.fbrm_template, context)
        cmd.extend(build_view_args(args, include_mgi=False, include_fbrm=True))
        if args.same_aspect_diagnostics:
            cmd.append("--same-aspect-diagnostics")
        cmd.extend(split_extra(args.fbrm_extra))
    elif step == "combo":
        cmd = format_template(args.combo_template, context)
        cmd.extend(build_view_args(args, include_mgi=True, include_fbrm=True))
        if args.same_aspect_diagnostics:
            cmd.append("--same-aspect-diagnostics")
        cmd.extend(split_extra(args.combo_extra))
    else:
        raise ValueError(f"Unsupported per-experiment step: {step}")

    return cmd


def build_gallery_command(args: argparse.Namespace) -> list[str]:
    context = make_context(args, None)
    cmd = format_template(args.gallery_template, context)
    cmd.extend(split_extra(args.gallery_extra))
    return cmd


def build_view_args(
    args: argparse.Namespace,
    *,
    include_mgi: bool,
    include_fbrm: bool,
) -> list[str]:
    out: list[str] = []
    if args.x_limits:
        out.extend(["--x-range", args.x_limits[0], args.x_limits[1]])
    if include_mgi and args.mgi_y_limits:
        out.extend(["--mgi-y-range", args.mgi_y_limits[0], args.mgi_y_limits[1]])
    if include_fbrm and args.fbrm_y_limits:
        out.extend(["--fbrm-y-range", args.fbrm_y_limits[0], args.fbrm_y_limits[1]])
    return out


def command_to_string(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def make_log_paths(project_root: Path, log_dir_arg: Path) -> tuple[Path, Path]:
    log_dir = log_dir_arg if log_dir_arg.is_absolute() else project_root / log_dir_arg
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run-onset-workflow-{timestamp}.log"
    summary_path = log_dir / f"run-onset-workflow-summary-{timestamp}.csv"
    return log_path, summary_path


def write_log_header(log_file, *, args: argparse.Namespace, experiments: Sequence[Experiment]) -> None:
    log_file.write(f"{SCRIPT_NAME}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n")
    log_file.write(f"project_root: {args.project_root.resolve()}\n")
    log_file.write(f"data_root: {args.data_root.resolve()}\n")
    log_file.write(f"mode: {'run' if args.run else 'dry-run'}\n")
    log_file.write(f"steps: {','.join(args.steps)}\n")
    log_file.write(f"clean_mode: {args.clean_mode}\n")
    log_file.write(f"auto_prerequisites: {args.auto_prerequisites}\n")
    log_file.write(f"fbrm_reference_mode: {args.fbrm_reference_mode}\n")
    log_file.write(f"experiments: {len(experiments)}\n")
    log_file.write("\n")


def append_command_log(log_file, *, title: str, cmd: Sequence[str], cwd: Path) -> None:
    log_file.write("\n" + "-" * 80 + "\n")
    log_file.write(f"{title}\n")
    log_file.write(f"cwd: {cwd}\n")
    log_file.write(f"cmd: {command_to_string(cmd)}\n")
    log_file.flush()


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_file,
    dry_run: bool,
) -> tuple[str, int | None, float, str]:
    start = datetime.now()
    if dry_run:
        return "dry-run", None, 0.0, "not executed"

    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    end = datetime.now()
    elapsed = (end - start).total_seconds()

    if proc.stdout:
        log_file.write("\n[stdout]\n")
        log_file.write(proc.stdout)
        if not proc.stdout.endswith("\n"):
            log_file.write("\n")

    if proc.stderr:
        log_file.write("\n[stderr]\n")
        log_file.write(proc.stderr)
        if not proc.stderr.endswith("\n"):
            log_file.write("\n")

    log_file.flush()

    status = "ok" if proc.returncode == 0 else "failed"
    return status, proc.returncode, elapsed, ""


def append_result(results: list[StepResult], result: StepResult) -> None:
    results.append(result)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_summary_csv(path: Path, results: Sequence[StepResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "step",
            "status",
            "returncode",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "cwd",
            "command",
            "message",
        ])
        for r in results:
            writer.writerow([
                r.experiment,
                r.step,
                r.status,
                "" if r.returncode is None else r.returncode,
                r.started_at,
                r.finished_at,
                f"{r.elapsed_seconds:.3f}",
                r.cwd,
                r.command,
                r.message,
            ])


def write_workflow_marker(
    *,
    directory: Path,
    project_root: Path,
    experiment: str,
    step: str,
    command: Sequence[str],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".generated-by-onset-workflow.json"
    payload = {
        "generated_by": SCRIPT_NAME,
        "generated_at": now_iso(),
        "project_root": str(project_root),
        "experiment": experiment,
        "step": step,
        "command": list(command),
    }
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def marker_dirs_for_step(exp: Experiment, step: str) -> list[Path]:
    if step in ("pchip", "mgi"):
        return [exp.pchip_dir]
    if step == "fbrm":
        return [exp.fbrm_onsets_dir]
    if step == "combo":
        return [exp.combo_dir]
    return []


def run_clean_if_requested(
    args: argparse.Namespace,
    *,
    project_root: Path,
    experiments: Sequence[Experiment],
    log_file,
    results: list[StepResult],
) -> bool:
    if args.clean_mode == "none":
        return True

    clean_script = ensure_script_exists(project_root, "clean-onset-outputs.py")

    cmd = [
        args.python,
        str(clean_script),
        "--archive",
        "--project-root",
        str(args.data_root.resolve()),
    ]

    # main() has validated the active code root and discovered the experiments.
    # A separate data tree intentionally has no scripts/ directory; bypass only
    # the cleaner's single-tree root check, retaining its target protections.
    if args.data_root.resolve() != project_root.resolve() and experiments:
        cmd.append("--allow-untrusted-root")

    if args.experiments:
        cmd.append("--experiments")
        cmd.extend(exp.name for exp in experiments)

    # If gallery is not part of this workflow run, avoid cleaning galleries.
    if "gallery" not in args.steps:
        cmd.append("--no-include-galleries")

    append_command_log(log_file, title="cleanup: archive generated workflow outputs", cmd=cmd, cwd=project_root)
    print(f"[clean] {command_to_string(cmd)}")

    started = now_iso()
    status, returncode, elapsed, message = run_command(cmd, cwd=project_root, log_file=log_file, dry_run=args.dry_run)
    finished = now_iso()

    append_result(
        results,
        StepResult(
            experiment="",
            step="clean",
            command=command_to_string(cmd),
            cwd=str(project_root),
            status=status,
            returncode=returncode,
            started_at=started,
            finished_at=finished,
            elapsed_seconds=elapsed,
            message=message,
        ),
    )

    if status == "failed":
        print("[error] cleanup failed")
        return False

    return True


def run_pchip_prerequisites_for_mgi(
    args: argparse.Namespace,
    *,
    project_root: Path,
    exp: Experiment,
    log_file,
    results: list[StepResult],
) -> bool:
    """Generate missing PCHIP CSVs required by the MGI step.

    This is intentionally missing-only. Even when --force is used for the
    selected mgi step, prerequisite generation does not overwrite existing
    PCHIP CSVs unless they are actually missing.
    """
    missing_specs = missing_pchip_prerequisite_specs(args, exp)

    if not missing_specs:
        branches = ",".join(mgi_input_branch_names(exp)) or "none"
        print(f"[prereq:mgi] {exp.name}: PCHIP prerequisites already available ({branches})")
        return True

    missing_names = ",".join(branch for branch, _inp, _out, _tpl in missing_specs)
    print(f"[prereq:mgi] {exp.name}: generating missing PCHIP branch(es): {missing_names}")

    if args.run:
        exp.pchip_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    generated_branches: list[str] = []

    for branch, input_csv, expected_csv, template in missing_specs:
        step_name = f"pchip_{branch}_prereq"

        if not input_csv.is_file():
            message = f"prerequisite input missing: {input_csv}"
            print(f"[skip:{step_name}] {exp.name}: {message}")
            append_result(
                results,
                StepResult(
                    experiment=exp.name,
                    step=step_name,
                    command="",
                    cwd=str(project_root),
                    status="skipped",
                    returncode=None,
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    elapsed_seconds=0.0,
                    message=message,
                ),
            )
            all_ok = False
            if not args.continue_on_error:
                return False
            continue

        cmd = build_pchip_branch_command(args, exp, branch=branch, template=template)
        cwd = project_root

        append_command_log(log_file, title=f"{exp.name}: {step_name}", cmd=cmd, cwd=cwd)
        print(f"[{exp.name}:{step_name}] {command_to_string(cmd)}")

        started = now_iso()
        status, returncode, elapsed, message = run_command(cmd, cwd=cwd, log_file=log_file, dry_run=args.dry_run)
        finished = now_iso()

        if args.dry_run:
            message = f"dry-run prerequisite for MGI branch: {branch}"

        append_result(
            results,
            StepResult(
                experiment=exp.name,
                step=step_name,
                command=command_to_string(cmd),
                cwd=str(cwd),
                status=status,
                returncode=returncode,
                started_at=started,
                finished_at=finished,
                elapsed_seconds=elapsed,
                message=message,
            ),
        )

        if status == "ok" and args.run:
            generated_branches.append(branch)

        if status == "failed":
            all_ok = False
            if not args.continue_on_error:
                return False

    if args.run and generated_branches:
        write_workflow_marker(
            directory=exp.pchip_dir,
            project_root=project_root,
            experiment=exp.name,
            step="pchip_prerequisite_for_mgi",
            command=[
                "branches",
                ",".join(generated_branches),
            ],
        )

    return all_ok


def run_combo_prerequisites(
    args: argparse.Namespace,
    *,
    project_root: Path,
    exp: Experiment,
    log_file,
    results: list[StepResult],
) -> bool:
    """Generate missing MGI/FBRM prerequisite outputs required by combo."""
    ok = True

    if combo_missing_mgi_outputs(exp):
        print(f"[prereq:combo] {exp.name}: generating missing MGI prerequisite outputs")
        ok = run_per_experiment_step(
            args,
            project_root=project_root,
            exp=exp,
            step="mgi",
            log_file=log_file,
            results=results,
        )
        if not ok and not args.continue_on_error:
            return False
    else:
        print(f"[prereq:combo] {exp.name}: MGI prerequisites already available")

    if combo_missing_fbrm_outputs(exp):
        print(f"[prereq:combo] {exp.name}: generating missing FBRM prerequisite outputs")
        ok = run_per_experiment_step(
            args,
            project_root=project_root,
            exp=exp,
            step="fbrm",
            log_file=log_file,
            results=results,
        )
        if not ok and not args.continue_on_error:
            return False
    else:
        print(f"[prereq:combo] {exp.name}: FBRM prerequisites already available")

    return ok


def run_pchip_step(
    args: argparse.Namespace,
    *,
    project_root: Path,
    exp: Experiment,
    log_file,
    results: list[StepResult],
) -> bool:
    missing_inputs = step_inputs_missing(exp, "pchip")
    if missing_inputs:
        message = "missing input(s): " + ", ".join(str(p) for p in missing_inputs)
        print(f"[skip:pchip] {exp.name}: {message}")
        append_result(
            results,
            StepResult(
                experiment=exp.name,
                step="pchip",
                command="",
                cwd=str(project_root),
                status="skipped",
                returncode=None,
                started_at=now_iso(),
                finished_at=now_iso(),
                elapsed_seconds=0.0,
                message=message,
            ),
        )
        return True

    if args.run:
        exp.pchip_dir.mkdir(parents=True, exist_ok=True)

    print(f"[branches] {exp.name}: {branch_state_line(exp)}")

    all_ok = True
    executed_or_planned_branches: list[str] = []

    for branch, input_csv, expected_csv, template in pchip_branch_specs(args, exp):
        step_name = f"pchip_{branch}"

        if not input_csv.is_file():
            message = f"optional input missing: {input_csv}"
            print(f"[skip:{step_name}] {exp.name}: {message}")
            append_result(
                results,
                StepResult(
                    experiment=exp.name,
                    step=step_name,
                    command="",
                    cwd=str(project_root),
                    status="skipped",
                    returncode=None,
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    elapsed_seconds=0.0,
                    message=message,
                ),
            )
            continue

        if not args.force and expected_csv.exists():
            message = "expected output already exists; use --force to rerun"
            print(f"[skip:{step_name}] {exp.name}: {message}")
            append_result(
                results,
                StepResult(
                    experiment=exp.name,
                    step=step_name,
                    command="",
                    cwd=str(project_root),
                    status="skipped",
                    returncode=None,
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    elapsed_seconds=0.0,
                    message=message,
                ),
            )
            continue

        cmd = build_pchip_branch_command(args, exp, branch=branch, template=template)
        cwd = project_root

        append_command_log(log_file, title=f"{exp.name}: {step_name}", cmd=cmd, cwd=cwd)
        print(f"[{exp.name}:{step_name}] {command_to_string(cmd)}")

        started = now_iso()
        status, returncode, elapsed, message = run_command(cmd, cwd=cwd, log_file=log_file, dry_run=args.dry_run)
        finished = now_iso()

        append_result(
            results,
            StepResult(
                experiment=exp.name,
                step=step_name,
                command=command_to_string(cmd),
                cwd=str(cwd),
                status=status,
                returncode=returncode,
                started_at=started,
                finished_at=finished,
                elapsed_seconds=elapsed,
                message=message,
            ),
        )

        if status == "ok" and args.run:
            executed_or_planned_branches.append(branch)

        if status == "failed":
            all_ok = False
            if not args.continue_on_error:
                return False

    if args.run and executed_or_planned_branches:
        write_workflow_marker(
            directory=exp.pchip_dir,
            project_root=project_root,
            experiment=exp.name,
            step="pchip",
            command=[
                "branches",
                ",".join(executed_or_planned_branches),
            ],
        )

    if not exp.rgb_tr_sg_csv.is_file():
        message = f"optional SG input missing: {exp.rgb_tr_sg_csv}"
        print(f"[skip:pchip_sg] {exp.name}: {message}")
        append_result(
            results,
            StepResult(
                experiment=exp.name,
                step="pchip_sg",
                command="",
                cwd=str(project_root),
                status="skipped",
                returncode=None,
                started_at=now_iso(),
                finished_at=now_iso(),
                elapsed_seconds=0.0,
                message=message,
            ),
        )

    return all_ok


def run_per_experiment_step(
    args: argparse.Namespace,
    *,
    project_root: Path,
    exp: Experiment,
    step: str,
    log_file,
    results: list[StepResult],
) -> bool:
    # Every per-experiment workflow pass resolves/creates the protected
    # human-editable onset configuration. The config is the truth source for
    # MGI/FBRM onset counts; dry-run reports what would happen without writing.
    _onset_cfg = get_onset_config_for_experiment(args, exp)

    if step == "pchip":
        return run_pchip_step(
            args,
            project_root=project_root,
            exp=exp,
            log_file=log_file,
            results=results,
        )

    if step == "mgi" and args.auto_prerequisites:
        if mgi_needs_pchip_prerequisites(args, exp):
            ok = run_pchip_prerequisites_for_mgi(
                args,
                project_root=project_root,
                exp=exp,
                log_file=log_file,
                results=results,
            )
            if not ok and not args.continue_on_error:
                return False
        else:
            branches = ",".join(mgi_input_branch_names(exp)) or "none"
            print(f"[prereq:mgi] {exp.name}: PCHIP prerequisites already available ({branches})")

    if step == "combo" and args.auto_prerequisites:
        if combo_missing_mgi_outputs(exp) or combo_missing_fbrm_outputs(exp):
            ok = run_combo_prerequisites(
                args,
                project_root=project_root,
                exp=exp,
                log_file=log_file,
                results=results,
            )
            if not ok and not args.continue_on_error:
                return False
        else:
            print(f"[prereq:combo] {exp.name}: MGI/FBRM prerequisites already available")

    missing_inputs = step_inputs_missing(exp, step)
    if missing_inputs:
        message = "missing input(s): " + ", ".join(str(p) for p in missing_inputs)
        if step == "mgi" and args.dry_run and args.auto_prerequisites:
            message += "; MGI would run after prerequisite generation in --run mode"
        if step == "combo" and args.dry_run and args.auto_prerequisites:
            message += "; combo would run after prerequisite generation in --run mode"
        print(f"[skip:{step}] {exp.name}: {message}")
        append_result(
            results,
            StepResult(
                experiment=exp.name,
                step=step,
                command="",
                cwd=str(project_root),
                status="skipped",
                returncode=None,
                started_at=now_iso(),
                finished_at=now_iso(),
                elapsed_seconds=0.0,
                message=message,
            ),
        )
        return True

    if not args.force and output_exists(exp, step):
        message = "expected output already exists; use --force to rerun"
        if step == "mgi":
            branches = ",".join(mgi_input_branch_names(exp)) or "none"
            message += f"; MGI input branches available: {branches}; {onset_config_state_line(args, exp)}"
        if step == "fbrm":
            message += f"; {onset_config_state_line(args, exp)}"
        if step == "combo":
            message += f"; {combo_state_line(exp)}; {onset_config_state_line(args, exp)}"
        print(f"[skip:{step}] {exp.name}: {message}")
        append_result(
            results,
            StepResult(
                experiment=exp.name,
                step=step,
                command="",
                cwd=str(project_root),
                status="skipped",
                returncode=None,
                started_at=now_iso(),
                finished_at=now_iso(),
                elapsed_seconds=0.0,
                message=message,
            ),
        )
        return True

    if step == "mgi":
        branches = mgi_input_branch_names(exp)
        branches_text = ",".join(branches) if branches else "none"
        print(
            f"[mgi-inputs] {exp.name}: branches={branches_text}; "
            f"{branch_state_line(exp)}; {onset_config_state_line(args, exp)}"
        )

    if step == "fbrm":
        ref = fbrm_reference_onsets_path(exp)
        ref_available = ref.is_file()
        ref_used = args.fbrm_reference_mode == "auto" and ref_available
        ref_text = str(ref) if ref_used else "none"
        print(
            f"[fbrm-inputs] {exp.name}: {fbrm_state_line(exp)}; "
            f"reference_mode={args.fbrm_reference_mode}; reference_used={'yes' if ref_used else 'no'}; "
            f"reference={ref_text}; {onset_config_state_line(args, exp)}"
        )

    if step == "combo":
        print(f"[combo-inputs] {exp.name}: {combo_state_line(exp)}; {onset_config_state_line(args, exp)}")

    if args.run:
        if step == "pchip":
            exp.pchip_dir.mkdir(parents=True, exist_ok=True)
        elif step == "fbrm":
            exp.fbrm_onsets_dir.mkdir(parents=True, exist_ok=True)
        elif step == "combo":
            exp.combo_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_step_command(args, exp, step)
    cwd = project_root

    append_command_log(log_file, title=f"{exp.name}: {step}", cmd=cmd, cwd=cwd)
    print(f"[{exp.name}:{step}] {command_to_string(cmd)}")

    started = now_iso()
    status, returncode, elapsed, message = run_command(cmd, cwd=cwd, log_file=log_file, dry_run=args.dry_run)
    finished = now_iso()

    if step == "mgi":
        branches = ",".join(mgi_input_branch_names(exp)) or "none"
        branch_message = f"MGI input branches: {branches}; {onset_config_state_line(args, exp)}"
        message = branch_message if not message else f"{message}; {branch_message}"

    if step == "fbrm":
        ref_available = fbrm_reference_onsets_path(exp).is_file()
        ref_used = args.fbrm_reference_mode == "auto" and ref_available
        fbrm_message = (
            "FBRM input: ts-fbrm-tr.csv; "
            f"MGI reference available: {'yes' if ref_available else 'no'}; "
            f"reference used: {'yes' if ref_used else 'no'}; "
            f"{onset_config_state_line(args, exp)}"
        )
        message = fbrm_message if not message else f"{message}; {fbrm_message}"

    if step == "combo":
        combo_message = f"{combo_state_line(exp)}; {onset_config_state_line(args, exp)}"
        message = combo_message if not message else f"{message}; {combo_message}"

    append_result(
        results,
        StepResult(
            experiment=exp.name,
            step=step,
            command=command_to_string(cmd),
            cwd=str(cwd),
            status=status,
            returncode=returncode,
            started_at=started,
            finished_at=finished,
            elapsed_seconds=elapsed,
            message=message,
        ),
    )

    if status == "ok" and args.run:
        for marker_dir in marker_dirs_for_step(exp, step):
            write_workflow_marker(
                directory=marker_dir,
                project_root=project_root,
                experiment=exp.name,
                step=step,
                command=cmd,
            )

    return status != "failed"


def run_gallery_step(
    args: argparse.Namespace,
    *,
    project_root: Path,
    log_file,
    results: list[StepResult],
) -> bool:
    cmd = build_gallery_command(args)
    cwd = project_root

    append_command_log(log_file, title="gallery", cmd=cmd, cwd=cwd)
    print(f"[gallery] {command_to_string(cmd)}")

    started = now_iso()
    status, returncode, elapsed, message = run_command(cmd, cwd=cwd, log_file=log_file, dry_run=args.dry_run)
    finished = now_iso()

    append_result(
        results,
        StepResult(
            experiment="",
            step="gallery",
            command=command_to_string(cmd),
            cwd=str(cwd),
            status=status,
            returncode=returncode,
            started_at=started,
            finished_at=finished,
            elapsed_seconds=elapsed,
            message=message,
        ),
    )

    if status == "ok" and args.run:
        galleries_dir = args.data_root.resolve() / "galleries"
        if galleries_dir.exists():
            marker = galleries_dir / ".generated-by-onset-workflow.json"
            payload = {
                "generated_by": SCRIPT_NAME,
                "generated_at": now_iso(),
                "project_root": str(project_root),
                "data_root": str(args.data_root.resolve()),
                "step": "gallery",
                "command": list(cmd),
            }
            marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return status != "failed"


def main() -> int:
    args = parse_args()
    args._onset_config_cache = {}
    project_root = args.project_root.resolve()
    args.project_root = project_root
    data_root = args.data_root.resolve()
    args.data_root = data_root

    if not project_root.is_dir():
        print(f"[error] project root is not a directory: {project_root}", file=sys.stderr)
        return 2

    if not looks_like_project_root(project_root):
        print(
            "[error] project root must contain the active scripts/ directory.",
            file=sys.stderr,
        )
        return 2

    if not data_root.is_dir():
        print(f"[error] data root is not a directory: {data_root}", file=sys.stderr)
        return 2

    required_scripts = [
        "xy-pchip.py",
        "bw-pchip-onsets-t1t2.py",
        "fbrm-onsets.py",
        "make-mgi-fbrm-combo.py",
        "make-onset-gallery.py",
        "onset_config.py",
    ]
    if args.clean_mode == "archive":
        required_scripts.append("clean-onset-outputs.py")

    for script_name in required_scripts:
        try:
            ensure_script_exists(project_root, script_name)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2

    experiments = discover_experiments(args)
    if not experiments and any(step in args.steps for step in PER_EXPERIMENT_STEPS):
        print("[error] no experiments selected/found", file=sys.stderr)
        return 2

    log_path, summary_path = make_log_paths(project_root, args.log_dir)
    results: list[StepResult] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        write_log_header(log_file, args=args, experiments=experiments)

        print(f"Project root: {project_root}")
        print(f"Data root: {data_root}")
        print(f"Mode: {'run' if args.run else 'dry-run'}")
        print(f"Steps: {','.join(args.steps)}")
        print(f"Auto prerequisites: {'on' if args.auto_prerequisites else 'off'}")
        print(f"FBRM reference mode: {args.fbrm_reference_mode}")
        print(f"Experiments: {len(experiments)}")
        print(f"Log: {log_path}")
        print(f"Summary: {summary_path}")
        print()

        if args.clean_mode != "none":
            ok = run_clean_if_requested(
                args,
                project_root=project_root,
                experiments=experiments,
                log_file=log_file,
                results=results,
            )
            if not ok and not args.continue_on_error:
                write_summary_csv(summary_path, results)
                return 1

        for step in args.steps:
            if step == "gallery":
                continue

            for exp in experiments:
                ok = run_per_experiment_step(
                    args,
                    project_root=project_root,
                    exp=exp,
                    step=step,
                    log_file=log_file,
                    results=results,
                )
                if not ok and not args.continue_on_error:
                    write_summary_csv(summary_path, results)
                    print()
                    print(f"[failed] stopped at {exp.name}:{step}")
                    return 1

        if "gallery" in args.steps:
            ok = run_gallery_step(
                args,
                project_root=project_root,
                log_file=log_file,
                results=results,
            )
            if not ok and not args.continue_on_error:
                write_summary_csv(summary_path, results)
                return 1

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"finished_at: {datetime.now().isoformat(timespec='seconds')}\n")

    write_summary_csv(summary_path, results)

    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]

    print()
    print("Workflow summary")
    print("----------------")
    print(f"Results: {len(results)}")
    print(f"Failed:  {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Log:     {log_path}")
    print(f"Summary: {summary_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
